'''
Aum Sri Sai Ram
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

from thop import profile  
from thop import clever_format


class eca_layer(nn.Module):
   
    def __init__(self, channel, k_size = 5):
        super(eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
     
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        #print(y.size())

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class FERNet(nn.Module):
    def __init__(self,num_classes=7, num_regions=4):
        super(FERNet, self).__init__()
        
        self.base= LightCNN_29Layers_v2(num_classes=7)
        
        checkpoint = torch.load('/kaggle/input/lightcnnn/pytorch/default/1/LightCNN_29Layers_V2_checkpoint.pth.tar',weights_only=True)
        pretrained_state_dict = dict(checkpoint['state_dict'])        
        keys = list(pretrained_state_dict.keys())        
        [pretrained_state_dict.pop(key) for key in keys if ('3' in key or '4' in key or 'fc' in key)]   # for light cnn 29 v2    
        new_dict = dict(zip(list(self.base.state_dict().keys()), list(pretrained_state_dict.values())))
        self.base.load_state_dict(new_dict, strict = True)
        
        
        
        '''comment below 2 lines for freezing basemode parameters
        for param in self.base.parameters():
            param.requires_grad = False
        '''
        
        self.num_regions = num_regions
        
        self.eca = nn.ModuleList([eca_layer(192,3) for i in range(num_regions+1)])  
        
        self.globalavgpool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)  for i in range(num_regions+1)])                                 
        self.region_net = nn.ModuleList([ nn.Sequential( nn.Linear(192,256), nn.ReLU()) for i in range(num_regions+1)])       
        
        self.classifiers =  nn.ModuleList([ nn.Linear(256+256, num_classes, bias = False) for i in range(num_regions+1)])
        self.s = 30.0
        
    def forward(self, x1, x2):
        print("Input to base:" ,x1.shape)
        
        x1 = self.base(x1)
        print("Output from base:", x1.shape)
        
        
        x2 = self.base(x2) 
        
        
        
        bs, c, w, h = x1.size()
        
        region_size = int(x1.size(2) / (self.num_regions/2) ) 
        
        patches1 = x1.unfold(2, region_size, region_size).unfold(3,region_size,region_size)  
        
        patches1 = patches1.contiguous().view(bs, c, -1, region_size, region_size).permute(0,2,1,3,4)
        patches2 = x2.unfold(2, region_size, region_size).unfold(3,region_size,region_size)         
        patches2 = patches2.contiguous().view(bs, c, -1, region_size, region_size).permute(0,2,1,3,4)
        print("Patch shape x1:", patches1.shape)
        print("Patch shape x2:", patches2.shape)
        
                 
        output = []
        for i in range(int(self.num_regions)):
            f1 = patches1[:,i,:,:,:] 
            print(f"Patch {i} shape before ECA:", f1.shape)
            
            f1 = self.eca[i](f1) 
            
            print(f"Patch {i} after ECA:", f1.shape)
            f1 = self.globalavgpool[i](f1).squeeze(3).squeeze(2)
            print(f"Patch {i} after GAP:", f1.shape)
            f1 =  self.region_net[i](f1)
            print(f"Patch {i} after Linear + ReLU:", f1.shape)
            
            f2 = patches2[:,i,:,:,:] 
            print(f"Patch {i} shape before ECA:", f2.shape)
            
            f2 = self.eca[i](f2) 
            print(f"Patch {i} after ECA:", f2.shape)
            f2 = self.globalavgpool[i](f2).squeeze(3).squeeze(2)
            print(f"Patch {i} after GAP:", f2.shape)
            f2 =  self.region_net[i](f2)
            (f"Patch {i} after Linear + ReLU:", f2.shape)
            f = torch.cat((f1,f2),dim=1) 
            
            for W in self.classifiers[i].parameters():
                W = F.normalize(W, p=2, dim=1)         
            f  = F.normalize(f, p=2, dim=1)
            
            f = self.s * self.classifiers[i](f)   
            output.append(f)      

        
        output_stacked = torch.stack(output, dim = 2)
        
       
        
        y1 = self.globalavgpool[4](self.eca[4](x1)).squeeze(3).squeeze(2)
        #y1 = self.globalavgpool[4](x1).squeeze(3).squeeze(2)     
        y1 = self.region_net[4](y1)
        print("Y1 PRINT",y1.size())
        
        y2 = self.globalavgpool[4](self.eca[4](x2)).squeeze(3).squeeze(2)
        #y2 = self.globalavgpool[4](x2).squeeze(3).squeeze(2)      
        y2 = self.region_net[4](y2)
        print("PRINT",y2.size())
        
        for W in self.classifiers[4].parameters():
                W = F.normalize(W, p=2, dim=1)
                
        y = torch.cat((y1,y2),dim=1) 
        print("Y=",y.shape)
        y  = F.normalize(y, p=2, dim=1)
        #y = torch.cat((y1,y2),dim=1)
        
        output_global = self.classifiers[4](y).unsqueeze(2)
        output_final = torch.cat((output_stacked,output_global),dim=2)
        
        return output_final
       
       
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
   model = model.to('cuda:0')
   print(count_parameters(model))
   x = torch.rand(1,  1, 128, 128).to('cuda:0')
   #x = torch.rand(1,  192, 16, 16).to('cuda:0')
   y = model(x, x) 
   #print(x.size())
   #print(y.size())
   macs, params = profile(model, inputs=(x,x ))
   macs, params = clever_format([macs, params], "%.3f")
   print(macs,params)
   #print(y.size()) 
  
   for name, param in model.named_parameters():
       print(name, param.size())  
      
