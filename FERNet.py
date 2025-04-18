from torchsummary import summary

model = FERNet().to('cuda:0')

# Input shape should match your model (1-channel 128x128)
summary(model, input_size=[(1, 128, 128), (1, 128, 128)])
