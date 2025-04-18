# Rebuild the model exactly the same way
model = FERNet(num_classes=7, num_regions=4)
model = torch.nn.DataParallel(model).to(device)

# Load best weights
model.load_state_dict(torch.load("/kaggle/working/fernet_kmu_best.pth",weights_only=True))
model.eval()
