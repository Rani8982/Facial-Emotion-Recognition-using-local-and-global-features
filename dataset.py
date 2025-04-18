from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import AutoAugment from your custom implementation
# Assuming it's in the same notebook/file or already defined
# from your_augmentation_module import AutoAugment

# Compose transform with AutoAugment
train_transform = transforms.Compose([
    AutoAugment(),                           # Custom AutoAugment policy
    transforms.Resize((128, 128)),          # Resize image
    transforms.ToTensor(),                  # Convert to tensor
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize
])

# For test data, no heavy augmentation — just resize and normalize
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root='/kaggle/input/raf-db-dataset/DATASET/train',
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root='/kaggle/input/raf-db-dataset/DATASET/test',
    transform=test_transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Sanity check
print(f"Train set: {len(train_dataset)} images")
print(f"Test set: {len(test_dataset)} images")
