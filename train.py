import torch
import torch.nn as nn
import torch.optim as optim
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume you already imported FERNet and AutoAugment
# from fernet_model import FERNet
# from auto_augment import AutoAugment
# from utils import count_parameters (optional)

# Model
model = FERNet(num_classes=7, num_regions=4)
model = torch.nn.DataParallel(model).to(device)
print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=0.001, weight_decay=4e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

# Accuracy Metric
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = [correct[:k].reshape(-1).float().sum(0).mul_(100.0 / target.size(0)) for k in topk]
    return res

# Train One Epoch
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, images)

        loss = sum(criterion(outputs[:, :, j], labels) for j in range(5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_pred = torch.mean(outputs, dim=2)
        acc1 = accuracy(avg_pred, labels, topk=(1,))
        correct += acc1[0].item()

        if i % 50 == 0:
            print(f"[Epoch {epoch}][Batch {i}] Loss: {loss.item():.4f} Acc@1: {acc1[0].item():.2f}%")

    avg_loss = running_loss / len(train_loader)
    avg_acc = correct / len(train_loader)
    print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Train Acc@1: {avg_acc:.2f}%")

# Validation
def validate(epoch):
    model.eval()
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, images)
            loss = sum(criterion(outputs[:, :, j], labels) for j in range(5))
            val_loss += loss.item()

            avg_pred = torch.mean(outputs, dim=2)
            acc1 = accuracy(avg_pred, labels, topk=(1,))
            correct += acc1[0].item()

    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = correct / len(test_loader)
    print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f} | Val Acc@1: {avg_val_acc:.2f}%")
    scheduler.step(avg_val_loss)
    return avg_val_acc

# Main Training Loop
best_acc = 0.0
for epoch in range(50):
    train_one_epoch(epoch)
    val_acc = validate(epoch)

    if val_acc > best_acc:
        best_acc = val_acc
        print("Saving best model...")
        torch.save(model.state_dict(), "fernet_kmu_best.pth")
