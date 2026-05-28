import model
import data

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

build_transform = data.build_transform

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


train_full = ImageFolder("characters", transform = build_transform(training=True))
val_full = ImageFolder("characters", transform=build_transform(training=False))


n = len(train_full)
perm = torch.randperm(n)
n_val = n // 10
val_ds = Subset(val_full, perm[:n_val].tolist())
train_ds = Subset(train_full, perm[n_val:].tolist())

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

model = model.CharCNN(num_classes=len(train_full.classes)).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch+1}: Val Accuracy = {correct/total:.3f}")


test_ds = ImageFolder("real_characters", transform=build_transform(training=False))
remap = {test_ds.class_to_idx[c]: train_full.class_to_idx[c] for c in test_ds.classes}
test_ds.samples = [(path, remap[label]) for path, label in test_ds.samples]
test_ds.targets = [label for _, label in test_ds.samples]

test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"\nReal-plate test accuracy = {correct/total:.3f}  ({correct}/{total})")

torch.save(model.state_dict(), "char_cnn.pt")
import json; json.dump(train_full.classes, open("classes.json", "w"))