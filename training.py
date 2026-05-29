import model
import data

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

BuildTransform = data.BuildTransform

Device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


TrainFull = ImageFolder("Characters", transform = BuildTransform(Training=True))
ValFull = ImageFolder("Characters", transform=BuildTransform(Training=False))


n = len(TrainFull)
Perm = torch.randperm(n)
nVal = n // 10
ValDataSet = Subset(ValFull, Perm[:nVal].tolist())
TrainDataSet = Subset(TrainFull, Perm[nVal:].tolist())

TrainLoader = DataLoader(TrainDataSet, batch_size=64, shuffle=True, num_workers=0)
ValLoader = DataLoader(ValDataSet, batch_size=64, shuffle=False, num_workers=0)

Model = model.CharCNN(NumClasses=len(TrainFull.classes)).to(Device)
Optim = torch.optim.Adam(Model.parameters(), lr=1e-3)
LossFN = torch.nn.CrossEntropyLoss()

for Epoch in range(50):
    Model.train()
    for x, y in TrainLoader:
        x, y = x.to(Device), y.to(Device)
        Logits = Model(x)
        Loss = LossFN(Logits, y)
        Optim.zero_grad()
        Loss.backward()
        Optim.step()
    
    Model.eval()
    Correct = Total = 0
    with torch.no_grad():
        for x, y in ValLoader:
            x, y = x.to(Device), y.to(Device)
            Prediction = Model(x).argmax(dim=1)
            Correct += (Prediction == y).sum().item()
            Total += y.size(0)
    print(f"Epoch {Epoch+1}: Val Accuracy = {Correct/Total:.3f}")


TestDataSet = ImageFolder("RealCharacters", transform=BuildTransform(Training=False))
ReMap = {TestDataSet.class_to_idx[c]: TrainFull.class_to_idx[c] for c in TestDataSet.classes}
TestDataSet.samples = [(path, ReMap[label]) for path, label in TestDataSet.samples]
TestDataSet.targets = [label for _, label in TestDataSet.samples]

TestLoader = DataLoader(TestDataSet, batch_size=64, shuffle=False, num_workers=0)

Model.eval()
Correct = Total = 0
with torch.no_grad():
    for x, y in TestLoader:
        x, y = x.to(Device), y.to(Device)
        Prediction = Model(x).argmax(dim=1)
        Correct += (Prediction == y).sum().item()
        Total += y.size(0)
print(f"\nReal-plate test accuracy = {Correct/Total:.3f}  ({Correct}/{Total})")

torch.save(Model.state_dict(), "CharCNNWeights.pt")
import json; json.dump(TrainFull.classes, open("CharClasses.json", "w"))