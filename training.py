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
    RunningLoss = 0.0
    for x, y in TrainLoader:
        x, y = x.to(Device), y.to(Device)
        Logits = Model(x)
        Loss = LossFN(Logits, y)
        Optim.zero_grad()
        Loss.backward()
        Optim.step()
        RunningLoss += Loss.item() * y.size(0)
    TrainLoss = RunningLoss / len(TrainDataSet)
    
    Model.eval()
    Correct = Total = 0
    with torch.no_grad():
        for x, y in ValLoader:
            x, y = x.to(Device), y.to(Device)
            Prediction = Model(x).argmax(dim=1)
            Correct += (Prediction == y).sum().item()
            Total += y.size(0)
    print(f"Epoch {Epoch+1}: Train Loss = {TrainLoss:.4f}, Val Accuracy = {Correct/Total:.3f}")


TestDataSet = ImageFolder("RealCharacters", transform=BuildTransform(Training=False))
ReMap = {TestDataSet.class_to_idx[c]: TrainFull.class_to_idx[c] for c in TestDataSet.classes}
TestDataSet.samples = [(path, ReMap[label]) for path, label in TestDataSet.samples]
TestDataSet.targets = [label for _, label in TestDataSet.samples]

TestLoader = DataLoader(TestDataSet, batch_size=64, shuffle=False, num_workers=0)


from collections import defaultdict

Model.eval()
Correct = Total = 0
PerClassTotal = defaultdict(int)
PerClassCorrect = defaultdict(int)
IdxToClass = {v: k for k, v in TrainFull.class_to_idx.items()}
with torch.no_grad():
    for x, y in TestLoader:
        x, y = x.to(Device), y.to(Device)
        Prediction = Model(x).argmax(dim=1)
        Correct += (Prediction == y).sum().item()
        Total += y.size(0)
        for true, pred in zip(y.tolist(), Prediction.tolist()):
            PerClassTotal[true] += 1
            if true == pred:
                PerClassCorrect[true] += 1
print(f"\nReal-plate test accuracy overall = {Correct/Total:.3f}  ({Correct}/{Total})")
Right = [(IdxToClass[c], PerClassCorrect[c], PerClassTotal[c])
         for c in sorted(PerClassTotal) if PerClassCorrect[c] < PerClassTotal[c]]
if Right:
    print("Characters it got right:")
    for Char, CharCorrect, CharTotal in Right:
        print(f"  {Char}: {CharCorrect}/{CharTotal}")
else:
    print("All characters were classified correctly!")


torch.save(Model.state_dict(), "CharCNNWeights.pt")
import json; json.dump(TrainFull.classes, open("CharClasses.json", "w"))