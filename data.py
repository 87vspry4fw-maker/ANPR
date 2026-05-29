import os
import cv2
import random
import torchvision.transforms as transforms
from pathlib import Path
import segmentation
from segmentationV2 import SegmentV2

Segment = segmentation.SegmentCharacters
Validate = segmentation.SegmentationIsValid

def BuildTransform(Training):
    if Training:
        Augments = [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        Augments = []
    
    ImageSteps = [
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
    ] 
    if Training:   
        TensorSteps = [
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), value=0),
            transforms.Normalize((0.5,), (0.5,))
        ]
    else:
        TensorSteps = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

    return transforms.Compose(ImageSteps + Augments + TensorSteps)

def CreateFolder(Crop, PlateChar, Index, PlateString, ScriptDir, DestFolder="Characters"):
    CharDir = os.path.join(ScriptDir, DestFolder, PlateChar)
    os.makedirs(CharDir, exist_ok=True)
    NewFilename = f"{PlateString}_{Index}.png"
    NewPath = os.path.join(CharDir, NewFilename)
    cv2.imwrite(NewPath, Crop)

def SplitDataset():
    ScriptDir = os.path.dirname(os.path.abspath(__file__))
    WhiteDir = os.path.join(ScriptDir, "UKLicencePlateDataset", "whiteplate_normal")
    YellowDir = os.path.join(ScriptDir, "UKLicencePlateDataset", "yellowplate_normal")
    PlateDirs = [WhiteDir, YellowDir]

    Passed = Discarded = TotalCrops = 0
    PerFolderLimit = 2500

    for PlatesDir in PlateDirs:
        Files = [f for f in os.listdir(PlatesDir) if f.lower().endswith(".png")]
        if PerFolderLimit:
            random.seed(0)                                   # reproducible subset
            Files = random.sample(Files, min(PerFolderLimit, len(Files)))
        for Filename in Files:
            PlatePath = os.path.join(PlatesDir, Filename)
            PlateString = os.path.splitext(Filename)[0]
            Crops = Segment(PlatePath, HasBand=True)
            Correct = Validate(Crops, PlateString)

            if Correct:
                Passed += 1
                for i in range(len(Crops)):
                    Crop = Crops[i]
                    PlateChar = PlateString[i]
                    CreateFolder(Crop, PlateChar, i, PlateString, ScriptDir)
                    TotalCrops += 1
            else:
                Discarded += 1

    Labels = sorted(d.name for d in (Path(ScriptDir) / "Characters").iterdir() if d.is_dir())
    print(f"\nPlates pass count-check: {Passed}")
    print(f"Plates discarded: {Discarded}")
    print(f"Total character Images: {TotalCrops}")
    print(f"Labels ({len(Labels)}): {Labels}")

def BuildRealCharacters():
    ScriptDir = os.path.dirname(os.path.abspath(__file__))
    RealPlatesDir = os.path.join(ScriptDir, "RealPlates")

    Passed = Discarded = TotalCrops = 0
    for Filename in os.listdir(RealPlatesDir):
        if not Filename.lower().endswith(".png"):
            continue
        PlatePath = os.path.join(RealPlatesDir, Filename)
        PlateString = os.path.splitext(Filename)[0].upper()     # "GU73CFA"
        Crops = SegmentV2(PlatePath, HasBand=False)              # real-photo segmenter; band already trimmed
        if Validate(Crops, PlateString):                        # same count-check as synthetic
            Passed += 1
            for i, Crop in enumerate(Crops):
                CreateFolder(Crop, PlateString[i], i, PlateString, ScriptDir,
                              DestFolder="RealCharacters")
                TotalCrops += 1
        else:
            Discarded += 1

    print(f"\nReal plates passed count-check: {Passed}")
    print(f"Real plates discarded (count mismatch): {Discarded}")
    print(f"Real character crops written: {TotalCrops}")

  
if __name__ == "__main__":
    import sys
    Mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if Mode == "synthetic":
        SplitDataset()
    elif Mode == "real":
        BuildRealCharacters()
    else:
        raise SystemExit("Usage: python data.py [synthetic|real]")