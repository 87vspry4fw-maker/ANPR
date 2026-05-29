import os
import cv2
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import segmentation

Segment = segmentation.segment_characters
Validate = segmentation.segmentation_is_valid

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

def create_folder(crop, plate_char, index, plate_string, script_dir):
    CharDir = os.path.join(script_dir, "Characters", plate_char)
    os.makedirs(CharDir, exist_ok=True)
    NewFilename = f"{plate_string}_{index}.png"
    NewPath = os.path.join(CharDir, NewFilename)
    cv2.imwrite(NewPath, crop)

def split_dataset():
    ScriptDir = os.path.dirname(os.path.abspath(__file__))
    WhiteDir = os.path.join(ScriptDir, "UKLicencePlateDataset", "whiteplate_normal")
    YellowDir = os.path.join(ScriptDir, "UKLicencePlateDataset", "yellowplate_normal")
    PlateDirs = [WhiteDir, YellowDir]

    for PlatesDir in PlateDirs:
        for Filename in os.listdir(PlatesDir):
            if not Filename.lower().endswith(".png"):
                continue
            PlatePath = os.path.join(PlatesDir, Filename)
            PlateString = os.path.splitext(Filename)[0]
            print("PATH:", PlatePath)
            Crops = Segment(PlatePath, HasBand=True)
            Correct = Validate(Crops, PlateString)

            if Correct:
                for i in range(len(Crops)):
                    Crop = Crops[i]
                    PlateChar = PlateString[i]
                    create_folder(Crop, PlateChar, i, PlateString, ScriptDir)


ScriptDir = Path(__file__).resolve().parent
CharDir = ScriptDir / "Characters"

ClassNames = sorted([d.name for d in CharDir.iterdir() if d.is_dir()])
ClassToIndex = {name: idx for idx, name in enumerate(ClassNames)}
IndexToClass = {i: name for name, i in ClassToIndex.items()}

class CharDataset(Dataset):
    def __init__(self, Root, ClassToIndex, Transform=None):
        self.Transform = Transform
        self.Samples = []
        for Name, Index in ClassToIndex.items():
            ClassDir = Root / Name
            for ImagePath in sorted(ClassDir.iterdir()):
                if ImagePath.is_file():
                    self.Samples.append((ImagePath, Index))
    
    def __len__(self):
        return len(self.Samples)
    
    def __getitem__(self, i):
        ImagePath, Label = self.Samples[i]
        image = Image.open(ImagePath).convert("L")
        if self.Transform:
            image = self.Transform(image)
        return image, Label