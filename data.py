import os
import cv2
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation

segment = segmentation.segment_characters
validate = segmentation.segmentation_is_valid

def build_transform(training):
    if training:
        augments = [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        augments = []
    
    image_steps = [
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
    ] 
    if training:   
        tensor_steps = [
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), value=0),
            transforms.Normalize((0.5,), (0.5,))
        ]
    else:
        tensor_steps = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

    return transforms.Compose(image_steps + augments + tensor_steps)

def create_folder(crop, plate_char, index, plate_string, script_dir):
    char_dir = os.path.join(script_dir, "characters", plate_char)
    os.makedirs(char_dir, exist_ok=True)
    new_filename = f"{plate_string}_{index}.png"
    new_path = os.path.join(char_dir, new_filename)
    cv2.imwrite(new_path, crop)

def split_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    white_dir = os.path.join(script_dir, "UKLicencePlateDataset", "whiteplate_normal")
    yellow_dir = os.path.join(script_dir, "UKLicencePlateDataset", "yellowplate_normal")
    plate_dirs = [white_dir, yellow_dir]

    for plates_dir in plate_dirs:
        for filename in os.listdir(plates_dir):
            if not filename.lower().endswith(".png"):
                continue
            plate_path = os.path.join(plates_dir, filename)
            plate_string = os.path.splitext(filename)[0]
            print("PATH:", plate_path)
            crops = segment(plate_path, has_band=True)
            correct = validate(crops, plate_string)

            if correct:
                for i in range(len(crops)):
                    crop = crops[i]
                    plate_char = plate_string[i]
                    create_folder(crop, plate_char, i, plate_string, script_dir)


script_dir = Path(__file__).resolve().parent
char_dir = script_dir / "characters"

class_names = sorted([d.name for d in char_dir.iterdir() if d.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {i: name for name, i in class_to_idx.items()}

class CharDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.transform = transform
        self.samples = []
        for name, idx in class_to_idx.items():
            class_dir = root / name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file():
                    self.samples.append((img_path, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        img_path, label = self.samples[i]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label