import os
import torch
import segmentation

segment = segmentation.segment_characters
segment_validation = segmentation.segmentation_is_valid


this_file = __file__
this_file_absolute = os.path.abspath(this_file)
script_dir = os.path.dirname(this_file_absolute)
data_dir = os.path.join(script_dir, 'data')

def build_transform(training=True):
    return torch.torchvision.transforms.Compose([
        torch.torchvision.transforms.Grayscale(),
        torch.torchvision.transforms.Resize((64, 64)),
        augments,
        torch.torchvision.transforms.ToTensor(),
        torch.torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    if training:
        augments = torch.torchvision.transforms.Compose([
            torch.torchvision.transforms.RandomRotation(10),
            torch.torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.1)),
            torch.torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    else:
        augments = None
    