import os
import torchvision.transforms as transforms
import segmentation

segment = segmentation.segment_characters
segment_validation = segmentation.segmentation_is_valid

def data_loader():
    this_file = __file__
    this_file_absolute = os.path.abspath(this_file)
    script_dir = os.path.dirname(this_file_absolute)
    data_dir = os.path.join(script_dir, 'data')

def build_transform(training=True):
    if training:
        augments = [
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        augments = []
    
    image_steps = [
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
    ]    
    tensor_steps = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

    return transforms.Compose(image_steps + augments + tensor_steps)
