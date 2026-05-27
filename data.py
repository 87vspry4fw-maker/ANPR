import os
import cv2
import torchvision.transforms as transforms
import segmentation

segment = segmentation.segment_characters
validate = segmentation.segmentation_is_valid

def data_loader():
    this_file = __file__
    this_file_absolute = os.path.abspath(this_file)
    script_dir = os.path.dirname(this_file_absolute)
    data_dir = os.path.join(script_dir, 'data')
    return data_dir

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
        crops = segment(plate_path)
        print(validate(crops, plate_string))
        print(filename)
        print(plate_string)

