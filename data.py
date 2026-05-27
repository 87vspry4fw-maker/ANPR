import os
import cv2
import torchvision.transforms as transforms
import segmentation

segment = segmentation.segment_characters
validate = segmentation.segmentation_is_valid

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

def create_folder(crop, plate_char, index):
    char_dir = os.path.join(script_dir, "characters", plate_char)
    os.makedirs(char_dir, exist_ok=True)
    new_filename = f"{plate_char}_{index}.png"
    new_path = os.path.join(char_dir, new_filename)
    cv2.imwrite(new_path, crop)

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
        correct = validate(crops, plate_string)

        if correct:
            for i in range(len(crops)):
                crop = crops[i]
                plate_char = plate_string[i]
                create_folder(crop, plate_char, i)