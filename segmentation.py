import cv2

def segment_characters(image_path, has_band=True):
    plate = cv2.imread(image_path)
    grey = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    width = grey.shape[1]
    if has_band:
        band_fraction = 0.16
        cutoff = int(width * band_fraction)
        grey = grey[:, cutoff:]
    plate_height = grey.shape[0]
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if h > 0.4 * plate_height and 0.17 < aspect_ratio < 1.0:  # Filter out small contours that are unlikely to be characters
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda box: box[0])
    char_images = []
    for (x, y, w, h) in boxes:
        crop = grey[y:y+h, x:x+w]
        char_images.append(crop)
    return char_images

def segmentation_is_valid(char_images, plate_string):
    expected = len(plate_string.replace(" ", ""))
    return len(char_images) == expected