import cv2

def SegmentCharacters(ImagePath, HasBand=True):
    Plate = cv2.imread(ImagePath)
    Grey = cv2.cvtColor(Plate, cv2.COLOR_BGR2GRAY)
    Width = Grey.shape[1]
    if HasBand:
        BandFraction = 0.16
        Cutoff = int(Width * BandFraction)
        Grey = Grey[:, Cutoff:]
    PlateHeight = Grey.shape[0]
    _, Binary = cv2.threshold(Grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    Contours, _ = cv2.findContours(Binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Boxes = []
    for Contour in Contours:
        x, y, w, h = cv2.boundingRect(Contour)
        AspectRatio = w / h
        if h > 0.4 * PlateHeight and 0.17 < AspectRatio < 1.0:  # Filter out small contours that are unlikely to be characters
            Boxes.append((x, y, w, h))
    Boxes.sort(key=lambda box: box[0])
    CharImages = []
    for (x, y, w, h) in Boxes:
        Crop = Grey[y:y+h, x:x+w]
        CharImages.append(Crop)
    return CharImages

def SegmentationIsValid(CharImages, PlateString):
    Expected = len(PlateString.replace(" ", ""))
    return len(CharImages) == Expected