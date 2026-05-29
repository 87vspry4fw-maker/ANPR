import cv2
from pathlib import Path

ScriptDir = Path(__file__).resolve().parent
InputDir = ScriptDir / "RealPlates"
OutputDir = ScriptDir / "Debugging"

# --- tunable parameters (all scaled to the normalized height) ---
TargetH = 128                          # normalize every plate to this height first
BandFraction = 0.16
BGKernel = ((TargetH // 2) | 1,) * 2  # background-estimate kernel (~65) for illumination flattening
BlockSize = (TargetH // 3) | 1        # adaptive-threshold block size (odd, ~43)
AdaptC = 15                            # adaptive-threshold constant; raise to drop more noise
MorphKernel = (3, 3)                   # close/open cleanup kernel
MinHFrac = 0.4                        # keep contours taller than 40% of plate height
MinAR, MaxAR = 0.17, 1.0


def Preprocess(ImagePath, HasBand):
    Grey = cv2.imread(str(ImagePath), cv2.IMREAD_GRAYSCALE)
    if HasBand:
        Cut = int(Grey.shape[1] * BandFraction)
        Grey = Grey[:, Cut:]
    # 4. normalize size so the kernels below mean the same thing on every plate
    Scale = TargetH / Grey.shape[0]
    Grey = cv2.resize(Grey, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_AREA)
    # 2. flatten illumination: estimate background and divide it out
    BG = cv2.morphologyEx(Grey, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BGKernel))
    Flat = cv2.divide(Grey, BG, scale=255)
    Flat = cv2.medianBlur(Flat, 3)   # kill salt-and-pepper speckle before thresholding
    return Grey, Flat


def Binarise(Flat):
    # 1. adaptive (local) threshold instead of global Otsu
    Binary = cv2.adaptiveThreshold(Flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, BlockSize, AdaptC)
    # 3. morphology: close re-joins fragmented characters, open removes speckle
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MorphKernel)
    Binary = cv2.morphologyEx(Binary, cv2.MORPH_CLOSE, k)
    Binary = cv2.morphologyEx(Binary, cv2.MORPH_OPEN, k)
    return Binary


def MergeAndSplit(Boxes):
    # 5. use the known structure: characters are ~equal width. Fold fragments together,
    #    cut over-wide blobs (touching characters) apart, using the median width as reference.
    if not Boxes:
        return Boxes
    Boxes = sorted(Boxes, key=lambda b: b[0])
    MedianWidth = sorted(b[2] for b in Boxes)[len(Boxes) // 2]   # median width ≈ one character

    # MERGE: a tiny/negative gap whose union is still ~1 char wide = a fragmented character
    Merged = [list(Boxes[0])]
    for x, y, w, h in Boxes[1:]:
        px, py, pw, ph = Merged[-1]
        Gap = x - (px + pw)
        UnionW = max(px + pw, x + w) - min(px, x)
        if Gap < 0.15 * MedianWidth and UnionW <= 1.2 * MedianWidth:   # union ~1 char wide = a fragment, not two chars
            nx, ny = min(px, x), min(py, y)
            Merged[-1] = [nx, ny, max(px + pw, x + w) - nx, max(py + ph, y + h) - ny]
        else:
            Merged.append([x, y, w, h])

    # SPLIT: a box much wider than one char = touching characters → cut into equal slices
    Out = []
    for x, y, w, h in Merged:
        n = round(w / MedianWidth)
        if n >= 2 and w > 1.6 * MedianWidth:
            Piece = w // n
            Out.extend((x + k * Piece, y, Piece, h) for k in range(n))
        else:
            Out.append((x, y, w, h))
    return Out


def SegmentV2(ImagePath, HasBand=False, Debug=False):
    Grey, Flat = Preprocess(ImagePath, HasBand)
    Binary = Binarise(Flat)
    PlateH = Grey.shape[0]
    Contours, _ = cv2.findContours(Binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    Boxes = []
    Vis = cv2.cvtColor(Grey, cv2.COLOR_GRAY2BGR) if Debug else None
    for c in Contours:
        x, y, w, h = cv2.boundingRect(c)
        AR = w / h
        Keep = h > MinHFrac * PlateH and MinAR < AR < MaxAR
        if Keep:
            Boxes.append((x, y, w, h))
        elif Debug:
            cv2.rectangle(Vis, (x, y), (x + w, y + h), (0, 0, 255), 1)   # dropped contour = red

    Boxes = sorted(MergeAndSplit(Boxes), key=lambda b: b[0])
    if Debug:
        for x, y, w, h in Boxes:
            cv2.rectangle(Vis, (x, y), (x + w, y + h), (0, 200, 0), 2)   # final box = green
    Crops = [Grey[y:y + h, x:x + w] for (x, y, w, h) in Boxes]   # crop from PLAIN grey (matches training look)
    if Debug:
        return Crops, Binary, Vis
    return Crops


if __name__ == "__main__":
    OutputDir.mkdir(exist_ok=True)
    ok = 0
    for p in sorted(InputDir.glob("*.png")):
        Plate = p.stem.upper()
        Crops, Binary, Vis = SegmentV2(p, HasBand=False, Debug=True)
        cv2.imwrite(str(OutputDir / f"{Plate}-Binary.png"), Binary)
        cv2.imwrite(str(OutputDir / f"{Plate}-Boxes.png"), Vis)
        Match = len(Crops) == len(Plate)
        ok += Match
        print(f"[{'OK' if Match else 'XX'}] {Plate}: {len(Crops)}/{len(Plate)} chars")
    print(f"\n{ok}/17 plates at the correct character count")
    print(f"Annotated images (binary + boxes) in {OutputDir.name}/")
