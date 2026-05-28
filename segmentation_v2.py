import cv2
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "RealPlates"
OUTPUT_DIR = SCRIPT_DIR / "debug_v2"

# --- tunable parameters (all scaled to the normalized height) ---
TARGET_H = 128                          # normalize every plate to this height first
BAND_FRACTION = 0.16
BG_KERNEL = ((TARGET_H // 2) | 1,) * 2  # background-estimate kernel (~65) for illumination flattening
BLOCK_SIZE = (TARGET_H // 3) | 1        # adaptive-threshold block size (odd, ~43)
ADAPT_C = 15                            # adaptive-threshold constant; raise to drop more noise
MORPH_KERNEL = (3, 3)                   # close/open cleanup kernel
MIN_H_FRAC = 0.4                        # keep contours taller than 40% of plate height
MIN_AR, MAX_AR = 0.17, 1.0


def _preprocess(image_path, has_band):
    grey = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if has_band:
        cut = int(grey.shape[1] * BAND_FRACTION)
        grey = grey[:, cut:]
    # 4. normalize size so the kernels below mean the same thing on every plate
    scale = TARGET_H / grey.shape[0]
    grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # 2. flatten illumination: estimate background and divide it out
    bg = cv2.morphologyEx(grey, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BG_KERNEL))
    flat = cv2.divide(grey, bg, scale=255)
    flat = cv2.medianBlur(flat, 3)   # kill salt-and-pepper speckle before thresholding
    return grey, flat


def _binarize(flat):
    # 1. adaptive (local) threshold instead of global Otsu
    binary = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, BLOCK_SIZE, ADAPT_C)
    # 3. morphology: close re-joins fragmented characters, open removes speckle
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return binary


def _merge_and_split(boxes):
    # 5. use the known structure: characters are ~equal width. Fold fragments together,
    #    cut over-wide blobs (touching characters) apart, using the median width as reference.
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    mw = sorted(b[2] for b in boxes)[len(boxes) // 2]   # median width ≈ one character

    # MERGE: a tiny/negative gap whose union is still ~1 char wide = a fragmented character
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        px, py, pw, ph = merged[-1]
        gap = x - (px + pw)
        union_w = max(px + pw, x + w) - min(px, x)
        if gap < 0.15 * mw and union_w <= 1.2 * mw:   # union ~1 char wide = a fragment, not two chars
            nx, ny = min(px, x), min(py, y)
            merged[-1] = [nx, ny, max(px + pw, x + w) - nx, max(py + ph, y + h) - ny]
        else:
            merged.append([x, y, w, h])

    # SPLIT: a box much wider than one char = touching characters → cut into equal slices
    out = []
    for x, y, w, h in merged:
        n = round(w / mw)
        if n >= 2 and w > 1.6 * mw:
            piece = w // n
            out.extend((x + k * piece, y, piece, h) for k in range(n))
        else:
            out.append((x, y, w, h))
    return out


def segment_v2(image_path, has_band=False, debug=False):
    grey, flat = _preprocess(image_path, has_band)
    binary = _binarize(flat)
    plate_h = grey.shape[0]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    vis = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR) if debug else None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / h
        keep = h > MIN_H_FRAC * plate_h and MIN_AR < ar < MAX_AR
        if keep:
            boxes.append((x, y, w, h))
        elif debug:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)   # dropped contour = red

    boxes = sorted(_merge_and_split(boxes), key=lambda b: b[0])
    if debug:
        for x, y, w, h in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)   # final box = green
    crops = [grey[y:y + h, x:x + w] for (x, y, w, h) in boxes]   # crop from PLAIN grey (matches training look)
    if debug:
        return crops, binary, vis
    return crops


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    ok = 0
    for p in sorted(INPUT_DIR.glob("*.png")):
        plate = p.stem.upper()
        crops, binary, vis = segment_v2(p, has_band=False, debug=True)
        cv2.imwrite(str(OUTPUT_DIR / f"{plate}_binary.png"), binary)
        cv2.imwrite(str(OUTPUT_DIR / f"{plate}_boxes.png"), vis)
        match = len(crops) == len(plate)
        ok += match
        print(f"[{'OK' if match else 'XX'}] {plate}: {len(crops)}/{len(plate)} chars")
    print(f"\n{ok}/9 plates at the correct character count")
    print(f"Annotated images (binary + boxes) in {OUTPUT_DIR.name}/")
