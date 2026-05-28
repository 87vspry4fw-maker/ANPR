import sys
import json
from pathlib import Path

import torch
from PIL import Image

from model import CharCNN
from data import build_transform
from segmentation_v2 import segment_v2
from plate_format import constrained_index, DIGIT_POSITIONS

script_dir = Path(__file__).resolve().parent
weights = script_dir / "char_cnn.pt"
classes = script_dir / "classes.json"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if not weights.exists() or not classes.exists():
    raise SystemExit("Missing char_cnn.pt / classes.json. Train and save the model first.")

classes = json.load(open(classes))
model = CharCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(weights, map_location=device))
model.eval()

transform = build_transform(training=False)


def predict_plate(image_path):
    crops = segment_v2(str(image_path), has_band=False)    # real photos: no GB band to cut
    if not crops:
        print("No characters found. Is the image cropped to just the plate (no GB band)?")
        return ""

    use_format = len(crops) == 7   # the LL DD LLL rule only applies to standard 7-char plates
    raw_chars, final_chars, confs = [], [], []
    with torch.no_grad():
        for i, crop in enumerate(crops):
            # .convert("RGB") matches how ImageFolder loaded crops during training
            x = transform(Image.fromarray(crop).convert("RGB")).unsqueeze(0).to(device)  # [1,1,64,64]
            logits = model(x)[0].cpu()                  # [num_classes]
            probs = torch.softmax(logits, dim=0)
            raw_chars.append(classes[int(logits.argmax())])

            if use_format:
                idx = constrained_index(logits.tolist(), classes, i)
            else:
                idx = int(logits.argmax())
            final_chars.append(classes[idx])
            confs.append(float(probs[idx]))

    raw_plate = "".join(raw_chars)
    final_plate = "".join(final_chars)

    print(f"Raw prediction:   {raw_plate}")
    if use_format:
        print(f"Format-corrected: {final_plate}   (UK LL DD LLL)")
        for i, (r, f) in enumerate(zip(raw_chars, final_chars)):
            if r != f:
                slot = "digit" if i in DIGIT_POSITIONS else "letter"
                print(f"  pos {i}: {r} -> {f}  (must be a {slot})")
    else:
        print(f"(segmented {len(crops)} characters, not 7 - format rule skipped)")
    print("Per-character confidence:")
    for c, p in zip(final_chars, confs):
        print(f"  {c}  {p:.2f}")
    return final_plate


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predictions.py <path-to-plate-image>")
    predict_plate(sys.argv[1])
