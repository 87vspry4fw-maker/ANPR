import sys
import json
from pathlib import Path

import torch
from PIL import Image

from model import CharCNN
from data import BuildTransform
from segmentation_v2 import SegmentV2
from plate_format import ConstrainedIndex, DigitPositions

ScriptDir = Path(__file__).resolve().parent
Weights = ScriptDir / "CharCNNWeights.pt"
Classes = ScriptDir / "CharClasses.json"

Device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if not Weights.exists() or not Classes.exists():
    raise SystemExit("Missing CharCNNWeights.pt / CharClasses.json. Train and save the model first.")

Classes = json.load(open(Classes))
Model = CharCNN(NumClasses=len(Classes)).to(Device)
Model.load_state_dict(torch.load(Weights, map_location=Device))
Model.eval()

Transform = BuildTransform(Training=False)


def PredictPlate(ImagePath):
    Crops = SegmentV2(str(ImagePath), HasBand=False)    # real photos: no GB band to cut
    if not Crops:
        print("No characters found. Is the image cropped to just the plate (no GB band)?")
        return ""

    UseFormat = len(Crops) == 7   # the LL DD LLL rule only applies to standard 7-char plates
    RawChars, FinalChars, Confs = [], [], []
    with torch.no_grad():
        for i, Crop in enumerate(Crops):
            # .convert("RGB") matches how ImageFolder loaded crops during training
            x = Transform(Image.fromarray(Crop).convert("RGB")).unsqueeze(0).to(Device)  # [1,1,64,64]
            Logits = Model(x)[0].cpu()                  # [num_classes]
            Probabilities = torch.softmax(Logits, dim=0)
            RawChars.append(Classes[int(Logits.argmax())])

            if UseFormat:
                Index = ConstrainedIndex(Logits.tolist(), Classes, i)
            else:
                Index = int(Logits.argmax())
            FinalChars.append(Classes[Index])
            Confs.append(float(Probabilities[Index]))

    RawPlates = "".join(RawChars)
    FinalPlates = "".join(FinalChars)

    print(f"Raw prediction:   {RawPlates}")
    if UseFormat:
        print(f"Format-corrected: {FinalPlates}   (UK LL DD LLL)")
        for i, (r, f) in enumerate(zip(RawChars, FinalChars)):
            if r != f:
                Slot = "digit" if i in DigitPositions else "letter"
                print(f"  pos {i}: {r} -> {f}  (must be a {Slot})")
    else:
        print(f"(segmented {len(Crops)} characters, not 7 - format rule skipped)")
    print("Per-character confidence:")
    for c, p in zip(FinalChars, Confs):
        print(f"  {c}  {p:.2f}")
    return FinalPlates


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predictions.py <path-to-plate-image>")
    PredictPlate(sys.argv[1])
