"""UK number-plate format rules (pure logic — no model, no images, no I/O).

Current UK plates follow the layout "LL DD LLL": the two characters at positions
2 and 3 are always digits, and positions 0, 1, 4, 5, 6 are always letters. Knowing
a character's position therefore tells us whether it *must* be a letter or a digit,
which lets us correct cross-category misreads (e.g. a digit-slot "O" must really be
a "0", a letter-slot "0" must really be an "O").
"""

DigitPositions = {2, 3}   # 0-indexed positions that must be digits in "LL DD LLL"


def PositionWantsDigit(Position):
    return Position in DigitPositions


def ConstrainedIndex(Scores, Classes, Position):
    """Pick the best class index that is *allowed* at this plate position.

    scores:   per-class scores (e.g. model logits), one float per class.
    classes:  class names aligned with scores, e.g. ['0','1',...,'A','B',...].
    position: 0-indexed character position on the plate.

    Returns the index of the highest-scoring class whose type — letter vs digit —
    matches what the position requires. Falls back to the plain best score if no
    class qualifies.
    """
    WantDigit = PositionWantsDigit(Position)
    BestIndex, BestScore = None, float("-inf")
    for i, (Name, Score) in enumerate(zip(Classes, Scores)):
        Allowed = Name.isdigit() if WantDigit else Name.isalpha()
        if Allowed and Score > BestScore:
            BestIndex, BestScore = i, Score
    if BestIndex is None:
        BestIndex = max(range(len(Scores)), key=lambda i: Scores[i])
    return BestIndex
