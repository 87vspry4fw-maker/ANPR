"""UK number-plate format rules (pure logic — no model, no images, no I/O).

Current UK plates follow the layout "LL DD LLL": the two characters at positions
2 and 3 are always digits, and positions 0, 1, 4, 5, 6 are always letters. Knowing
a character's position therefore tells us whether it *must* be a letter or a digit,
which lets us correct cross-category misreads (e.g. a digit-slot "O" must really be
a "0", a letter-slot "0" must really be an "O").
"""

DIGIT_POSITIONS = {2, 3}   # 0-indexed positions that must be digits in "LL DD LLL"


def position_wants_digit(position):
    return position in DIGIT_POSITIONS


def constrained_index(scores, classes, position):
    """Pick the best class index that is *allowed* at this plate position.

    scores:   per-class scores (e.g. model logits), one float per class.
    classes:  class names aligned with scores, e.g. ['0','1',...,'A','B',...].
    position: 0-indexed character position on the plate.

    Returns the index of the highest-scoring class whose type — letter vs digit —
    matches what the position requires. Falls back to the plain best score if no
    class qualifies.
    """
    want_digit = position_wants_digit(position)
    best_idx, best_score = None, float("-inf")
    for i, (name, score) in enumerate(zip(classes, scores)):
        allowed = name.isdigit() if want_digit else name.isalpha()
        if allowed and score > best_score:
            best_idx, best_score = i, score
    if best_idx is None:
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx
