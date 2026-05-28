"""
What is a unit test?
--------------------
A "unit test" is a small, automated check that one specific piece of code
(usually a single function) behaves the way we expect. Instead of running the
whole program by hand and eyeballing the output, we write code that calls the
function with known inputs and then *asserts* — i.e. demands — that the result
matches what we expect. If the result is wrong, the test fails loudly and
points us at the broken function. The big payoff is that next month, when you
change something seemingly unrelated, these tests will catch it for you
*automatically* if you've accidentally broken old behaviour.

What does this file demonstrate?
--------------------------------
This file tests the function `segmentation_is_valid` from segmentation.py.
That function is a good first target because it's "pure": it takes simple
inputs (a list and a string), does no file I/O, and returns a single boolean.
There are exactly two tests below — one happy-path test (correct input,
expected output) and one error-case test (bad input, expected exception).
That's the whole file. Keeping it small on purpose; you'll add more later.

How to run these tests:
    pip install -r requirements-dev.txt
    pytest -v
"""

# We import the function we want to test. pytest.ini tells pytest that the
# repo root is on the import path, so `import segmentation` works even though
# this file lives in tests/.
import segmentation

# `pytest` is the testing framework. We import it here only because the
# second test uses `pytest.raises` to check for an expected error.
import pytest


def test_segmentation_is_valid_returns_true_when_counts_match():
    # ---- Arrange ----
    # Set up the inputs the function needs. The "Arrange / Act / Assert"
    # pattern is a common way to structure a test so it's easy to read.
    #
    # Here we pretend segmentation already ran and produced 7 character
    # images. We don't need real images — the function only cares about the
    # *length* of the list, so any 7 placeholder values work.
    fake_char_images = ["img1", "img2", "img3", "img4", "img5", "img6", "img7"]

    # A real UK plate string. Note the space in the middle: the function is
    # supposed to ignore spaces when counting expected characters, so this
    # string has 7 actual characters ("AB12CDE") even though it's 8 chars
    # long with the space. That's exactly the behaviour we want to verify.
    plate_string = "AB12 CDE"

    # ---- Act ----
    # Call the function under test with our arranged inputs.
    result = segmentation.segmentation_is_valid(fake_char_images, plate_string)

    # ---- Assert ----
    # `assert <expression>` tells pytest: "this must be truthy; if it isn't,
    # fail this test and show me why." If the function returns False here,
    # pytest will print a clear failure message pointing to this line.
    # We expect True because 7 images matches 7 non-space characters.
    assert result is True


def test_segmentation_is_valid_raises_when_plate_string_is_not_a_string():
    # ---- Arrange ----
    # The function calls `plate_string.replace(" ", "")` internally, so it
    # only works if `plate_string` is actually a string. If we pass something
    # that isn't (like None), Python will raise an AttributeError because
    # None has no `.replace` method. This test pins down that behaviour.
    fake_char_images = ["img1", "img2"]
    bad_plate_string = None  # deliberately the wrong type

    # ---- Act + Assert ----
    # `pytest.raises(SomeError)` is a context manager that says "I expect the
    # code inside this `with` block to raise SomeError. If it doesn't, fail
    # the test." We use it instead of a bare `assert` because the *whole
    # point* of this test is that calling the function blows up — there's no
    # return value to check.
    with pytest.raises(AttributeError):
        segmentation.segmentation_is_valid(fake_char_images, bad_plate_string)
