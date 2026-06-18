# Tests

This folder holds **unit tests** — small automated checks that confirm
individual functions behave the way we expect. They run in seconds and
catch bugs you would otherwise only notice by running the whole pipeline.

## One-time setup

From the repo root:

```bash
pip install -r requirements-dev.txt
```

That installs `pytest`, the test runner.

## Running the tests

```bash
pytest -v
```

`-v` means "verbose" — it prints each test name and a green PASSED or red
FAILED next to it. Without `-v` you only get dots.

## Adding a new test

1. Open (or create) a file in `tests/` whose name starts with `test_`.
2. Inside it, write a function whose name also starts with `test_`.
3. Set up some inputs, call the function you want to check, and use
   `assert <something>` to state what should be true.
4. Run `pytest -v` again. Your new test should show up in the list.

That's it. Pytest auto-discovers anything named `test_*`.

## Ideas for tests you could write next

Pick one of these — they're all small, isolated, and use only what you've
already seen in `test_segmentation.py`:

- **`segmentation_is_valid` with a mismatch** — pass 5 images and a plate
  string with 7 characters; assert the function returns `False`. This is
  the negative case for the happy-path test that's already in the file.
- **`segmentation_is_valid` with an empty plate** — pass `[]` and `""`;
  assert it returns `True`. Edge cases like "what about empty inputs?"
  often reveal bugs.
- **`build_transform` in `data.py`** — call it with `training=True` and
  again with `training=False`; assert the two return different things
  (e.g. compare `len(result.transforms)`). This proves the training-only
  augmentations actually get added.
