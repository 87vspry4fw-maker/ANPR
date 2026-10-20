from torchvision import transforms
from data import BuildTransform
import plateFormat

import sqlite3
from datetime import datetime

from database import CarParkDB
from prediction import PredictPlate
from segmentationV2 import SegmentV2


def test_training_and_evaluation_transforms_are_different():
    training_transform = BuildTransform(True)
    evaluation_transform = BuildTransform(False)

    #Both should return a Compose object
    assert isinstance(training_transform, transforms.Compose)
    assert isinstance(evaluation_transform, transforms.Compose)

    #Training should have more transformations than evaluation
    assert len(training_transform.transforms) > len(evaluation_transform.transforms)

    #Training should include data augmentation
    assert any(
        isinstance(t, (
            transforms.RandomAffine,
            transforms.GaussianBlur,
            transforms.ColorJitter,
            transforms.RandomErasing,
        ))
        for t in training_transform.transforms
    )

    #Evaluation should not include data augmentation
    assert not any(
        isinstance(t, (
            transforms.RandomAffine,
            transforms.GaussianBlur,
            transforms.ColorJitter,
            transforms.RandomErasing,
        ))
        for t in evaluation_transform.transforms
    )


def test_database_is_created(tmp_path):
    db_path = tmp_path / "test_ANPR.db"

    CarParkDB(db_path=str(db_path))

    assert db_path.exists()


def test_normalise_removes_spaces_and_capitalises_plate():
    result = CarParkDB._normalise(" ab 12 cde ")

    assert result == "AB12CDE"


def test_normalise_handles_lowercase_plate():
    result = CarParkDB._normalise("ab12cde")

    assert result == "AB12CDE"


def test_now_returns_iso_format_datetime():
    result = CarParkDB._now()

    #Check that the returned string can be parsed
    parsed = datetime.fromisoformat(result)

    assert isinstance(parsed, datetime)


def test_is_banned_returns_false_for_nonexistent_plate(tmp_path):
    db_path = tmp_path / "test_ANPR.db"
    db = CarParkDB(db_path=str(db_path))

    assert not db.is_banned("NONEXISTENT")


def test_ban_and_is_banned(tmp_path):
    db_path = tmp_path / "test_ANPR.db"
    db = CarParkDB(db_path=str(db_path))

    plate = "AB12CDE"
    reason = "Test reason"

    #Initially, the plate should not be banned
    assert not db.is_banned(plate)

    #Ban the plate
    db.ban(plate, reason)

    #Now, the plate should be banned
    assert db.is_banned(plate)


def test_ban_records_event_in_ban_events_table(tmp_path):
    db_path = tmp_path / "test_ANPR.db"
    db = CarParkDB(db_path=str(db_path))

    plate = "AB12CDE"
    reason = "Test reason"

    #Ban the plate
    db.ban(plate, reason)

    #Check that the ban event is recorded in the ban_events table
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            "SELECT * FROM ban_events WHERE plate = ? AND action = 'ban'", (plate,)
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row["reason"] == reason


def test_ban_updates_vehicles_table(tmp_path):
    db_path = tmp_path / "test_ANPR.db"
    db = CarParkDB(db_path=str(db_path))

    plate = "AB12CDE"
    reason = "Test reason"

    #Ban the plate
    db.ban(plate, reason)

    #Check that the vehicles table is updated
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            "SELECT * FROM vehicles WHERE plate = ?", (plate,)
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row["is_banned"] == 1


def test_plate_format_validation():
    scores = [0.1, 0.9, 0.2, 0.8]  #Example scores for classes
    classes = ['A', 'B', '0', '1']
    position = 2  #This position requires a digit

    best_index = plateFormat.ConstrainedIndex(scores, classes, position)
    assert best_index == 3  #Expected index for a digit at position 2


def test_predict_plate_with_no_characters(tmp_path):
    #Create a blank image (no characters)
    from PIL import Image
    import numpy as np

    blank_image_path = tmp_path / "blank_plate.jpg"
    blank_image = Image.new('RGB', (200, 100), color='white')
    blank_image.save(blank_image_path)

    #Call the PredictPlate function
    from prediction import PredictPlate

    final_plate, char_confidences = PredictPlate(blank_image_path)

    #Since there are no characters, expect an empty string and an empty list
    assert final_plate == ""
    assert char_confidences == []


def test_predict_plate_with_seven_characters(tmp_path):
    im_path = "/Users/vahnee/Library/Mobile Documents/com~apple~CloudDocs/A-Level/Computer Science/NEA/ANPR/UKLicencePlateDataset/yellowplate_normal/AA31TCM.png"
    plate, confidences = PredictPlate(im_path)

    assert len(plate) == 7
    assert confidences is not None

def test_segment_v2_with_no_characters(tmp_path):
    #Create a blank image (no characters)
    from PIL import Image
    import numpy as np

    blank_image_path = tmp_path / "blank_plate.jpg"
    blank_image = Image.new('RGB', (200, 100), color='white')
    blank_image.save(blank_image_path)

    #Call the SegmentV2 function
    from segmentationV2 import SegmentV2

    crops = SegmentV2(blank_image_path)

    #Since there are no characters, expect an empty list of crops
    assert crops == []


def test_segment_v2_with_seven_characters(tmp_path):
    im_path = "/Users/vahnee/Library/Mobile Documents/com~apple~CloudDocs/A-Level/Computer Science/NEA/ANPR/UKLicencePlateDataset/yellowplate_normal/AA31TCM.png"
    crops = SegmentV2(im_path)

    assert len(crops) == 7