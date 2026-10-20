
import pytest

from database import CarParkDB


@pytest.fixture
def carpark_db(tmp_path):
#Create a fresh CarParkDB for each test
    db_path = tmp_path / "test_ANPR.db"
    db = CarParkDB(db_path=str(db_path))

    yield db
