import pandas as pd
import pytest

def test_load_coord_station(coord_station):
    assert not coord_station.empty
    for col in coord_station.columns[:5]:
        assert coord_station[col].isnull().sum() == 0,  f"NaN détecté dans la colonne {col}"

def test_lat_lon_ranges(coord_station):
    assert coord_station["latitude"].between(41, 44).all()
    assert coord_station["longitude"].between(8, 10).all()

