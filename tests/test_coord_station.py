import pandas as pd
import pytest

def test_coord_station_required_columns(coord_station):
    required_columns = {"POSTE", "nom", "lon", "lat"}
    missing = required_columns - set(coord_station.columns)
    assert not missing, f"Colonnes manquantes dans coord_station : {missing}"

def test_load_coord_station(coord_station):
    assert not coord_station.empty
    for col in coord_station.columns[:5]:
        assert coord_station[col].isnull().sum() == 0,  f"NaN détecté dans la colonne {col}"

def test_lat_lon_ranges(coord_station):
    assert coord_station["lat"].between(41, 44).all()
    assert coord_station["lon"].between(8, 10).all()

