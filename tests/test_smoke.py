import pandas as pd
import pytest

def test_smoke_pipeline(meteoday_get, data_for_test):
    assert len(meteoday_get) > 0
    assert len(data_for_test) > 0
