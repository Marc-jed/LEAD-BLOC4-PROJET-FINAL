import pytest
import pandas as pd
import datetime
import os
import boto3
from pathlib import Path
from sqlalchemy import create_engine
from Fire_retrain.features_fusion import run_features_and_fusion


@pytest.fixture(scope="session")
def neon_engine():
    # En CI, on skip proprement
    if os.getenv("CI") == "true":
        pytest.skip("Skipping DB tests in CI environment")

    try:
        host = os.environ["DB_HOST"]
        db = os.environ["DB_NAME"]
        user = os.environ["DB_USER"]
        password = os.environ["DB_PASS"]
        port = os.environ.get("DB_PORT", "5432")
    except KeyError as e:
        pytest.fail(f"Missing DB environment variable: {e}")

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)

@pytest.fixture(scope="session")
def s3_client():
    try:
        S3_BUCKET = os.environ["S3_BUCKET_NAME"]
        AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
        AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
    except KeyError as e:
        pytest.fail(f"Missing AWS environment variable: {e}")
        
    s3 = boto3.client('s3', aws_access_key_id = AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except Exception as e:
        pytest.fail(f"S3 bucket not accessible: {e}")
    return s3

@pytest.fixture(scope="session")
def meteoday_get(neon_engine):
    query = f"""SELECT * FROM meteoday WHERE date = CURRENT_DATE - INTERVAL '1 day'"""
    df=pd.read_sql(query,neon_engine)
    if df.empty:
        pytest.skip("No meteoday data for yesterday")
    return df

@pytest.fixture(scope="session")
def data_for_test(neon_engine):
    query = f"""SELECT * FROM data_prediction WHERE date = CURRENT_DATE - INTERVAL '1 day'"""
    df=pd.read_sql(query,neon_engine)
    if df.empty:
        pytest.skip("No meteoday data for yesterday")
    return df

@pytest.fixture(scope="session")
def coord_station(s3_client, tmp_path_factory):
    bucket = os.environ["S3_BUCKET_NAME"]
    s3_key = "dataset/coord_stations.csv"
    

    tmp_dir = tmp_path_factory.mktemp("s3_data")
    local_file = tmp_dir / "coord_stations.csv"

    s3_client.download_file(bucket, s3_key, str(local_file))

    df = pd.read_csv(local_file, sep=",", low_memory=False)

    if df.empty:
        pytest.fail("Downloaded coord_stations dataset is empty")

    return df

@pytest.fixture(scope="session")
def historique_feu(s3_client, tmp_path_factory):
    bucket = os.environ["S3_BUCKET_NAME"]
    s3_key = "dataset/historique_incendie_corse_cleaned_2025.csv"
    

    tmp_dir = tmp_path_factory.mktemp("s3_data")
    local_file = tmp_dir / "historique_incendie.csv"

    s3_client.download_file(bucket, s3_key, str(local_file))

    df = pd.read_csv(local_file, sep=",", low_memory=False)

    if df.empty:
        pytest.fail("Downloaded historique incendie dataset is empty")

    return df

@pytest.fixture(scope="session")
def prediction_get(neon_engine):
    # Charger les données depuis néon
    query = f"""SELECT * FROM data_prediction WHERE date = CURRENT_DATE - INTERVAL '1 day' """
    df=pd.read_sql(query,neon_engine)
    return df


REQUIRED_INPUT_COLUMNS = {
    "poste", "date", "rr", "drr", "tn", "htn", "tx", "htx", "tm", "tmnx",
    "tnsol", "tn50", "tampli", "tntxm", "ffm", "fxi", "dxi", "hxi",
    "fxy", "dxy", "hxy", "fxi3s", "hxi3s", "un", "hun", "ux", "hux",
    "dhumi40", "dhumi80", "tsvm", "um", "orag", "brume",
    "etpmon", "etpgrille"
}

@pytest.fixture(scope="session")
def fusion_features():
    """
    Fixture globale : retourne un DataFrame fusionné
    minimal mais valide pour les tests ML.
    """
    df = pd.DataFrame([{
        **{col: 1 for col in REQUIRED_INPUT_COLUMNS},
        "date": pd.Timestamp("2024-07-01"),
        "poste": "20002001",
    }])

    return run_features_and_fusion(df)
