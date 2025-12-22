import pytest
import pandas as pd
import datetime
import os
import boto3
from pathlib import Path
from sqlalchemy import create_engine


@pytest.fixture(scope="session")
def neon_engine():
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
        
    s3 = boto3.client('s3', aws_access_key_id = AWS_ACCESS_KEY_ID, aws_secrect_key_id=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except Exception as e:
        pytest.fail(f"S3 bucket not accessible: {e}")
    return s3

@pytest.fixture(scope="session")
def meteoday_get(neon_engine):
    query = f"""SELECT * FROM meteoday WHERE Date = CURRENT_DATE - INTERVAL '1 day'"""
    df=pd.read_sql(query,neon_engine)
    if df.empty:
        pytest.skip("No meteoday data for yesterday")
    return df

@pytest.fixture(scope="session")
def data_for_test(s3_client, tmp_path_factory):
    bucket = os.environ["S3_BUCKET_NAME"]
    s3_key = os.environ["S3_PREDICTION_DATA_KEY"]
    

    tmp_dir = tmp_path_factory.mktemp("s3_data")
    local_file = tmp_dir / "predictions.csv"

    s3_client.download_file(bucket, s3_key, str(local_file))

    df = pd.read_csv(local_file, sep=";", low_memory=False)

    if df.empty:
        pytest.fail("Downloaded prediction dataset is empty")

    return df

@pytest.fixture(scope="session")
def coord_station(s3_client, tmp_path_factory):
    bucket = os.environ["S3_BUCKET_NAME"]
    s3_key = os.environ["S3_COORD_DATA_KEY"]
    

    tmp_dir = tmp_path_factory.mktemp("s3_data")
    local_file = tmp_dir / "coord_station.csv"

    s3_client.download_file(bucket, s3_key, str(local_file))

    df = pd.read_csv(local_file, sep=";", low_memory=False)

    if df.empty:
        pytest.fail("Downloaded coord_station dataset is empty")

    return df

@pytest.fixture(scope="session")
def historique_feu(s3_client, tmp_path_factory):
    bucket = os.environ["S3_BUCKET_NAME"]
    s3_key = os.environ["S3_FEU_DATA_KEY"]
    

    tmp_dir = tmp_path_factory.mktemp("s3_data")
    local_file = tmp_dir / "historique_incendie.csv"

    s3_client.download_file(bucket, s3_key, str(local_file))

    df = pd.read_csv(local_file, sep=";", low_memory=False)

    if df.empty:
        pytest.fail("Downloaded coord_station dataset is empty")

    return df

@pytest.fixture(scope="session")
def prediction_get(neon_engine):
    # Charger les données depuis néon
    query = f"""SELECT * FROM data_prediction WHERE Date = CURRENT_DATE - INTERVAL '1 day' """
    df=pd.read_sql(query,neon_engine)
    return df
