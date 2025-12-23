import os
import pytest
import pandas as pd
import boto3
from sqlalchemy import create_engine


@pytest.fixture(scope="session")
def neon_engine():
    if os.getenv("CI") == "true":
        pytest.skip("Skipping DB access in CI")

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

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )

    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except Exception as e:
        pytest.fail(f"S3 bucket not accessible: {e}")

    return s3


@pytest.fixture(scope="session")
def coord_station(s3_client, tmp_path_factory):
    bucket = os.environ["S3_BUCKET_NAME"]
    s3_key = "dataset/coord_stations.csv"

    tmp_dir = tmp_path_factory.mktemp("s3_data")
    local_file = tmp_dir / "coord_stations.csv"

    s3_client.download_file(bucket, s3_key, str(local_file))
    df = pd.read_csv(local_file, sep=",", low_memory=False)

    if df.empty:
        pytest.fail("coord_stations dataset is empty")

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
        pytest.fail("historique_feu dataset is empty")

    return df
