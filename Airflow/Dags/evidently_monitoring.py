import logging
import requests
import pandas as pd
import numpy as np
import io
import os
import warnings
import evidently
from evidently.ui.workspace import CloudWorkspace
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from datetime import datetime
from sqlalchemy import create_engine
from airflow import DAG
from airflow.hooks.S3_hook import S3Hook
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.email import EmailOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook




defaults_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2
    }

def fetch_reference_data(ti, **kwarg):
    s3 = S3Hook(aws_conn_id='aws_default')
    bucket = Variable.get('S3BucketName')
    key = "dataset/liste_station.csv"
    path = '/tmp'
    df = s3.download_file(bucket_name=bucket, key=key, local_path=path)
    df = pd.read_csv(df)
    df["0"] = df["0"].astype('int')
    hook = PostgresHook(postgres_conn_id='neon_default')
    all_station = []
    for poste in df["0"]:
        df_ref = hook.get_pandas_df(f"SELECT um, tn, rr FROM data_prediction WHERE poste = '{poste}' AND date BETWEEN (CURRENT_DATE - INTERVAL '1 YEAR' - INTERVAL '30 DAYS') AND (CURRENT_DATE - INTERVAL '1 YEAR')")
        if not df_ref.empty:
            df_ref["poste"] = poste
            all_station.append(df_ref)
    final_station = pd.concat(all_station, ignore_index=True)
    path = '/tmp/data_reference.csv'
    final_station.to_csv(path, index=False)
    ti.xcom_push(key="data_reference_path", value=path)
    logging.info('Données référentes récupérées')

def fetch_current_data(ti, **kwarg):
    s3 = S3Hook(aws_conn_id='aws_default')
    bucket = Variable.get('S3BucketName')
    key = "dataset/liste_station.csv"
    path = '/tmp'
    df = s3.download_file(bucket_name=bucket, key=key, local_path=path)
    df = pd.read_csv(df)
    df["0"] = df["0"].astype('int')
    hook = PostgresHook(postgres_conn_id='neon_default')
    all_station1 = []
    for poste in df["0"]:
        df_ref1 = hook.get_pandas_df(f"SELECT um, tn, rr FROM data_prediction WHERE poste = '{poste}' AND date >= CURRENT_DATE - INTERVAL '30 DAYS'")
        if not df_ref1.empty:
            df_ref1["poste"] = poste
            all_station1.append(df_ref1)
    station_final = pd.concat(all_station1, ignore_index=True)
    path = '/tmp/current_data.csv'
    station_final.to_csv(path, index=False)
    ti.xcom_push(key="current_data_path", value=path)
    logging.info('Données courantes récupérées sur les 30 derniers jours')

def run_evidently(ti):
    ref_path = ti.xcom_pull(
        key="data_reference_path",
        task_ids="fetch_reference_data"
    )
    cur_path = ti.xcom_pull(
        key="current_data_path",
        task_ids="fetch_current_data"
    )

    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)

    ref_df = ref_df.fillna(0)
    cur_df = cur_df.fillna(0)

    def rename_col(df):
        return df.rename(columns={'rr' : 'precipitation', 'um' : 'humidite', 'tn' : 'temperature'})

    ref_df = rename_col(ref_df)
    cur_df = rename_col(cur_df)

    
    if ref_df.empty or cur_df.empty:
        raise ValueError("Reference or current dataset is empty")

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ])
    # On va prendre 2 stations pour controler les données
    postes_selectionnes = {20004014, 20303002}
    for poste in postes_selectionnes:
        cur_df_poste = cur_df[cur_df["poste"] == poste]
        ref_df_poste = ref_df[ref_df["poste"] == poste]

        report_run = report.run(
            current_data=cur_df_poste,
            reference_data=ref_df_poste
        )

        ws = CloudWorkspace(
            token=Variable.get('TOKEN_EVIDENT'),
            url="https://app.evidently.cloud",
        )
        project_id=Variable.get('projet_id')
        ws.add_run(
            project_id,
            report_run,
            include_data=True
        )

    logging.info('Données envoyées sur le cloud evidently')


with DAG(
    dag_id="evidently_dags",
    default_args=defaults_args,
    schedule_interval="@daily",
    catchup=False,
    start_date=datetime(2023, 1, 1),
    description="Monitoring des données météos"
) as dag:
    
    reference_data = PythonOperator(
        task_id="fetch_reference_data",
        python_callable = fetch_reference_data
    )
    current_data = PythonOperator(
    task_id="fetch_current_data",
    python_callable = fetch_current_data
    )
    fetch_evidently = PythonOperator(
    task_id="run_evidently",
    python_callable = run_evidently
    )

    reference_data >> current_data >> fetch_evidently 



