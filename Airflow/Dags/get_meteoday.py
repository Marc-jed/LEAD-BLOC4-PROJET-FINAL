import pandas as pd
import requests
import time
import tqdm
import os
import io
import boto3
import logging
import datetime
import glob
import numpy as np
from sqlalchemy import create_engine
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.hooks.S3_hook import S3Hook
from airflow.models import Variable
from datetime import date, timedelta, timezone
from airflow.providers.postgres.hooks.postgres import PostgresHook

default_args = {
    "owner": "airflow",
    "start_date": datetime.datetime(2024, 1, 1),
    "retries": 1
}

# appelle du département par code postal
def get_meteo(ti, **kwargs):
    
    api = Variable.get("meteoapi")
    dep = '20'
    url = 'https://public-api.meteofrance.fr/public/DPClim/v1/liste-stations/quotidienne'

    params = {
        'id-departement': dep,
        'parametre': 'temperature'
    }

    headers = {
        'accept': '*/*',
        'apikey': api
    }

    corse = requests.get(url, headers=headers, params=params)

    print('erreur corse', corse.status_code)

    # on applique un mask pour ne prendre que les stations ouvertes
    corse_df = pd.DataFrame(corse.json())
    corse_df['posteOuvert'] = corse_df['posteOuvert'].astype(bool)
    mask = corse_df['posteOuvert'] == True
    corse_df = corse_df[mask]

    # on va chercher la météo de la veille
    id = corse_df['id']
    all_paths = []

    for i in id:
        yesterday = datetime.datetime.now(timezone.utc) - timedelta(days=1)
        yesterday_format = yesterday.strftime('%Y-%m-%d')
        date_debut = f'{yesterday_format}T00:00:00Z'
        date_fin = f'{yesterday_format}T23:59:59Z'

        url = "https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne"
        params = {
            "id-station": i,
            "date-deb-periode": date_debut,
            "date-fin-periode": date_fin
        }
        headers = {
            "accept": "*/*",
            "apikey": api
        }

        corse1 = requests.get(url, headers=headers, params=params)
        print('erreur corse1', corse1.status_code)
        corse1_json = corse1.json()

        corse1_df = pd.DataFrame(corse1_json).reset_index()

        name = dep + '_' + corse_df.loc[corse_df['id'] == i, 'nom'].values[0] + '_' + str(yesterday_format)
    
        # Extract the 'return' value if the response is a dict
        id_cmde = corse1_df.iloc[0,1]
    

        url = "https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier"
        params = {
            "id-cmde": id_cmde
        }
        headers = {
            "accept": "*/*",
            "apikey": api
        }
        corse2 = requests.get(url, headers=headers, params=params)

        nom_station = corse_df.loc[corse_df['id'] == i, 'nom'].values[0].replace(' ', '_')
        path = f"/tmp/{dep}_{nom_station}_{yesterday_format}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Enregistrement du fichier
        with open(path, 'w', encoding='utf-8') as f:
            f.write(corse2.text)
        time.sleep(60 / 15)  # 60 seconds divided by 20 requests
        all_paths.append(path)
    ti.xcom_push(key='meteo_paths', value=all_paths)
    ti.xcom_push(key='meteo_format', value=yesterday_format)        
    logging.info("Météo data fetched and saved successfully.")

def compile_meteo_data(ti, **kwargs):
    all_files = ti.xcom_pull(task_ids='get_meteo', key='meteo_paths')

    # Liste pour stocker les DataFrames
    list_of_dfs = []

    # Lire chaque fichier CSV et ajouter son contenu à la liste
    for file in all_files:
        try:
            # Essaye d'abord avec le séparateur ;
            df = pd.read_csv(file, sep=';', on_bad_lines='skip', engine='python')

            # Si le DataFrame n'a qu'une seule colonne, essaie avec ,
            if df.shape[1] == 1:
                df = pd.read_csv(file, sep=',', on_bad_lines='skip', engine='python')

            # Si toujours 1 seule colonne, essaie avec tabulation
            if df.shape[1] == 1:
                df = pd.read_csv(file, sep='\t', on_bad_lines='skip', engine='python')

            list_of_dfs.append(df)

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture du fichier : {file}")
            print("➡️ Erreur :", e)
        
    # Concaténer tous les DataFrames en un seul
    corse_df = pd.concat(list_of_dfs, ignore_index=True)
    # Écrire le DataFrame combiné dans un nouveau fichier CSV
    meteo_date = ti.xcom_pull(task_ids="get_meteo", key="meteo_format")
    csv_path = f"/tmp/compile-meteo-corse-{meteo_date}.csv"
    corse_df.to_csv(csv_path, index=False) 
    # Push le chemin du fichier vers XCom
    ti.xcom_push(key="meteo-compile_csv_path", value=csv_path)
    logging.info("Météo data compiled successfully.")

def upload_compile_meteoday_to_neon(ti, **kwargs):
    path = ti.xcom_pull(task_ids='compile_meteo_data', key='meteo-compile_csv_path')
    hook = PostgresHook(postgres_conn_id="neon_default")
    df = pd.read_csv(path)
    df["POSTE"] = df["POSTE"].astype(int).astype(str)
    df["DATE"] = pd.to_datetime(df["DATE"], format='ISO8601')
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date

    for col in df.columns:
        if col != "DATE" and df[col].dtype == "object":
            df[col] = (
                df[col]
                .str.replace(",", ".", regex=False)
                .astype(float, errors="ignore")
            )
    del_col = {"production en échec (la commande contient une plage d'absence de données)","production en échec (la station ne contient pas de donnée pour cette période)"}
    df = df.drop(columns=[c for c in del_col if c in df.columns])
    df.columns = df.columns.str.lower()
    
    hook.insert_rows(
        table="meteoday",
        rows=df.to_records(index=False).tolist(),
        target_fields=list(df.columns),
        commit_every=1000,
        replace=False
    )
    logging.info("Compiled CSV uploaded to NEON successfully.")

def call_api_transform():
    url = "https://gdleds-fire-retrain.hf.space/features"
    resp = requests.post(url, timeout=900)


with DAG(
    dag_id="new_meteo",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="traitement donnée flux meteo et feux corse"
) as dag:

    fetch_weather = PythonOperator(
        task_id="get_meteo",
        python_callable=get_meteo,
    )
    compile_meteo = PythonOperator(
        task_id="compile_meteo_data",
        python_callable=compile_meteo_data
    )
    upload_compile_csv = PythonOperator(
        task_id="upload_compile_meteoday_to_neon",
        python_callable=upload_compile_meteoday_to_neon
    )
    fetch_call_api = PythonOperator(
        task_id="call_api_transform",
        python_callable=call_api_transform
    )

    fetch_weather >> compile_meteo >> upload_compile_csv >> fetch_call_api
   
