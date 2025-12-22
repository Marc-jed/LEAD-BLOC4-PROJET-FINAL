import logging
import json
import requests
import pandas as pd
import numpy as np
import io
import os
import joblib
import pickle
import plotly.express as px
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
from sksurv.metrics import concordance_index_censored
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

defaults_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2
    }

def fetch_prediction_data(ti, **kwargs):
    hook=PostgresHook(postgres_conn_id="neon_default")
    # engine = hook.get_sqlalchemy_engine()
    df = hook.get_pandas_df("SELECT * FROM data_prediction WHERE date >= NOW() - INTERVAL '540 DAYS'")
    path="/tmp/data_prediction.csv"
    df.to_csv(path,index=False)
    ti.xcom_push(key="data_prediction_path", value=path)
    logging.info("données des 540 derniers jours chargées")

def analyse_model_data(ti, **kwargs):
    path = ti.xcom_pull(task_ids="fetch_prediction_data", key="data_prediction_path")
    df = pd.read_csv(path)
    features = [
    "rr","um","tn","tx","jours_sans_pluie","jours_tx_sup_30",
    "etpgrille_7j","compteur_jours_vers_prochain_feu", "moyenne_temperature_mois","moyenne_precipitations_mois","moyenne_vitesse_vent_mois","compteur_feu_log"
    ]
    features = [f for f in features if f in df.columns]
    df["décompte"] = df["décompte"].fillna(0)
    df = df.rename(columns={"feu_prévu": "event", "décompte": "duration"})
    bucket = Variable.get('S3BucketName')
    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_client = s3_hook.get_conn() 

    def get_latest_model_key():

        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix="mlflow/models/"
            )

            if "Contents" not in response:
                raise ValueError("Aucun modèle trouvé dans le bucket S3.")

            # Filtrer uniquement les .joblib
            models = [
                obj for obj in response.get("Contents", [])
                if obj["Key"].endswith(".joblib")
            ]

            if not models:
                raise ValueError("Aucun fichier .joblib trouvé.")

            # Trier par LastModified (date d'upload dans S3)
            models.sort(key=lambda x: x["LastModified"], reverse=True)

            latest_key = models[0]["Key"]
            print(f"Dernier modèle détecté : {latest_key}")
            return latest_key

        except Exception as e:
            raise RuntimeError(f"Erreur récupération modèle S3 : {e}")

    def load_latest_model():
        latest_model_key = get_latest_model_key()
        response = s3_client.get_object(Bucket=bucket, Key=latest_model_key)
        buffer = io.BytesIO(response["Body"].read())
        model = joblib.load(buffer)
        logging.info(f"Chargement modèle : {latest_model_key}")
        return model
        
    model = load_latest_model()
    
    # # modele baseline
    # def load_baseline_S3():
    #     response = s3_client.list_objects_v2(Bucket=bucket, Prefix="mlflow/models/baseline_")

    #     baselines = [obj for obj in response.get("Contents", [])]
    #     latest = sorted(baselines, key=lambda x: x["LastModified"], reverse=True)[0]["Key"]

    #     obj = s3_client.get_object(Bucket=bucket, Key=latest)
    #     buffer = io.BytesIO(obj["Body"].read())
    #     data = pickle.load(buffer)

    #     return data["baseline_survival"], data["baseline_hazard"]

    # baseline_survival, baseline_hazard = load_baseline_S3()

    log_hr_pred = model.predict(df[features])
    HR = np.exp(log_hr_pred)

    c_index = concordance_index_censored(
    df["event"].astype(bool),
    df["duration"].astype(float),
    log_hr_pred
    )[0]

    logging.info(f"\nC-index (test) : {c_index:.3f}")
    ti.xcom_push(key="c-index", value=c_index)
    if c_index >= 0.69:
        report_text1 = f"""
        <h3> Rapport d'analyse c_index</h3>
        <p> Le modèle a une performence conforme <b>{c_index:.3f}<b></p>
        """
        ti.xcom_push(key="text1", value=report_text1)
    else:
        report_text = f"""
        <h3>Rapport d'analyse c_index</h3>
        <p>Dernier résultat du c_index <b>{c_index:.3f}</b></p>
        <p>Résulat inférieur à 0.69, le model est réentrainé</p>
        """
        ti.xcom_push(key="text", value=report_text)
        
def check_model(ti,**kwargs):
    score_C_index=ti.xcom_pull(key="c-index", task_ids="analyse_model_data")
    if score_C_index < 0.69:
        return "call_retrain_api"
    else: 
        return "sendmail_model_ok"

def load_metrics_neon(ti, **kwargs):
    path = ti.xcom_pull(key="c-index", task_ids="analyse_model_data")
    raw_path = {"date" : datetime.today(), "resultat" : path, "status" : "checking_model"}
    df = pd.DataFrame([raw_path])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    hook=PostgresHook(postgres_conn_id="neon_default")
    hook.insert_rows(
        table="metrics",
        rows=df.to_records(index=False).tolist(),
        target_fields=list(df.columns),
        commit_every=1000,
        replace=False
    )
    
def call_retrain_api():
    url = "https://gdleds-fire-retrain.hf.space/retrain"
    resp = requests.post(url, timeout=900)

    if resp.status_code != 200:
        raise ValueError(f"Erreur API retrain : {resp.status_code} | {resp.text}")

    logging.info("Réentraînement lancé avec succès")

with DAG(
    dag_id="check_model",
    default_args=defaults_args,
    description="DAG for reporting daily BI",
    schedule_interval="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False,

) as dag:
    fetch_prediction_data_task = PythonOperator(
        task_id="fetch_prediction_data",
        python_callable=fetch_prediction_data
    )
    analyse_model_data_task=PythonOperator(
        task_id="analyse_model_data",
        python_callable= analyse_model_data
    )
    branch_check_model=BranchPythonOperator(
        task_id="check_model",
        python_callable=check_model
    ) 
    model_ok_sendmail = EmailOperator(
        task_id="sendmail_model_ok",
        to="gdleds31@gmail.com",
        subject="Analyse C-INDEX du model",
        html_content= "{{ti.xcom_pull(key='text1', task_ids='analyse_model_data')}}"
    )
    fetch_call_retrain_api=PythonOperator(
        task_id="call_retrain_api",
        python_callable=call_retrain_api
    )
    # continue_pipeline = EmptyOperator(
    #     task_id="continue_pipeline")
    fetch_metrics_neon = PythonOperator(
        task_id='load_metrics_neon',
        python_callable=load_metrics_neon
    )
    sendemail_check_model = EmailOperator(
        task_id="sendemail_check_model",
        to="gdleds31@gmail.com",
        subject="Analyse C-INDEX du model",
        html_content= "{{ti.xcom_pull(key='text', task_ids='analyse_model_data')}}"
    )
    fetch_prediction_data_task >> analyse_model_data_task >> branch_check_model 
    branch_check_model >> fetch_call_retrain_api >> sendemail_check_model 
    branch_check_model >> model_ok_sendmail >> fetch_metrics_neon
