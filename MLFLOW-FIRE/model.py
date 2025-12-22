import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
import sklearn
import warnings
from scipy.special import expit, logit
import sksurv.datasets
import numpy as np
import joblib
import pickle
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import DMatrix
from xgboost import train
from lifelines import CoxPHFitter
from itertools import product
from tqdm import tqdm
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_to_structured
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterGrid
from sksurv.datasets import load_breast_cancer
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv
from dotenv import load_dotenv
import boto3
import mlflow
import os
import io

from sksurv.ensemble import GradientBoostingSurvivalAnalysis


load_dotenv(dotenv_path=".secrets")

mlflow.set_tracking_uri('https://gdleds-mlflow-fire2.hf.space')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = os.getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT')
os.environ['S3_BUCKET'] = os.getenv('S3_BUCKET')

# Log configurations au démarrage 
print("=== Configuration MLflow ===")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Artifact Store: {os.getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT')}")
print(f"AWS Access: {'Configuré' if os.getenv('AWS_ACCESS_KEY_ID') else 'Manquant'}")

s3 = boto3.client('s3')
try:
   response = s3.list_objects_v2(Bucket=os.getenv('S3_BUCKET'))
   print("S3 contents:", response.get('Contents', []))
except Exception as e:
   print("S3 error:", e)


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
set_config(display="text")
s3.download_file(os.getenv('S3_BUCKET'), 'dataset/dataset_pour_prediction_2025-2.csv', 'predictions.csv')
df=pd.read_csv('predictions.csv', sep=';', low_memory=False)
df.columns = df.columns.str.lower()
############
mask = df["date"] >= "2025-01-01"
df = df[~mask]
###########
df['feu_prévu'] = df['feu_prévu'].astype(bool)
df_clean = df.copy()

features = [
    "rr","um","tn","tx","jours_sans_pluie","jours_tx_sup_30",
    "etpgrille_7j","compteur_jours_vers_prochain_feu", "moyenne_temperature_mois","moyenne_precipitations_mois","moyenne_vitesse_vent_mois","compteur_feu_log"

    ]
features = [f for f in features if f in df_clean.columns]

# Nous mettons à 0 les NAN de la colonne décompte
df_clean["décompte"] = df_clean["décompte"].fillna(0)



# 🔹 Préparation des données réelles
df_clean = df_clean.rename(columns={"feu_prévu": "event", "décompte": "duration"})
y_structured = Surv.from_dataframe("event", "duration", df_clean)

X = df_clean[features]
y = y_structured

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

event_train = y_train["event"]
duration_train = y_train["duration"]
event_test = y_test["event"]
duration_test = y_test["duration"]

def astype_float(X):
    return X.astype(float)

#  Pipeline XGBoost survie avec StandardScaler
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(
        objective="survival:cox",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        tree_method="hist",
        random_state=42
    ))
])

def train_evaluate_model_with_mlflow(model, X_train, X_test, y_train, y_test, model_name):
   print(f"\n=== Démarrage entraînement {model_name} ===")
   print(f"Tracking URI: {mlflow.get_tracking_uri()}")
   print(f"Registry URI: {mlflow.get_registry_uri()}")
   
   mlflow.set_experiment("fire_survival")
   print(f"Experiment: fire_survival")
   s3 = boto3.client('s3')

   with mlflow.start_run() as run:
        print(f"Run ID: {run.info.run_id}")
        
        print("Entraînement du modèle...")
        model.fit(X_train, duration_train, xgb__sample_weight=event_train)
        
        #save model to S3
        print("Enregistrement du modèle sur S3...")
        model_path = f"mlflow/models/{model_name}_{run.info.run_id}.joblib"

        # mlflow.sklearn.log_model(model, "model")

        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        s3.put_object(
            Bucket=os.getenv('S3_BUCKET'),
            Key=model_path,
            Body=buffer.getvalue()
        )
        print("Modèle enregistré")
  

        print(X[features].dtypes)
        print(np.isfinite(X[features].to_numpy()).all())
        print(X[features].isna().mean().sort_values(ascending=False).head(10))



       
       #  Prédictions réelles (log(HR)) sur données test
        log_hr_test = model.predict(X_test)

        print("Events test:", event_test.sum())
        print("Durée min/max:", duration_test.min(), duration_test.max())
        print("log_hr min/max:", np.min(log_hr_test), np.max(log_hr_test))

        #  Jeu factice pour estimer le modèle de Cox

        df_fake = pd.DataFrame({
            "duration": duration_train,
            "event": event_train,
            "const": 1
        })
        dtrain_fake = DMatrix(df_fake[["const"]])
        dtrain_fake.set_float_info("label", df_fake["duration"])
        dtrain_fake.set_float_info("label_lower_bound", df_fake["duration"])
        dtrain_fake.set_float_info("label_upper_bound", df_fake["duration"])
        dtrain_fake.set_float_info("weight", df_fake["event"])

        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "learning_rate": 0.1,
            "max_depth": 1,
            "verbosity": 0
        }
        bst_fake = train(params, dtrain_fake, num_boost_round=100)

        log_hr_fake = bst_fake.predict(dtrain_fake)
        df_risque = pd.DataFrame({
            "duration": duration_train,
            "event": event_train,
            "log_risque": log_hr_fake
        })
        # insertion de bruit pour aider le modèle à converger
        df_risque["log_risque"] += np.random.normal(0, 1e-4, size=len(df_risque))

        # Modèle de Cox factice
        cph = CoxPHFitter()
        cph.fit(df_risque, duration_col="duration", event_col="event", show_progress=False)
        
        # Sauvegarde dans S3
        baseline_path = f"mlflow/models/baseline_{run.info.run_id}.pkl"
        baseline_data = {"baseline_survival" : cph.baseline_survival_,
                         "baseline_hazard" : cph.baseline_cumulative_hazard_}
        buffer_baseline = io.BytesIO()
        pickle.dump(baseline_data, buffer_baseline)
        buffer_baseline.seek(0)

        s3.put_object(
            Bucket=os.getenv('S3_BUCKET'),
            Key=baseline_path,
            Body=buffer_baseline.getvalue()
        )
        print("Baseline H0(t) enregistrée sur S3 :", baseline_path)

        # Évaluation avec le c-index
        c_index = concordance_index_censored(event_test, duration_test, log_hr_test)[0]
        print(f"\nC-index (test) : {c_index:.3f}")

        
        print("\nEnregistrement des métriques...")
        mlflow.log_metric("c_index", c_index)

        input_example = X_train.iloc[:1]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Enregistrer dans le Registry
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="fire_survival"
        )
        return model, run.info.run_id

if __name__ == "__main__":
    xgb_final = pipeline
    _, run_id = train_evaluate_model_with_mlflow(
        xgb_final, X_train, X_test, y_train, y_test, "xgboost_survivalCOX_model"
    )
    print(f"Run ID: {run_id}")


