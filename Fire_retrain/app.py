from fastapi import FastAPI
from retrain import run_retraining
from features_fusion import run_features_and_fusion
import requests

app = FastAPI()

@app.post("/retrain")
def retrain():
    result = run_retraining()
    requests.post("https://gdleds-fire-retrain.hf.space/reload-model", timeout=120)
    return result

@app.post("/features")
def transform_data():
    result = run_features_and_fusion()
    return result

@app.get("/")
def home():
    return {"message" : "Bienvenue sur l'API de retrain du modèle XGBoostSurvivalCox pour la prédiction des risques d'incendie en Corse"}
