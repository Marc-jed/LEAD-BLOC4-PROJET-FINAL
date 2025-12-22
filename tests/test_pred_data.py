import pandas as pd
import pytest

def data_prediction(prediction_get):
    assert prediction_get.shape[1] == 75, "Le dataset final doit avoir 75 colonnes"
    assert prediction_get["évènement"].dtypes == bool
    assert prediction_get["Feu_prévu"].dtypes == bool
    assert (prediction_get["décompte"].dropna() >=0).all(), "la colonne décompte doit contenir un chiffre positif ou 0"