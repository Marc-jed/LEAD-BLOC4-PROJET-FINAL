import pandas as pd
import pytest
from Fire_retrain.features_fusion import run_features_and_fusion
from tests.conftest import REQUIRED_INPUT_COLUMNS


def test_features_fusion_accepts_required_input_columns():
    df = pd.DataFrame([{
        **{col: 0 for col in REQUIRED_INPUT_COLUMNS},
        "date": pd.Timestamp("2024-07-01"),
        "poste": "20002001",
    }])

    output = run_features_and_fusion(df)

    assert isinstance(output, pd.DataFrame)
    assert not output.empty, "La fusion retourne un DataFrame vide"



def test_prediction_output_is_valid(fusion_features):
    assert "feu_prévu" in fusion_features.columns
    assert fusion_features["feu_prévu"].notna().all()
    assert fusion_features["feu_prévu"].isin([0, 1]).all(), \
        "feu_prévu doit être binaire (0/1)"



def test_prediction_required_features_present(fusion_features):
    required = {
        "rr",
        "um",
        "tn",
        "tx",
        "jours_sans_pluie",
        "jours_tx_sup_30",
        "etpgrille_7j",
        "compteur_jours_vers_prochain_feu",
        "moyenne_temperature_mois",
        "moyenne_precipitations_mois",
        "moyenne_vitesse_vent_mois",
        "compteur_feu_log",
    }

    missing = required - set(fusion_features.columns)
    assert not missing, f"Features manquantes pour la prédiction : {missing}"




