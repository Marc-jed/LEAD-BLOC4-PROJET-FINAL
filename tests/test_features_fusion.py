import os
import pytest
import pandas as pd
from Fire_retrain.features_fusion import run_features_and_fusion


@pytest.mark.integration
def test_features_fusion_pipeline_contract():

    if os.getenv("CI") == "true":
        pytest.skip("features_fusion requires DB and S3")

    output = run_features_and_fusion()

    assert isinstance(output, pd.DataFrame)
    assert not output.empty, "Le pipeline retourne un DataFrame vide"

    # Colonnes indispensables à la prédiction
    required_features = {
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
        "feu_prévu",
    }

    missing = required_features - set(output.columns)
    assert not missing, f"Colonnes manquantes dans la sortie : {missing}"
