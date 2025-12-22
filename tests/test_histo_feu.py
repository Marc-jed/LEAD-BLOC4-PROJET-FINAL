import pandas as pd
import pytest

def test_data_feu(historique_feu):
    assert historique_feu.shape[1] == 24, "Le fichier feu doit avoir 24 colonnes"
    for col in historique_feu.columns[:5]:
        nan_ratio = historique_feu[col].isnull().mean()
        assert nan_ratio < 0.01, f"Trop de NaN dans la colonne {col} ({nan_ratio:.2%})"
