import pandas as pd
import pytest

def test_data_feu(historique_feu):
    assert historique_feu.shape[1] == 24, "Le fichier feu doit avoir 24 colonnes"
    for col in historique_feu.columns[:5]:
        assert historique_feu[col].isnull().sum() == 0, f"NaN détecté dans la colonne {col}"
