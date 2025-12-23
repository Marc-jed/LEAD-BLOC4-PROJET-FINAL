import pandas as pd
import pytest

def test_data_feu(historique_feu):
    assert historique_feu.shape[1] == 19, "Le fichier feu doit avoir 24 colonnes"
    assert historique_feu['Feux'].isin([1]).all(), f'il y a une valeur différente de 1'
    for col in historique_feu.columns[:5]:
        nan_ratio = historique_feu[col].isnull().mean()
        assert nan_ratio < 0.01, f"Trop de NaN dans la colonne {col} ({nan_ratio:.2%})"
