import pandas as pd
import pickle
import os


def charger_donnees(chemin, separateur=','):
    """Charge un fichier CSV et affiche un résumé des dimensions."""
    df = pd.read_csv(chemin, sep=separateur)
    print(f'Données chargées : {df.shape[0]:,} lignes x {df.shape[1]} colonnes')
    return df


def sauvegarder_modele(modele, chemin):
    """Sauvegarde un modèle scikit-learn / XGBoost / LightGBM au format pickle."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, 'wb') as f:
        pickle.dump(modele, f)
    print(f'Modèle sauvegardé : {chemin}')


def charger_modele(chemin):
    """Charge un modèle sauvegardé au format pickle."""
    with open(chemin, 'rb') as f:
        modele = pickle.load(f)
    print(f'Modèle chargé : {chemin}')
    return modele
