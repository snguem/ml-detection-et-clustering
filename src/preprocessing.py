import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


# ─────────────────────────────────────────────────
# EXERCICE 1 — Détection de fraude
# ─────────────────────────────────────────────────

def supprimer_colonnes_inutiles(df):
    return df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])


def feature_engineering_fraude(df):
    df = df.copy()
    df['diff_solde_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['diff_solde_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['erreur_solde_orig'] = df['diff_solde_orig'] - df['amount']
    df['erreur_solde_dest'] = df['diff_solde_dest'] - df['amount']
    return df


def encoder_type_transaction(df):
    le = LabelEncoder()
    df = df.copy()
    df['type'] = le.fit_transform(df['type'])
    return df, le


def splitter_donnees(df, cible='isFraud', taille_test=0.2, graine=42):
    X = df.drop(columns=[cible])
    y = df[cible]
    return train_test_split(X, y, test_size=taille_test, stratify=y, random_state=graine)


def normaliser_fraude(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def appliquer_smote(X_train, y_train, graine=42):
    smote = SMOTE(random_state=graine)
    return smote.fit_resample(X_train, y_train)


# ─────────────────────────────────────────────────
# EXERCICE 2 — Segmentation client
# ─────────────────────────────────────────────────

def imputer_valeurs_manquantes(df):
    df = df.copy()
    mediane_income = df['Income'].median()
    df['Income'] = df['Income'].fillna(mediane_income)
    return df


def feature_engineering_client(df):
    df = df.copy()
    df['Age'] = 2024 - df['Year_Birth']
    df['TotalSpent'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts']
                        + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
    df['TotalPurchases'] = (df['NumDealsPurchases'] + df['NumWebPurchases']
                            + df['NumCatalogPurchases'] + df['NumStorePurchases'])
    df['HasChildren'] = ((df['Kidhome'] + df['Teenhome']) > 0).astype(int)
    df['TotalCampaignes'] = (df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3']
                             + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response'])
    return df


def encoder_variables_categorielles(df):
    df = df.copy()
    education_map = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
    df['Education'] = df['Education'].map(education_map)
    df['Marital_Status'] = df['Marital_Status'].replace(
        {'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'}
    )
    marital_map = {'Single': 0, 'Divorced': 1, 'Widow': 1, 'Married': 2, 'Together': 2}
    df['Marital_Status'] = df['Marital_Status'].map(marital_map)
    return df


def selectionner_features_clustering(df):
    features = [
        'Age', 'Income', 'Education', 'Marital_Status', 'HasChildren',
        'Recency', 'TotalSpent', 'TotalPurchases', 'NumWebVisitsMonth',
        'TotalCampaignes', 'Complain'
    ]
    return df[features]


def normaliser_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def appliquer_pca(X_scaled, n_composantes=2):
    pca = PCA(n_components=n_composantes, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca
