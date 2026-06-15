import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)


def evaluer_modele(nom, y_true, y_pred, y_proba):
    """Calcule les métriques de classification et affiche le rapport détaillé."""
    metriques = {
        'Modèle': nom,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'F1-Score': round(f1_score(y_true, y_pred), 4),
        'ROC-AUC': round(roc_auc_score(y_true, y_proba), 4)
    }
    print(f'\n--- {nom} ---')
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraude']))
    return metriques


def evaluer_clustering(X, labels):
    """Calcule les métriques d'évaluation des clusters."""
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    metriques = {
        'Silhouette Score': round(silhouette_score(X, labels), 4),
        'Davies-Bouldin Score': round(davies_bouldin_score(X, labels), 4)
    }
    for nom, val in metriques.items():
        print(f'{nom} : {val}')
    return metriques
