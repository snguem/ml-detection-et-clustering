from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────────
# EXERCICE 1 — Classification supervisée
# ─────────────────────────────────────────────────

def entrainer_regression_logistique(X_train, y_train, params=None):
    if params is None:
        params = {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
    modele = LogisticRegression(**params)
    modele.fit(X_train, y_train)
    return modele, params


def entrainer_random_forest(X_train, y_train, params=None):
    if params is None:
        params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    modele = RandomForestClassifier(**params)
    modele.fit(X_train, y_train)
    return modele, params


def entrainer_xgboost(X_train, y_train, params=None):
    if params is None:
        params = {
            'n_estimators': 100, 'max_depth': 6,
            'learning_rate': 0.1, 'eval_metric': 'logloss', 'random_state': 42
        }
    modele = xgb.XGBClassifier(**params)
    modele.fit(X_train, y_train)
    return modele, params


def entrainer_lightgbm(X_train, y_train, params=None):
    if params is None:
        params = {
            'n_estimators': 100, 'max_depth': 6,
            'learning_rate': 0.1, 'random_state': 42, 'verbose': -1
        }
    modele = lgb.LGBMClassifier(**params)
    modele.fit(X_train, y_train)
    return modele, params


def construire_reseau_neurones(input_dim):
    modele = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    modele.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc'), keras.metrics.Recall(name='recall')]
    )
    return modele


# ─────────────────────────────────────────────────
# EXERCICE 2 — Clustering non supervisé
# ─────────────────────────────────────────────────

def entrainer_kmeans(X, k, params=None):
    if params is None:
        params = {'n_clusters': k, 'random_state': 42, 'n_init': 10}
    modele = KMeans(**params)
    labels = modele.fit_predict(X)
    return modele, labels, params


def entrainer_dbscan(X, params=None):
    if params is None:
        params = {'eps': 0.8, 'min_samples': 10}
    modele = DBSCAN(**params)
    labels = modele.fit_predict(X)
    return modele, labels, params


def entrainer_agglomerative(X, params=None):
    if params is None:
        params = {'n_clusters': 4, 'linkage': 'ward'}
    modele = AgglomerativeClustering(**params)
    labels = modele.fit_predict(X)
    return modele, labels, params


def entrainer_gmm(X, params=None):
    if params is None:
        params = {'n_components': 4, 'random_state': 42, 'covariance_type': 'full'}
    modele = GaussianMixture(**params)
    labels = modele.fit_predict(X)
    return modele, labels, params
