import os
import sys
import glob
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, f1_score, recall_score,
    precision_score, accuracy_score, silhouette_score, davies_bouldin_score,
)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import shap

from src.preprocessing import (
    imputer_valeurs_manquantes,
    feature_engineering_client,
    encoder_variables_categorielles,
    selectionner_features_clustering,
    normaliser_features,
    appliquer_pca,
)
from src.models import (
    entrainer_regression_logistique,
    entrainer_random_forest,
    entrainer_xgboost,
    entrainer_lightgbm,
    entrainer_kmeans,
    entrainer_dbscan,
    entrainer_agglomerative,
    entrainer_gmm,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard ML — Analyse & Modélisation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DONNEES_DIR = os.path.join(BASE_DIR, "donnees", "brutes")


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données de fraude…")
def charger_fraude():
    fichiers = glob.glob(os.path.join(DONNEES_DIR, "*fraude*")) + \
               glob.glob(os.path.join(DONNEES_DIR, "*tection*"))
    if not fichiers:
        st.error("Fichier détection_de_fraude.csv introuvable.")
        st.stop()
    df = pd.read_csv(fichiers[0], sep=";")
    return df


@st.cache_data(show_spinner="Chargement des données clients…")
def charger_clients():
    chemin = os.path.join(DONNEES_DIR, "data_cluster_exercice_2.csv")
    return pd.read_csv(chemin, sep=";")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Exercice 1 — Fraude
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Entraînement des modèles de détection de fraude…")
def entrainer_modeles_fraude():
    df = charger_fraude()

    # Échantillon stratifié pour la vitesse : toutes les fraudes + 60k normaux
    df_fraude = df[df["isFraud"] == 1]
    df_normal = df[df["isFraud"] == 0].sample(n=60_000, random_state=42)
    df_s = pd.concat([df_fraude, df_normal]).sample(frac=1, random_state=42)

    df_s = df_s.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])
    df_s["diff_solde_orig"] = df_s["oldbalanceOrg"] - df_s["newbalanceOrig"]
    df_s["diff_solde_dest"] = df_s["newbalanceDest"] - df_s["oldbalanceDest"]
    df_s["erreur_solde_orig"] = df_s["diff_solde_orig"] - df_s["amount"]
    df_s["erreur_solde_dest"] = df_s["diff_solde_dest"] - df_s["amount"]

    le = LabelEncoder()
    df_s["type"] = le.fit_transform(df_s["type"])

    X = df_s.drop(columns=["isFraud"])
    y = df_s["isFraud"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_sc, y_train)

    resultats = []
    modeles = {}

    def evaluer(nom, y_pred, y_proba):
        resultats.append({
            "Modèle": nom,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        })
        return y_pred, y_proba

    # Régression Logistique
    lr, _ = entrainer_regression_logistique(X_res, y_res)
    modeles["Régression Logistique"] = (lr, scaler)
    y_pred_lr = lr.predict(X_test_sc)
    y_proba_lr = lr.predict_proba(X_test_sc)[:, 1]
    evaluer("Régression Logistique", y_pred_lr, y_proba_lr)

    # Random Forest
    rf, _ = entrainer_random_forest(X_res, y_res)
    modeles["Random Forest"] = (rf, scaler)
    y_pred_rf = rf.predict(X_test_sc)
    y_proba_rf = rf.predict_proba(X_test_sc)[:, 1]
    evaluer("Random Forest", y_pred_rf, y_proba_rf)

    # XGBoost
    xgb_m, _ = entrainer_xgboost(X_res, y_res)
    modeles["XGBoost"] = (xgb_m, scaler)
    y_pred_xgb = xgb_m.predict(X_test_sc)
    y_proba_xgb = xgb_m.predict_proba(X_test_sc)[:, 1]
    evaluer("XGBoost", y_pred_xgb, y_proba_xgb)

    # LightGBM
    lgb_m, _ = entrainer_lightgbm(X_res, y_res)
    modeles["LightGBM"] = (lgb_m, scaler)
    y_pred_lgb = lgb_m.predict(X_test_sc)
    y_proba_lgb = lgb_m.predict_proba(X_test_sc)[:, 1]
    evaluer("LightGBM", y_pred_lgb, y_proba_lgb)

    df_res = pd.DataFrame(resultats).set_index("Modèle")
    df_res = df_res.sort_values("F1-Score", ascending=False)

    probas = {
        "Régression Logistique": y_proba_lr,
        "Random Forest": y_proba_rf,
        "XGBoost": y_proba_xgb,
        "LightGBM": y_proba_lgb,
    }
    preds = {
        "Régression Logistique": y_pred_lr,
        "Random Forest": y_pred_rf,
        "XGBoost": y_pred_xgb,
        "LightGBM": y_pred_lgb,
    }

    return {
        "df_resultats": df_res,
        "probas": probas,
        "preds": preds,
        "y_test": y_test,
        "X_test_sc": X_test_sc,
        "X_test": X_test,
        "feature_names": feature_names,
        "xgb_model": xgb_m,
        "scaler": scaler,
        "le": le,
        "modeles": modeles,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Exercice 2 — Segmentation
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Entraînement des algorithmes de clustering…")
def entrainer_modeles_clustering():
    df = charger_clients()
    df = imputer_valeurs_manquantes(df)
    df = feature_engineering_client(df)
    df = encoder_variables_categorielles(df)
    X_df = selectionner_features_clustering(df)
    X_scaled, scaler = normaliser_features(X_df)
    X_pca, pca = appliquer_pca(X_scaled, n_composantes=2)

    # Choix k optimal
    silhouettes = []
    inerties = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, lbl))
        inerties.append(km.inertia_)
    k_optimal = list(k_range)[silhouettes.index(max(silhouettes))]

    resultats = []

    # K-Means
    km_m, labels_km, p_km = entrainer_kmeans(X_scaled, k=k_optimal)
    m_km = {
        "Algorithme": "K-Means",
        "Silhouette Score": round(silhouette_score(X_scaled, labels_km), 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_scaled, labels_km), 4),
        "N Clusters": k_optimal,
    }
    resultats.append(m_km)

    # DBSCAN
    db_m, labels_db, _ = entrainer_dbscan(X_scaled)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    masque_db = labels_db != -1
    if n_clusters_db >= 2:
        m_db = {
            "Algorithme": "DBSCAN",
            "Silhouette Score": round(silhouette_score(X_scaled[masque_db], labels_db[masque_db]), 4),
            "Davies-Bouldin": round(davies_bouldin_score(X_scaled[masque_db], labels_db[masque_db]), 4),
            "N Clusters": n_clusters_db,
        }
        resultats.append(m_db)

    # Agglomératif
    agg_m, labels_agg, _ = entrainer_agglomerative(X_scaled, params={"n_clusters": k_optimal, "linkage": "ward"})
    m_agg = {
        "Algorithme": "Agglomératif",
        "Silhouette Score": round(silhouette_score(X_scaled, labels_agg), 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_scaled, labels_agg), 4),
        "N Clusters": k_optimal,
    }
    resultats.append(m_agg)

    # GMM
    gmm_m, labels_gmm, _ = entrainer_gmm(X_scaled, params={"n_components": k_optimal, "random_state": 42})
    m_gmm = {
        "Algorithme": "GMM",
        "Silhouette Score": round(silhouette_score(X_scaled, labels_gmm), 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_scaled, labels_gmm), 4),
        "N Clusters": k_optimal,
    }
    resultats.append(m_gmm)

    df_res = pd.DataFrame(resultats).set_index("Algorithme")
    df_res = df_res.sort_values("Silhouette Score", ascending=False)
    meilleur = df_res.index[0]

    labels_map = {"K-Means": labels_km, "Agglomératif": labels_agg, "GMM": labels_gmm}
    labels_best = labels_map.get(meilleur, labels_km)

    # Attribution des noms de segments
    df_clust = X_df.copy()
    df_clust["Cluster"] = labels_best
    profils = df_clust.groupby("Cluster").mean().round(2)
    ordre = profils["TotalSpent"].argsort()[::-1].values
    noms_possibles = ["Premium", "Digital", "Promo-Sensible", "Dormant", "Autre1", "Autre2", "Autre3"]
    mapping = {cluster: noms_possibles[i] for i, cluster in enumerate(ordre)}
    df_clust["Segment"] = df_clust["Cluster"].map(mapping)

    return {
        "df_resultats": df_res,
        "meilleur": meilleur,
        "labels_best": labels_best,
        "labels_km": labels_km,
        "labels_agg": labels_agg,
        "labels_gmm": labels_gmm,
        "labels_db": labels_db,
        "X_pca": X_pca,
        "pca": pca,
        "X_df": X_df,
        "X_scaled": X_scaled,
        "df_clust": df_clust,
        "profils": profils,
        "mapping": mapping,
        "k_optimal": k_optimal,
        "k_range": list(k_range),
        "silhouettes": silhouettes,
        "inerties": inerties,
        "scaler": scaler,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHAP (calculé séparément car lent)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Calcul des valeurs SHAP…")
def calculer_shap(_xgb_model, X_test_sc, feature_names):
    X_df = pd.DataFrame(X_test_sc[:500], columns=feature_names)
    explainer = shap.TreeExplainer(_xgb_model)
    shap_values = explainer.shap_values(X_df)
    return shap_values, X_df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de visualisation
# ─────────────────────────────────────────────────────────────────────────────
COULEURS = px.colors.qualitative.Set2


def fig_roc(probas, y_test):
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray"))
    for i, (nom, y_proba) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{nom} (AUC={auc:.3f})",
            line=dict(width=2, color=COULEURS[i]),
        ))
    fig.update_layout(
        title="Courbes ROC — Comparaison des modèles",
        xaxis_title="Taux de Faux Positifs",
        yaxis_title="Taux de Vrais Positifs",
        legend=dict(x=0.6, y=0.1),
        height=480,
    )
    return fig


def fig_confusion(y_test, y_pred, nom):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Réel"),
        x=["Normal", "Fraude"], y=["Normal", "Fraude"],
        title=f"Matrice de confusion — {nom}",
    )
    fig.update_layout(height=380)
    return fig


def fig_pca_clusters(X_pca, labels, mapping, pca, titre="Clusters (PCA 2D)"):
    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": [str(l) for l in labels],
        "Segment": [mapping.get(l, f"C{l}") for l in labels],
    })
    fig = px.scatter(
        df_plot, x="PC1", y="PC2", color="Segment",
        hover_data=["Cluster"],
        title=titre,
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=480, legend_title="Segment")
    return fig


def fig_profils_radar(profils, mapping):
    features = profils.columns.tolist()
    profils_norm = (profils - profils.min()) / (profils.max() - profils.min() + 1e-9)

    fig = go.Figure()
    for cluster_id, row in profils_norm.iterrows():
        nom = mapping.get(cluster_id, f"C{cluster_id}")
        fig.add_trace(go.Scatterpolar(
            r=row.values.tolist() + [row.values[0]],
            theta=features + [features[0]],
            fill="toself", name=nom,
            opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Profils normalisés des segments (radar)",
        height=480,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
def page_fraude():
    st.header("Détection de fraude bancaire")
    st.caption("Jeu de données : 1 048 575 transactions | Variable cible : `isFraud`")

    with st.spinner("Chargement et entraînement en cours…"):
        data = entrainer_modeles_fraude()

    df_res = data["df_resultats"]
    probas = data["probas"]
    preds = data["preds"]
    y_test = data["y_test"]
    X_test_sc = data["X_test_sc"]
    feature_names = data["feature_names"]
    xgb_model = data["xgb_model"]

    onglets = st.tabs([
        "📊 Métriques", "📈 Courbes ROC", "🔲 Matrice de confusion",
        "🔍 SHAP", "🎯 Prédiction interactive",
    ])

    # ── Onglet 1 : Métriques ────────────────────────────────────────────────
    with onglets[0]:
        st.subheader("Comparaison des modèles")
        meilleur = df_res.index[0]
        st.success(f"Meilleur modèle : **{meilleur}** (F1-Score = {df_res.loc[meilleur, 'F1-Score']:.4f})")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1-Score", f"{df_res.loc[meilleur, 'F1-Score']:.4f}")
        col2.metric("Recall", f"{df_res.loc[meilleur, 'Recall']:.4f}")
        col3.metric("Precision", f"{df_res.loc[meilleur, 'Precision']:.4f}")
        col4.metric("ROC-AUC", f"{df_res.loc[meilleur, 'ROC-AUC']:.4f}")

        st.divider()

        # Tableau stylisé
        st.dataframe(
            df_res.style
            .set_properties(**{"background-color": "#1e2a3a", "color": "#ffffff", "border-color": "#3a4a5a"})
            .highlight_max(subset=["F1-Score", "Recall", "ROC-AUC"], color="#1a6b3c")
            .highlight_min(subset=["F1-Score", "Recall", "ROC-AUC"], color="#7a1c2a")
            .background_gradient(subset=["Accuracy", "Precision"], cmap="Blues")
            .format("{:.4f}"),
            use_container_width=True,
        )

        # Graphique barres
        fig = px.bar(
            df_res.reset_index(), x="Modèle",
            y=["F1-Score", "Recall", "Precision", "ROC-AUC"],
            barmode="group", title="Métriques comparatives par modèle",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=420, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Rappel métier** : dans la détection de fraude, le **Recall** est "
            "la métrique prioritaire — une fraude non détectée coûte bien plus "
            "qu'une fausse alarme."
        )

    # ── Onglet 2 : ROC ──────────────────────────────────────────────────────
    with onglets[1]:
        st.subheader("Courbes ROC — Comparaison")
        st.plotly_chart(fig_roc(probas, y_test), use_container_width=True)
        st.caption(
            "La courbe ROC mesure la capacité discriminante indépendamment du seuil. "
            "Un AUC proche de 1 indique un modèle quasi-parfait."
        )

    # ── Onglet 3 : Matrice de confusion ─────────────────────────────────────
    with onglets[2]:
        st.subheader("Matrice de confusion")
        modele_choisi = st.selectbox("Choisir un modèle", list(preds.keys()), index=0)
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.plotly_chart(
                fig_confusion(y_test, preds[modele_choisi], modele_choisi),
                use_container_width=True,
            )
        with col_right:
            st.markdown("### Interprétation")
            cm = confusion_matrix(y_test, preds[modele_choisi])
            vn, fp, fn, vp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            st.metric("Vrais Négatifs (Normal → Normal)", f"{vn:,}")
            st.metric("Faux Positifs (Normal → Fraude)", f"{fp:,}", delta=f"-{fp}", delta_color="inverse")
            st.metric("Faux Négatifs (Fraude → Normal)", f"{fn:,}", delta=f"-{fn}", delta_color="inverse")
            st.metric("Vrais Positifs (Fraude → Fraude)", f"{vp:,}")
            st.markdown(f"""
| Erreur | Impact métier |
|---|---|
| Faux Positif ({fp}) | Blocage d'une transaction légitime |
| Faux Négatif ({fn}) | Fraude non détectée — **coût élevé** |
""")

    # ── Onglet 4 : SHAP ─────────────────────────────────────────────────────
    with onglets[3]:
        st.subheader("Interprétabilité — SHAP (XGBoost)")
        st.caption("Valeurs SHAP calculées sur les 500 premiers exemples du jeu de test.")

        shap_values, X_shap = calculer_shap(xgb_model, X_test_sc, feature_names)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Importance globale (valeur absolue moyenne)**")
            mean_abs = np.abs(shap_values).mean(axis=0)
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance SHAP": mean_abs,
            }).sort_values("Importance SHAP", ascending=True)
            fig_imp = px.bar(
                df_imp, x="Importance SHAP", y="Feature",
                orientation="h", title="Feature Importance SHAP",
                color="Importance SHAP", color_continuous_scale="Blues",
            )
            fig_imp.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            st.markdown("**Summary Plot (beeswarm)**")
            fig_shap, ax = plt.subplots(figsize=(7, 5))
            shap.summary_plot(shap_values, X_shap, show=False, max_display=12, plot_size=None)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

        st.info(
            "Les variables de **feature engineering** (`erreur_solde_orig`, `diff_solde_orig`) "
            "dominent systématiquement l'importance SHAP, validant leur pertinence métier."
        )

    # ── Onglet 5 : Prédiction interactive ───────────────────────────────────
    with onglets[4]:
        st.subheader("Simuler une transaction")
        st.caption("Renseignez les caractéristiques d'une transaction pour obtenir une prédiction.")

        col1, col2, col3 = st.columns(3)
        with col1:
            type_tx = st.selectbox("Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
            montant = st.number_input("Montant (€)", min_value=0.0, value=5000.0, step=100.0)
            step = st.number_input("Step (heure)", min_value=1, value=1)
        with col2:
            old_bal_orig = st.number_input("Solde émetteur avant", min_value=0.0, value=10000.0)
            new_bal_orig = st.number_input("Solde émetteur après", min_value=0.0, value=5000.0)
        with col3:
            old_bal_dest = st.number_input("Solde destinataire avant", min_value=0.0, value=0.0)
            new_bal_dest = st.number_input("Solde destinataire après", min_value=0.0, value=5000.0)

        if st.button("Prédire", type="primary"):
            le = data["le"]
            scaler = data["scaler"]
            modele_pred = data["modeles"]["XGBoost"][0]

            type_enc = le.transform([type_tx])[0] if type_tx in le.classes_ else 0
            diff_orig = old_bal_orig - new_bal_orig
            diff_dest = new_bal_dest - old_bal_dest
            err_orig = diff_orig - montant
            err_dest = diff_dest - montant

            X_input = pd.DataFrame([[
                step, type_enc, montant,
                old_bal_orig, new_bal_orig,
                old_bal_dest, new_bal_dest,
                diff_orig, diff_dest, err_orig, err_dest,
            ]], columns=feature_names)

            X_input_sc = scaler.transform(X_input)
            proba = modele_pred.predict_proba(X_input_sc)[0, 1]
            prediction = int(proba >= 0.5)

            st.divider()
            if prediction == 1:
                st.error(f"🚨 **FRAUDE DÉTECTÉE** — Probabilité : {proba:.1%}")
            else:
                st.success(f"✅ **Transaction normale** — Probabilité de fraude : {proba:.1%}")

            fig_jauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Risque de fraude (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if prediction == 1 else "darkgreen"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 70], "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
                },
            ))
            fig_jauge.update_layout(height=320)
            st.plotly_chart(fig_jauge, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_segmentation():
    st.header("Segmentation client")
    st.caption("Jeu de données : 2 240 clients | Clustering non supervisé")

    with st.spinner("Clustering en cours…"):
        data = entrainer_modeles_clustering()

    df_res = data["df_resultats"]
    meilleur = data["meilleur"]
    labels_best = data["labels_best"]
    X_pca = data["X_pca"]
    pca = data["pca"]
    X_df = data["X_df"]
    X_scaled = data["X_scaled"]
    df_clust = data["df_clust"]
    profils = data["profils"]
    mapping = data["mapping"]
    k_optimal = data["k_optimal"]
    k_range = data["k_range"]
    silhouettes = data["silhouettes"]
    inerties = data["inerties"]

    onglets = st.tabs([
        "📊 Métriques", "🗺️ Visualisation PCA",
        "👤 Profils des segments", "🎯 Client interactif", "💼 Recommandations",
    ])

    # ── Onglet 1 : Métriques ────────────────────────────────────────────────
    with onglets[0]:
        st.subheader("Comparaison des algorithmes de clustering")
        st.success(f"Meilleur algorithme : **{meilleur}** | k optimal = **{k_optimal}**")

        col1, col2 = st.columns(2)
        col1.metric(
            "Silhouette Score (↑ meilleur)",
            f"{df_res.loc[meilleur, 'Silhouette Score']:.4f}",
        )
        col2.metric(
            "Davies-Bouldin (↓ meilleur)",
            f"{df_res.loc[meilleur, 'Davies-Bouldin']:.4f}",
        )

        st.divider()
        st.dataframe(
            df_res.style
            .set_properties(**{"background-color": "#1e2a3a", "color": "#ffffff", "border-color": "#3a4a5a"})
            .highlight_max(subset=["Silhouette Score"], color="#1a6b3c")
            .highlight_min(subset=["Davies-Bouldin"], color="#1a6b3c")
            .highlight_min(subset=["Silhouette Score"], color="#7a1c2a")
            .highlight_max(subset=["Davies-Bouldin"], color="#7a1c2a")
            .format({"Silhouette Score": "{:.4f}", "Davies-Bouldin": "{:.4f}"}),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Méthode Elbow + Silhouette — Choix de k")

        fig_elbow = make_subplots(rows=1, cols=2, subplot_titles=("Méthode Elbow", "Silhouette Score"))
        fig_elbow.add_trace(
            go.Scatter(x=k_range, y=inerties, mode="lines+markers",
                       line=dict(color="steelblue", width=2), name="Inertie"),
            row=1, col=1,
        )
        fig_elbow.add_trace(
            go.Scatter(x=k_range, y=silhouettes, mode="lines+markers",
                       line=dict(color="coral", width=2), name="Silhouette"),
            row=1, col=2,
        )
        fig_elbow.add_vline(x=k_optimal, line_dash="dash", line_color="green",
                            annotation_text=f"k={k_optimal}", row=1, col=2)
        fig_elbow.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_elbow, use_container_width=True)

    # ── Onglet 2 : PCA ──────────────────────────────────────────────────────
    with onglets[1]:
        st.subheader("Visualisation des clusters en 2D — PCA")

        algo_choisi = st.radio(
            "Algorithme à visualiser",
            ["K-Means", "Agglomératif", "GMM"],
            horizontal=True,
        )
        lbl_map = {
            "K-Means": data["labels_km"],
            "Agglomératif": data["labels_agg"],
            "GMM": data["labels_gmm"],
        }
        labels_viz = lbl_map[algo_choisi]
        st.plotly_chart(
            fig_pca_clusters(X_pca, labels_viz, mapping, pca, titre=f"Segments — {algo_choisi}"),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Variance expliquée par les 2 composantes principales**")
            var_df = pd.DataFrame({
                "Composante": ["PC1", "PC2"],
                "Variance expliquée": pca.explained_variance_ratio_,
            })
            fig_var = px.bar(var_df, x="Composante", y="Variance expliquée",
                             color="Composante",
                             color_discrete_sequence=["steelblue", "coral"],
                             title="Variance expliquée par PCA")
            fig_var.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_var, use_container_width=True)

        with col2:
            seg_counts = df_clust["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Effectif"]
            fig_pie = px.pie(
                seg_counts, names="Segment", values="Effectif",
                title="Répartition des clients par segment",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Onglet 3 : Profils ──────────────────────────────────────────────────
    with onglets[2]:
        st.subheader("Profils moyens des segments")

        profils_named = profils.copy()
        profils_named.index = [mapping.get(i, f"C{i}") for i in profils.index]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tableau des profils moyens**")
            st.dataframe(profils_named.T.style.background_gradient(cmap="YlOrRd", axis=1),
                         use_container_width=True)

        with col2:
            st.markdown("**Radar — Profils normalisés**")
            st.plotly_chart(fig_profils_radar(profils, mapping), use_container_width=True)

        st.divider()
        st.subheader("Comparaison par variable clé")
        feature_sel = st.selectbox("Variable à comparer", X_df.columns.tolist(), index=1)

        df_box = df_clust.copy()
        df_box["Segment"] = df_box["Cluster"].map(mapping)
        fig_box = px.box(
            df_box, x="Segment", y=feature_sel, color="Segment",
            title=f"Distribution de `{feature_sel}` par segment",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Onglet 4 : Client interactif ────────────────────────────────────────
    with onglets[3]:
        st.subheader("Identifier le segment d'un nouveau client")
        st.caption("Renseignez le profil client pour obtenir sa segmentation.")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Âge", 18, 90, 45)
            income = st.number_input("Revenu annuel (€)", 0, 200_000, 50_000, step=1000)
            education = st.selectbox("Éducation", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"])
        with col2:
            marital = st.selectbox("Situation", ["Single", "Married", "Divorced", "Widow", "Together"])
            has_children = st.checkbox("A des enfants", value=False)
            recency = st.slider("Récence (jours depuis dernier achat)", 0, 99, 30)
        with col3:
            total_spent = st.number_input("Dépenses totales (€)", 0, 3000, 500, step=50)
            total_purchases = st.number_input("Nombre total d'achats", 0, 50, 10)
            web_visits = st.slider("Visites web / mois", 0, 20, 5)
            total_cmp = st.slider("Campagnes acceptées", 0, 6, 1)
            complain = st.checkbox("A déjà réclamé", value=False)

        if st.button("Segmenter ce client", type="primary"):
            edu_map = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
            mar_map = {"Single": 0, "Divorced": 1, "Widow": 1, "Married": 2, "Together": 2}

            X_client = pd.DataFrame([[
                age, income, edu_map[education], mar_map[marital],
                int(has_children), recency, total_spent,
                total_purchases, web_visits, total_cmp, int(complain),
            ]], columns=X_df.columns.tolist())

            scaler = data["scaler"]
            km_model = entrainer_kmeans.__wrapped__ if hasattr(entrainer_kmeans, "__wrapped__") else None

            X_client_sc = scaler.transform(X_client)
            # Utiliser K-Means déjà entraîné (via cache)
            from sklearn.cluster import KMeans as _KM
            km_reload = _KM(n_clusters=k_optimal, random_state=42, n_init=10)
            km_reload.fit(X_scaled)
            cluster_pred = int(km_reload.predict(X_client_sc)[0])
            segment = mapping.get(cluster_pred, f"C{cluster_pred}")

            couleurs_seg = {
                "Premium": "🔵", "Digital": "🟢",
                "Promo-Sensible": "🟠", "Dormant": "⚫",
            }
            icone = couleurs_seg.get(segment, "🔘")
            st.success(f"{icone} **Segment identifié : {segment}** (Cluster {cluster_pred})")

            profil_ref = profils_named.loc[segment] if segment in profils_named.index else None
            if profil_ref is not None:
                st.markdown("### Comparaison avec le profil moyen du segment")
                comp = pd.DataFrame({
                    "Ce client": X_client.iloc[0],
                    f"Moyenne {segment}": profil_ref,
                })
                st.dataframe(comp.T.style.background_gradient(cmap="Blues"), use_container_width=True)

    # ── Onglet 5 : Recommandations ──────────────────────────────────────────
    with onglets[4]:
        st.subheader("Recommandations business par segment")

        recommandations = {
            "Premium": {
                "icone": "💎",
                "description": "Clients à fort revenu, dépenses élevées, engagés sur plusieurs campagnes.",
                "objectif": "Fidélisation et montée en gamme",
                "actions": [
                    "Programme VIP avec avantages exclusifs",
                    "Invitations à des événements privés",
                    "Offres de produits luxe personnalisées",
                    "Programme de parrainage avec récompenses premium",
                ],
                "kpi": "Taux de rétention, panier moyen, NPS",
            },
            "Digital": {
                "icone": "💻",
                "description": "Clients actifs en ligne, achats web fréquents, profil jeune.",
                "objectif": "Activation et cross-sell",
                "actions": [
                    "Campagnes email/push personnalisées",
                    "Recommandations produits basées sur l'historique",
                    "Abonnements et services en ligne",
                    "Offres flash avec expiration courte",
                ],
                "kpi": "CTR email, taux de conversion web, fréquence d'achat",
            },
            "Promo-Sensible": {
                "icone": "🏷️",
                "description": "Clients à revenu modeste, achats déclenchés par les promotions.",
                "objectif": "Conversion et augmentation de la fréquence",
                "actions": [
                    "Promotions temporaires et ventes flash",
                    "Programme de points de fidélité",
                    "Bundle pricing (offres groupées)",
                    "Coupons de réduction ciblés par catégorie",
                ],
                "kpi": "Taux d'utilisation des coupons, fréquence d'achat, revenu par visite",
            },
            "Dormant": {
                "icone": "💤",
                "description": "Clients inactifs, faibles dépenses, non-répondants aux campagnes.",
                "objectif": "Réactivation ou désengagement maîtrisé",
                "actions": [
                    "Campagne win-back avec offre de retour attractive",
                    "Sondage satisfaction pour identifier les freins",
                    "Réduction des coûts de contact (email uniquement)",
                    "Désabonnement des inactifs depuis >12 mois",
                ],
                "kpi": "Taux de réactivation, coût par client réactivé",
            },
        }

        segments_presents = list(mapping.values())
        for segment, infos in recommandations.items():
            if segment not in segments_presents:
                continue
            with st.expander(f"{infos['icone']} **{segment}** — {infos['objectif']}", expanded=True):
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.markdown(f"**Description** : {infos['description']}")
                    st.markdown(f"**Objectif** : {infos['objectif']}")
                    st.markdown(f"**KPI à suivre** : _{infos['kpi']}_")

                    # Taille du segment
                    n_seg = (df_clust["Segment"] == segment).sum()
                    pct = n_seg / len(df_clust) * 100
                    st.metric("Clients dans ce segment", f"{n_seg} ({pct:.1f}%)")

                with col2:
                    st.markdown("**Actions recommandées :**")
                    for action in infos["actions"]:
                        st.markdown(f"- {action}")

        st.divider()
        st.subheader("Bilan — Comparaison des segments")

        recap = df_clust.groupby("Segment").agg(
            Effectif=("Cluster", "count"),
            Revenu_Moyen=("Income", "mean"),
            Depenses_Moyennes=("TotalSpent", "mean"),
            Age_Moyen=("Age", "mean"),
            Campagnes_Moyennes=("TotalCampaignes", "mean"),
        ).round(1)
        recap.index.name = "Segment"
        st.dataframe(recap.style.background_gradient(cmap="YlOrRd"), use_container_width=True)

        fig_recap = px.bar(
            recap.reset_index(),
            x="Segment", y=["Revenu_Moyen", "Depenses_Moyennes"],
            barmode="group", title="Revenu vs Dépenses moyennes par segment",
            color_discrete_sequence=["steelblue", "coral"],
        )
        fig_recap.update_layout(height=380)
        st.plotly_chart(fig_recap, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar + routing
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.title("📊 Dashboard ML")
        st.markdown("**Analyse & Modélisation — Data Science**")
        st.divider()

        page = st.radio(
            "Navigation",
            ["Détection de fraude", "Segmentation client"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### À propos")
        st.markdown(
            "Plateforme MLOps de suivi et de visualisation des performances "
            "des modèles de Machine Learning déployés en production."
        )
        st.markdown("**Détection de fraude** : LR · RF · XGBoost · LightGBM")
        st.markdown("**Segmentation client** : K-Means · DBSCAN · Agglomératif · GMM")

        st.divider()
        if st.button("🔄 Réinitialiser le cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if page == "Détection de fraude":
        page_fraude()
    else:
        page_segmentation()


if __name__ == "__main__":
    main()
