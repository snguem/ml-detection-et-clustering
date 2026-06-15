# Détection de fraude & Segmentation client

Auteur : **Steeve NGUEMA** — Développeur web (Spring / Angular)  
Date : Juin 2026

---

## Présentation

Ce projet regroupe deux analyses de Machine Learning conduites sur des données financières et marketing, accompagnées d'une plateforme MLOps permettant de visualiser et de suivre les résultats en production.

**Projet 1 — Détection de fraude bancaire**  
Développement d'un système de classification supervisée pour identifier automatiquement les transactions frauduleuses dans un flux de paiements mobiles.

**Projet 2 — Segmentation client**  
Identification de segments de clients homogènes à partir de données CRM, pour orienter les stratégies marketing de fidélisation et de ciblage.

---

## Structure du projet

```
projet/
│
├── donnees/
│   ├── brutes/                          # Fichiers CSV sources — ne pas modifier
│   │   ├── détection_de_fraude.csv      # Projet 1 — 1 048 575 transactions
│   │   └── data_cluster_exercice_2.csv  # Projet 2 — 2 240 clients
│   └── traitees/                        # Données après prétraitement (générées)
│
├── notebooks/
│   ├── 01_fraude.ipynb                  # Analyse complète — détection de fraude
│   └── 02_segmentation.ipynb            # Analyse complète — segmentation client
│
├── src/
│   ├── utils.py          # Chargement et sauvegarde des modèles
│   ├── preprocessing.py  # Nettoyage, encodage, normalisation, SMOTE
│   ├── models.py         # Entraînement des modèles supervisés et de clustering
│   └── evaluation.py     # Métriques et visualisations
│
├── mlops/
│   ├── api/main.py        # API FastAPI — endpoints de prédiction
│   └── dashboard/app.py   # Dashboard Streamlit — visualisation des résultats
│
├── rapports/
│   ├── rapport_detection_fraude.html    # Rapport HTML — Projet 1
│   └── rapport_segmentation_client.html # Rapport HTML — Projet 2
│
├── mlruns/               # Suivi des expériences MLflow (généré automatiquement)
└── requirements.txt
```

---

## Installation

### Prérequis

- Python 3.9 ou supérieur
- Git

### 1. Cloner le dépôt

### 2. Créer un environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

> **Note Windows** : si le chemin du projet contient des espaces, créez l'environnement dans un chemin court, par exemple `C:\venv_projet\`.

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer les notebooks

```bash
jupyter notebook
```

Ouvrir `notebooks/01_fraude.ipynb` puis `notebooks/02_segmentation.ipynb` et exécuter toutes les cellules dans l'ordre.  
Les modèles entraînés seront sauvegardés automatiquement et les runs MLflow enregistrés dans `mlruns/`.

### 5. Lancer le dashboard Streamlit

```bash
streamlit run mlops/dashboard/app.py
```

Accès depuis le navigateur : [http://localhost:8501](http://localhost:8501)

### 6. (Optionnel) Consulter les expériences MLflow

```bash
mlflow ui
```

Accès depuis le navigateur : [http://localhost:5000](http://localhost:5000)

---

## Projet 1 — Détection de fraude bancaire

**Fichier** : `notebooks/01_fraude.ipynb`  
**Données** : 1 048 575 transactions financières mobiles, séparateur `;`  
**Objectif** : classifier chaque transaction comme légitime ou frauduleuse (`isFraud`)

### Démarche

**1. Chargement et description des données**  
Le jeu de données est issu d'une simulation de transactions mobiles. Il ne présente aucune valeur manquante. Les montants varient fortement, ce qui justifie une normalisation préalable.

**2. Analyse exploratoire (EDA)**  
Le taux de fraude est inférieur à 0,2 % — le déséquilibre des classes est extrême. Les fraudes se concentrent exclusivement sur les types `CASH_OUT` (retrait) et `TRANSFER` (virement). Des anomalies de soldes caractéristiques (solde émetteur vidé, solde destinataire inchangé) sont identifiées comme signaux discriminants.

**3. Prétraitement**  
- Suppression des colonnes non pertinentes (`nameOrig`, `nameDest`, `isFlaggedFraud`)
- Création de 4 variables synthétiques à partir des anomalies de soldes (`diff_solde_orig`, `diff_solde_dest`, `erreur_solde_orig`, `erreur_solde_dest`)
- Encodage de la variable `type`, split stratifié train/test, normalisation (`StandardScaler`)
- Rééquilibrage par **SMOTE** appliqué uniquement sur le jeu d'entraînement

**4. Modélisation (suivi MLflow)**  
Cinq algorithmes comparés : Régression Logistique, Random Forest, XGBoost, LightGBM, Réseau de neurones (MLP Keras).

**5. Évaluation**  
Métriques prioritaires : **Recall** et **F1-Score** sur la classe fraude. Courbes ROC superposées, matrice de confusion du meilleur modèle.

**6. Interprétabilité**  
Importance des variables XGBoost et analyse **SHAP** (graphiques bar et beeswarm) pour justifier les décisions de classification auprès des équipes métier et réglementaires. Les variables synthétiques de soldes ressortent systématiquement parmi les plus déterminantes.

---

## Projet 2 — Segmentation client

**Fichier** : `notebooks/02_segmentation.ipynb`  
**Données** : 2 240 clients issus d'un CRM marketing (informations socio-démographiques, comportements d'achat, canaux, réponses aux campagnes)  
**Objectif** : identifier des segments homogènes pour personnaliser les actions marketing

### Démarche

**1. Chargement et description des données**  
29 variables disponibles. Seule la variable `Income` présente 24 valeurs manquantes (~1 %). Les colonnes constantes (`Z_CostContact`, `Z_Revenue`) et les identifiants sont supprimés.

**2. Analyse exploratoire (EDA)**  
La clientèle est majoritairement âgée de 35 à 65 ans. Les vins représentent près de 50 % des dépenses totales. Le magasin physique reste le canal dominant, suivi du web. Les taux de réponse aux campagnes sont faibles (7–15 %), typiques du marketing direct.

**3. Prétraitement**  
- Imputation des valeurs manquantes par la médiane
- Création de 5 variables synthétiques : `Age`, `TotalSpent`, `TotalPurchases`, `HasChildren`, `CampaignAccepted`
- Encodage ordinal (`Education`, `Marital_Status`), normalisation (`StandardScaler`)
- PCA 2D pour la visualisation uniquement (le clustering s'effectue sur les 11 dimensions complètes)

**4. Modélisation**  
Quatre algorithmes comparés : K-Means (avec méthode Elbow + Silhouette pour choisir `k`), DBSCAN, Clustering Agglomératif, Gaussian Mixture Models (GMM).

**5. Évaluation**  
Silhouette Score et Davies-Bouldin Score. Visualisation des clusters en 2D via PCA.

**6. Interprétation métier des clusters**

| Segment | Profil | Levier d'action |
|---|---|---|
| **Clients Premium** | Revenu élevé, dépenses maximales, multi-canal, fidèles aux campagnes | Programme VIP, offres exclusives, parrainage |
| **Clients Digitaux** | Jeunes, orientés web, revenu moyen, achats fréquents en ligne | Campagnes email/push, recommandations personnalisées |
| **Clients Promo-Sensibles** | Revenu modeste, sensibles aux promotions, peu fidèles | Promotions temporaires, bons de réduction, programme de points |
| **Clients Dormants** | Inactifs depuis longtemps, faibles dépenses, peu réactifs | Campagnes de réactivation, offres de retour |

---

## Dashboard MLOps

Le dashboard Streamlit centralise la visualisation des résultats des deux projets sans avoir à rouvrir les notebooks.

- **Détection de fraude** : tableau comparatif des métriques, courbes ROC, analyse SHAP
- **Segmentation client** : métriques de clustering, méthode Elbow, profils moyens par segment, visualisation PCA

```bash
streamlit run mlops/dashboard/app.py
```

---

## Rapports

Les rapports HTML autonomes (sans code Python, style professionnel) sont disponibles dans le dossier `rapports/`. Ils peuvent être ouverts directement dans un navigateur ou imprimés.

Pour les régénérer après modification des notebooks, **l'environnement virtuel doit être actif** :

```bash
# Windows — activer le venv d'abord
C:\venv_fraude\Scripts\Activate.ps1
python rapports/generer_rapports.py

# Ou directement sans activer le venv
C:\venv_fraude\Scripts\python.exe rapports/generer_rapports.py
```

> **Important** : ne pas utiliser le Python système (`python.exe` ou `C:\PythonXXX\python.exe`) — les dépendances (`nbconvert`, `pandas`, etc.) sont installées uniquement dans le venv.

---

## Stack technique

| Domaine | Outils |
|---|---|
| Manipulation des données | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Réseau de neurones | TensorFlow / Keras |
| Rééquilibrage | imbalanced-learn (SMOTE) |
| Interprétabilité | SHAP |
| Suivi des expériences | MLflow |
| Dashboard | Streamlit |
| API | FastAPI, Uvicorn |
