# Fire-retrain – API de réentraînement et préparation des features

Ce dossier contient une **API FastAPI** dédiée au **réentraînement automatique du modèle de prédiction des risques d’incendies de forêt en Corse** ainsi qu’à la **préparation des données météorologiques** utilisées pour la prédiction.

Cette API s’inscrit dans une architecture MLOps pilotée par **Airflow**, **MLflow**, **AWS S3** et une **base de données Neon**, et alimente une application de visualisation et de suivi via **Streamlit**.

---

## Contenu du dossier

```
Fire-retrain/
├── Dockerfile
├── app.py
├── retrain.py
├── features_fusion.py
├── requirements.txt
└── .secrets (à créer)
```

### Description des fichiers

* **Dockerfile**
  Construit l’image Docker nécessaire au déploiement de l’API FastAPI et installe l’ensemble des dépendances.

* **app.py**
  Point d’entrée de l’application FastAPI.
  Définit et expose les endpoints `/retrain` et `/features`.

* **retrain.py**
  Contient la logique de réentraînement du modèle :

  * relance l’entraînement du modèle de risque d’incendie,
  * enregistre les métriques et paramètres dans MLflow,
  * sauvegarde le modèle sur AWS S3,
  * enregistre les résultats (métriques, dates, version du modèle) dans une base de données Neon pour le suivi.

* **features_fusion.py**
  Gère le nettoyage, la transformation et la préparation des données météorologiques nécessaires à la prédiction du risque d’incendie.

* **requirements.txt**
  Liste l’ensemble des dépendances Python nécessaires à l’API et aux traitements ML.

* **.secrets** (non versionné)
  Fichier de configuration des variables d’environnement sensibles (AWS, MLflow, Neon, etc.).

---

## Rôle de l’API dans l’architecture

Cette API joue un rôle central entre :

* les **pipelines Airflow**,
* la **chaîne MLOps** (MLflow + S3),
* les **bases de données opérationnelles**,
* et l’**application Streamlit**.

Elle permet à la fois :

* l’automatisation du **réentraînement du modèle**,
* la **préparation des features météo** pour les prédictions quotidiennes.

---

## Endpoints disponibles

### `POST /retrain`

Endpoint déclenché automatiquement par **Airflow**.

#### Cas d’usage

* Un DAG Airflow surveille les performances du modèle.
* Si le **c-index passe sous le seuil de 0.69**, l’endpoint `/retrain` est appelé.

#### Actions réalisées

* Réentraînement du modèle de prédiction du risque d’incendie
* Enregistrement des métriques et paramètres dans MLflow
* Sauvegarde du modèle sur AWS S3
* Insertion des résultats dans une base de données Neon

  * utilisée pour alimenter un tableau de suivi dans l’application Streamlit

---

### `POST /features`

Endpoint déclenché par le DAG **`get_meteoday`**.

#### Cas d’usage

* Récupération quotidienne des données météorologiques
* Nettoyage et transformation des données
* Construction des features nécessaires à la prédiction du risque d’incendie

#### Actions réalisées

* Préparation des données météo
* Enregistrement des features finales dans une base de données Neon
* Les données sont ensuite consommées par l’application Streamlit pour la prédiction

---

## Configuration des variables d’environnement

### Création du fichier `.secrets`

Créer un fichier `.secrets` à la racine du projet avec les variables suivantes (exemple) :

```bash
MLFLOW_DEFAULT_ARTIFACT_ROOT = 
AWS_ACCESS_KEY_ID= 
AWS_SECRET_ACCESS_KEY= 
BACKEND_STORE_URI= 
PORT = 
S3_BUCKET=


DB_USER = 
DB_NAME = 
DB_PASSWORD = 
DB_HOST = 
```

Ce fichier :

* ne doit pas être versionné,
* doit être ajouté au `.gitignore`.

---

## Objectif MLOps du projet

Ce service permet de :

* automatiser le **cycle de vie du modèle**,
* garantir un **niveau de performance minimal** via le retraining conditionnel,
* centraliser les métriques et modèles,
* préparer les données de prédiction de manière robuste et traçable.

Il constitue une brique essentielle de l’architecture de **prédiction des risques d’incendies de forêt en Corse**.


