
# Stream-fire – Application de visualisation du risque d’incendies

Ce dossier contient une **application Streamlit** dédiée à la **visualisation et au suivi du risque d’incendies de forêt en Corse**.

L’application constitue la **couche de restitution** du projet et s’appuie sur l’ensemble de l’infrastructure mise en place (Airflow, MLflow, AWS S3, Neon). Elle permet de consulter les prédictions mises à jour quotidiennement, d’explorer les données historiques et de visualiser les moyens de lutte contre les incendies.

---

## Contenu du dossier

```
Stream-fire/
├── Dockerfile
├── app.py
└── requirements.txt
```

### Description des fichiers

* **Dockerfile**
  Construit l’image Docker nécessaire au déploiement de l’application Streamlit sur Hugging Face Spaces et installe les dépendances requises.

* **app.py**
  Point d’entrée de l’application Streamlit.
  Contient l’ensemble des pages, visualisations et logiques d’accès aux données.

* **requirements.txt**
  Liste des dépendances Python nécessaires à l’exécution de l’application.

---

## Fonctionnalités de l’application

L’application Streamlit permet :

### Présentation du projet

* Contexte des incendies de forêt en Corse
* Objectifs du projet
* Description de l’architecture technique et MLOps

### Carte quotidienne du risque d’incendie

* Visualisation des **prédictions de risque d’incendie**
* Carte mise à jour automatiquement chaque jour
* Données préparées et mises à disposition par les pipelines Airflow

### Moyens de lutte contre les incendies

* Carte recensant l’ensemble des **moyens de lutte contre les incendies de forêt en Corse**
* Vision géographique des ressources disponibles

### Analyse historique des feux en France

* Données historiques couvrant la période **2006 – 2024**
* Exploration interactive :

  * tendances temporelles,
  * répartitions géographiques,
  * statistiques descriptives.

---

## Déploiement sur Hugging Face Spaces

L’application est conçue pour être déployée sur **Hugging Face Spaces** avec une configuration Docker.

### Étapes principales

1. Créer un nouvel espace Hugging Face :

   * Type : Docker
   * Framework : Streamlit

2. Ajouter le contenu du dossier `Stream-fire` à l’espace.

3. Hugging Face utilisera automatiquement le **Dockerfile** pour construire et lancer l’application.

---

## Configuration des variables d’environnement

Les variables sensibles (accès aux bases de données, services cloud, etc.) doivent être configurées **directement dans les Settings de l’espace Hugging Face**.

### Exemples de variables à configurer

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

Aucun fichier `.secrets` n’est stocké dans le dépôt pour cette application.

---

## Objectif du module Streamlit

Cette application permet de :

* rendre les résultats du projet accessibles à un public non technique,
* centraliser la visualisation des prédictions et des données historiques,
* fournir un outil de suivi et d’aide à la décision autour du risque d’incendies de forêt.

Elle constitue la **vitrine fonctionnelle** du projet de prédiction des risques d’incendies de forêt en Corse.


