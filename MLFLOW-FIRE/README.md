# MLflow Fire – Model Tracking & Registry

Ce dossier contient l’infrastructure et le code nécessaires pour **déployer MLflow**, **entraîner/enregistrer un modèle de prédiction du risque d’incendie de forêt en Corse**, et **stocker les artefacts et métriques sur AWS S3**, avec une interface MLflow hébergée sur **Hugging Face Spaces**.

L’objectif est de fournir un socle reproductible pour le **suivi d’expériences (experiment tracking)** et la **gestion de modèles** dans un contexte MLOps.

---

## Contenu du dossier

```
MLFLOW-FIRE/
├── Dockerfile
├── requirements.txt
├── model.py
└── .secrets (à créer)
```

### Description des fichiers

* **Dockerfile**
  Permet de construire une image Docker contenant MLflow et ses dépendances, destinée à être déployée sur Hugging Face Spaces.

* **requirements.txt**
  Liste l’ensemble des dépendances Python nécessaires à l’exécution du projet (MLflow, bibliothèques ML, accès S3, etc.).

* **model.py**
  Script principal qui :

  * entraîne ou charge un modèle de prédiction du risque d’incendie,
  * enregistre le modèle dans un bucket S3 via MLflow,
  * logge les métriques et paramètres dans MLflow.

* **.secrets** (non versionné)
  Fichier contenant les variables d’environnement sensibles nécessaires à la connexion AWS et à MLflow.

---

## Prérequis

* Un compte **AWS** avec accès à :

  * S3 (bucket pour les artefacts MLflow)
* Un compte **Hugging Face**
* Docker installé en local (pour les tests éventuels)
* Python ≥ 3.9

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

```

Aucun fichier `.secrets` n’est stocké dans le dépôt pour cette application.

---

## Déploiement de MLflow sur Hugging Face Spaces

1. Créer un **nouvel espace Hugging Face** :

   * Type : Docker
   * Visibilité : selon vos besoins (public / privé)

2. Ajouter le contenu de ce dossier à l’espace Hugging Face.

3. Hugging Face utilisera automatiquement le **Dockerfile** pour construire et lancer MLflow.

4. Une fois l’espace déployé, récupérer l’URL publique de MLflow, par exemple :

```
https://username-mlflow-fire.hf.space
```

---

## Configuration de MLflow dans `model.py`

Avant de lancer le script `model.py`, **modifier la variable `mlflow_tracking_uri`** avec l’URL de votre espace Hugging Face :

```python
mlflow.set_tracking_uri("https://username-mlflow-fire.hf.space")
```

Cette étape est indispensable pour que :

* les métriques soient visibles dans l’interface MLflow,
* les runs soient correctement enregistrés.

---

## Exécution du script

Une fois :

* MLflow déployé sur Hugging Face,
* les variables AWS configurées,
* le tracking URI mis à jour,

lancer simplement :

```bash
python model.py
```

Le script :

* exécute le pipeline de modélisation,
* enregistre le modèle sur S3,
* logge les métriques, paramètres et artefacts dans MLflow.

---

## Objectif MLOps du projet

Ce dossier s’inscrit dans un projet plus large de **prédiction du risque d’incendie de forêt en Corse**, et vise à :

* structurer le suivi d’expériences ML,
* centraliser les métriques et modèles,
* préparer l’industrialisation (CI/CD, retraining, monitoring).



