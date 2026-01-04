
# Db_neon – Initialisation des bases de données Neon

Ce dossier contient les éléments nécessaires à la **création et à l’initialisation des bases de données sur Neon** utilisées dans le projet de prédiction des risques d’incendies de forêt en Corse.

Il est destiné à être exécuté **une seule fois** (ou ponctuellement) lors de la mise en place de l’infrastructure ou lors d’une réinitialisation des bases.

---

## Contenu du dossier

```
Db_neon/
├── create_database.ipynb
└── .secrets (à créer)
```

### Description des fichiers

* **create-database.ipynb**
  Notebook permettant :

  * de se connecter à Neon,
  * de créer les bases de données et/ou schémas nécessaires au projet,
  * d’initialiser les tables utilisées par :

    * l’API de réentraînement,
    * l’application Streamlit de suivi et de prédiction.

* **.secrets** (non versionné)
  Fichier contenant les variables d’environnement nécessaires à la connexion à Neon.

---

## Prérequis

* Un compte **Neon**
* Un environnement Python avec Jupyter Notebook ou JupyterLab
* Accès réseau à la base Neon

---

## Configuration des variables d’environnement

### Création du fichier `.secrets`

Créer un fichier `.secrets` à la racine du dossier avec les informations de connexion à Neon :

```bash
MLFLOW_DEFAULT_ARTIFACT_ROOT =
AWS_ACCESS_KEY_ID= 
AWS_SECRET_ACCESS_KEY= 
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

## Exécution du notebook

1. Charger les variables d’environnement depuis le fichier `.secrets`.
2. Ouvrir le notebook :

```bash
jupyter notebook create-database.ipynb
```

3. Exécuter les cellules dans l’ordre afin de :

   * créer les bases et tables nécessaires,
   * valider la bonne connexion à Neon.

---

## Rôle dans l’architecture globale

Les bases créées via ce notebook sont utilisées par :

* l’API **Fire-retrain** pour stocker les résultats de réentraînement et les features,
* Airflow pour les dags get_meteoday, check_model, evidently_monitoring
* l’application **Streamlit** pour :

  * le suivi des performances du modèle,
  * la récupération des données nécessaires à la prédiction du risque d’incendie.


