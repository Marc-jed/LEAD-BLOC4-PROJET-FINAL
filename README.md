# 🔥 Projet FIRE – Prédiction et Monitoring du Risque d’Incendie

##  Présentation du projet

Le projet **FIRE** met en œuvre une **architecture data et MLOps complète** dédiée à la **prédiction du risque d’incendie**, intégrant :

* la collecte automatisée de données météorologiques,
* l’enrichissement et la fusion de features,
* l’entraînement et le réentraînement de modèles de Machine Learning,
* le monitoring de la performance des modèles,
* le déploiement de services temps réel,
* et l’orchestration des workflows via Airflow.

L’objectif est de construire un **système fiable, automatisé et monitoré**, capable d’évoluer dans le temps face aux dérives de données et de modèles.

---

##  Objectifs

* Prédire le risque d’incendie à partir de données météo et historiques
* Automatiser l’ensemble du pipeline data et ML
* Surveiller la performance du modèle dans le temps
* Réentraîner le modèle en cas de dérive ou perte de performance
* Garantir la traçabilité des expériences et des modèles
* Déployer des services exploitables en production

---

##  Architecture globale

Le projet repose sur une architecture modulaire composée de :

* **Airflow** : orchestration des pipelines
* **PostgreSQL (Neon)** : stockage des données structurées
* **MLflow** : suivi des expériences et versioning des modèles
* **FastAPI** : APIs de prédiction et de réentraînement
* **Evidently** : monitoring de la dérive des données et du modèle
* **Docker** : containerisation des composants
* **Jenkins** : CI/CD
* **Pytest** : tests automatisés

---

##  Structure du projet

```
├── Airflow/                # Orchestration des workflows
│   ├── Dags/
│   │   ├── check_model_dags.py
│   │   ├── Evidently_monitoring.py
│   │   └── get_meteoday.py
│   ├── docker-compose.yaml
│   └── Dockerfile
│
├── Db_neon/                # Base de données PostgreSQL (Neon)
│   ├── README.md
│   └── create_database.ipynb
│
├── Fire_retrain/           # API de réentraînement du modèle
│   ├── app.py
│   ├── retrain.py
│   ├── features_fusion.py
│   └── Dockerfile
│
├── MLFLOW-FIRE/             # Suivi expérimental et modèles
│   ├── model.py
│   └── Dockerfile
│
├── Stream-fire/             # API de prédiction temps réel
│   ├── app.py
│   └── Dockerfile
│
├── Jenkins/                 # CI/CD
│   └── Jenkinsfile
│
├── tests/                   # Tests unitaires et fonctionnels
│   ├── test_coord_station.py
│   ├── test_features_fusion.py
│   └── test_histo_feu.py
│
├── README.md                # Documentation principale
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

---

##  Pipelines principaux

###  Pipeline de collecte & traitement

* Récupération quotidienne des données météo
* Enrichissement et fusion des features
* Stockage dans PostgreSQL

###  Pipeline ML

* Entraînement et réentraînement du modèle
* Suivi des métriques via MLflow
* Versioning des modèles

###  Monitoring

* Détection de dérive des données et du modèle (Evidently)
* Surveillance des performances
* Déclenchement du réentraînement si nécessaire

###  Déploiement

* API FastAPI pour la prédiction
* Chargement dynamique du dernier modèle valide

---

##  Tests & Qualité

* Tests unitaires sur la logique métier
* Validation des transformations de données
* Vérification de la cohérence des features
* Intégration continue via Jenkins

---

##  Bonnes pratiques mises en œuvre

* Architecture modulaire et scalable
* Séparation entraînement / prédiction
* Monitoring continu du modèle
* Traçabilité complète des modèles
* Déploiement sans interruption de service
* Automatisation des workflows data & ML

---

##  Conclusion

Le projet **FIRE** illustre une approche moderne et industrielle de la **Data Science appliquée aux risques environnementaux**, combinant ingénierie data, MLOps et monitoring avancé pour garantir des modèles fiables, maintenables et évolutifs.

* ou une **slide d’architecture FIRE** prête à présenter

