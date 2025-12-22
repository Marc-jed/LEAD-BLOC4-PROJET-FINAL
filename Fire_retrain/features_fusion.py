import pandas as pd
import io
import os
import boto3
import datetime
import json
import joblib
import requests
import boto3
import numpy as np
from datetime import date, timedelta, timezone
from geopy.distance import geodesic
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv('.secrets')

def run_features_and_fusion():
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = os.getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT')
    os.environ['S3_BUCKET'] = os.getenv('S3_BUCKET')
    s3 = boto3.client("s3")

    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")

    query = ("""SELECT * FROM meteoday WHERE date >= CURRENT_DATE - INTERVAL '1825 DAYS'""")
    df = pd.read_sql(query, engine)
    df = pd.DataFrame(df)
    df = df.copy()


    print(df.info())
    print('done ....')
    
    df["poste"] = df["poste"].astype('Float64')

    print(df.head(2))
    print('done ....')
    print('1-nombe de date null :',df['date'].isnull().sum())

    bucket = os.getenv('S3_BUCKET')
    s3.download_file(bucket,"dataset/liste_station.csv", "liste_station.csv" )
    df_station = pd.read_csv("liste_station.csv")
    list_station = df_station["0"].unique().tolist()

    print('done ....')
    print(list_station)

    # on ne garde que les stations présentent dans la liste station météo corse
    mask = df['poste'].isin(list_station)
    df= df[mask]

    print('done ....')
    print(df.head())

    df['date'] = pd.to_datetime(df['date'], format='ISO8601')
    # Traitements de colonnes innutiles
    drop_cols = list(set([
        'PMERM','PMERMIN','QPMERMIN','FF2M','QFF2M','FXI2','QFXI2','DXI2','QDXI2','HXI2','QHXI2','DXI3S','QDXI3S','DHUMEC','QDHUMEC','INST','QINST','GLOT','QGLOT','DIFT','QDIFT','DIRT','QDIRT','SIGMA','QSIGMA','INFRART','QINFRART','UV_INDICEX','QUV_INDICEX','NB300','QNB300','BA300','QBA300','NEIG','QNEIG','BROU','QBROU','GRESIL','GRELE','QGRELE','ROSEE','QROSEE','VERGLAS','QVERGLAS','SOLNEIGE','QSOLNEIGE','GELEE','QGELEE','FUMEE','QFUMEE','UV','QUV','TMERMAX','QTMERMAX','TMERMIN','QTMERMIN','HNEIGEF','QHNEIGEF','NEIGETOTX','QNEIGETOTX','NEIGETOT06','QNEIGETOT06','QRR','QDRR','QTN','QHTN','QTX','QHTX','QTM','QTMNX','QTNSOL','QTN50','DG','QDG','QTAMPLI','QTNTXM','QPMERM','QFFM','QFXI','QDXI','QHXI','QFXY','QDXY','QHXY','QFXI3S','QHXI3S','QUN','QHUN','QUX','QHUX','QDHUMI40','QDHUMI80','QTSVM','QUM','QORAG','QGRESIL','QBRUME','ECLAIR','QECLAIR','QETPMON','QETPGRILLE'
    ]))
    drop_cols = [d.lower() for d in drop_cols]
    df = df.drop(columns=drop_cols, axis=1)

    # convertion des colonnes object en float64
    for column in df.columns:
        if column != "date" and df[column].dtype == 'object':
            df[column] = df[column].astype('Float64')

    # fonction de moyenne lissante 

    df['moyenne_precipitations_mois'] = (df['rr'].rolling(window=31, min_periods=10).mean().round(2))
    df['moyenne_vitesse_vent_mois'] = (df['ffm'].rolling(window=31, min_periods=10).mean().round(2))
    df['moyenne_temperature_mois'] = (df['tn'].rolling(window=31, min_periods=10).mean().round(2))

    # Fonction pour compter les jours consécutifs sans pluie
    def compter_jours_sans_pluie(groupe):
        compteur = 0
        jours_sans_pluie = []
        for rr in groupe['rr']:
            if pd.isna(rr):
                jours_sans_pluie.append(np.nan)
            elif rr == 0:
                compteur += 1
                jours_sans_pluie.append(compteur)
            else:
                compteur = 0
                jours_sans_pluie.append(compteur)
        return pd.Series(jours_sans_pluie, index=groupe.index)

    # Fonction pour compter les jours consécutifs avec TX > 30
    def compter_jours_chauds(groupe):
        compteur = 0
        jours_chauds = []
        for tx in groupe['tx']:
            if pd.isna(tx):
                jours_chauds.append(np.nan)
            elif tx > 30:
                compteur += 1
                jours_chauds.append(compteur)
            else:
                compteur = 0
                jours_chauds.append(compteur)
        return pd.Series(jours_chauds, index=groupe.index)

    # Application des 2 fonctions
    df= df.sort_values(['poste', 'date']) 
    df['jours_tx_sup_30'] = (
        df.groupby('poste')['tx']
        .transform(lambda s: compter_jours_chauds(
            pd.DataFrame({'tx': s}, index=s.index)
        ))
        .astype(float)
    )

    df['jours_sans_pluie'] = (
        df.groupby('poste')['rr']
        .transform(lambda s: compter_jours_sans_pluie(
            pd.DataFrame({'rr': s}, index=s.index)
        ))
        .astype(float)
    )


    df["etpgrille_7j"] = df.groupby("poste")["etpgrille"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['poste'] = df['poste'].astype('Int64')

    print('done ....')
    print('2-nombe de date null :',df['date'].isnull().sum())

    # appelle du fichier cordonnées gps des stations météo
    bucket = os.getenv('S3_BUCKET')
    s3.download_file(bucket, "dataset/coord_stations.csv", "coordonnée_station.csv")
    df_coord_station = pd.read_csv("coordonnée_station.csv", sep=',')
    df_coord_station.drop(columns={'Unnamed: 0'}, inplace=True)
    df_coord_station.columns = df_coord_station.columns.str.lower()

    # on ne garde que les stations présentent dans la liste station météo corse
    list_station2 = df["poste"].unique().tolist()
    mask = df_coord_station['poste'].isin(list_station2)
    df_coord_station2= df_coord_station[mask]

    print('done ....')
    print(df_coord_station2.head(2))

    # on crée notre fichier météo avec les coordonnées gps des stations
    df_meteo=pd.merge(df_coord_station2, df, on="poste", how='left')

    print(df_meteo.head(2))
    print('done ....')
    print('3-nombe de date null :',df_meteo['date'].isnull().sum())

    # on nettoie/supprime et modifie les colonnes
    df_meteo['date'] = pd.to_datetime(df_meteo['date'], format='ISO8601')
    df_meteo['nom'] = df_meteo['nom'].astype('string')
    df_stations_unique = df_meteo[['nom', 'lat', 'lon','poste']].drop_duplicates()

    print('done ....')
    print(df_stations_unique.head(2))

    # charge depuis le S3 du fichier incendie corse
    s3.download_file(bucket, "dataset/historique_incendie_corse_cleaned_2025.csv", "historique_incendie.csv")
    df_feux = pd.read_csv("historique_incendie.csv", sep=',')
    df_feux.columns = df_feux.columns.str.lower()

    def find_nearest_station(row, stations):
        p_feu = (row["latitude_feu"], row["longitude_feu"])
        distances = stations.apply(
            lambda s: geodesic(p_feu, (s["lat"], s["lon"])).km, axis=1
        )
        idx = distances.idxmin()
        post = stations.loc[idx, "poste"]
        return pd.Series([idx, distances[idx], post])

    df_feux[["station_index", "dist_km",'poste']] = df_feux.apply(
        find_nearest_station,
        stations=df_stations_unique,
        axis=1
        )
    df_fire = df_feux[['date', 'poste', 'feux', 'ville', 'latitude_feu', 'longitude_feu', 'surface_parcourue_m2', 'surface_forêt_m2', 'surface_maquis_garrigues_m2', 'autres_surfaces_naturelles_hors_forêt_m2', 'surfaces_agricoles_m2', 'autres_surfaces_m2', 'surface_autres_terres_boisées_m2', 'surfaces_non_boisées_naturelles_m2', 'surfaces_non_boisées_artificialisées_m2', 'surfaces_non_boisées_m2', 'type_de_peuplement', 'nature']]
    df_final = pd.merge(df_stations_unique, df_fire, on=['poste'], how='left')

    print('done ....')
    print('4-nombe de date null :',df_final['date'].isnull().sum())

    df_final['date'] = pd.to_datetime(df_final['date'])
    df_meteo['date'] = pd.to_datetime(df_meteo['date'])
    df_final.drop(columns={'lat', 'lon','poste'}, inplace=True)
    df_feux_corse = pd.merge(df_meteo, df_final, on=('date','nom'), how='left') 

    print('done ....')
    print('5-nombe de date null :',df_feux_corse['date'].isnull().sum())

    df_feux_corse = df_feux_corse.sort_values(["poste", "date"])
    # Suppression des doublons crée lors du merge, on garde la première valeur
    df_feux_corse = df_feux_corse.drop_duplicates(subset=["poste","date"], keep='first')

    def days_until_next_fire(g):
        # index des feux
        fire_idx = g.index[g["feux"] == 1]

        # tableau vide
        days = pd.Series(index=g.index, dtype=float)

        for i in range(len(g)):
            current_date = g.loc[g.index[i], "date"]

            # dates de feux dans le futur
            future_fires = g.loc[fire_idx, "date"]
            future_fires = future_fires[future_fires >= current_date]

            if len(future_fires) == 0:
                days[g.index[i]] = None  # pas de feu après
            else:
                days[g.index[i]] = (future_fires.iloc[0] - current_date).days

        return days

    df_feux_corse["décompte"] = df_feux_corse.groupby("poste").apply(days_until_next_fire).reset_index(level=0, drop=True)

    df_feux_corse['feux'] = df_feux_corse.feux.fillna(0).astype(int)
    df_feux_corse= df_feux_corse.sort_values(['nom', 'date'])
    # Création de la colonne évènement pour indiquer si un feu a eu lieu
    df_feux_corse['évènement'] = df_feux_corse['feux'] == 1
    # Nouvelle colonne initialisée à NaN
    df_feux_corse["compteur_jours_vers_prochain_feu"] = pd.NA

    # Traitement par ville
    for nom, groupe in df_feux_corse.groupby("nom"):
        groupe = groupe.sort_values("date")
        indices_feux = groupe[groupe["évènement"] == True].index.tolist()
        
        for i in range(len(indices_feux) - 1):
            debut = indices_feux[i]
            fin = indices_feux[i + 1]
            
            # Remplir les jours entre les deux feux avec un compteur croissant
            for j, idx in enumerate(range(debut, fin)):
                df_feux_corse.loc[idx, "compteur_jours_vers_prochain_feu"] = j

    # # nombre de jour sans feu + log et carré
    df_feux_corse['compteur_feu_log'] = df_feux_corse['compteur_jours_vers_prochain_feu'].apply(lambda x: np.log1p(x) if pd.notnull(x) else np.nan)

    df_feux_corse['année'] = df_feux_corse['date'].dt.year
    df_feux_corse['mois'] = df_feux_corse['date'].dt.month


    # Trier par ville et par date
    df_feux_corse = df_feux_corse.sort_values(['nom', 'date'])
    # Création de la colone Feu prévu pour le modèle Survival
    df_feux_corse.loc[:, "feu_prévu"] = df_feux_corse["décompte"].notna().astype(int)
    df_feux_corse['feu_prévu'] = df_feux_corse['feu_prévu'].astype(bool)
    df_feux_corse['date'] = pd.to_datetime(df_feux_corse['date']).dt.date

    print('done ....')
    print('visu du format date de feux corse:', df_feux_corse['date'].tail(),'.........', df_feux_corse['date'].info())
    
    yesterday = (datetime.date.today() - datetime.timedelta(days=1))

    print('done ....')
    print("date d'hier:", yesterday)

    df_push = df_feux_corse[df_feux_corse["date"] == yesterday]

    print('done ....')
    print(df_push.head())
    print('6-nombe de date null :',df_feux_corse['date'].isnull().sum(), 'nombre de post null :', df_feux_corse['poste'].isnull().sum())

   
    print("--------------------Opération terminée-------------------")

    df_push.to_sql('data_prediction', engine, if_exists="append", index=False)

    print("-----------Envoie sur la database neon effectué----------")
