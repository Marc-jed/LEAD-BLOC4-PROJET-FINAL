# Image de base
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système utiles (optionnel mais pratique pour scikit-learn, xgboost etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier ton code
COPY . .


EXPOSE 7860

# Lancer FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
