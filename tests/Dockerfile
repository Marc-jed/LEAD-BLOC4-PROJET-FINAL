
FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances systèmes nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    zlib1g-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    liblzma-dev \
    lzma \
    libgdbm-dev \
    libdb-dev \
    libgmp-dev \
    libmpfr-dev \
    libssl-dev \
    libpcap-dev \
    libjpeg-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Définir le PYTHONPATH pour que les modules soient trouvés
ENV PYTHONPATH=/home

# Copier les fichiers de dépendances et installer les dépendances Python
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt


RUN mkdir tests
# Copier les fichiers du projet dans le container
# COPY test_etl_meteo.py /app/tests
# COPY conftest.py /app/tests
# COPY test_data.py /app/tests
COPY . /app/tests

# Définir le point d'entrée pour exécuter les tests avec pytest
CMD ["python", "-m", "pytest", "tests/"]