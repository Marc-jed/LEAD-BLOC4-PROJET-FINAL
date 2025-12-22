# FROM apache/airflow:2.10.4-python3.10

# USER root

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gosu \
#     libpq-dev \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN pip install --no-cache-dir "evidently==0.6.7" "pydantic<2"

# USER airflow

# COPY requirements.txt .

# RUN pip install -r requirements.txt

FROM apache/airflow:2.10.4-python3.10

# ---- System dependencies (root only) ----
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies (airflow user ONLY) ----
USER airflow

# Evidently pinned + compatibility
RUN pip install --no-cache-dir \
    "evidently==0.7.17" \
    "pydantic<2"

# Project Python deps
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
