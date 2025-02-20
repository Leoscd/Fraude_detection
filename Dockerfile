FROM python:3.9-slim

WORKDIR /app

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código y archivos
COPY . .

# Variables de entorno
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production

# Crear directorios necesarios si no existen
RUN mkdir -p /app/mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/

# Verificar la estructura después de copiar
RUN ls -l /app/mlartifacts