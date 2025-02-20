FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias
RUN apt-get update && apt-get install -y curl

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el modelo a una ubicación simple
COPY mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl /app/model.pkl

# Copiar el resto del código
COPY . .

# Variables de entorno
ENV MODEL_PATH=/app/model.pkl
ENV PORT=8000

# Comando para iniciar
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}