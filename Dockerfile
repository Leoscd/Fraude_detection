FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias
RUN apt-get update && apt-get install -y curl

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto (incluido mlartifacts)
COPY . .

# Variables de entorno
ENV PORT=8000

# Verificar la estructura (para debugging)
RUN ls -l /app/mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/

# Comando para iniciar
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}