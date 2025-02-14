FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/logs /app/mlartifacts

# Variables de entorno
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production

# Script de inicio
RUN echo '#!/bin/sh\nPORT="${PORT:-8000}"\nexec uvicorn src.api.app:app --host 0.0.0.0 --port $PORT' > /app/start.sh
RUN chmod +x /app/start.sh

# Configuración para Railway
CMD ["/app/start.sh"]