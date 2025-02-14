FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements primero
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/logs /app/mlartifacts

# Variables de entorno por defecto
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production
ENV PORT=8000

# Script de inicio
RUN echo '#!/bin/bash\n\
port="${PORT:-8000}"\n\
uvicorn src.api.app:app --host 0.0.0.0 --port $port\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Comando para iniciar la aplicación
CMD ["/app/start.sh"]