FROM python:3.9-slim

WORKDIR /app

# Instalar curl para el healthcheck
RUN apt-get update && apt-get install -y curl

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código y archivos
COPY . .

# Crear estructura de directorios necesaria
RUN mkdir -p /app/mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/

# Variables de entorno
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production
ENV PORT=8000

# Verificar estructura
RUN ls -la /app
RUN ls -la /app/mlartifacts

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Comando para iniciar la aplicación
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}