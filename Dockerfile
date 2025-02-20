FROM python:3.9-slim

WORKDIR /app

# Instalar curl para el healthcheck
RUN apt-get update && apt-get install -y curl

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código y archivos
COPY . .

# Variables de entorno
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production
ENV PORT=8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Comando para iniciar la aplicación
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}