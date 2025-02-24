FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Crear directorio para el modelo
RUN mkdir -p /app/models

# Variables de entorno
ENV PORT=8000
ENV MODEL_DIR=/app/models
ENV MODEL_PATH=/app/models/model.pkl
ENV ENVIRONMENT=production
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
# MODEL_URL se configura directamente en Railway

# Verificar la estructura después de copiar
RUN echo "=== Contenido de /app ===" && \
    ls -la /app && \
    echo "=== Contenido de models ===" && \
    ls -la /app/models || echo "directorio de modelos vacío"

# Ejecutar la aplicación
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}