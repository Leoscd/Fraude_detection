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

# Crear directorio para el modelo (por si acaso)
RUN mkdir -p /app/models

# Variables de entorno
ENV PORT=8000
ENV MODEL_DIR=/app/models
ENV MODEL_PATH=/app/models/model.pkl

# IMPORTANTE: Asegurarnos de que el modelo esté en el lugar correcto
# Esto moverá el modelo desde donde está en tu repo a donde la app espera encontrarlo
RUN if [ -f /app/mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl ]; then \
        mkdir -p /app/models && \
        cp /app/mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl /app/models/model.pkl && \
        echo "Modelo copiado exitosamente"; \
    else \
        echo "ERROR: No se encontró el archivo del modelo"; \
        exit 1; \
    fi

# Verificar la estructura después de copiar
RUN echo "=== Contenido de /app/models ===" && \
    ls -la /app/models

# Comando para iniciar
RUN pip install gunicorn
CMD gunicorn src.api.app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --timeout 120