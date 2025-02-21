FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias
RUN apt-get update && apt-get install -y curl

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Verificar la estructura después de copiar (para debugging)
RUN echo "=== Contenido de /app ===" && \
    ls -la /app && \
    echo "=== Contenido de mlartifacts (si existe) ===" && \
    ls -la mlartifacts || echo "mlartifacts no encontrado"

# Variables de entorno
ENV PORT=8000

# Comando para iniciar
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}