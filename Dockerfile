FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/mlartifacts

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production
ENV PORT=8000

# Usar un script de inicio
RUN echo '#!/bin/bash\nport="${PORT:-8000}"\nexec uvicorn src.api.app:app --host 0.0.0.0 --port "${port}"' > start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]