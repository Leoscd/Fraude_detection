FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/mlartifacts

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production

# Usar shell form en lugar de exec form para el CMD
CMD uvicorn src.api.app:app --host 0.0.0.0 --port $PORT