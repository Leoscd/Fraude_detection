FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# NO definir PORT aquí, dejarlo para Railway
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production

# Comando simple y directo
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}