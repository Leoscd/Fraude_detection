FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/mlartifacts

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_STAGE=Production

# Script de inicio más robusto
RUN echo '#!/bin/bash\n\
REAL_PORT=${PORT:-8000}\n\
echo "Starting server on port $REAL_PORT"\n\
exec uvicorn src.api.app:app --host 0.0.0.0 --port "$REAL_PORT"' > start.sh

RUN chmod +x start.sh

CMD ["./start.sh"]