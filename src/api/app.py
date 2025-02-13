from fastapi import FastAPI, HTTPException
import mlflow
import numpy as np
import logging
from src.api.models import PredictionInput, PredictionOutput
from src.utils.mlflow_utils import get_best_model_info
import json
from datetime import datetime
import os
from fastapi.responses import JSONResponse

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

# Inicializar FastAPI
app = FastAPI(title="Fraud Detection API")

# Configurar MLflow - Ajustado para Railway
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)

@app.get("/health")
async def health_check():
    try:
        # Verificar estado del modelo
        model_status = "loaded" if model is not None else "not_loaded"
        
        # Verificar conexión con MLflow
        mlflow_status = "connected"
        try:
            mlflow.get_tracking_uri()
        except Exception as e:
            mlflow_status = f"error: {str(e)}"
        
        # Verificar acceso a logs
        logs_status = "accessible" if os.path.exists("logs") else "not_accessible"
        
        return {
            "status": "healthy" if all([
                model_status == "loaded",
                mlflow_status == "connected",
                logs_status == "accessible"
            ]) else "degraded",
            "components": {
                "model": model_status,
                "mlflow": mlflow_status,
                "logs": logs_status
            },
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "production")
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

def log_prediction(input_data, prediction, probability):
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data.dict(),
            "prediction": prediction,
            "probability": float(probability)
        }
        
        log_path = "logs/prediction_logs.json"
        logger.info(f"Intentando guardar log en: {log_path}")
        os.makedirs("logs", exist_ok=True)
        
        with open(log_path, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
            
        logger.info("Log guardado exitosamente")
    except Exception as e:
        logger.error(f"Error guardando log: {str(e)}")

# [El resto de tu código actual permanece igual...]
@app.get("/")
def read_root():
    return {"message": "Fraud Detection API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if model is None:
        logger.error("Modelo no disponible para predicciones")
        raise HTTPException(status_code=500, detail="Modelo no disponible")
        
    try:
        features = np.array([[
            input_data.V14,
            input_data.V10,
            input_data.V4,
            input_data.V12,
            input_data.V1
        ]])
        logger.info(f"Datos de entrada: {features}")
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        logger.info(f"Predicción: {prediction}")
        logger.info(f"Probabilidad: {probability}")
        
        # Agregar el logging aquí
        log_prediction(input_data, prediction, probability)
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

    
if __name__ == "__main__":  
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)