from fastapi import FastAPI, HTTPException
import mlflow
import numpy as np
import logging
from src.api.models import PredictionInput, PredictionOutput
import json
from datetime import datetime
import os
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
model = None

def load_model():
    global model
    try:
        logger.info("=== INICIO PROCESO DE CARGA DEL MODELO ===")
        
        # Construir ruta absoluta
        base_dir = "/app"  # Railway usa /app como WORKDIR
        model_path = os.path.join(
            base_dir,
            "mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl"
        )
        
        logger.info(f"Intentando cargar modelo desde: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"✗ Modelo no encontrado en: {model_path}")
            return None

        import joblib
        loaded_model = joblib.load(model_path)
        
        # Verificar que el modelo tiene los métodos necesarios
        if hasattr(loaded_model, 'predict') and hasattr(loaded_model, 'predict_proba'):
            logger.info("✓ Modelo cargado y verificado exitosamente")
            return loaded_model
        else:
            logger.error("✗ El modelo cargado no tiene los métodos requeridos")
            return None
            
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        logger.error("Traceback completo:", exc_info=True)
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== INICIANDO APLICACIÓN ===")
    try:
        loaded_model = load_model()
        if loaded_model is not None:
            global model
            model = loaded_model
            logger.info("✓ Modelo cargado exitosamente en startup")
        else:
            logger.error("✗ No se pudo cargar el modelo en startup")
    except Exception as e:
        logger.error(f"Error en startup: {str(e)}")
    yield
    logger.info("=== CERRANDO APLICACIÓN ===")



# 5. Inicializar FastAPI
app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

# 6. Definir endpoints
@app.get("/health")
async def health_check():
    try:
        current_dir = os.getcwd()
        model_path = "mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_status": {
                "is_loaded": model is not None,
                "model_path": model_path,
                "model_exists": os.path.exists(model_path),
                "current_directory": current_dir,
                "directory_contents": os.listdir(current_dir)
            },
            "environment": os.getenv("ENVIRONMENT", "production")
        }
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    # Verificar el modelo de manera más informativa
    if model is None:
        logger.error("Estado del modelo: No inicializado")
        raise HTTPException(
            status_code=500,
            detail="Modelo no disponible - Error en la inicialización"
        )
    
    try:
        features = np.array([[
            input_data.V14,
            input_data.V10,
            input_data.V4,
            input_data.V12,
            input_data.V1
        ]])
        logger.info(f"Procesando predicción con features: {features}")
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        logger.info(f"Predicción completada: {prediction}, prob: {probability}")
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción: {str(e)}"
        )

# 7. Arranque de la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)

