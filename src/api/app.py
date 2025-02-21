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
        
        model_path = "/app/mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl"
        logger.info(f"Intentando cargar modelo desde: {model_path}")

        # Intentar diferentes métodos de carga
        try:
            import joblib
            logger.info("Intentando cargar con joblib...")
            model = joblib.load(model_path)
        except Exception as e1:
            logger.warning(f"Error con joblib: {str(e1)}")
            try:
                from sklearn.externals import joblib
                logger.info("Intentando cargar con sklearn.externals.joblib...")
                model = joblib.load(model_path)
            except Exception as e2:
                logger.warning(f"Error con sklearn.joblib: {str(e2)}")
                try:
                    import pickle
                    logger.info("Intentando cargar con pickle...")
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                except Exception as e3:
                    logger.error("Todos los métodos de carga fallaron")
                    raise Exception(f"No se pudo cargar el modelo: {str(e3)}")

        if model is not None:
            logger.info("✓ Modelo cargado exitosamente")
            return model
            
    except Exception as e:
        logger.error(f"Error final en carga del modelo: {str(e)}")
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

