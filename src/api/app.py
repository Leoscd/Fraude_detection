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

# 1. Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Inicializar variables globales
model = None

# 3. Función de carga del modelo
def load_model():
    global model
    try:
        logger.info("=== INICIO PROCESO DE CARGA DEL MODELO ===")
        
        # Ir dos niveles arriba para llegar a la raíz del proyecto
        project_root = os.path.dirname(os.path.dirname(os.getcwd()))
        
        # Ruta al modelo
        base_path = "mlartifacts"
        model_dir = "426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model"
        model_path = os.path.join(project_root, base_path, model_dir, "model.pkl")
        
        logger.info(f"Intentando cargar modelo desde: {model_path}")
        
        # Cargar el modelo
        import joblib
        model = joblib.load(model_path)
        logger.info("✓ Modelo cargado exitosamente")
        
        return model
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        logger.error("Traceback completo:", exc_info=True)
        raise e

# 4. Configurar lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        logger.info("Aplicación iniciada correctamente")
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {str(e)}")
    yield
    logger.info("Aplicación cerrada")

# 5. Inicializar FastAPI
app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

# 6. Definir endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "production")
    }

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
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 7. Arranque de la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)

