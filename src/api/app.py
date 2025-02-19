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

# Variable global para el modelo
model = None

def load_model():
    global model
    try:
        logger.info("=== INICIO PROCESO DE CARGA DEL MODELO ===")
        
        # 1. Verificar el directorio actual y listar contenido
        current_dir = os.getcwd()
        logger.info(f"1. Directorio actual: {current_dir}")
        logger.info(f"1.1 Contenido del directorio actual: {os.listdir(current_dir)}")
        
        # 2. Intentar diferentes rutas
        possible_paths = [
            "/app/mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl",
            "mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl",
            "./mlartifacts/426660670654388389/fa4a6618c80747fdab8e573b58f17030/artifacts/random_forest_model/model.pkl"
        ]
        
        # 3. Intentar cada ruta
        for path in possible_paths:
            logger.info(f"2. Intentando ruta: {path}")
            if os.path.exists(path):
                logger.info(f"3. ✓ Archivo encontrado en: {path}")
                import joblib
                model = joblib.load(path)
                logger.info("4. ✓ Modelo cargado exitosamente")
                
                # Verificar que el modelo tiene los métodos necesarios
                if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                    logger.info("5. ✓ Modelo verificado correctamente")
                    return model
                else:
                    raise ValueError("El modelo no tiene los métodos requeridos")
        
        # Si no se encontró el modelo en ninguna ruta
        raise FileNotFoundError(f"No se encontró el modelo en ninguna de las rutas intentadas")
        
    except Exception as e:
        logger.error(f"ERROR en carga del modelo: {type(e).__name__}")
        logger.error(f"Mensaje de error: {str(e)}")
        logger.error("Traceback completo:", exc_info=True)
        raise e
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== INICIANDO APLICACIÓN ===")
    try:
        logger.info("1. Iniciando carga del modelo...")
        load_model()
        logger.info("2. Modelo cargado correctamente")
    except Exception as e:
        logger.error("=== ERROR EN INICIALIZACIÓN ===")
        logger.error(f"Tipo de error: {type(e).__name__}")
        logger.error(f"Mensaje: {str(e)}")
    yield
    logger.info("=== CERRANDO APLICACIÓN ===")

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

