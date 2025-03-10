from fastapi import FastAPI, HTTPException
import mlflow
import numpy as np
import logging
from src.api.models import PredictionInput, PredictionOutput
import json
from datetime import datetime
import os
import requests
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
        
        # Rutas y URLs del modelo
        model_dir = os.getenv("MODEL_DIR", "/app/models")
        model_path = os.getenv("MODEL_PATH", os.path.join(model_dir, "model.pkl"))
        model_url = os.getenv("MODEL_URL", None)
        
        # Crear directorio para el modelo si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # MODIFICADO: Verificar primero si el modelo existe localmente
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            logger.info(f"✓ Modelo encontrado localmente ({file_size} bytes)")
            
            # Cargar el modelo
            import joblib
            loaded_model = joblib.load(model_path)
            logger.info("✓ Modelo cargado exitosamente desde archivo local")
            
            if hasattr(loaded_model, 'predict') and hasattr(loaded_model, 'predict_proba'):
                logger.info("✓ Modelo verificado correctamente")
                return loaded_model
            else:
                logger.error("✗ El modelo local no tiene los métodos necesarios")
                # Continuar con la descarga en línea como respaldo
        else:
            logger.info(f"Modelo no encontrado en la ruta local: {model_path}")
            # Continuar con la descarga en línea
        
        # El resto de la función permanece igual - intentar descargar si hay URL
        if model_url:
            try:
                logger.info(f"Intentando descargar modelo desde: {model_url}")
                
                # Configurar timeout para evitar bloqueos
                response = requests.get(model_url, timeout=60)
                
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✓ Modelo descargado exitosamente a: {model_path}")
                else:
                    logger.error(f"✗ Error al descargar modelo. Código: {response.status_code}")
                    logger.error(f"Respuesta: {response.text[:200]}...")
            except Exception as e:
                logger.error(f"✗ Error en la descarga del modelo: {str(e)}")
                # Continuar con la carga local si falla la descarga
        else:
            logger.info("No se proporcionó MODEL_URL, intentando carga local")
        
        # Verificar existencia del archivo (ya sea cargado localmente o descargado)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            logger.info(f"✓ Archivo encontrado ({file_size} bytes)")
            
            # Cargar el modelo
            import joblib
            loaded_model = joblib.load(model_path)
            logger.info("✓ Modelo cargado exitosamente")
            
            # Verificar que el modelo tiene los métodos necesarios
            if hasattr(loaded_model, 'predict') and hasattr(loaded_model, 'predict_proba'):
                logger.info("✓ Modelo verificado correctamente")
                return loaded_model
            else:
                logger.error("✗ El modelo no tiene los métodos necesarios")
                return None
        else:
            logger.error(f"✗ Archivo no encontrado: {model_path}")
            # Mostrar contenido del directorio para debug
            parent_dir = os.path.dirname(model_path)
            if os.path.exists(parent_dir):
                logger.info(f"Contenido de {parent_dir}: {os.listdir(parent_dir)}")
            return None
            
    except Exception as e:
        logger.error(f"Error en carga del modelo: {str(e)}")
        logger.error("Traceback completo:", exc_info=True)
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== INICIANDO APLICACIÓN ===")
    # Ya no intentamos cargar el modelo al iniciar
    yield
    logger.info("=== CERRANDO APLICACIÓN ===")

# Inicializar FastAPI
app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

# Definir endpoints
@app.get("/health")
async def health_check():
    try:
        # Información sobre el estado del modelo
        model_dir = os.getenv("MODEL_DIR", "/app/models")
        model_path = os.getenv("MODEL_PATH", os.path.join(model_dir, "model.pkl"))
        model_url = os.getenv("MODEL_URL", "No configurado")
        
        # Verificar existencia y tamaño
        model_exists = os.path.exists(model_path)
        model_size = os.path.getsize(model_path) if model_exists else 0
        
        status_info = {
            "status": "healthy",  # Siempre devolver healthy
            "timestamp": datetime.now().isoformat(),
            "model_status": {
                "is_loaded": model is not None,
                "model_path": model_path,
                "model_url": model_url,
                "model_exists": model_exists,
                "model_file_size": model_size,
            },
            "environment": os.getenv("ENVIRONMENT", "production")
        }
        
        return status_info  # Siempre devuelve 200
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return {"status": "error", "error": str(e)}  # Aún devuelve 200

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    global model
    
    # Cargar el modelo bajo demanda si no está cargado
    if model is None:
        logger.info("Modelo no cargado, intentando cargar bajo demanda")
        model = load_model()
        
    # Verificar si se pudo cargar el modelo
    if model is None:
        logger.error("Estado del modelo: No inicializado")
        raise HTTPException(
            status_code=503,
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

@app.get("/debug")
async def debug_info():
    return {
        "environment_variables": {
            "MODEL_URL": os.getenv("MODEL_URL", "No configurado"),
            "MODEL_PATH": os.getenv("MODEL_PATH", "No configurado"),
            "MODEL_DIR": os.getenv("MODEL_DIR", "No configurado")
        },
        "file_system": {
            "current_directory": os.getcwd(),
            "app_directory_exists": os.path.exists("/app"),
            "models_directory_exists": os.path.exists("/app/models"),
            "models_directory_contents": os.listdir("/app/models") if os.path.exists("/app/models") else []
        },
        "model_status": {
            "is_loaded": model is not None,
            "can_access_url": check_url_access(os.getenv("MODEL_URL", ""))
        }
    }

def check_url_access(url):
    try:
        if not url:
            return {"status": "error", "message": "URL no proporcionada"}
        response = requests.head(url, timeout=5)
        return {
            "status": "success" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", "desconocido"),
            "content_length": response.headers.get("Content-Length", "desconocido")
        }
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "No se pudo conectar con la URL (conexión rechazada)"}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Tiempo de espera agotado al intentar acceder a la URL"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/load-model")
async def load_model_endpoint():
    global model
    try:
        if model is not None:
            return {"status": "success", "message": "Modelo ya está cargado"}
            
        loaded_model = load_model()
        if loaded_model is not None:
            model = loaded_model
            return {"status": "success", "message": "Modelo cargado exitosamente"}
        else:
            return {"status": "error", "message": "No se pudo cargar el modelo"}
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}
    
        
# Arranque de la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)