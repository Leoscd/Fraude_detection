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
from src.monitoring.drift_detector import DriftDetector
import traceback
import joblib
from src.monitoring.drift_visualizer import DriftVisualizer
from fastapi.responses import HTMLResponse
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
model = None
drift_detector = None
def load_model():
    global model
    try:
        logger.info("=== INICIO PROCESO DE CARGA DEL MODELO ===")
                 
        # Verificar primero en la ubicación específica para desarrollo local
        local_path = r"D:\Proyectos Personales ML\Fraud detection\mlartifacts\426660670654388389\fa4a6618c80747fdab8e573b58f17030\artifacts\random_forest_model\model.pkl"
        if os.path.exists(local_path):
            logger.info(f"Encontrado modelo en ruta local de desarrollo: {local_path}")
            model = joblib.load(local_path)
            return model
            
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
    global model, drift_detector
    
    try:
        # Inicializar el detector de drift
        reference_path = os.getenv("REFERENCE_STATS_PATH", "data/reference_stats.json")
        logger.info(f"Buscando archivo de referencia en: {reference_path}")
        
        if os.path.exists(reference_path):
            logger.info(f"✓ Archivo de referencia encontrado")
            
            # Configurar ruta para el historial de drift
            dir_path = os.path.dirname(reference_path)
            history_path = os.path.join(dir_path, "drift_history.json")
            
            # Inicializar el detector con la ruta para el historial
            drift_detector = DriftDetector(reference_path, history_path=history_path)
            logger.info(f"✓ Detector de drift inicializado correctamente")
        else:
            logger.error(f"✗ Archivo de referencia NO encontrado en {reference_path}")
            # Imprimir los archivos en el directorio para debugging
            dir_path = os.path.dirname(reference_path)
            if os.path.exists(dir_path):
                logger.info(f"Archivos en {dir_path}: {os.listdir(dir_path)}")
    except Exception as e:
        logger.error(f"Error al inicializar detector de drift: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    yield
    
    # Mostrar estadísticas al cerrar
    if drift_detector:
        logger.info(f"Muestras acumuladas al cerrar: {len(drift_detector.samples) if hasattr(drift_detector, 'samples') else 'N/A'}")
    
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

@app.get("/monitoring/drift")
async def check_drift():
    try:
        if not drift_detector:
            return {"status": "not_configured"}
        
        # Solución temporal: implementar check_drift aquí si no existe en la clase
        if not hasattr(drift_detector, 'check_drift'):
            print("Usando implementación alternativa de check_drift")
            
            # Verificar si hay suficientes datos
            if len(drift_detector.samples) < 100:
                return {
                    "status": "insufficient_data",
                    "count": len(drift_detector.samples)
                }
            
            # Implementación temporal de check_drift
            results = {}
            drift_detected = False
            
            for feature in drift_detector.reference:
                # Llamar al método calculate_psi existente
                psi = drift_detector.calculate_psi(feature)
                if psi is not None:
                    is_drifting = bool(psi > drift_detector.threshold)
                    
                    if is_drifting:
                        drift_detected = True
                        
                    results[feature] = {
                        "psi": float(psi),
                        "drifting": is_drifting
                    }
            
            return {
                "status": "drift_detected" if drift_detected else "normal",
                "features": results,
                "sample_count": len(drift_detector.samples),
                "timestamp": datetime.now().isoformat()
            }
        
        # Usar el método de la clase si existe
        drift_result = drift_detector.check_drift()
        
        if drift_result["status"] == "drift_detected":
            logger.warning(f"¡ALERTA! Data drift detectado: {drift_result}")
        
        return drift_result
    except Exception as e:
        import traceback
        error_details = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        logger.error(f"Error en check_drift: {error_details}")
        return error_details


@app.get("/")
def read_root():
    return {"message": "Fraud Detection API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    global model, drift_detector
    
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
        # Registrar datos para monitoreo de drift ANTES de la predicción
        # (esto nos permite capturar datos incluso si la predicción falla)
        if drift_detector:
            # Convertir a diccionario para evitar incompatibilidades de tipo
            sample_data = {
                "V14": float(input_data.V14),
                "V10": float(input_data.V10),
                "V4": float(input_data.V4),
                "V12": float(input_data.V12),
                "V1": float(input_data.V1)
            }
            
            try:
                drift_detector.add_sample(sample_data)
                logger.info(f"✓ Muestra registrada para monitoreo. Total acumulado: {len(drift_detector.samples)}")
            except Exception as drift_error:
                logger.error(f"✗ Error al registrar muestra para monitoreo: {str(drift_error)}")
        else:
            logger.warning("⚠ Detector de drift no inicializado. No se registró la muestra.")
        
        # Ahora procesamos la predicción
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
        
        logger.info(f"✓ Predicción completada: {prediction}, prob: {probability}")
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        logger.error(f"✗ Error en predicción: {str(e)}")
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
    

@app.get("/monitoring/drift-dashboard", response_class=HTMLResponse)
async def get_drift_dashboard():
    """
    Endpoint para visualizar el dashboard de drift.
    
    Returns:
        Una página HTML con el dashboard de visualización de drift.
    """
    try:
        # Verificar que el detector de drift está inicializado
        if not drift_detector:
            return """
            <html>
                <body>
                    <h1>Error: Detector de drift no inicializado</h1>
                    <p>El sistema de monitoreo de drift no ha sido configurado correctamente.</p>
                </body>
            </html>
            """
        
        # Obtener la ruta del historial de drift
        history_path = getattr(drift_detector, 'history_path', None)
        if not history_path or not os.path.exists(history_path):
            return """
            <html>
                <body>
                    <h1>No hay datos de drift disponibles</h1>
                    <p>Aún no se han recolectado suficientes datos para visualizar el drift.</p>
                </body>
            </html>
            """
        
        # Inicializar el visualizador de drift
        visualizer = DriftVisualizer(
            history_path=history_path,
            reference_path=drift_detector.reference_stats_path if hasattr(drift_detector, 'reference_stats_path') else None
        )
        
        # Generar el dashboard
        dashboard_html = visualizer.generate_drift_dashboard()
        
        # Crear una plantilla HTML básica
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard de Monitoreo de Data Drift</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                h1 {{
                    color: #343a40;
                    margin-bottom: 30px;
                }}
                .alert {{
                    margin-top: 20px;
                }}
                .dashboard {{
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Dashboard de Monitoreo de Data Drift</h1>
                
                <div class="dashboard">
                    {dashboard_html}
                </div>
                
                <div class="mt-4">
                    <p>Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        return html_content
    except Exception as e:
        logger.error(f"Error al generar dashboard de drift: {str(e)}")
        return f"""
        <html>
            <body>
                <h1>Error al generar dashboard</h1>
                <p>{str(e)}</p>
                <pre>{traceback.format_exc()}</pre>
            </body>
        </html>
        """

@app.get("/api/drift/history")
async def get_drift_history():
    """
    Endpoint para obtener el historial de drift en formato JSON.
    
    Returns:
        Historial completo de métricas de drift
    """
    try:
        if not drift_detector or not hasattr(drift_detector, 'history_path'):
            return {"error": "Detector de drift no inicializado correctamente"}
        
        if not os.path.exists(drift_detector.history_path):
            return {"error": "Archivo de historial no encontrado", "path": drift_detector.history_path}
        
        with open(drift_detector.history_path, 'r') as f:
            history = json.load(f)
        
        return history
    except Exception as e:
        logger.error(f"Error al obtener historial de drift: {str(e)}")
        return {"error": str(e)}

@app.get("/api/drift/summary")
async def get_drift_summary():
    """
    Endpoint para obtener un resumen del estado actual de drift.
    
    Returns:
        Resumen del estado actual de drift
    """
    try:
        if not drift_detector:
            return {"status": "not_configured"}
        
        # Usar el endpoint existente de check_drift
        drift_result = await check_drift()
        
        # Extraer el estado de drift para cada feature
        if drift_result.get("status") == "insufficient_data":
            return {
                "status": "insufficient_data",
                "features_status": {},
                "sample_count": drift_result.get("count", 0)
            }
        
        # Si hay error o no hay información, manejar apropiadamente
        if not drift_result.get("features") or drift_result.get("status") == "error":
            return {
                "status": drift_result.get("status", "error"),
                "features_status": {},
                "error": drift_result.get("error", "Error desconocido")
            }
        
        # Extraer estado por feature
        features_status = {}
        for feature, data in drift_result.get("features", {}).items():
            psi = data.get("psi", 0)
            status = "stable"
            if psi >= 0.2:
                status = "critical"
            elif psi >= 0.1:
                status = "warning"
            
            features_status[feature] = {
                "psi": psi,
                "status": status,
                "drifting": data.get("drifting", False)
            }
        
        # Determinar estado general
        overall_status = "stable"
        if any(f["status"] == "critical" for f in features_status.values()):
            overall_status = "critical"
        elif any(f["status"] == "warning" for f in features_status.values()):
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "features_status": features_status,
            "sample_count": drift_result.get("sample_count", 0),
            "timestamp": drift_result.get("timestamp", datetime.now().isoformat())
        }
    except Exception as e:
        logger.error(f"Error al obtener resumen de drift: {str(e)}")
        return {"status": "error", "error": str(e)}

@app.get("/api/drift/features")
async def get_drift_features():
    """
    Endpoint para obtener la lista de features monitoreadas.
    
    Returns:
        Lista de features y sus estadísticas básicas
    """
    try:
        if not drift_detector:
            return {"error": "Detector de drift no inicializado correctamente"}
        
        features = {}
        for feature, stats in drift_detector.reference.items():
            features[feature] = {
                "mean": stats.get("mean", 0),
                "std": stats.get("std", 0)
            }
        
        return {"features": features}
    except Exception as e:
        logger.error(f"Error al obtener features: {str(e)}")
        return {"error": str(e)}
       
# Arranque de la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)