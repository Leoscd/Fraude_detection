import numpy as np
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, reference_path, threshold=0.2, history_path=None):
        """
        Inicializa el detector con estadísticas de referencia.
        
        Args:
            reference_path: Ruta al archivo JSON con estadísticas de referencia
            threshold: Umbral de PSI para considerar que hay drift
            history_path: Ruta donde se guardará el historial de métricas de drift
        """
        logger.info(f"Inicializando DriftDetector con archivo: {reference_path}")
        self.threshold = threshold
        self.samples = []
        self.reference_stats_path = reference_path  # Guardar la ruta para referencia
        
        # Configurar ruta del historial
        if history_path is None:
            # Usar la misma carpeta que reference_path
            dir_path = os.path.dirname(reference_path)
            self.history_path = os.path.join(dir_path, "drift_history.json")
        else:
            self.history_path = history_path
            
        logger.info(f"Historial de drift se guardará en: {self.history_path}")
        
        try:
            # Cargar estadísticas de referencia
            with open(reference_path) as f:
                self.reference = json.load(f)
            logger.info(f"Estadísticas de referencia cargadas. Features: {list(self.reference.keys())}")
        except FileNotFoundError:
            error_msg = f"No se encontró el archivo de estadísticas de referencia: {reference_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError:
            error_msg = f"Error al decodificar el archivo JSON: {reference_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Cargar historial existente o crear uno nuevo
        self._load_history()
    
    def _load_history(self):
        """Carga el historial de drift desde el archivo o crea uno nuevo."""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    self.history = json.load(f)
                logger.info(f"Historial de drift cargado con {len(self.history.get('timestamps', []))} registros")
            else:
                # Inicializar un historial vacío
                self.history = {
                    "timestamps": [],
                    "drift_scores": {feature: [] for feature in self.reference.keys()}
                }
                logger.info("Creado nuevo historial de drift")
        except Exception as e:
            logger.error(f"Error al cargar historial de drift: {str(e)}")
            # Inicializar un historial vacío en caso de error
            self.history = {
                "timestamps": [],
                "drift_scores": {feature: [] for feature in self.reference.keys()}
            }
    
    def _save_history(self):
        """Guarda el historial de drift en el archivo JSON."""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Historial de drift guardado en {self.history_path}")
        except Exception as e:
            logger.error(f"Error al guardar historial de drift: {str(e)}")
    
    def add_sample(self, sample_data):
        """
        Registra una nueva muestra para análisis.
        
        Args:
            sample_data: Diccionario con los valores de las features
        """
        try:
            # Primero verificar que las features necesarias estén presentes
            expected_features = set(self.reference.keys())
            provided_features = set(sample_data.keys())
            missing_features = expected_features - provided_features
        
            if missing_features:
                logger.warning(f"Advertencia: Faltan features en la muestra: {missing_features}")
        
            # Después agregar la muestra
            self.samples.append({
                "timestamp": datetime.now().isoformat(),
                **sample_data
            })
        
            logger.info(f"Muestra registrada: {sample_data}")
            logger.info(f"Total de muestras acumuladas: {len(self.samples)}")
            
            # Verificar si tenemos suficientes muestras para calcular drift
            if len(self.samples) >= 100 and len(self.samples) % 50 == 0:
                # Cada 50 muestras después de alcanzar 100, verificamos y guardamos el drift
                drift_result = self.check_drift()
                logger.info(f"Verificación automática de drift: {drift_result['status']}")
                
            return True
        except Exception as e:
            logger.error(f"Error al agregar muestra: {str(e)}")
            return False
        
    def calculate_psi(self, feature):
        """
        Calcula PSI (Population Stability Index) para una feature.
        
        Args:
            feature: Nombre de la feature para calcular PSI
            
        Returns:
            Valor PSI o None si no hay suficientes muestras
        """
        try:
            if len(self.samples) < 100:
                return None
                
            # Extraer valores de la feature de las muestras
            values = [s[feature] for s in self.samples if feature in s]
            
            # Obtener histograma de referencia
            ref_hist = np.array(self.reference[feature]["histogram"])
            ref_bins = np.array(self.reference[feature]["bins"])
            
            # Calcular histograma actual con los mismos bins
            current_hist, _ = np.histogram(values, bins=ref_bins)
            
            # Normalizar y evitar ceros
            ref_hist = np.maximum(ref_hist / sum(ref_hist), 1e-6)
            current_hist = np.maximum(current_hist / sum(current_hist), 1e-6)
            
            # Calcular PSI
            psi = np.sum((current_hist - ref_hist) * np.log(current_hist / ref_hist))
            return psi
        except Exception as e:
            logger.error(f"Error al calcular PSI para {feature}: {str(e)}")
            return None
    
    def check_drift(self):
        """
        Verifica si hay drift en alguna feature y actualiza el historial.
        
        Returns:
            Diccionario con resultados del análisis de drift
        """
        try:
            if len(self.samples) < 100:
                return {
                    "status": "insufficient_data",
                    "count": len(self.samples)
                }
            
            results = {}
            drift_detected = False
            current_time = datetime.now().isoformat()
            
            # Añadir timestamp al historial
            self.history["timestamps"].append(current_time)
            
            for feature in self.reference:
                psi = self.calculate_psi(feature)
                if psi is not None:
                    # Convertir explícitamente np.bool_ a bool de Python
                    is_drifting = bool(psi > self.threshold)
                    
                    if is_drifting:
                        drift_detected = True
                    
                    # Convertir valores numpy a tipos nativos de Python
                    psi_value = float(psi)
                    
                    # Guardar en resultados
                    results[feature] = {
                        "psi": psi_value,
                        "drifting": is_drifting
                    }
                    
                    # Guardar en el historial
                    if feature not in self.history["drift_scores"]:
                        self.history["drift_scores"][feature] = []
                    
                    self.history["drift_scores"][feature].append(psi_value)
            
            # Guardar el historial actualizado
            self._save_history()
            
            return {
                "status": "drift_detected" if bool(drift_detected) else "normal",
                "features": results,
                "sample_count": len(self.samples),
                "timestamp": current_time
            }
        except Exception as e:
            logger.error(f"Error al verificar drift: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }