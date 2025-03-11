import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import traceback

logger = logging.getLogger(__name__)

class DriftVisualizer:
    def __init__(self, history_path, reference_path=None):
        """
        Inicializa el visualizador de drift.
        
        Args:
            history_path: Ruta al archivo JSON donde se almacena el historial de drift
            reference_path: Ruta al archivo JSON con estadísticas de referencia (opcional)
        """
        logger.info(f"Inicializando DriftVisualizer con historial: {history_path}")
        self.history_path = history_path
        self.reference_path = reference_path
        
        # Cargar historial
        self._load_data()
    
    def _load_data(self):
        """Carga los datos necesarios para las visualizaciones."""
        try:
            # Cargar historial de drift
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    self.history = json.load(f)
                logger.info(f"Historial de drift cargado con {len(self.history.get('timestamps', []))} registros")
                # Verificar la estructura del historial
                if 'timestamps' not in self.history or 'drift_scores' not in self.history:
                    logger.warning(f"Estructura de historial incorrecta: {list(self.history.keys())}")
            else:
                logger.warning(f"Archivo de historial no encontrado: {self.history_path}")
                self.history = {"timestamps": [], "drift_scores": {}}
            
            # Cargar estadísticas de referencia si están disponibles
            if self.reference_path and os.path.exists(self.reference_path):
                with open(self.reference_path, 'r') as f:
                    self.reference = json.load(f)
                logger.info(f"Estadísticas de referencia cargadas. Features: {list(self.reference.keys())}")
            else:
                self.reference = None
                logger.info("No se cargaron estadísticas de referencia")
        except Exception as e:
            logger.error(f"Error al cargar datos para visualización: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.history = {"timestamps": [], "drift_scores": {}}
            self.reference = None
    
    def generate_feature_drift_plot(self, feature_name):
        """
        Genera un gráfico de línea para mostrar el drift de una feature a lo largo del tiempo.
        
        Args:
            feature_name: Nombre de la feature para visualizar
            
        Returns:
            Objeto de figura de Plotly
        """
        try:
            # Verificar si tenemos datos para esta feature
            if (not self.history.get("timestamps") or 
                feature_name not in self.history.get("drift_scores", {}) or
                not self.history["drift_scores"].get(feature_name)):
                # Si no hay datos, devuelve un gráfico vacío
                fig = go.Figure()
                fig.update_layout(
                    title=f"No hay datos de drift para la feature '{feature_name}'",
                    xaxis_title="Tiempo",
                    yaxis_title="PSI Score"
                )
                return fig
            
            # Obtener datos para el gráfico
            timestamps = self.history.get("timestamps", [])
            scores = self.history["drift_scores"].get(feature_name, [])
            
            # Convertir timestamps a objetos datetime para mejor visualización
            try:
                datetime_timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            except (ValueError, TypeError) as e:
                logger.error(f"Error al parsear timestamps: {str(e)}")
                datetime_timestamps = [datetime.now() for _ in timestamps]
            
            # Asegurar que hay la misma cantidad de timestamps que scores
            if len(datetime_timestamps) > len(scores):
                logger.warning(f"Más timestamps que scores para {feature_name}. Recortando timestamps.")
                datetime_timestamps = datetime_timestamps[-len(scores):]
            elif len(scores) > len(datetime_timestamps):
                logger.warning(f"Más scores que timestamps para {feature_name}. Recortando scores.")
                scores = scores[-len(datetime_timestamps):]
            
            # Crear un DataFrame para facilitar la visualización
            df = pd.DataFrame({
                'timestamp': datetime_timestamps,
                'psi_score': scores
            })
            
            # Crear gráfico
            fig = go.Figure()
            
            # Añadir la línea principal de PSI
            fig.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['psi_score'],
                mode='lines+markers',
                name='PSI Score',
                line=dict(color='blue', width=2),
                marker=dict(
                    size=8,
                    color=df['psi_score'].apply(
                        lambda x: 'green' if x < 0.1 else ('orange' if x < 0.2 else 'red')
                    )
                )
            ))
            
            # Añadir áreas sombreadas para los umbrales
            if len(df) > 0:
                min_time = min(df['timestamp'])
                max_time = max(df['timestamp'])
                
                # Área verde (estable)
                fig.add_trace(go.Scatter(
                    x=[min_time, max_time, max_time, min_time],
                    y=[0, 0, 0.1, 0.1],
                    fill="toself",
                    fillcolor="rgba(0,255,0,0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="none",
                    showlegend=True,
                    name="Estable (PSI < 0.1)"
                ))
                
                # Área amarilla (alerta)
                fig.add_trace(go.Scatter(
                    x=[min_time, max_time, max_time, min_time],
                    y=[0.1, 0.1, 0.2, 0.2],
                    fill="toself",
                    fillcolor="rgba(255,255,0,0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="none",
                    showlegend=True,
                    name="Alerta (0.1 ≤ PSI < 0.2)"
                ))
                
                # Área roja (crítico)
                fig.add_trace(go.Scatter(
                    x=[min_time, max_time, max_time, min_time],
                    y=[0.2, 0.2, 0.5, 0.5],
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="none",
                    showlegend=True,
                    name="Crítico (PSI ≥ 0.2)"
                ))
            
            # Formatear el diseño del gráfico
            fig.update_layout(
                title=f'Drift para la feature "{feature_name}" a lo largo del tiempo',
                xaxis_title="Fecha y Hora",
                yaxis_title="PSI Score",
                legend_title="Umbrales de Drift",
                hovermode="x unified",
                yaxis=dict(range=[0, max(0.5, max(scores) * 1.1) if scores else 0.5])
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error al generar gráfico para {feature_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Crear un gráfico de error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error al generar gráfico para '{feature_name}'",
                xaxis_title="Tiempo",
                yaxis_title="PSI Score",
                annotations=[
                    dict(
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        text=f"Error: {str(e)}",
                        showarrow=False,
                        font=dict(color="red")
                    )
                ]
            )
            return fig
    
    def generate_drift_heatmap(self):
        """
        Genera un mapa de calor para visualizar el drift de todas las features.
        
        Returns:
            Objeto de figura de Plotly
        """
        try:
            # Verificar si tenemos datos para mostrar
            if (not self.history.get("timestamps") or 
                not self.history.get("drift_scores") or 
                all(len(scores) == 0 for scores in self.history["drift_scores"].values())):
                # Si no hay datos, devuelve un gráfico vacío
                fig = go.Figure()
                fig.update_layout(
                    title="No hay datos de drift para visualizar",
                    xaxis_title="Tiempo",
                    yaxis_title="Features",
                    annotations=[
                        dict(
                            x=0.5,
                            y=0.5,
                            xref="paper",
                            yref="paper",
                            text="Aún no hay suficientes datos para mostrar el mapa de calor",
                            showarrow=False,
                            font=dict(size=14)
                        )
                    ]
                )
                return fig
            
            # Obtener la lista de features
            features = list(self.history["drift_scores"].keys())
            
            # Convertir timestamps a formato legible
            try:
                timestamps = [datetime.fromisoformat(ts).strftime('%Y-%m-%d %H:%M') 
                             for ts in self.history["timestamps"]]
            except (ValueError, TypeError) as e:
                logger.error(f"Error al formatear timestamps: {str(e)}")
                timestamps = [f"Punto {i+1}" for i in range(len(self.history["timestamps"]))]
            
            # Crear matriz de calor
            heat_data = []
            for feature in features:
                scores = self.history["drift_scores"].get(feature, [])
                
                # Asegurarse de que hay tantos scores como timestamps
                if len(scores) < len(timestamps):
                    # Rellenar con None los valores faltantes
                    logger.warning(f"Faltan scores para feature {feature}. Rellenando con None.")
                    scores = scores + [None] * (len(timestamps) - len(scores))
                elif len(scores) > len(timestamps):
                    # Recortar los scores extra
                    logger.warning(f"Hay más scores que timestamps para feature {feature}. Recortando.")
                    scores = scores[:len(timestamps)]
                
                heat_data.append(scores)
            
            # Verificar que tenemos datos para mostrar
            if not heat_data or not any(heat_data):
                fig = go.Figure()
                fig.update_layout(
                    title="No hay suficientes datos para visualizar",
                    xaxis_title="Tiempo",
                    yaxis_title="Features"
                )
                return fig
            
            # Crear figura
            fig = go.Figure(data=go.Heatmap(
                z=heat_data,
                x=timestamps,
                y=features,
                colorscale=[
                    [0, 'green'],        # PSI = 0
                    [0.1/0.3, 'lime'],   # PSI = 0.1
                    [0.2/0.3, 'yellow'], # PSI = 0.2
                    [1, 'red']           # PSI >= 0.3
                ],
                zmin=0,
                zmax=0.3,
                colorbar=dict(
                    title="PSI Score",
                    tickvals=[0, 0.1, 0.2, 0.3],
                    ticktext=["0 (Sin drift)", "0.1 (Moderado)", "0.2 (Significativo)", "0.3+ (Crítico)"]
                )
            ))
            
            # Actualizar diseño
            fig.update_layout(
                title="Mapa de calor de Data Drift por Feature",
                xaxis_title="Tiempo",
                yaxis_title="Features",
                height=max(400, 100 + len(features) * 30),  # Ajustar altura según el número de features
                xaxis=dict(
                    tickangle=-45,
                    type='category'
                )
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error al generar mapa de calor: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Crear un gráfico de error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error al generar mapa de calor: {str(e)}",
                xaxis_title="Tiempo",
                yaxis_title="Features"
            )
            return fig
    
    def generate_drift_dashboard(self):
        """
        Genera un dashboard completo con todas las visualizaciones de drift.
        
        Returns:
            HTML del dashboard
        """
        try:
            # Verificar que tenemos datos para mostrar
            features = list(self.history.get("drift_scores", {}).keys())
            
            if not features:
                # Si no hay features, mostrar un mensaje informativo
                return self._generate_empty_dashboard("No hay features para visualizar")
            
            # Crear una figura para cada feature
            feature_figures = {}
            for feature in features:
                feature_figures[feature] = self.generate_feature_drift_plot(feature)
            
            # Crear el mapa de calor
            heatmap_fig = self.generate_drift_heatmap()
            
            # Generar el dashboard
            dashboard_html = self._generate_dashboard_html(heatmap_fig, feature_figures)
            
            return dashboard_html
        
        except Exception as e:
            logger.error(f"Error al generar dashboard: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Crear un mensaje de error
            return f"""
            <div style="color: red; padding: 20px; border: 1px solid red; border-radius: 5px;">
                <h3>Error al generar dashboard</h3>
                <p>{str(e)}</p>
                <pre>{traceback.format_exc()}</pre>
            </div>
            """
    
    def _generate_empty_dashboard(self, message):
        """Genera un dashboard vacío con un mensaje informativo."""
        return f"""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>{message}</h3>
            <p>Espere a que se recolecten más datos para visualizar el drift.</p>
        </div>
        """
    
    def _generate_dashboard_html(self, heatmap_fig, feature_figures):
        """
        Genera el HTML del dashboard con todas las visualizaciones.
        
        Args:
            heatmap_fig: Figura del mapa de calor
            feature_figures: Diccionario de figuras por feature
            
        Returns:
            HTML del dashboard
        """
        # Convertir el mapa de calor a HTML
        heatmap_html = heatmap_fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Generar HTML para cada feature
        features_html = ""
        for feature, fig in feature_figures.items():
            feature_html = fig.to_html(full_html=False, include_plotlyjs=False)
            features_html += f"""
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-body">
                        {feature_html}
                    </div>
                </div>
            </div>
            """
        
        # Generar el HTML completo del dashboard
        dashboard_html = f"""
        <div class="container-fluid">
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body">
                            {heatmap_html}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                {features_html}
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-light">
                            <h4>Interpretación de métricas PSI</h4>
                        </div>
                        <div class="card-body">
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Sin drift significativo
                                    <span class="badge bg-success rounded-pill">PSI < 0.1</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Drift moderado - Requiere atención
                                    <span class="badge bg-warning text-dark rounded-pill">0.1 ≤ PSI < 0.2</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Drift significativo - Acción requerida
                                    <span class="badge bg-danger rounded-pill">PSI ≥ 0.2</span>
                                </li>
                            </ul>
                            
                            <div class="mt-3">
                                <h5>¿Qué significa PSI?</h5>
                                <p>El Population Stability Index (PSI) mide cuánto ha cambiado la distribución de los datos 
                                respecto a la distribución de referencia del conjunto de entrenamiento. Valores más altos 
                                indican mayores cambios en la distribución, lo que puede afectar el rendimiento del modelo.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return dashboard_html