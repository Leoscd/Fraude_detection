# Sistema de Detección de Fraude con Machine Learning

<div align="center">
  <img src="https://img.shields.io/badge/ML-Fraud%20Detection-blue" alt="ML Badge">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue" alt="Python Badge">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange" alt="Jupyter Badge">
  <img src="https://img.shields.io/badge/Status-Active-green" alt="Status Badge">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License Badge">
</div>

<div align="center">
  <img src="docs/images/fraud-detection-banner.png" alt="Fraud Detection Banner" width="600px">
</div>

## 📋 Descripción

Sistema avanzado para la detección de fraudes financieros utilizando técnicas de Machine Learning y análisis de datos en tiempo real. Esta solución identifica patrones sospechosos y anomalías en transacciones financieras, permitiendo a instituciones financieras y empresas protegerse contra actividades fraudulentas.

### 🎯 Problema que resuelve
El fraude financiero genera pérdidas millonarias anualmente para empresas e instituciones. Este sistema proporciona una capa de protección proactiva mediante la aplicación de algoritmos avanzados que detectan transacciones potencialmente fraudulentas antes de que se completen.

## 🔑 Características Principales

- **Detección de Anomalías**: Algoritmos que identifican comportamientos inusuales en las transacciones
- **Análisis en Tiempo Real**: Capacidad para procesar y analizar datos transaccionales instantáneamente
- **Dashboard Interactivo**: Interfaz visual para monitorear y analizar alertas de fraude
- **Implementación Dockerizada**: Facilidad de despliegue en cualquier entorno mediante contenedores Docker
- **Seguimiento de Experimentos**: Uso de MLflow para versionar modelos y experimentos

<div align="center">
  <img src="docs/images/dashboard-preview.png" alt="Dashboard Preview" width="800px">
</div>

## 🏗️ Arquitectura del Sistema

<div align="center">
  <img src="docs/images/architecture-diagram.png" alt="Architecture Diagram" width="700px">
</div>

El sistema sigue una arquitectura modular compuesta por:
1. **Módulo de Ingesta de Datos**: Captura y valida transacciones en tiempo real
2. **Preprocesamiento**: Limpieza y transformación de datos para el análisis
3. **Motor de Detección**: Aplicación de diversos algoritmos de ML
4. **API REST**: Interfaces para integración con sistemas externos
5. **Dashboard**: Visualización en tiempo real de las métricas y alertas

## 🚀 Instalación

### Requisitos Previos
- Python 3.7+
- Docker y Docker Compose
- Git

### Instalación Rápida

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Leoscd/Fraude_detection.git
   cd Fraude_detection
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Iniciar servicios con Docker:
   ```bash
   docker-compose up -d
   ```

Para instrucciones detalladas, consulta nuestra [Guía de Instalación](docs/INSTALLATION.md).

## 📊 Ejemplos de Uso

### Detección Básica

```python
from src.models import load_model
from src.preprocessing import preprocess_transaction

# Cargar modelo pre-entrenado
model = load_model("models/fraud_detector_v1.pkl")

# Datos de ejemplo
transaction_data = {
    "amount": 1500.00,
    "time": "2023-03-15T14:23:08",
    "location": "Online",
    "card_present": False,
    "user_id": "USR123456"
}

# Realizar predicción
processed_data = preprocess_transaction(transaction_data)
prediction = model.predict(processed_data)
probability = model.predict_proba(processed_data)[0][1]

print(f"Fraude: {'Sí' if prediction[0] == 1 else 'No'}")
print(f"Probabilidad: {probability:.2%}")
```

Para más ejemplos, consulta nuestra [documentación de la API](docs/API.md) y [notebooks de ejemplo](notebooks/).

## 🤖 Modelos Implementados

- **Gradient Boosting**: Alta precisión en clasificación de transacciones
- **Isolation Forest**: Especializado en detección de anomalías
- **Redes Neuronales**: Para patrones complejos y relaciones no lineales
- **Ensemble Models**: Combinación de múltiples modelos para mayor robustez

## 📈 Dashboard

Para acceder al dashboard de visualización:
1. Asegúrese de que todos los servicios estén corriendo
2. Navegue a `http://localhost:8050` en su navegador

<div align="center">
  <img src="docs/images/dashboard-metrics.png" alt="Dashboard Metrics" width="800px">
</div>

## 📚 Documentación

- [Guía de Instalación](docs/INSTALLATION.md)
- [Documentación de la API](docs/API.md)
- [Modelos y Características](docs/MODELS.md)
- [Guía de Contribución](CONTRIBUTING.md)

## 🛠️ Tecnologías Utilizadas

- **Lenguajes**: Python
- **Frameworks de ML**: Scikit-learn, XGBoost
- **Visualización**: Plotly, Dash
- **Despliegue**: Docker, Docker Compose
- **API**: FastAPI
- **Experimentación**: MLflow

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, revise primero [nuestra guía de contribución](CONTRIBUTING.md).

1. Fork el repositorio
2. Cree una nueva rama (`git checkout -b feature/amazing-feature`)
3. Haga sus cambios y commit (`git commit -m 'Add some amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abra un Pull Request

## 📬 Contacto

Leonardo Díaz - [GitHub Profile](https://github.com/Leoscd) - [LinkedIn](https://www.linkedin.com/in/leonardoadriandiaz/)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - consulte el archivo [LICENSE](LICENSE) para más detalles.
