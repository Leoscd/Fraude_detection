# Guía de Instalación Detallada

Esta guía proporciona instrucciones paso a paso para instalar y configurar el Sistema de Detección de Fraude en diferentes entornos.

## Tabla de Contenidos
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación Local](#instalación-local)
- [Instalación con Docker](#instalación-con-docker)
- [Configuración Avanzada](#configuración-avanzada)
- [Solución de Problemas](#solución-de-problemas)

## Requisitos del Sistema

### Hardware Recomendado
- **CPU**: 4+ núcleos
- **RAM**: 8GB mínimo (16GB recomendado)
- **Almacenamiento**: 10GB de espacio libre

### Software Requerido
- **Sistema Operativo**: 
  - Linux (Ubuntu 18.04+, CentOS 7+)
  - macOS 10.15+
  - Windows 10/11 con WSL2
- **Python**: 3.7, 3.8 o 3.9
- **Git**: Cualquier versión reciente
- **Docker** (opcional): 19.03.0+
- **Docker Compose** (opcional): 1.27.0+

## Instalación Local

### 1. Preparación del Entorno

```bash
# Clonar el repositorio
git clone https://github.com/Leoscd/Fraude_detection.git
cd Fraude_detection

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Actualizar herramientas de Python
pip install --upgrade pip setuptools wheel
```

### 2. Instalación de Dependencias

```bash
# Instalar dependencias principales
pip install -r requirements.txt

# Instalar dependencias de desarrollo (opcional)
pip install -r requirements-dev.txt
```

### 3. Configuración del Entorno

Crea un archivo `.env` en el directorio raíz:

```
# Configuración de la Base de Datos
DB_USERNAME=postgres
DB_PASSWORD=postgrespassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraud_detection

# Configuración de MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=fraud_detection

# Configuración del Servicio
API_PORT=8000
DASHBOARD_PORT=8050
LOG_LEVEL=INFO
```

### 4. Inicialización de la Base de Datos

```bash
# Instalar PostgreSQL si es necesario
# Para Ubuntu:
# sudo apt install postgresql postgresql-contrib

# Crear base de datos
# sudo -u postgres createdb fraud_detection

# Inicializar esquema
python scripts/init_database.py
```

### 5. Iniciar los Servicios

```bash
# Iniciar MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000 &

# Iniciar API
python src/api/main.py &

# Iniciar Dashboard
python src/dashboard/app.py &
```

## Instalación con Docker

### 1. Preparación

```bash
# Clonar el repositorio
git clone https://github.com/Leoscd/Fraude_detection.git
cd Fraude_detection
```

### 2. Configuración de Variables de Entorno

Crea un archivo `.env` en el directorio raíz (ver sección anterior).

### 3. Construcción e Inicio de Contenedores

```bash
# Construir imágenes
docker-compose build

# Iniciar servicios
docker-compose up -d
```

### 4. Verificación

```bash
# Verificar que los contenedores estén funcionando
docker-compose ps

# Ver logs
docker-compose logs -f
```

### 5. Detener Servicios

```bash
# Detener servicios
docker-compose down

# Detener y eliminar volúmenes (elimina datos persistentes)
docker-compose down -v
```

## Configuración Avanzada

### Configuración de SSL

Para habilitar HTTPS en la API:

1. Genera certificados SSL:
   ```bash
   mkdir -p certificates
   openssl req -x509 -newkey rsa:4096 -nodes -out certificates/cert.pem -keyout certificates/key.pem -days 365
   ```

2. Actualiza la configuración en `config/api_config.yaml`:
   ```yaml
   ssl:
     enabled: true
     cert_file: certificates/cert.pem
     key_file: certificates/key.pem
   ```

### Integración con Servicios de Monitoreo

El sistema puede integrarse con Prometheus y Grafana:

1. Habilita la exportación de métricas en `config/api_config.yaml`:
   ```yaml
   monitoring:
     prometheus_enabled: true
     metrics_endpoint: /metrics
   ```

2. Agrega servicios adicionales en `docker-compose.yml`:
   ```yaml
   prometheus:
     image: prom/prometheus
     volumes:
       - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
     ports:
       - "9090:9090"
     
   grafana:
     image: grafana/grafana
     ports:
       - "3000:3000"
     depends_on:
       - prometheus
   ```

## Solución de Problemas

### Problemas Comunes

#### Error de Conexión a la Base de Datos

```
Error: Could not connect to database
```

**Solución**:
1. Verifica que PostgreSQL esté instalado y funcionando
2. Confirma que las credenciales en `.env` sean correctas
3. Asegúrate de que la base de datos exista:
   ```bash
   sudo -u postgres psql -c "CREATE DATABASE fraud_detection;"
   ```

#### Error en la Inicialización de MLflow

```
Error: MLflow server failed to start
```

**Solución**:
1. Verifica que el puerto 5000 no esté en uso:
   ```bash
   lsof -i :5000
   ```
2. Asegúrate de tener permisos para crear archivos en el directorio actual
3. Prueba con una ruta de almacenamiento diferente:
   ```bash
   mlflow server --backend-store-uri sqlite:///$(pwd)/mlflow.db --default-artifact-root $(pwd)/mlartifacts --host 0.0.0.0 --port 5000
   ```

#### Error en Docker

```
Error: Cannot connect to the Docker daemon
```

**Solución**:
1. Asegúrate de que Docker esté instalado y funcionando:
   ```bash
   systemctl status docker
   # o
   service docker status
   ```
2. Agrega tu usuario al grupo docker:
   ```bash
   sudo usermod -aG docker $USER
   # Luego cierra sesión y vuelve a iniciar
   ```

### Logs y Diagnóstico

Para habilitar logging detallado:

```bash
# Configurar nivel de log
export LOG_LEVEL=DEBUG

# Redirigir logs a un archivo
python src/api/main.py > api.log 2>&1 &
```

Para verificar el estado de los servicios:

```bash
# API
curl http://localhost:8000/health

# MLflow
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Dashboard (solo verifica que esté respondiendo)
curl -I http://localhost:8050
```

## Recursos Adicionales

- [Documentación oficial de Docker](https://docs.docker.com/)
- [Documentación de MLflow](https://www.mlflow.org/docs/latest/index.html)
- [Guía de PostgreSQL](https://www.postgresql.org/docs/)

Si continúas experimentando problemas, por favor [abre un issue](https://github.com/Leoscd/Fraude_detection/issues/new/choose) con los detalles de tu entorno y los errores encontrados.
