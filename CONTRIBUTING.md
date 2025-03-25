# Guía de Contribución

¡Gracias por tu interés en contribuir al Sistema de Detección de Fraude! Este documento proporciona lineamientos y mejores prácticas para contribuir al proyecto.

## Código de Conducta

Al participar en este proyecto, te comprometes a mantener un entorno abierto y acogedor. Por favor, sé respetuoso con otros contribuyentes y usuarios.

## ¿Cómo puedo contribuir?

### Reportando Bugs

Si encuentras un bug, por favor crea un issue en GitHub con la siguiente información:
- Título descriptivo del problema
- Pasos detallados para reproducir el bug
- Comportamiento esperado vs. comportamiento observado
- Capturas de pantalla si aplica
- Entorno (sistema operativo, versión de Python, etc.)

### Sugiriendo Mejoras

Las sugerencias de mejoras son bienvenidas. Por favor, crea un issue con:
- Título claro de la propuesta
- Descripción detallada de la mejora
- Justificación de por qué esta mejora sería valiosa
- Si es posible, ejemplos o mockups

### Pull Requests

1. Primero, crea un issue describiendo qué planeas implementar
2. Fork el repositorio
3. Crea una nueva rama desde `main`:
   ```bash
   git checkout -b feature/tu-caracteristica
   ```
4. Realiza tus cambios siguiendo las convenciones de código
5. Asegúrate de incluir tests para tu código
6. Actualiza la documentación si es necesario
7. Ejecuta los tests existentes
8. Envía tu pull request a la rama `main`

## Estilo de Código

### Python

- Sigue [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Utiliza docstrings en formato [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Usa nombres descriptivos para variables y funciones
- Mantén las funciones pequeñas y con un solo propósito

### Documentación

- Mantén la documentación actualizada
- Usa Markdown para todos los documentos
- Incluye ejemplos cuando sea posible

## Desarrollo Local

### Configuración del Entorno

```bash
# Clonar el repositorio
git clone https://github.com/Leoscd/Fraude_detection.git
cd Fraude_detection

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Para herramientas de desarrollo
```

### Ejecución de Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar tests con coverage
pytest --cov=src tests/
```

### Linting y Formatting

```bash
# Ejecutar linting
flake8 src tests

# Formatear código
black src tests
```

## Estructura del Proyecto

```
fraude_detection/
├── config/                  # Archivos de configuración
├── data/                    # Datasets y archivos de datos
├── docs/                    # Documentación
├── notebooks/               # Jupyter notebooks para análisis
├── src/                     # Código fuente principal
│   ├── preprocessing/       # Procesamiento de datos
│   ├── models/              # Implementación de modelos
│   ├── api/                 # API REST
│   └── dashboard/           # Código del dashboard
├── tests/                   # Tests unitarios y de integración
├── .github/                 # Configuraciones de GitHub
└── docker/                  # Archivos Docker
```

## Proceso de Release

1. Las contribuciones se integran a la rama `main`
2. Periódicamente, se crean releases desde `main`
3. Las versiones siguen [Semantic Versioning](https://semver.org/)
4. Los cambios significativos se documentan en CHANGELOG.md

## Recursos Adicionales

- [Documentación oficial del proyecto](docs/)
- [Guía de instalación](docs/INSTALLATION.md)
- [Documentación de la API](docs/API.md)

Gracias por contribuir al proyecto. Tu ayuda es muy valorada y contribuye a mejorar la detección de fraude para todos.
