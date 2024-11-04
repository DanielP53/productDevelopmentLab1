# Reproducción del Proyecto con DVC

Este proyecto incluye un pipeline de DVC que realiza preprocesamiento, entrenamiento, optimización y evaluación de modelos. Sigue las instrucciones a continuación para reproducir el proyecto y revisar los resultados.

## Requisitos Previos

- Python 3.8 o superior
- `pip` para la instalación de paquetes de Python
- `dvc` para la gestión del pipeline de datos

## Instalación y Configuración

1. **Clona el repositorio desde GitHub**:
   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git
   cd tu_repositorio
   
2. **Crea un entorno virtual e instala las dependencias**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
3. **Instala DVC: Si DVC no está instalado, instálalo con**:

   ```bash
   pip install dvc
   
4. **Ejecucion del pipeline**:

   ```bash
   dvc repro

5. **Verificacion de resultados**:
   ```bash
   cat results/metrics.csv
   cat results/metrics.md
