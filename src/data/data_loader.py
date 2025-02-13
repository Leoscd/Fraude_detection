# src/data/data_loader.py
import pandas as pd
import os
import yaml

class DataLoader:
    def __init__(self):
        # Obtener la ruta absoluta al directorio raíz del proyecto
        current_dir = os.path.dirname(os.path.abspath(__file__))  # directorio actual (data)
        project_root = os.path.dirname(os.path.dirname(current_dir))  # directorio raíz del proyecto
        
        # Construir rutas
        self.config_path = os.path.join(project_root, 'config', 'config.yaml')
        print(f"Buscando config en: {self.config_path}")  # Para debug
        
        # Cargar configuración
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Construir ruta a los datos
            self.data_path = os.path.join(
                project_root,
                'data',
                'raw',
                'data2.csv'
            )
            print(f"Ruta de datos: {self.data_path}")  # Para debug
            
        except Exception as e:
            print(f"Error en inicialización: {str(e)}")
            raise
    
    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            print("Datos cargados exitosamente")
            return df
        except Exception as e:
            print(f"Error cargando datos: {str(e)}")
            return None