# src/utils/config_handler.py
import yaml
import os
from typing import Dict, Any

class ConfigHandler:
    def __init__(self):
        # Obtener la ruta absoluta al directorio raíz
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        self.config_path = os.path.join(root_dir, 'config', 'config.yaml')
        print(f"Buscando config en: {self.config_path}")  # Debug
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    return yaml.safe_load(file)
            else:
                print(f"No se encontró el archivo en: {self.config_path}")
                return None
        except Exception as e:
            print(f"Error al cargar la configuración: {str(e)}")
            return None

    def get_dataset_config(self, dataset_name: str):
        if self.config and 'data' in self.config:
            return self.config['data']['datasets'].get(dataset_name)
        return None