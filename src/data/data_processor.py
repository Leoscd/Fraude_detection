import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..utils.config_handler import ConfigHandler

class DataPreprocessor:
    def __init__(self):
        self.config = ConfigHandler()
    
    def preprocess_features(self, df: pd.DataFrame):
        """Preprocesar características"""
        pass

    def handle_missing_values(self, df: pd.DataFrame):
        """Manejar valores faltantes"""
        pass