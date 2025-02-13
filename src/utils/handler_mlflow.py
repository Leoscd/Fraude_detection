import mlflow
from .config_handler import ConfigHandler

class MLFlowHandler:
    def __init__(self):
        self.config = ConfigHandler()
        mlflow_config = self.config.config['mlflow']
        
        # Configurar MLflow
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        self.experiment_name = mlflow_config['experiments']['fraud_detection']['name']
        
        # Crear o cargar experimento
        try:
            self.experiment = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)

    def start_run(self, run_name=None):
        """Iniciar un nuevo run de MLflow"""
        return mlflow.start_run(experiment_id=self.experiment.experiment_id, 
                              run_name=run_name)

    def log_params(self, params):
        """Registrar parámetros"""
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        """Registrar métricas"""
        mlflow.log_metrics(metrics)

    def log_model(self, model, model_name):
        """Registrar modelo"""
        mlflow.sklearn.log_model(model, model_name)