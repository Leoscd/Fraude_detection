import mlflow


def get_best_model_info():
    """
    Obtiene el mejor modelo basado en métricas (ejemplo: accuracy)
    Returns:
        dict: Información del mejor modelo incluyendo run_id
    """
    try:
        # Obtener todos los experimentos
        experiment = mlflow.get_experiment_by_name("fraud_detection_experiment")
        if experiment is None:
            raise Exception("No se encontró el experimento")

        # Buscar todos los runs del experimento
        runs = mlflow.search_runs(
            experiment_ids=experiment.experiment_id,
            order_by=["metrics.accuracy DESC"]  # Ordenar por accuracy descendente
        )

        if runs.empty:
            raise Exception("No se encontraron modelos registrados")

        # Obtener el mejor run (primer resultado)
        best_run = runs.iloc[0]
        
        return {
            'run_id': best_run.run_id,
            'accuracy': best_run['metrics.accuracy'],
            'model_path': f"runs:/{best_run.run_id}/random_forest_model"
        }
    except Exception as e:
        print(f"Error obteniendo información del modelo: {e}")
        return None

# Uso en app.py
try:
    best_model_info = get_best_model_info()
    if best_model_info:
        model = mlflow.sklearn.load_model(best_model_info['model_path'])
        print(f"Modelo cargado exitosamente. Accuracy: {best_model_info['accuracy']}")
    else:
        raise Exception("No se pudo obtener información del modelo")
except Exception as e:
    print(f"Error loading model: {e}")