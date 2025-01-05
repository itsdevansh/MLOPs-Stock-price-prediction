import logging
from zenml import step
from zenml.client import Client
import mlflow
from zenml.client import Client
import numpy as np
from src.evaluation import MAE
from tensorflow.python.keras.models import Sequential

experiment_tracker = Client().active_stack.experiment_tracker.name

@step(experiment_tracker=experiment_tracker)
def evaluate_model(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray
    ) -> np.float_:

    try:
        prediction = model.predict(X_test)
        mae_class = MAE()
        mae = mae_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mae", mae)

        return mae
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
    finally:
        mlflow.end_run()
    
    