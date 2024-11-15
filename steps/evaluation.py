import logging
from zenml import step
from zenml.client import Client
import mlflow

import numpy as np
from src.evaluation import MAE
from keras.models import Sequential

# experiment_tracker = Client().active_stack.experiment_tracker

@step
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
    
    