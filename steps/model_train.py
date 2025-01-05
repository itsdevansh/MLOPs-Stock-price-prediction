import logging
from zenml import step
from zenml.client import Client
import numpy as np

from src.model_dev import LSTMModel
from tensorflow.python.keras.models import Sequential
import mlflow

from zenml import get_step_context

experiment_tracker = Client().active_stack.experiment_tracker.name

@step(experiment_tracker=experiment_tracker, enable_cache=False)
def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ticker: str,
    action: str =["train", "update", "nothing"]
) -> tuple[Sequential, str]:
    
    try:
        model = LSTMModel()
        trained_model, model_uri = model.train(X_train, y_train, ticker, action)
        return trained_model, model_uri
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e