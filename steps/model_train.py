import logging
from zenml import step
from zenml.client import Client
import numpy as np

from src.model_dev import LSTMModel
from keras.models import Sequential
from zenml.model.model import Model
import mlflow

from zenml import get_step_context

# experiment_tracker = Client().active_stack.experiment_tracker

@step
def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[Sequential, str]:
    
    try:
        model = LSTMModel()
        trained_model, model_uri = model.train(X_train, y_train)
        return trained_model, model_uri
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e