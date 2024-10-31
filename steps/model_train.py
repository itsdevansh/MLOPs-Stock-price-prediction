import logging
from zenml import step
import numpy as np

from src.model_dev import LSTMModel
from keras.models import Sequential

@step
def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Sequential:
    
    try:
        model = None
        if model_name == 'Long Short Term Memory':
            model = LSTMModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {model_name} not supported")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e