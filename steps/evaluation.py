import logging
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

import numpy as np
from src.evaluation import MAE
from keras.models import Sequential

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

        return mae
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
    
    