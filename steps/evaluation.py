import logging
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

import numpy as np
from src.evaluation import MSE
from keras.models import Sequential

@step
def evaluate_model(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray
    ) -> np.float_:

    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        return mse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
    
    