import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error

class Evlaution(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evlaution):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try: 
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
