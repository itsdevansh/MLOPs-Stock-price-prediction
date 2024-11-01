import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error

class Evlaution(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MAE(Evlaution):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try: 
            logging.info("Calculating MAE")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"MSE: {mae}")
            return mae
        except Exception as e:
            logging.error(f"Error in calculating MAE: {e}")
            raise e
