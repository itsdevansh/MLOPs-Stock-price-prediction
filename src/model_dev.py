import logging
from abc import ABC, abstractmethod
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
import mlflow.keras.save
import tensorflow as tf
import os

import mlflow
import mlflow.keras

from zenml import get_step_context

class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LSTMModel(Model):

    def train(self, X_train, y_train, **kwargs):
        try:
            mlflow.keras.autolog(registered_model_name="lstm_alpha", save_exported_model=True)
            current_run = mlflow.start_run()
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1], X_train.shape[2],)))
            model.add(LSTM(units=30))
            model.add(Dense(units = 1))
            model.compile(optimizer = 'adam', loss = 'mse')
            model.fit(X_train, y_train, 20)

            # with mlflow.start_run() as run:
            #     print('artifact uri:', mlflow.get_artifact_uri())
            #     mlflow.keras.save.log_model(model=model, registered_model_name="lstm_alpha", artifact_path=mlflow.get_artifact_uri().replace("file:///", ""))
            #     run_id = run.info.run_id

            run_id = current_run.info.run_id
            model_uri=f"runs:/{run_id}/model"

            logging.info("Model training completed")
            return model, model_uri
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
        finally:
            mlflow.end_run()


        

