import logging
from abc import ABC, abstractmethod
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
import mlflow.keras.save
import tensorflow as tf
import os
from zenml.client import Client
import mlflow
import mlflow.keras

from datetime import datetime

class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LSTMModel(Model):

    def train(self, X_train, y_train, ticker, action, **kwargs):
        try:
            if action == "update":
                runs = mlflow.search_runs(
                    experiment_ids=["0"],  # Replace with your experiment ID
                    filter_string=f"params.dataset = '{ticker}'",
                    order_by=["params.start_time DESC"]
                )
                print("RUNS: ", runs)
                if len(runs) != 0:
                    # Load the latest model
                    latest_run = runs[0]
                    model_uri = f"runs:/{latest_run.info.run_id}/model"
                    model = mlflow.keras.load_model(model_uri)
                    logging.info(f"Loaded model from run ID: {latest_run.info.run_id}.")
            elif action == "train":
                # create a new model
                model = Sequential()
                model.add(Input(shape=(X_train.shape[1], X_train.shape[2],)))
                model.add(LSTM(units=30))
                model.add(Dense(units = 1))
                model.compile(optimizer = 'adam', loss = 'mse')
                logging.info("New model created for training.")

            mlflow.end_run()
            current_run = mlflow.start_run()
            mlflow.keras.autolog(registered_model_name=f"lstm_{ticker}", save_exported_model=True)
            mlflow.log_param("dataset", ticker)
            mlflow.log_param("start_date", datetime.now().strftime('%Y-%m-%d'))
            model.fit(X_train, y_train, 20)
            run_id = current_run.info.run_id
            model_uri=f"runs:/{run_id}/model"
            logging.info("Model training completed")
            return model, model_uri
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
        finally:
            mlflow.end_run()


        

