import logging
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.optimizers import Adam
import mlflow.keras
from datetime import datetime

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LSTMModel(Model):
    def train(self, X_train, y_train, ticker, action, **kwargs):
        try:
            model = None  # Initialize model variable
            
            if action == "update":
                runs = mlflow.search_runs(
                    experiment_ids=["0"],  # Replace with your experiment ID
                    filter_string=f"params.dataset = '{ticker}'",
                    order_by=["params.start_time DESC"]
                )
                logging.info(f"RUNS: {runs}")
                if len(runs) != 0:
                    # Load the latest model
                    latest_run = runs[0]
                    model_uri = f"runs:/{latest_run.info.run_id}/model"
                    model = mlflow.keras.load_model(model_uri)
                    logging.info(f"Loaded model from run ID: {latest_run.info.run_id}.")
            
            # If model is still None (either action != "update" or no runs found)
            if model is None:
                # Create a new Sequential model
                model = Sequential([
                    LSTM(units=50, return_sequences=True, 
                         input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dropout(0.2),
                    LSTM(units=50, return_sequences=False),
                    Dropout(0.2),
                    Dense(units=25),
                    Dense(units=1)
                ])
                
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                logging.info("New model created for training.")

            mlflow.end_run()  # End any existing runs
            current_run = mlflow.start_run()
            
            try:
                mlflow.keras.autolog(registered_model_name=f"lstm_{ticker}", save_exported_model=True)
                mlflow.log_param("dataset", ticker)
                mlflow.log_param("start_date", datetime.now().strftime('%Y-%m-%d'))
                
                # Train the model with proper epochs parameter
                history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
                
                run_id = current_run.info.run_id
                model_uri = f"runs:/{run_id}/model"
                logging.info("Model training completed")
                
                return model, model_uri
                
            finally:
                mlflow.end_run()
                
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e