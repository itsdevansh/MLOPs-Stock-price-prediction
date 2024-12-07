from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

import mlflow
import logging

@pipeline
def train_pipeline(ticker: str):

    df, ticker, action = ingest_df(ticker)
    if action != "nothing":
        X_train, X_test, y_train, y_test = clean_df(df, ticker)
        model, model_uri = train_model(X_train, X_test, y_train, y_test, ticker, action)
        mae = evaluate_model(model, X_test, y_test)
        print(model_uri)
        return model, model_uri
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
        return model, model_uri
