import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import mlflow_deployment
from zenml.integrations.mlflow.steps import mlflow_deployer
from zenml import pipeline
from zenml.client import Client
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentConfig, MLFlowDeploymentService
from typing import Optional
from mlflow.tracking import MlflowClient, artifact_utils
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from pipelines.training_pipeline import train_pipeline
from steps.model_deploy import deploy_model
# from zenml.steps import Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model


@pipeline(enable_cache=True)
def continuous_deployment_pipeline(ticker: str = "MSFT"):
    """Run a training job and deploy an MLflow model deployment."""
    # Run the training pipeline
    model, model_uri = train_pipeline(ticker=ticker)
    

    
    

# @pipeline(enable_cache=False)
# def inference_pipeline():
#     """Run a batch inference job with data loaded from an API."""
#     # Load batch data for inference
#     batch_data = dynamic_importer()

#     # Load the deployed model service
#     model_deployment_service = prediction_service_loader(
#         pipeline_name="continuous_deployment_pipeline",
#         step_name="mlflow_model_deployer_step",
#     )

#     # Run predictions on the batch data
#     predictor(service=model_deployment_service, input_data=batch_data)