import mlflow.models
from zenml import step
import mlflow
import mlflow.keras
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentConfig, MLFlowDeploymentService
import logging
from keras.models import Sequential
from zenml.client import Client

# from zenml.services.service_type import

@step(enable_cache=False)
def deploy_model(model: Sequential, model_uri: str):
    
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer

    # Configure MLFlow deployer using the latest BaseModelDeployerStepConfig
    deploy_config = MLFlowDeploymentConfig(
        model_name="lstm_alpha",
        model_uri=model_uri,
        tracking_uri=mlflow.get_tracking_uri(),
        flavor="keras",  # Use keras flavor for deployment
        workers=3,
    )

    service = model_deployer.deploy_model(config=deploy_config, service_type=MLFlowDeploymentService.SERVICE_TYPE)
    logging.info(f"The deployed service info: {model_deployer.get_model_server_info(service)}")
    return service
    # # Use ZenML's mlflow_model_deployer_step to deploy the model
    # mlflow_model_deployer_step(
    #     model=model_uri,
    #     model_name="lstm_alpha",
    #     # model_uri=model_uri,
    #     # tracking_uri=mlflow.get_tracking_uri(),
    #     # flavor="keras",  # Use keras flavor for deployment
    #     workers=3,
    #     timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT,)