from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":

    # print(Client().active_stack.experiment_tracker.get)
    train_pipeline.with_options(config_path='config.yaml')(ticker='MSFT')