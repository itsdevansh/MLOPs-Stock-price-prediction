from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":

    train_pipeline.with_options(config_path='config.yaml')(ticker='MSFT')