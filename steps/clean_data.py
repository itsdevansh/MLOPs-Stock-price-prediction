import logging

import pandas as pd
import numpy as np
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessingStrategy, FeatureSelectionStrategy
from typing_extensions import Annotated
from typing import Tuple

@step(enable_cache=False)
def clean_df(df: pd.DataFrame, ticker: str) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    try:
        preprocess_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        feature_select_strategy = FeatureSelectionStrategy(train_split=0.7)
        data_cleaning = DataCleaning(preprocessed_data, feature_select_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy(train_split=0.7)
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e