import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from tools.technical_indicators import calculate_technical_indicators


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessingStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # Handle DatetimeIndex if present
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()

            # Convert Date column to datetime if it exists
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Ensure all price and volume columns are numeric
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            # Calculate technical indicators
            data = calculate_technical_indicators(data)
            
            # Drop rows with missing values
            data = data.dropna(axis=0)

            # Drop the Date column as it's not needed for modeling
            if 'Date' in data.columns:
                data = data.drop('Date', axis=1)

            return data
        except Exception as e:
            logging.error(f"Error while processing data: {e}")
            raise e


class FeatureSelectionStrategy(DataStrategy):
    def __init__(self, train_split: float) -> None:
        self.train_split = train_split

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # Using Random Forest Regressor for feature selection
            rfr = RandomForestRegressor(n_estimators=300, random_state=42)

            train_data = data.iloc[:int(len(data) * self.train_split)]
            test_data = data.iloc[int(len(data) * self.train_split):]

            # Handle non-numeric columns
            numeric_cols = train_data.select_dtypes(include=[np.number]).columns
            train_data = train_data[numeric_cols]
            test_data = test_data[numeric_cols]

            # Scaling the data
            scaler = MinMaxScaler()
            scaled_train_data = pd.DataFrame(
                scaler.fit_transform(train_data),
                columns=train_data.columns
            )
            
            scaled_test_data = pd.DataFrame(
                scaler.transform(test_data),
                columns=test_data.columns
            )

            X_train = scaled_train_data.drop('Close', axis=1)
            Y_train = scaled_train_data['Close'].values

            # Train Random Forest for feature selection
            rfr.fit(X_train, Y_train)
            
            # Get feature importances
            importances = rfr.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            # Select important features
            selected_features = X_train.columns[indices].tolist()
            selected_features.append('Close')  # Add target variable
            imp_feature_data = data[selected_features]
            
            return imp_feature_data
        except Exception as e:
            logging.error(f"Error in feature selection: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    def __init__(self, train_split: float) -> None:
        self.train_split = train_split

    def handle_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            train_data = data.iloc[:int(len(data) * self.train_split)]
            test_data = data.iloc[int(len(data) * self.train_split):]

            # Scaling the data
            scaler = MinMaxScaler()
            scaled_train_data = pd.DataFrame(
                scaler.fit_transform(train_data),
                columns=train_data.columns
            )
            
            scaled_test_data = pd.DataFrame(
                scaler.transform(test_data),
                columns=test_data.columns
            )

            x_train = []
            y_train = []
            x_test = []
            y_test = []

            # Create sequences for LSTM
            lookback = 100
            for i in range(lookback, len(scaled_train_data)):
                x_train.append(
                    scaled_train_data.drop('Close', axis=1).iloc[i-lookback:i].values
                )
                y_train.append(scaled_train_data['Close'].iloc[i])

            for i in range(lookback, len(scaled_test_data)):
                x_test.append(
                    scaled_test_data.drop('Close', axis=1).iloc[i-lookback:i].values
                )
                y_test.append(scaled_test_data['Close'].iloc[i])

            return (
                np.array(x_train),
                np.array(x_test),
                np.array(y_train),
                np.array(y_test)
            )
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e