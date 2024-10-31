import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from tools.technical_indicators import calculate_technical_indicators

class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        pass

class DataPreprocessingStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        
        try:
            data = calculate_technical_indicators(data)
            data = data.dropna(axis=0)
            return data
        except Exception as e:
            logging.error(f"Error while processing data: {e}")
            raise e
    
class FeatureSelectionStrategy(DataStrategy):

    def __init__(self, train_split: float) -> None:
        self.train_split = train_split

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:

        # Using Random Forest Regressor as the feature selection strategy
        rfr = RandomForestRegressor(n_estimators=300, random_state=42)

        train_data = pd.DataFrame(data.iloc[0:int(len(data)*self.train_split), :])
        test_data = pd.DataFrame(data.iloc[int(len(data)*self.train_split):len(data), :])

        # Scaling the data
        scaler = MinMaxScaler()
        scaled_train_data = pd.DataFrame(scaler.fit_transform(train_data))
        scaled_train_data.columns = train_data.columns
        
        scaled_test_data = pd.DataFrame(scaler.transform(test_data))
        scaled_test_data.columns = test_data.columns

        X_train = train_data
        Y_train = train_data['Close'].values
        rfr.fit(X_train, Y_train)
        # Get feature importances from the trained model
        importances = rfr.feature_importances_
        # Sort the feature importances in descending order
        indices = np.argsort(importances)[::-1][:15]
        imp_feature_data = data.iloc[:, indices]
        imp_feature_data['Close'] = data['Close']
        return imp_feature_data

class DataDivideStrategy(DataStrategy):

    def __init__(self, train_split: float) -> None:
        self.train_split = train_split

    def handle_data(self, data: pd.DataFrame) -> np.ndarray:
        try:
            train_data = pd.DataFrame(data.iloc[0:int(len(data)*self.train_split), :])
            test_data = pd.DataFrame(data.iloc[int(len(data)*self.train_split):len(data), :])

            # Scaling the data
            scaler = MinMaxScaler()
            scaled_train_data = pd.DataFrame(scaler.fit_transform(train_data))
            scaled_train_data.columns = train_data.columns
            
            scaled_test_data = pd.DataFrame(scaler.transform(test_data))
            scaled_test_data.columns = test_data.columns

            x_train = []
            y_train = [] 
            x_test = []
            y_test = []

            # Convert stock data into timesteps for LSTM 
            for i in range(100, scaled_train_data.shape[0]):
                x_train.append(scaled_train_data.iloc[i-100: i, data.columns != 'Close'])
                y_train.append(scaled_train_data['Close'][i])
           
            for i in range(100, scaled_test_data.shape[0]):
                x_test.append(scaled_test_data.iloc[i-100: i, data.columns != 'Close'])
                y_test.append(scaled_test_data['Close'][i])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_test, y_test = np.array(x_test), np.array(y_test)
            return x_train, x_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error while dividing data: {e}")
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