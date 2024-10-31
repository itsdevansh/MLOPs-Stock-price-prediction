import logging
from abc import ABC, abstractmethod
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
import tensorflow as tf
import os

class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LSTMModel(Model):

    def train(self, X_train, y_train, **kwargs):

        try:
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1], X_train.shape[2],)))
            model.add(LSTM(units=30))
            model.add(Dense(units = 1))
            model.compile(optimizer = 'adam', loss = 'mse', metrics=['mae'])
            model.fit(X_train, y_train, 20)
            logging.info("Model training completed")
            return model
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e

        

