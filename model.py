from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
import tensorflow as tf
import os


def LSTMModel(x_train, LSTM_cells=30):
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2],)))
    model.add(LSTM(units=LSTM_cells))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mse', metrics=['mae'])
    return model

def lstm_train(x_train, y_train, update=False, epochs=20):
    print('Training model...')

    checkpoint_path = "models/lstm.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model = LSTMModel(x_train, 30)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    if os.path.exists(checkpoint_path) and update:

        # Load the saved state_dict into the model
        model.load_weights(checkpoint_path)

    model.fit(x_train, y_train, epochs, callbacks=[cp_callback])

    print("Model trained and saved!")
    return checkpoint_path

def lstm_test(x_test, y_test, checkpoint_path="models/lstm.weights.h5"):

    model = LSTMModel(x_test, 30)

    model.load_weights(checkpoint_path)

    model.evaluate(x_test, y_test)
