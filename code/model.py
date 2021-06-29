import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_model():
    model = get_candlestick_rnn_model()
    model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=.1, momentum=.9, nesterov=True),
            loss='mse',
        )
    return model



def get_candlestick_rnn_model():
    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(50, recurrent_dropout=.5, return_sequences=True))
    model.add(keras.layers.LSTM(50, recurrent_dropout=.5, return_sequences=False))
    model.add(keras.layers.Dense(1))

    return model

