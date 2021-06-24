import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_model():
	model = keras.models.Sequential()
	model.add(keras.layers.Conv1D(32, 3, padding='same', input_shape=(20, 6)))
	model.add(CandlestickCNNModule())

	# model = Model()
	
	model.compile(optimizer=keras.optimizers.SGD(learning_rate=.1, momentum=.9, nesterov=True),
				  loss='mse',
		)
	return model




class CandlestickCNNModule(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(CandlestickCNNModule, self).__init__(**kwargs)

		self.main_layers = [
			keras.layers.Conv1D(32, 5, padding='same'),
			keras.layers.GlobalAveragePooling1D(),
			keras.layers.Flatten(),
			keras.layers.Dense(6),
		]

	def call(self, inputs):
		for layer in self.main_layers:
			inputs = layer(inputs)
		return inputs


class Model(keras.Model):
	def __init__(self, **kwargs):
		super(Model, self).__init__(**kwargs)

		self.candlestick_module = CandlestickCNNModule()


	def call(self, inputs):
		inputs = self.input_layer(inputs)
		return self.candlestick_module(inputs)