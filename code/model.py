import tensorflow as tf
from tensorflow import keras





class CandlestickCNN(keras.Model):
	def __init__(self, **kwargs):
		super(Model, self).__init__(**kwargs)


class Model(keras.Model):
	def __init__(self, **kwargs):
		super(Model, self).__init__(**kwargs)
		self.modules = [
			CandlestickCNN(),
		]