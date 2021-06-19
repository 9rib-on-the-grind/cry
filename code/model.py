import tensorflow as tf
from tensorflow import keras





class CandlestickCNN(keras.Model):
	def __init__(self, **kwargs):
		super(Model, self).__init__(**kwargs)

		self.model = keras.models.Sequential()
		self.model.add(keras.layer.conv1d(64, 7, 1, padding='same', input_shape=(None, )))

		filters = [64] + [128] + [256]



class Model(keras.Model):
	def __init__(self, **kwargs):
		super(Model, self).__init__(**kwargs)
		self.modules = [
			CandlestickCNN(),
		]




