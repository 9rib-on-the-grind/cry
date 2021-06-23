import tensorflow as tf
from tensorflow import keras


# class CandlestickCNN(keras.Model):
# 	def __init__(self, **kwargs):
# 		super(Model, self).__init__(**kwargs)


# class Model(keras.Model):
# 	def __init__(self, **kwargs):
# 		super(Model, self).__init__(**kwargs)
# 		self.modules = [
# 			CandlestickCNN(),
# 		]



def get_model():
	model = keras.models.Sequential()

	model.add(keras.layers.Conv1D(32, 5, padding='same', input_shape=(20, 6)))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(2))

	model.compile(
			optimizer='SGD',
			loss='mse'
		)

	return model