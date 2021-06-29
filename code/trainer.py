import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from datasethandler import DatasetHandler
from visualizer import show_graph



class Trainer:
	def __init__(self, client, model, trailing_candlesticks=5):
		self.client = client
		self.model = model
		self.trailing_candlesticks = trailing_candlesticks
		self.datasethandler = DatasetHandler(client, trailing_candlesticks)


	def build_new_dataset(self, symbol, interval, period):
		self.datasethandler.generate_dataset(symbol, interval, period)
		self.train_dataset, self.valid_dataset, self.test_dataset = self.datasethandler.get_datasets()
		# self.train_dataset, self.valid_dataset, self.test_dataset = self.datasethandler.get_datasets('percent')


	def train(self):
		self.model.fit(
				self.train_dataset,
				epochs=3,
				validation_data=self.valid_dataset,
	            callbacks=[keras.callbacks.ReduceLROnPlateau(patience=5, verbose=True)],
			)

		self.visual_validation()


	def visual_validation(self, n=20):
		inputs = self.train_dataset.take(20)
		predictions = self.model.predict(inputs)

		print('min prediction:', np.min(predictions))
		print('max prediction:', np.max(predictions))

		targets = [tar.numpy().reshape(-1) for (data, tar) in inputs.unbatch()]
		predictions = [pred.reshape(-1) for pred in predictions]

		plt.plot(targets[:n], color='green')
		plt.plot(predictions[:n], color='red')
		plt.plot([1] * n, color='black')
		plt.show()