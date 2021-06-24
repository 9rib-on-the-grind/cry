import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from datasethandler import DatasetHandler
from visualizer import show_graph


class Trainer:
	def __init__(self, client, model, trailing_candlesticks=20):
		self.client = client
		self.model = model
		self.trailing_candlesticks = trailing_candlesticks
		self.datasethandler = DatasetHandler(client, trailing_candlesticks)


	def build_new_dataset(self, symbol, interval, period):
		# self.datasethandler.generate_dataset(symbol, interval, period)
		# self.train_dataset, self.valid_dataset, self.test_dataset = self.datasethandler.get_datasets()
		self.train_dataset, self.valid_dataset, self.test_dataset = self.datasethandler.get_datasets('100_day')


	def train(self):
		self.model.fit(
				self.train_dataset,
				epochs=3,
			)
		self.visual_validation()


	def visual_validation(self, n=20):
		fig, axs = plt.subplots(nrows=4, ncols=n//4)
		
		inputs = self.train_dataset.take(1)
		predictions = self.model.predict(inputs)

		for ax, (data, target), pred in zip(axs.reshape(-1), inputs.unbatch(), predictions):
			data, target, pred = map(np.array, [data, target, pred])
			candlesticks = np.vstack([data[-3:], target.reshape((1, -1)), pred.reshape((1, -1))])
			show_graph(ax, candlesticks)

		# plt.tight_layout()
		plt.show()

	def simuilation(self):
		predictions = self.model.predict(self.test_dataset)