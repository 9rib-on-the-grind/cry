import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from datasethandler import DatasetHandler
from visualizer import show_graph


class Trainer:
	def __init__(self, client, model, trailing_candlesticks=30):
		self.client = client
		self.model = model
		self.trailing_candlesticks = trailing_candlesticks
		self.datasethandler = DatasetHandler(client, trailing_candlesticks)


	def build_new_dataset(self, symbol, interval, period):
		# self.datasethandler.generate_dataset(symbol, interval, period)
		# self.train_dataset, self.valid_dataset, self.test_dataset = self.datasethandler.get_datasets()
		self.train_dataset, self.valid_dataset, self.test_dataset = self.datasethandler.get_datasets('data')


	def train(self):
		self.model.fit(
				self.train_dataset,
				epochs=3,
			)

		# self.model.evaluate()
		# self.visual_validation()
		self.numeric_validation()



	def numeric_validation(self, n=20):
		inputs = self.test_dataset
		predictions = self.model.predict(inputs)

		tar_list, pred_list = [], []

		for (data, target), pred in zip(inputs.unbatch(), predictions):

			tar, pred = (self.datasethandler.denormalize(target.numpy()), 
							  self.datasethandler.denormalize(pred))
			tar, pred = (tar - 1) * 100, (pred - 1) * 100
			if tar[3] > 1 or tar[3] < -1:
				tar_list.append(tar[3])
				pred_list.append(pred[3])

		plt.plot(tar_list, color='green')
		plt.plot(pred_list, color='red')
		plt.plot([0] * len(tar_list), color='black')
		plt.show()
		


	# def visual_validation(self, n=20):
	# 	fig, axs = plt.subplots(nrows=4, ncols=n//4)
		
	# 	inputs = self.test_dataset.take(1)
	# 	predictions = self.model.predict(inputs)

	# 	for ax, (data, target), pred in zip(axs.reshape(-1), inputs.unbatch(), predictions):
	# 		data, target, pred = map(np.array, [data, target, pred])
	# 		candlesticks = np.vstack([data[-3:], target.reshape((1, -1)), pred.reshape((1, -1))])
	# 		show_graph(ax, candlesticks)

	# 		# ax.set_yticklabels([])
	# 		# ax.set_xticklabels([])

	# 	plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99, wspace=.07, hspace=.07)
	# 	plt.show()
