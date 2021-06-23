import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from datasethandler import DatasetHandler


class Trainer:
	def __init__(self, client, model, trailing_candlesticks=20):
		self.client = client
		self.model = model
		self.trailing_candlesticks = trailing_candlesticks
		self.datasethandler = DatasetHandler(client, trailing_candlesticks)


	def build_new_dataset(self, symbol, interval, period):
		# self.datasethandler.generate_dataset(symbol, interval, period)
		# self.train_dataset, self.valid_dataset, self.test_dataset = self.datasethandler.get_datasets()
		self.train_dataset, self.valid_dataset, self.test_dataset = self.datasethandler.get_datasets('100_days_norm')


	def train(self):
		self.model.fit(
				self.train_dataset,
				epochs=10,
			)