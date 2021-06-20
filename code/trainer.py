import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Trainer:
	def __init__(self, client):
		self.client = client
		self.create_directories()

	def create_directories(self):
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
		self.directory = f'./data/{timestamp}/'
		self.dataset_directory = self.directory + 'dataset/'
		os.mkdir(self.directory)
		os.makedirs(self.dataset_directory + 'train')
		os.makedirs(self.dataset_directory + 'valid')
		os.makedirs(self.dataset_directory + 'test')

	def build_dataset(self, symbol='BTCUSDT', interval='1m', period='10 days'):
		df = self.get_history(symbol, interval, period)
		self.split_into_files(df)


	def get_history(self, symbol='BTCUSDT', interval='1m', period='100 day'):
		klines = self.client.get_historical_klines(symbol=symbol, 
												   interval=interval, 
												   start_str= period + ' ago UTC')
		labels = ['Open time', 'Open', 'High', 'Low', 'Close', 
				  'Volume', 'Close time', 'Quote asset volume', 'Number of trades',
				  'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
		return pd.DataFrame(data=klines, columns=labels, dtype=float)

	def split_into_files(self, df, trailing_candlesticks=30, train=.7, valid=.2, test=.1):
		n = trailing_candlesticks
		atributes = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number of trades']
		df = df[atributes]
		
		ids = np.random.permutation(np.arange(n, len(df)))
		splits = [int(len(ids) * train), int(len(ids) * (train + valid))]
		indices_subsets = np.split(ids, splits)

		for ids, name in zip(indices_subsets, ('train', 'valid', 'test')):
			for i in ids:
				df[i-n:i].to_csv(self.dataset_directory + f'{name}/in_{i}.csv', index=False)
				df.iloc[i][['High', 'Close']].to_csv(self.dataset_directory + f'{name}/out_{i}.csv', index=False, header=False)