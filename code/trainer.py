import pandas as pd
import matplotlib.pyplot as plt



class Trainer:
	def __init__(self, client):
		self.client = client

	def get_history(self, symbol='BTCUSDT', interval='1m', period='100 day'):
		klines = self.client.get_historical_klines(symbol=symbol, 
												   interval=interval, 
												   start_str= period + ' ago UTC')
		labels = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
		df = pd.DataFrame(data=klines, columns=labels, dtype=float)

		df.to_csv('./data/test.csv', index=False)
		return df

