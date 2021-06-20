import matplotlib.pyplot as plt
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc


def show_graph(df=None, file=None):
	if df is not None or file is not None:
		if file is not None:
			df = pd.read_csv(file, dtype=float)
		df = df[['Open time', 'Open', 'High', 
		         'Low', 'Close']]
		df['Open time'] = range(len(df))

		plt.style.use('dark_background')
		fig, ax = plt.subplots()
		ax.grid(color='grey', linestyle='dashed', linewidth=.5, alpha=.5)
		  
		candlestick_ohlc(ax, df.values, width = 0.6,
		                 colorup = 'green', colordown = 'red', 
		                 alpha = 0.9)
		
		ax.set_xlabel('Time')
		ax.set_ylabel('Price')
		fig.tight_layout()
		plt.show()