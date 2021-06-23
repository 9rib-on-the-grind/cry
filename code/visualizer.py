import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc



def show_graph(ax, data):
	index = np.arange(len(data)).reshape((-1, 1))
	data = np.array(data)
	data = np.hstack([index, data])

	plt.style.use('dark_background')
	ax.grid(color='grey', linestyle='dashed', linewidth=.5, alpha=.5)
	  
	candlestick_ohlc(ax, data, width = 0.6,
	                 colorup = 'green', colordown = 'red', 
	                 alpha = 0.9)