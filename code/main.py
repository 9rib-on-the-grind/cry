import os

from dotenv import load_dotenv
from binance import Client
import pandas as pd

from trainer import Trainer
import visualizer


load_dotenv()

api_key = os.getenv('READONLY_API_KEY')
secret_key = os.getenv('READONLY_SECRET_KEY')

client = Client(api_key, secret_key)
trainer = Trainer(client)

df = trainer.get_history(
					symbol='BTCUSDT', 
					interval='1m', 
					period='1 day',
	)

# visualizer.show_graph(df)
