import os

from dotenv import load_dotenv
from binance import Client
import pandas as pd

from trainer import Trainer
import visualizer
import model


load_dotenv()

api_key = os.getenv('READONLY_API_KEY')
secret_key = os.getenv('READONLY_SECRET_KEY')

client = Client(api_key, secret_key)
model = model.get_model()
trainer = Trainer(client, model)

trainer.build_new_dataset(
				symbol='BTCUSDT', 
				interval='1m', 
				period='100 days',
				# period='90 minutes',
	)

trainer.train()