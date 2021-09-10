import os
import datetime

from dotenv import load_dotenv
from binance import Client
import pandas as pd

def get_history(symbol, interval):
    period = '4 years' if interval != '1m' else '3 months'
    klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=period)
    labels = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
              'Quote asset volume', 'Number of trades', 
              'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    df = pd.DataFrame(data=klines, columns=labels, dtype=float)
    return df

data_dir = './data/'

load_dotenv('.env')
api_key = os.getenv('READONLY_API_KEY')
secret_key = os.getenv('READONLY_SECRET_KEY')
client = Client(api_key, secret_key)

timeframes = ['1d', '12h', '8h', '6h', '4h', '2h', '1h', '30m', '15m', '5m', '3m']
pairs = ['BTCUSDT', 'ETHUSDT']

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
for pair in pairs:
    for timeframe in timeframes:
        print(f'writing {pair} | {timeframe}')
        dir = f'{data_dir}{timestamp}/{pair}/'
        os.makedirs(dir, exist_ok=True)
        history = get_history(pair, timeframe)
        history = history.iloc[:-1]
        history.to_csv(dir + timeframe + '.csv', index=False)