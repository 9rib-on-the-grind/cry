import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


class Trainer:
	def __init__(self, client, trailing_candlesticks=20):
		self.client = client
		self.create_directories()
		self.trailing_candlesticks = trailing_candlesticks

	def create_directories(self):
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
		self.directory = f'./data/{timestamp}/'
		self.dataset_directory = self.directory + 'dataset/'
		os.mkdir(self.directory)
		os.makedirs(self.dataset_directory + 'train')
		os.makedirs(self.dataset_directory + 'valid')
		os.makedirs(self.dataset_directory + 'test')

	def build_dataset(self, symbol='BTCUSDT', interval='1m', period='10 days'):
		self.history = self.get_history(symbol, interval, period)
		subsets = self.train_test_valid_split(train=.7, valid=.2, test=.1)
		for ids, name in zip(subsets, ('train', 'test', 'valid')):
			self.write_tfrecords(ids, name)

	def get_history(self, symbol='BTCUSDT', interval='1m', period='100 day'):
		klines = self.client.get_historical_klines(symbol=symbol, interval=interval, start_str= period + ' ago UTC')
		labels = ['Open time', 'Open', 'High', 'Low', 'Close', 
				  'Volume', 'Close time', 'Quote asset volume', 'Number of trades',
				  'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
		df = pd.DataFrame(data=klines, columns=labels, dtype=float)
		atributes = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number of trades']
		df = df[atributes]
		return df

	def train_test_valid_split(self, train, valid, test):
		ids = np.random.permutation(np.arange(self.trailing_candlesticks, len(self.history)))
		splits = [int(len(ids) * train), int(len(ids) * (train + valid))]
		subsets = np.split(ids, splits)
		return subsets


	def write_tfrecords(self, ids, name):

		# The following functions can be used to convert a value to a type compatible
		# with tf.train.Example.

		def _bytes_feature(value):
		    """Returns a bytes_list from a string / byte."""
		    if isinstance(value, type(tf.constant(0))):
		        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
		    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

		def _float_feature(value):
		    """Returns a float_list from a float / double."""
		    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

		def _int64_feature(value):
		    """Returns an int64_list from a bool / enum / int / uint."""
		    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

		def serialize_example(data, target):
			"""
			Creates a tf.train.Example message ready to be written to a file.
			"""
			# Create a dictionary mapping the feature name to the tf.train.Example-compatible
			# data type.
			feature = {
				'data': _bytes_feature(data),
				'target': _bytes_feature(target),
			}

			# Create a Features message using tf.train.Example.
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			return example.SerializeToString()

		def tf_serialize_example(args):
			tf_string = tf.py_function(serialize_example, *args, tf.string)
			return (tf.reshape(tf_string, ()))


		dataset = []
		for i in ids:
			data = self.history[i-self.trailing_candlesticks:i].to_numpy()
			data = tf.convert_to_tensor(data)
			target = self.history.iloc[i][['High', 'Close']].to_numpy()
			target = tf.convert_to_tensor(target)

			example = serialize_example(tf.io.serialize_tensor(data), tf.io.serialize_tensor(target))
			dataset.append(example)

		dataset = tf.data.Dataset.from_tensor_slices(dataset)

		filename = self.dataset_directory + name + '/rec.tfrecord'
		writer = tf.data.experimental.TFRecordWriter(filename)

		writer.write(dataset)