import datetime
import os

import pandas as pd
import numpy as np
import tensorflow as tf


class DatasetHandler:
	def __init__(self, client, trailing_candlesticks):
		self.client = client
		self.trailing_candlesticks = trailing_candlesticks
		self.output_attributes = ['Open', 'High', 'Low', 'Close']

		# min max values for past 100 days
		self.dfmin = np.array([0.93886857, 0.95632116, 0.93803297, 0.94201088])
		self.dfmax = np.array([1.04509891, 1.0432493, 1.03086067, 1.04629241])


	def create_directories(self):
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
		self.dataset_directory = f'./data/{timestamp}/'
		os.makedirs(self.dataset_directory + 'train')
		os.makedirs(self.dataset_directory + 'valid')
		os.makedirs(self.dataset_directory + 'test')


	def generate_dataset(self, symbol='BTCUSDT', interval='1m', period='90m'):
		self.create_directories()

		print('getting history')
		self.history = self.get_history(symbol, interval, period)
		print('preparing data')
		self.dataframe = self.prepare_data(self.history)

		subsets = self.train_test_valid_split(train=.7, valid=.2, test=.1)
		
		print('writing tfrecords')
		for ids, name in zip(subsets, ('train', 'test', 'valid')):
			print(name)
			self.write_tfrecords(ids, name)

	def get_datasets(self, dataset_directory=None):
		dataset_directory = f'./data/{dataset_directory}/' if dataset_directory is not None else self.dataset_directory
		train_filenames = tf.io.gfile.glob(dataset_directory + 'train/*.tfrecord')
		valid_filenames = tf.io.gfile.glob(dataset_directory + 'valid/*.tfrecord')
		test_filenames = tf.io.gfile.glob(dataset_directory + 'test/*.tfrecord')

		self.train_dataset = self.load_dataset(train_filenames)
		self.valid_dataset = self.load_dataset(valid_filenames)
		self.test_dataset = self.load_dataset(test_filenames)

		return self.train_dataset, self.valid_dataset, self.test_dataset


	def get_history(self, symbol, interval, period):
		klines = self.client.get_historical_klines(symbol=symbol, interval=interval, start_str= period + ' ago UTC')
		labels = ['Open time', 'Open', 'High', 'Low', 'Close', 
				  'Volume', 'Close time', 'Quote asset volume', 'Number of trades',
				  'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
		df = pd.DataFrame(data=klines, columns=labels, dtype=float)
		return df

	def prepare_data(self, df):

		atributes = ['Open', 'High', 'Low', 'Close']
		df = df[atributes]

		# body_center = (df['Open'] + df['Colse']) / 2
		# body_heigh = df['Close'] - df['Open']
		# shadow_center = (df['Low'] + df['High']) / 2
		# shadow_heigh = df['High'] - df['Low']

		df = df.iloc[1:] / df.iloc[:-1].values

		self.dfmin = df[self.output_attributes].min().to_numpy()
		self.dfmax = df[self.output_attributes].max().to_numpy()

		print('='*100)
		print(self.dfmin)
		print(self.dfmax)

		df = (df - df.min()) / (df.max() - df.min())

		return df

	def denormalize(self, data):
		return data * (self.dfmax - self.dfmin) + self.dfmin


	def train_test_valid_split(self, train, valid, test):
		ids = np.random.permutation(np.arange(self.trailing_candlesticks, len(self.dataframe)))
		splits = [int(len(ids) * train), int(len(ids) * (train + valid))]
		subsets = np.split(ids, splits)
		return subsets


	def write_tfrecords(self, ids, name):
		def _bytes_feature(value):
		    if isinstance(value, type(tf.constant(0))):
		        value = value.numpy()
		    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

		def _float_feature(value):
		    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

		def _int64_feature(value):
		    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

		def serialize_example(data, target):
			feature = {
				'data': _bytes_feature(data),
				'target': _bytes_feature(target),
			}
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			return example.SerializeToString()

		def tf_serialize_example(args):
			tf_string = tf.py_function(serialize_example, *args, tf.string)
			return (tf.reshape(tf_string, ()))


		dataset = []
		for i in ids:
			data = self.dataframe[i-self.trailing_candlesticks:i].to_numpy()
			target = self.dataframe.iloc[i][self.output_attributes].to_numpy()
			
			data = tf.convert_to_tensor(data)
			target = tf.convert_to_tensor(target)

			example = serialize_example(tf.io.serialize_tensor(data), tf.io.serialize_tensor(target))
			dataset.append(example)

		dataset = tf.data.Dataset.from_tensor_slices(dataset)

		filename = self.dataset_directory + name + '/rec.tfrecord'
		writer = tf.data.experimental.TFRecordWriter(filename)

		writer.write(dataset)


	def load_dataset(self, filenames):
		def deserialize(example):
			return tf.io.parse_single_example(example, features_description)

		def parse_tensors(example):
			example = {name: tf.io.parse_tensor(example[name], tf.float64) for name in example}
			return example['data'], example['target']

		def reshape_tensors(data, target):
			data = tf.reshape(data, (self.trailing_candlesticks, len(self.output_attributes)))
			target = tf.reshape(target, (len(self.output_attributes),))
			return data, target

		features_description = {
			'data': tf.io.FixedLenFeature([], tf.string),
			'target': tf.io.FixedLenFeature([], tf.string),
		}

		dataset = tf.data.TFRecordDataset(filenames)
		dataset = dataset.map(deserialize)
		dataset = dataset.map(parse_tensors)
		dataset = dataset.map(reshape_tensors)
		dataset = dataset.batch(32)

		return dataset