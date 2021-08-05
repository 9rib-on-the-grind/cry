import json
import collections

import numpy as np


def create_searchspace_config():
	"""Writes json file with parameter ranges to be searched."""

	def get_logspace(start, stop, num):
		space = list(sorted(set(np.logspace(start, stop, num, dtype=int))))
		return list(map(int, space))


	length = get_logspace(.7, 2.5, 20)
	patience = list(range(1, 8, 1))

	cfg_file = open('searchspace.json', 'w')

	nested_dict = lambda: collections.defaultdict(nested_dict)
	data = collections.defaultdict(nested_dict)


	"""
	Structure:
		rule: {
			'rule': {atribute: search_space}
			'indicators': {indicator: {atribute: search_space}}
		}
	"""

	# MACrossover
	data['MovingAverageCrossoverRule']['rule'] = {'patience': patience}
	for i in range(2):
		data['MovingAverageCrossoverRule']['indicators'][f'MovingAverageIndicator {i + 1}'] = {
			'length': length
		}

	# RSITrashold
	data['RelativeStrengthIndexTrasholdRule']['rule'] = {
		'patience': patience,
		'lower': list(range(20, 45, 5)),
		'upper': list(range(60, 85, 5)),
	}
	data['RelativeStrengthIndexTrasholdRule']['indicators']['RelativeStrengthIndexIndicator'] = {
		'length': length
	}
					

	json.dump(data, cfg_file, sort_keys=False, indent=4)


if __name__ == '__main__':
	create_searchspace_config()