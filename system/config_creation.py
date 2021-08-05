import json
import collections

import numpy as np

import experts



def create_searchspace_config():
	"""Writes json file with parameter ranges to be searched."""

	def get_logspace(start, stop, num):
		space = list(sorted(set(np.logspace(start, stop, num, dtype=int))))
		return list(map(int, space))

	length = get_logspace(.7, 2.5, 5)
	patience = list(range(2, 8, 2))

	cfg_file = open('searchspace.json', 'w')

	nested_dict = lambda: collections.defaultdict(nested_dict)
	data = collections.defaultdict(nested_dict)

	"""
	Structure:
		rule: {
			'parameters': {atribute: search_space}
			'indicators': [
				{
					'name': string,
					'parameters': {atribute: search_space}
				}
			]
		}
	"""

	# MACrossover
	data['MovingAverageCrossoverRule']['parameters'] = {'patience': patience}
	data['MovingAverageCrossoverRule']['indicators'] = [
		{'name': 'MovingAverageIndicator', 'parameters': {'length': length}},
		{'name': 'MovingAverageIndicator', 'parameters': {'length': length}},
	]

	# RSITrashold
	data['RelativeStrengthIndexTrasholdRule']['parameters'] = {
		'patience': patience,
		'lower': list(range(20, 45, 10)),
		'upper': list(range(60, 85, 10)),
	}
	data['RelativeStrengthIndexTrasholdRule']['indicators'] = [
		{'name': 'RelativeStrengthIndexIndicator', 'parameters': {'length': length}}
	]

	json.dump(data, cfg_file, indent=4)

def serialize_expert_to_json(filename: str = 'expert.json',
							 expert: experts.BaseExpert = None):
	def get_hierarchy(expert: experts.BaseExpert):
		state = {}
		state['name'] = expert.name
		if isinstance(expert, experts.RuleExpert):
			rule, indicators = expert._rule, expert._indicators
			state['rule'] = {'name': rule.__class__.__name__, 'parameters': rule.get_parameters()}
			state['indicators'] = [{'name': indicator.__class__.__name__, 
									'parameters': indicator.get_parameters()} 
																for indicator in indicators]
		else:
			state['inner experts'] = [get_hierarchy(exp) for exp in expert._inner_experts]
		return state

	hierarchy = get_hierarchy(expert)
	json.dump(hierarchy, open(filename, 'w'), indent=4)


if __name__ == '__main__':
	create_searchspace_config()