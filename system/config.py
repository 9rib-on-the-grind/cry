import json
import collections
from itertools import product

import numpy as np

import experts
import indicators
import rules


def create_searchspace_config():
	"""Writes json file with parameters searchspace.

	Structure:
		{
			rule: {
				'parameters': {attribute: search_space}
				'indicators': [
					{
						'name': string,
						'parameters': {attribute: search_space}
					}
				]
			}
		}
	"""
	
	def get_logspace(start, stop, num):
		space = list(sorted(set(np.logspace(start, stop, num, dtype=int))))
		return list(map(int, space))

	length = get_logspace(.7, 2.5, 5)
	patience = list(range(2, 8, 2))

	cfg_file = open('searchspace.json', 'w')

	nested_dict = lambda: collections.defaultdict(nested_dict)
	data = collections.defaultdict(nested_dict)


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

def get_experts_from_searchspace(rule_cls: rules.BaseRule,
                             	 cfg: str = 'searchspace.json') -> list[experts.RuleExpert]:

        cfg = json.load(open(cfg, 'r'))
        rule_name = rule_cls.__name__

        rule_parameters = cfg[rule_name]['parameters']
        indicators_lst = cfg[rule_name]['indicators']
        indicator_cls_names = list(ind['name'] for ind in indicators_lst)
        indicator_parameters = [ind['parameters'] for ind in indicators_lst]

        res = []
        for rule_params in product(*rule_parameters.values()):
            rule_kwargs = {key: val for key, val in zip(list(rule_parameters), rule_params)}
            
            indicator_combinations = [product(*ind.values()) for ind in indicator_parameters]
            for inds_params in product(*indicator_combinations):
                lst = []
                for cls_name, (attrs, params) in zip(indicator_cls_names, 
                                                     zip((param.keys() for param in indicator_parameters), inds_params)):
                    indicator_kwargs = {attr: val for attr, val in zip(attrs, params)}
                    indicator_cls = getattr(indicators, cls_name)
                    lst.append(indicator_cls(**indicator_kwargs))

                rule = rule_cls(**rule_kwargs)
                res.append(experts.RuleExpert(rule, lst))

        return res

def serialize_expert_to_json(filename: str = 'expert.json',
							 expert: experts.BaseExpert = None):
	"""
	Structure:
		{
			'name': string,
			'inner experts: [serialized experts]'
		}

		OR

		{
			'name': string,
			'rule': {
				'name': string,
				'parameters': {attribute: value}
			}
			'indicators': [
				{
					'name': 'string',
					'parameters': {attribute: value}
				}
			]
		}
	"""

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