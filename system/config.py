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

	def get_logspace(first, last, num, dtype=int):
		start = np.log10(first)
		stop = np.log10(last)
		space = list(sorted(set(np.logspace(start, stop, num, dtype=dtype))))
		return list(map(dtype, space))

	length1 = get_logspace(7, 250, 70) # 61 element
	patience1 = get_logspace(1, 50, 25) # 19 elements | 61 * 19     = 1 159

	length2 = get_logspace(7, 100, 25) # 24 elements
	patience2 = get_logspace(1, 50, 10) # 9 elements       | 24 * 24 * 9 = 5 184

	cfg_file = open('searchspace.json', 'w')

	nested_dict = lambda: collections.defaultdict(nested_dict)
	data = collections.defaultdict(nested_dict)


	# MACrossover
	data['MovingAverageCrossoverRule']['parameters'] = {'patience': patience2}
	data['MovingAverageCrossoverRule']['indicators'] = [
		{'name': 'MovingAverageIndicator', 'parameters': {'length': length2}},
		{'name': 'MovingAverageIndicator', 'parameters': {'length': length2}},
	]

	# RSITrashold
	data['RelativeStrengthIndexTrasholdRule']['parameters'] = {
		'patience': patience2,
		'lower': list(range(20, 45, 10)),
		'upper': list(range(60, 85, 10)),
	}
	data['RelativeStrengthIndexTrasholdRule']['indicators'] = [
		{'name': 'RelativeStrengthIndexIndicator', 'parameters': {'length': length2}}
	]

	# TRIXDirectionChange
	data['TripleExponentialDirectionChangeRule']['parameters'] = {'patience': patience1}
	data['TripleExponentialDirectionChangeRule']['indicators'] = [
		{'name': 'TripleExponentialIndicator', 'parameters': {'length': length1}},
	]

	# IchimokuTenkanKijunCrossover
	data['IchimokuKinkoHyoTenkanKijunCrossoverRule']['parameters'] = {'patience': patience2}
	data['IchimokuKinkoHyoTenkanKijunCrossoverRule']['indicators'] = [
		{'name': 'IchimokuKinkoHyoIndicator', 'parameters': {'short': length2, 'long': length2}},
	]

	# BBCrossover

	for name in ['BollingerBandsLowerMidCrossoverRule',
				 'BollingerBandsUpperMidCrossoverRule',
				 'BollingerBandsLowerUpperCrossoverRule']:
		data[name]['parameters'] = {'patience': patience2}
		data[name]['indicators'] = [
		{'name': 'BollingerBandsIndicator', 'parameters': {'length': length1, 
														   'mult': get_logspace(1, 3, 10, float)}},
		{'name': 'MovingAverageIndicator', 'parameters': {'length': [1]}},
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
		state['name'] = expert.__class__.__name__
		state['parameters'] = expert.get_parameters()
		if isinstance(expert, experts.RuleExpert):
			rule, indicators = expert._rule, expert._indicators
			rule = {'name': rule.__class__.__name__, 'parameters': rule.get_parameters()}
			indicators = [{'name': indicator.__class__.__name__, 
						   'parameters': indicator.get_parameters()} 
													for indicator in indicators]
			state['parameters'] = {'rule': rule, 'indicators': indicators}
			state['estimations'] = {'profit': expert._estimated_profit,
									'ntrades': expert._estimated_ntrades}

		else:
			state['inner experts'] = [get_hierarchy(exp) for exp in expert._inner_experts]
		return state

	hierarchy = get_hierarchy(expert)
	json.dump(hierarchy, open(filename, 'w'), indent=4)

def deserialize_expert_from_json(filename: str = 'expert.json'):
	def deserialize_expert_from_dict(hierarchy):
		if 'inner experts' in hierarchy:
			expert = getattr(experts, hierarchy['name'])(**hierarchy['parameters'])
			inner = [deserialize_expert_from_dict(exp) for exp in hierarchy['inner experts']]
			expert.set_experts(inner)
		else: # expert is RuleExpert
			rule = hierarchy['parameters']['rule']
			inds = hierarchy['parameters']['indicators']
			rule = getattr(rules, rule['name'])(**rule['parameters'])
			inds = [getattr(indicators, ind['name'])(**ind['parameters']) for ind in inds]
			expert = experts.RuleExpert(rule, inds)
			expert._estimated_profit = hierarchy['estimations']['profit']
			expert._estimated_ntrades = hierarchy['estimations']['ntrades']

		return expert

	hierarchy = json.load(open(filename, 'r'))
	expert = deserialize_expert_from_dict(hierarchy)
	return expert


if __name__ == '__main__':
	create_searchspace_config()