import json
import collections
from itertools import product
from pprint import pprint

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

    cfg_file = open('searchspace.json', 'w')

    nested_dict = lambda: collections.defaultdict(nested_dict)
    data = collections.defaultdict(nested_dict)

    ranges = {
        '1m': get_logspace(10, 360, 25),
        '15m': get_logspace(8, 192, 25),
        '1h': get_logspace(6, 168, 25),
        '4h': get_logspace(6, 180, 25),
        '1d': get_logspace(7, 365, 25),
    }
    patience = get_logspace(1, 50, 11)

    indicator_names = [
        'PriceIndicator',
        'MovingAverageIndicator',
        'ExponentialMovingAverageIndicator',
        'RelativeStrengthIndexIndicator',
        'TripleExponentialIndicator',
        'IchimokuKinkoHyoIndicator',
        'BollingerBandsIndicator',
        'MovingAverageConvergenceDivergenceIndicator',
    ]

    rule_names = [
        'MovingAverageCrossoverRule',
        'ExponentialMovingAverageCrossoverRule',
        'RelativeStrengthIndexTrasholdRule',
        'TripleExponentialDirectionChangeRule',
        'IchimokuKinkoHyoTenkanKijunCrossoverRule',
        'IchimokuKinkoHyoSenkouASenkouBCrossoverRule',
        'IchimokuKinkoHyoChikouCrossoverRule',
        'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule',
        'BollingerBandsLowerMidCrossoverRule',
        'BollingerBandsUpperMidCrossoverRule',
        'BollingerBandsLowerUpperCrossoverRule',
        'MovingAverageConvergenceDivergenceSignalLineCrossoverRule',
    ]

    rule_indicators  = {
        'MovingAverageCrossoverRule': ['MovingAverageIndicator'] * 2,
        'ExponentialMovingAverageCrossoverRule': ['ExponentialMovingAverageIndicator'] * 2,
        'RelativeStrengthIndexTrasholdRule': ['RelativeStrengthIndexIndicator'],
        'TripleExponentialDirectionChangeRule': ['TripleExponentialIndicator'],
        'IchimokuKinkoHyoTenkanKijunCrossoverRule': ['IchimokuKinkoHyoIndicator'],
        'IchimokuKinkoHyoSenkouASenkouBCrossoverRule': ['IchimokuKinkoHyoIndicator'],
        'IchimokuKinkoHyoChikouCrossoverRule': ['IchimokuKinkoHyoIndicator', 'PriceIndicator'],
        'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule': ['IchimokuKinkoHyoIndicator', 'PriceIndicator'],
        'BollingerBandsLowerMidCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
        'BollingerBandsUpperMidCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
        'BollingerBandsLowerUpperCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
        'MovingAverageConvergenceDivergenceSignalLineCrossoverRule': ['MovingAverageConvergenceDivergenceIndicator'],
    }

    for timeframe in ranges:

        space = ranges[timeframe]
        timeframe = data[timeframe]

        rule_parameters = {rule: {'patience': patience} for rule in rule_names}
        rule_parameters['RelativeStrengthIndexTrasholdRule'] |=  {
            'lower': list(range(20, 45, 10)),
            'upper': list(range(60, 85, 10)),
        }

        indicator_parameters = {indicator: {'length': space} for indicator in indicator_names}
        indicator_parameters['PriceIndicator'] = {}
        indicator_parameters['IchimokuKinkoHyoIndicator'] = {'short': space, 'long': space}
        indicator_parameters['BollingerBandsIndicator'] = {'length': space, 'mult': get_logspace(1.5, 3, 11, float)}
        indicator_parameters['MovingAverageConvergenceDivergenceIndicator'] = {'long': space, 'signal': space}

        for rule in rule_names:
            inds = []
            for indicator in rule_indicators[rule]:
                inds.append({'name': indicator, 'parameters': indicator_parameters[indicator]})
            timeframe[rule] = {'parameters': rule_parameters[rule], 'indicators': inds}

    json.dump(data, cfg_file, indent=4)

def get_experts_from_searchspace(timeframe: str,
                                 rule_name: str,
                                 cfg: str = 'searchspace.json') -> list[experts.RuleExpert]:
        cfg = json.load(open(cfg, 'r'))

        rule_parameters = cfg[timeframe][rule_name]['parameters']
        indicators_lst = cfg[timeframe][rule_name]['indicators']
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
                    try:
                        ind = getattr(indicators, cls_name)(**indicator_kwargs)
                        lst.append(ind)
                    except ValueError as err:
                        break

                else:
                    rule = getattr(rules, rule_name)(**rule_kwargs)
                    try:
                        res.append(experts.RuleExpert(rule, lst))
                    except ValueError as err:
                        pass

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
        name = expert.__class__.__name__
        uniq_name = f'{name}_{hex(id(expert))}'
        state['name'] = name
        state['unique name'] = uniq_name
        state['parameters'] = expert.get_parameters()
        if hasattr(expert, '_inner_experts'):
            state['inner experts'] = [get_hierarchy(exp) for exp in expert._inner_experts]
        else:
            rule, indicators = expert._rule, expert._indicators
            rule = {'name': rule.__class__.__name__, 'parameters': rule.get_parameters()}
            indicators = [{'name': indicator.__class__.__name__,
                           'parameters': indicator.get_parameters()}
                                                    for indicator in indicators]
            state['parameters'] = {'rule': rule, 'indicators': indicators}
            state['estimations'] = {'profit': expert._estimated_profit,
                                    'ntrades': expert._estimated_ntrades}
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