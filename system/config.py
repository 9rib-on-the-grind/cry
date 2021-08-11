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

    cfg_file = open('searchspace.json', 'w')

    nested_dict = lambda: collections.defaultdict(nested_dict)
    data = collections.defaultdict(nested_dict)

    ranges = {
        '1m': [10, 20, 30, 60, 120, 180],
        '15m': [8, 12, 24, 48, 96],
        '1h': [6, 12, 24, 48, 72, 168],
        '4h': [6, 12, 18, 30, 42],
        '1d': [7, 14, 30, 90, 182, 365]
    }

    indicator_names = [
        'PriceIndicator',
        'MovingAverageIndicator',
        'RelativeStrengthIndexIndicator',
        'TripleExponentialIndicator',
        'IchimokuKinkoHyoIndicator',
        'BollingerBandsIndicator',
    ]

    rule_names = [
        'MovingAverageCrossoverRule',
        'RelativeStrengthIndexTrasholdRule',
        'TripleExponentialDirectionChangeRule',
        'IchimokuKinkoHyoTenkanKijunCrossoverRule',
        'IchimokuKinkoHyoSenkouASenkouBCrossoverRule',
        'IchimokuKinkoHyoChikouCrossoverRule',
        'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule',
        'BollingerBandsLowerMidCrossoverRule',
        'BollingerBandsUpperMidCrossoverRule',
        'BollingerBandsLowerUpperCrossoverRule',
    ]

    rule_indicators  = {
        'MovingAverageCrossoverRule': ['MovingAverageIndicator'] * 2,
        'RelativeStrengthIndexTrasholdRule': ['RelativeStrengthIndexIndicator'],
        'TripleExponentialDirectionChangeRule': ['TripleExponentialIndicator'],
        'IchimokuKinkoHyoTenkanKijunCrossoverRule': ['IchimokuKinkoHyoIndicator'],
        'IchimokuKinkoHyoSenkouASenkouBCrossoverRule': ['IchimokuKinkoHyoIndicator'],
        'IchimokuKinkoHyoChikouCrossoverRule': ['IchimokuKinkoHyoIndicator', 'PriceIndicator'],
        'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule': ['IchimokuKinkoHyoIndicator', 'PriceIndicator'],
        'BollingerBandsLowerMidCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
        'BollingerBandsUpperMidCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
        'BollingerBandsLowerUpperCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
    }

    for timeframe in ranges:

        space = ranges[timeframe]
        timeframe = data[timeframe]

        rule_parameters = {rule: {'patience': space} for rule in rule_names}
        rule_parameters['RelativeStrengthIndexTrasholdRule'] |=  {
            'lower': list(range(20, 45, 10)),
            'upper': list(range(60, 85, 10)),
        }

        indicator_parameters = {indicator: {'length': space} for indicator in indicator_names}
        indicator_parameters['PriceIndicator'] = {}
        indicator_parameters['IchimokuKinkoHyoIndicator'] = {'short': space, 'long': space}
        indicator_parameters['BollingerBandsIndicator'] = {'length': space, 'mult': np.linspace(1.5, 3, 11).tolist()}

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