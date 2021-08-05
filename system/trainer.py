import json
from itertools import product
from copy import deepcopy

import rules
import indicators
import simulation
import trader
import experts



class Trainer:
    def __init__(self):
        pass

    def search(self, cfg: str = 'searchspace.json',
                     pair: str = 'BTC/USDT',
                     timeframe: str = '1h',
                     rule_cls: rules.BaseRule = None):

        cfg = json.load(open(cfg, 'r'))
        rule_name = rule_cls.__name__


        rule_searchspace = cfg[rule_name]['rule']
        inds_searchspace = cfg[rule_name]['indicators']
        indicators_cls_names = list(inds_searchspace.keys())
        indicators_keys = [list(ind.keys()) for ind in inds_searchspace.values()]

        search = []

        for rule_params in product(*rule_searchspace.values()):
            keys = list(rule_searchspace)
            rule_kwargs = {key: val for key, val in zip(keys, rule_params)}
            
            
            for inds_params in product(*[product(*ind.values()) for ind in inds_searchspace.values()]):
                indicators_lst = []
                for cls_name, (keys, params) in zip(indicators_cls_names, zip(indicators_keys, inds_params)):
                    cls_name = list(cls_name.split())[0]
                    ind_kwargs = {key: val for key, val in zip(keys, params)}
                    indicator = getattr(indicators, cls_name)
                    indicators_lst.append(indicator(**ind_kwargs))

                rule = rule_cls(**rule_kwargs)
                search.append((rule, indicators_lst))

        sim = simulation.Simulation()
        for idx, (rule, inds) in enumerate(search):
            expert = experts.RuleExpert(inds, rule)
            print(f'\niter [ {idx:>3} / {len(search):>3} ] : {expert.name}')
            sim.simulate(rule_exp=expert)






if __name__ == '__main__':
    trainer = Trainer()
    trainer.search(rule_cls=rules.MovingAverageCrossoverRule)