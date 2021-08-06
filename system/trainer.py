import json
from itertools import product
from copy import deepcopy
import time
import heapq

import pandas as pd

import rules
import indicators
import trader
import experts
import data
import config_creation



class Trainer:
    rule_classes = [
        rules.MovingAverageCrossoverRule,
        rules.RelativeStrengthIndexTrasholdRule,
    ]

    def __init__(self):
        pass

    def construct_system(self):
        pair = 'BTC/USDT'
        timeframes = ['1d', '4h', '15m', '1m']
        timeframes = ['1d', '4h', '1h']
        # timeframes = ['1d', '4h']
        # timeframes = ['1d']
        base, quote = pair.split('/')

        pair_expert = experts.PairExpert(base, quote)
        timeframe_lst = []

        for timeframe in timeframes:
            print(timeframe)
            candidates = [expert for rule_cls in self.rule_classes 
                                 for expert in self.get_candidates(pair, timeframe, rule_cls)]
            estimations = self.estimate_experts(pair, timeframe, candidates)
            chosen = self.best_rule_experts(estimations, percent=.2)
            
            timeframe_expert = experts.TimeFrameExpert(timeframe)
            timeframe_expert.set_experts(chosen)
            timeframe_lst.append(timeframe_expert)

        pair_expert.set_experts(timeframe_lst)
        pair_expert.show()
        config_creation.serialize_expert_to_json(expert=pair_expert)

    def get_candidates(self, pair: str,
                             timeframe: str,
                             rule_cls: rules.BaseRule,
                             cfg: str = 'searchspace.json') -> list[experts.RuleExpert]:

        cfg = json.load(open(cfg, 'r'))
        rule_name = rule_cls.__name__

        rule_parameters = cfg[rule_name]['parameters']
        indicators_lst = cfg[rule_name]['indicators']
        indicator_cls_names = list(ind['name'] for ind in indicators_lst)
        indicator_parameters = [ind['parameters'] for ind in indicators_lst]

        candidates = []

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
                candidates.append(experts.RuleExpert(rule, lst))

        return candidates

    def estimate_experts(self, pair: str,
                               timeframe: str,
                               experts: list[experts.RuleExpert]) -> list[tuple['profit', experts.RuleExpert]]:
        estimations = []
        for expert in experts:
            profit, ntrades = self.simulate_rule_expert(pair=pair, timeframe=timeframe, rule_expert=expert)
            estimations.append((profit, expert))
        return estimations

    def best_rule_experts(self, candidates: list[tuple['profit', 'expert']], *,
                                trashold: float = None,
                                nbest: int = None,
                                percent: 'float (0, 1)' = None) -> list[experts.RuleExpert]:
        nbest = nbest if percent is None else int(percent * len(candidates))
        if trashold is not None:
            return [expert for profit, expert in candidates if profit > trashold]
        elif nbest is not None:
            return [expert for profit, expert in heapq.nlargest(nbest, candidates, key=lambda x: x[0])]

    def simulate_rule_expert(self, pair: str, 
                                   timeframe: str,
                                   rule_expert: experts.RuleExpert,
                                   n: int = 1000) -> tuple['profit', 'number of trades']:

        history = self.load_history(pair, timeframe)
        init, new = history.iloc[-n-1000:-n], history.iloc[-n:-1].values
        init_data = data.DataMaintainer()
        init_data.add(data=init.values.T, keys=list(init), location=[pair, timeframe, 'History'])

        pair_trader = trader.PairTrader()
        pair_trader.set_data(init_data[pair])

        for indicator in rule_expert._indicators:
            indicator.set_data(init_data[pair, timeframe])

        pair_trader.set_expert(rule_expert)

        for update in new:
            pair_trader.update({timeframe: update})
            pair_trader.act(timeframe)
        return pair_trader.profit[-1] if pair_trader.profit else 0, len(pair_trader.trades)

    def load_history(self, pair: str = 'BTC/USDT', timeframe: str = '1h') -> pd.DataFrame:
        pair = pair.replace('/', '')
        filename = f'data/{pair}/{timeframe}.csv'
        return pd.read_csv(filename)



if __name__ == '__main__':
    trainer = Trainer()
    trainer.construct_system()