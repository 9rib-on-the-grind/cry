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
            rule_lst = []
            for rule_cls in self.rule_classes:
                candidates = self.search(pair=pair, timeframe=timeframe, rule_cls=rule_cls)
                rule_lst.extend(self.best_rule_experts(candidates, nbest=10))
            
            timeframe_expert = experts.TimeFrameExpert(timeframe)
            timeframe_expert.set_experts(rule_lst)
            timeframe_lst.append(timeframe_expert)

        pair_expert.set_experts(timeframe_lst)
        pair_expert.show()



    def best_rule_experts(self, candidates: list[tuple['profit', 'expert']], 
                                trashold: float = None,
                                nbest: int = None) -> list[experts.RuleExpert]:
        if trashold is not None:
            return [expert for profit, expert in candidates if profit > trashold]
        elif nbest is not None:
            return [expert for profit, expert in heapq.nlargest(nbest, candidates, key=lambda x: x[0])]

    def search(self, cfg: str = 'searchspace.json',
                     pair: str = 'BTC/USDT',
                     timeframe: str = '1h',
                     rule_cls: rules.BaseRule = None) -> list[tuple['profit', 'expert']]:

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

        res = []
        for idx, (rule, inds) in enumerate(search):
            expert = experts.RuleExpert(inds, rule)
            trades, profit = self.simulate_rule_expert(pair=pair, timeframe=timeframe, rule_expert=expert)
            # print(f'iter [ {idx:>3} / {len(search):>3} ] : {expert.name}')
            # print(f'{trades} trades, profit : {profit:.2f} %')
            res.append((profit, expert))
        return res



    def simulate_rule_expert(
            self, pair: str = 'BTC/USDT', 
                  timeframe: str = '1h',
                  n: int = 1000,
                  rule_expert: experts.RuleExpert = None):

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
        return len(pair_trader.trades), pair_trader.profit[-1] if pair_trader.profit else 0


    def load_history(self, pair: str = 'BTC/USDT', timeframe: str = '1h'):
        pair = pair.replace('/', '')
        filename = f'data/{pair}/{timeframe}.csv'
        return pd.read_csv(filename)





if __name__ == '__main__':
    trainer = Trainer()
    # trainer.search(rule_cls=rules.MovingAverageCrossoverRule)
    trainer.construct_system()