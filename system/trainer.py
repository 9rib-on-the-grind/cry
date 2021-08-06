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
import config



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
                                 for expert in config.get_experts_from_searchspace(rule_cls)]
            estimations = self.estimate_experts(pair, timeframe, candidates)
            chosen = self.best_rule_experts(estimations, percent=.2)
            
            timeframe_expert = experts.TimeFrameExpert(timeframe)
            timeframe_expert.set_experts(chosen)
            timeframe_lst.append(timeframe_expert)

        pair_expert.set_experts(timeframe_lst)
        pair_expert.show(detailed=True)
        config.serialize_expert_to_json(expert=pair_expert)

    def estimate_experts(self, pair: str,
                               timeframe: str,
                               experts: list[experts.RuleExpert]) -> list[tuple['profit', experts.RuleExpert]]:
        estimations = []
        for expert in experts:
            profit, ntrades = self.simulate_rule_expert(pair=pair, timeframe=timeframe, rule_expert=expert)
            expert._estimated_profit = profit
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