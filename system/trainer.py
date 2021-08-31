import json
from itertools import product, accumulate
from copy import deepcopy
import time
import heapq
from pprint import pprint
import multiprocessing as mp
import os
import sys
import pickle
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import rules
import indicators
import trader
import experts
import data
import config



class Trainer:
    rule_names = [
        'MovingAverageCrossoverRule',
        # 'ExponentialMovingAverageCrossoverRule',
        'RelativeStrengthIndexTrasholdRule',
        'TripleExponentialDirectionChangeRule',
        'IchimokuKinkoHyoTenkanKijunCrossoverRule',
        # 'IchimokuKinkoHyoSenkouASenkouBCrossoverRule',
        'IchimokuKinkoHyoChikouCrossoverRule',
        # 'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule',
        'BollingerBandsLowerUpperCrossoverRule',
        # 'BollingerBandsLowerMidCrossoverRule',
        # 'BollingerBandsUpperMidCrossoverRule',
        'MovingAverageConvergenceDivergenceSignalLineCrossoverRule',
    ]

    timeframes = ['1d', '4h', '1h', '15m'] # 41   75

    def __init__(self):
        self.loaded_history = {}

    def construct_system(self):
        timeframes = self.timeframes

        timeframes = ['4h', '1h', '15m']
        rules = [                                                            #  4h    1h     15m
            'MovingAverageCrossoverRule',                                    # 36%    21%    45%
            'ExponentialMovingAverageCrossoverRule',                         # 14%    18%    21%
            'RelativeStrengthIndexTrasholdRule',                             # 17%    14%   -30%
            'TripleExponentialDirectionChangeRule',                          # 23%    62%    33%
            'IchimokuKinkoHyoTenkanKijunCrossoverRule',                      # 48%    30%    58%
            'IchimokuKinkoHyoSenkouASenkouBCrossoverRule',                   # 28%    25%    56%
            'IchimokuKinkoHyoChikouCrossoverRule',                           # 28%    -6%     5%
            # 'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule',
            'BollingerBandsLowerUpperCrossoverRule',                         # 14%    -4%     0%
            'BollingerBandsLowerMidCrossoverRule',                           # 48%    35%    -1%
            'BollingerBandsUpperMidCrossoverRule',                           # 22%    13%     0%
            'MovingAverageConvergenceDivergenceSignalLineCrossoverRule',     # 25%    33%     8%
        ]

        config.create_searchspace_config()

        from_searchspace = False
        if from_searchspace:
            timeframe_lst = []
            for timeframe in timeframes:
                rule_cls_lst = []
                print(f'load timeframe [{timeframe}]')

                for rule in self.rule_names:
                    new = [expert for expert in config.get_experts_from_searchspace(timeframe, rule)]
                    print(' ' * 5 + f'{rule:<60} {len(new):>5} candidates')
                    rule_cls_expert = experts.RuleClassExpert(rule)
                    rule_cls_expert.set_experts(new)
                    rule_cls_lst.append(rule_cls_expert)

                timeframe_expert = experts.TimeFrameExpert(timeframe)
                timeframe_expert.set_experts(rule_cls_lst)
                timeframe_lst.append(timeframe_expert)

            pair_expert = experts.PairExpert(base, quote)
            pair_expert.set_experts(timeframe_lst)

        else:
            pair_expert = config.deserialize_expert_from_json()

        self.choose_branches(pair_expert, timeframes=timeframes, rules=rules, nleavs=999999)
        return pair_expert

    def choose_branches(self, expert: experts.BaseExpert, *,
                              timeframes: list[str] = None,
                              rules: list[str] = None,
                              nleavs: int = None):
        if isinstance(expert, experts.PairExpert) and timeframes is not None:
            expert._inner_experts = [exp for exp in expert._inner_experts if exp.timeframe in timeframes]
        elif isinstance(expert, experts.TimeFrameExpert) and rules is not None:
            expert._inner_experts = [exp for exp in expert._inner_experts if exp.rule in rules]
        elif isinstance(expert, experts.RuleClassExpert) and nleavs is not None:
            expert._inner_experts = expert._inner_experts[:nleavs]
        if hasattr(expert, '_inner_experts'):
            for exp in expert._inner_experts:
                self.choose_branches(exp, timeframes=timeframes, rules=rules, nleavs=nleavs)

    def simulate_pair_trader(self, pair_trader: trader.PairTrader, ndays: int, *, display: bool = False):
        def load_history(pair: str, timeframe: str) -> pd.DataFrame:
            if (pair, timeframe) not in self.loaded_history:
                filename = f"data/test_data/{pair.replace('/', '')}/{timeframe}.csv"
                self.loaded_history[(pair, timeframe)] = pd.read_csv(filename)
            return self.loaded_history[(pair, timeframe)]

        def construct_data(pair_trader: trader.PairTrader, ndays: int):
            init_data = data.DataMaintainer()
            new_data = {}
            start_time = load_history(pair_trader.pair, '1d')['Open time'].iloc[-ndays]
            for timeframe in pair_trader.timeframes:
                df = load_history(pair_trader.pair, timeframe)
                split = df['Open time'].searchsorted(start_time) - 1
                init, new = df.iloc[max(split-1000, 0): split].values.T, df.iloc[split:].values
                mapping = {key: val for key, val in zip(df, init)}
                init_data.add(mapping, location=[timeframe, 'Init'])
                new_data[timeframe] = new
            return init_data, new_data

        init_data, new_data = construct_data(pair_trader, ndays + 2)
        pair_trader.set_data(init_data)

        updates = defaultdict(dict) # close time -> update
        for timeframe, rows in new_data.items():
            for row in rows:
                close_time = row[6]
                updates[close_time][timeframe] = row

        for idx, (close_time, update) in enumerate(list(sorted(updates.items()))):
            pair_trader.update(update)
            pair_trader.act()
            if not idx % 100:
                print(f'done {100*idx/len(updates):>5.1f} %')

        if display:
            self.show_trades(pair_trader, new_data)
        return pair_trader.evaluate_profit(), len(pair_trader.trades)



def get_data():
    trainer = Trainer()

    expert = trainer.construct_system()

    reestimate = True
    reestimate = False
    if reestimate:
        pair_trader = trader.PairTrader('BTCUSDT')
        pair_trader.set_expert(expert)
        trainer.simulate_pair_trader(pair_trader, 360)

        pickle.dump(pair_trader.history, open('history', 'wb'))
        pickle.dump(expert.get_signals(), open('signals', 'wb'))

    history = pickle.load(open('history', 'rb'))
    signals = pickle.load(open('signals', 'rb'))

    return history, signals



if __name__ == '__main__':
    get_data()