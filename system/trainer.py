import json
from itertools import product, accumulate
from copy import deepcopy
import time
import heapq
from pprint import pprint

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
    rule_classes = [
        rules.MovingAverageCrossoverRule,
        rules.RelativeStrengthIndexTrasholdRule,
    ]

    def __init__(self):
        self.loaded_history = {}

    def construct_system(self):
        pair = 'BTC/USDT'
        timeframes = ['1d', '4h']
        # timeframes = ['1d']
        # timeframes = ['4h']
        base, quote = pair.split('/')

        pair_expert = experts.PairExpert(base, quote)
        timeframe_lst = []

        for timeframe in timeframes:
            print('searching', pair, timeframe)
            candidates = [expert for rule_cls in self.rule_classes 
                                 for expert in config.get_experts_from_searchspace(rule_cls)]
            self.estimate_experts(candidates, pair, timeframe)
            best = self.best_rule_experts(candidates, nbest=5)
            
            timeframe_expert = experts.TimeFrameExpert(timeframe)
            timeframe_expert.set_experts(best)
            timeframe_lst.append(timeframe_expert)

        pair_expert.set_experts(timeframe_lst)
        # pair_expert.init_weights()
        pair_expert.show()
        config.serialize_expert_to_json(expert=pair_expert)

    def estimate_experts(self, experts: list[experts.RuleExpert],
                               pair: str,
                               timeframe: str):
        ndays = {'1d': 300, '4h': 180, '1h': 90, '15m': 30, '1m': 2}
        for expert in experts:
            pair_trader = self.construct_pair_trader_from_rule_expert(expert, pair, timeframe)
            profit, ntrades = self.simulate_pair_trader(pair_trader, ndays=ndays[timeframe])
            expert._estimated_profit = profit
            expert._estimated_ntrades = ntrades

    def construct_pair_trader_from_rule_expert(self, rule_expert: experts.RuleExpert,
                                                     pair: str, 
                                                     timeframe: str) -> trader.PairTrader:
        pair_trader = trader.PairTrader(pair)
        pair_expert = experts.PairExpert(*pair.split('/'))
        timeframe_expert = experts.TimeFrameExpert(timeframe)
        timeframe_expert.set_experts([rule_expert])
        pair_expert.set_experts([timeframe_expert])
        pair_expert.init_weights()
        pair_trader.set_expert(pair_expert)
        return pair_trader

    def simulate_pair_trader(self, pair_trader: trader.PairTrader, ndays: int, *, display: bool = False):
        def load_history(pair: str, timeframe: str) -> pd.DataFrame:
            if (pair, timeframe) not in self.loaded_history:
                filename = f"data/test_data/{pair.replace('/', '')}/{timeframe}.csv"
                self.loaded_history[(pair, timeframe)] = pd.read_csv(filename)
            return self.loaded_history[(pair, timeframe)]

        def construct_data(pair_trader: trader.PairTrader, ndays: int):
            init_data = data.DataMaintainer()
            new_data = {}
            start_time = load_history(pair_trader.pair, '1d')['Close time'].iloc[-ndays]
            for timeframe in pair_trader.timeframes:
                df = load_history(pair_trader.pair, timeframe)
                split = df['Close time'].searchsorted(start_time)
                init, new = df.iloc[max(split-500, 0): split], df.iloc[split:].values
                init_data.add(data=init.values.T, keys=list(init), location=[timeframe, 'History'])
                new_data[timeframe] = new
            return init_data, new_data

        init_data, new_data = construct_data(pair_trader, ndays + 1)
        pair_trader.set_data(init_data)

        new_data_iter = {timeframe: iter(data) for timeframe, data in new_data.items()}
        minutes = {timeframe: n for timeframe, n in zip(['1m', '15m', '1h', '4h', '1d'], [1, 15, 60, 240, 1440])}
        simulation_length = minutes['1d'] * ndays

        for i in range(0, simulation_length, minutes[pair_trader.min_timeframe]):
            update = {}
            for timeframe in pair_trader.timeframes:
                if not i % minutes[timeframe]:
                    update[timeframe] = next(new_data_iter[timeframe])
            pair_trader.update(update)
            pair_trader.act()

        if display:
            self.show_trades(pair_trader, new_data)
        return pair_trader.evaluate_profit(), len(pair_trader.trades)

    def best_rule_experts(self, candidates: list[experts.RuleExpert], *,
                                trashold: float = None,
                                nbest: int = None,
                                percent: 'float (0, 1)' = None) -> list[experts.RuleExpert]:
        nbest = nbest if percent is None else int(percent * len(candidates))
        candidates = [expert for expert in candidates if expert._estimated_ntrades > 5 and expert._estimated_profit > 0]
        candidates.sort(reverse=True, key=lambda x: x._estimated_profit)
        if trashold is not None:
            return [expert for expert in candidates if expert._estimated_profit > trashold]
        elif nbest is not None:
            return candidates[:nbest]

    def fit_weights(self, epochs=20, population=7, nchildren=3):
        def fitness(pair_trader: trader.PairTrader) -> float:
            profit, ntrades = self.simulate_pair_trader(pair_trader, ndays=90)
            return profit if ntrades >= min_trades else -999

        def construct_pair_traider_from_weights(weights: list['weights', list['inner weights']]) -> trader.PairTrader:
            pair_trader = trader.PairTrader('BTC/USDT')
            expert = config.deserialize_expert_from_json()
            expert.set_weights(weights)
            pair_trader.set_expert(expert)
            return pair_trader

        def change_weights(weights: list['weights', list['inner weights']]):
            if not weights:
                return weights
            weights, inner = weights
            sigma = lr * np.exp(-decay * epoch)
            return [weights + np.random.normal(size=weights.shape, scale=sigma), 
                    [change_weights(weights) for weights in inner]]

        lr, decay = 1, .01
        min_trades = 10
        pair_trader.expert.show()
        parents = [pair_trader.expert.get_weights()]

        for epoch in range(epochs):
            print(f'[ {epoch+1:>3} / {epochs:<3} ]')
            children = []
            for weights in parents:
                children += [change_weights(weights) for _ in range(nchildren)]
            parents += children
            traders = [construct_pair_traider_from_weights(weights) for weights in parents]
            traders.sort(reverse=True, key=fitness)
            parents = [trader.expert.get_weights() for trader in traders][:population]

            best_trader = traders[0]
            best_trader.show_evaluation()

        pprint(best_trader.expert.get_weights())

        found_trader = trader.PairTrader('BTC/USDT')
        expert = config.deserialize_expert_from_json()
        expert.set_weights(best_trader.expert.get_weights())
        found_trader.set_expert(expert)
        self.simulate_pair_trader(found_trader, ndays=300, display=True)

    def show_trades(self, pair_trader: trader.PairTrader, new_data: dict):
        def config_axs(*axs):
            for ax in axs:
                ax.grid(True)
                ax.set_xlim(time[0], time[-1])
                ax.margins(x=.1)

        pair_trader.show_evaluation()

        timeframe = pair_trader.min_timeframe
        close = new_data[timeframe][:, 4] # Close price
        time = new_data[timeframe][:, 6] # Close time
        buy_time, buy_price = pair_trader.times[::2], [trade[2] for trade in pair_trader.trades[::2]]
        sell_time, sell_price = pair_trader.times[1::2], [trade[2] for trade in pair_trader.trades[1::2]]

        fig, axs = plt.subplots(nrows=3, figsize=(19.2, 10.8), dpi=100)
        fig.tight_layout()
        ax1, ax2, ax3 = axs.reshape(-1)
        config_axs(ax1, ax2, ax3)

        ax1.plot(time, close, color='black', linewidth=1)
        ax1.scatter(buy_time, buy_price, color='blue')
        ax1.scatter(sell_time, sell_price, color='red')

        ax2.plot(pair_trader.times, pair_trader.profit, linestyle=':')

        estimations = pair_trader.estimations
        ax3.plot(time[:len(estimations)], estimations)

        plt.show()



if __name__ == '__main__':
    trainer = Trainer()

    trainer.construct_system()

    pair_trader = trader.PairTrader('BTC/USDT')
    expert = config.deserialize_expert_from_json()
    expert.init_weights()
    pair_trader.set_expert(expert)

    trainer.fit_weights()