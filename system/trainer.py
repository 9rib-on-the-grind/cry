import json
from itertools import product, accumulate
from copy import deepcopy
import time
import heapq
from pprint import pprint
import multiprocessing as mp

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
        'RelativeStrengthIndexTrasholdRule',
        'TripleExponentialDirectionChangeRule',
        'IchimokuKinkoHyoTenkanKijunCrossoverRule',
        'IchimokuKinkoHyoSenkouASenkouBCrossoverRule',
        'IchimokuKinkoHyoChikouCrossoverRule',
        # 'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule',
        'BollingerBandsLowerUpperCrossoverRule',
        'BollingerBandsLowerMidCrossoverRule',
        'BollingerBandsUpperMidCrossoverRule',
    ]

    timeframes = ['1d', '4h', '1h', '15m', '1m']

    def __init__(self):
        self.loaded_history = {}

    def construct_system(self):
        timeframes = self.timeframes

        reestimate = False
        reestimate = True

        if reestimate:
            config.create_searchspace_config()
            pair = 'BTC/USDT'
            base, quote = pair.split('/')
            pair_expert = experts.PairExpert(base, quote)
            timeframe_lst = []
            for timeframe in timeframes:
                candidates = []

                for rule in self.rule_names:
                    new = [expert for expert in config.get_experts_from_searchspace(timeframe, rule)]
                    print(f'searching   {rule:<60} {len(new):>5} candidates')
                    self.estimate_experts(new, pair, timeframe)
                    # candidates += self.best_rule_experts(new, nbest=10)
                    candidates += new

                timeframe_expert = experts.TimeFrameExpert(timeframe)
                timeframe_expert.set_experts(candidates)
                timeframe_lst.append(timeframe_expert)

            pair_expert.set_experts(timeframe_lst)
            config.serialize_expert_to_json(filename='estimated_expert.json', expert=pair_expert)

        else:
            pair_expert = config.deserialize_expert_from_json('estimated_expert.json')

        # self.trim_bad_experts(pair_expert, trashold=.2)
        pair_expert.show(overview=False)
        pair_expert.show()
        config.serialize_expert_to_json(expert=pair_expert)

    def estimate_experts(self, experts: list[experts.RuleExpert],
                               pair: str,
                               timeframe: str):
        ndays = {'1d': 300, '4h': 180, '1h': 90, '15m': 30, '1m': 3}
        for expert in experts:
            if expert._estimated_profit is None:
                pair_trader = self.construct_pair_trader_from_rule_expert(expert, pair, timeframe)
                profit, ntrades = self.simulate_pair_trader(pair_trader, ndays=ndays[timeframe])
                # profit, ntrades = self.simulate_pair_trader(pair_trader, ndays=360, display=True)
                expert._estimated_profit = profit / ndays[timeframe]
                expert._estimated_ntrades = ntrades

    def construct_pair_trader_from_rule_expert(self, rule_expert: experts.RuleExpert,
                                                     pair: str, 
                                                     timeframe: str) -> trader.PairTrader:
        pair_trader = trader.PairTrader(pair)
        pair_expert = experts.PairExpert(*pair.split('/'))
        timeframe_expert = experts.TimeFrameExpert(timeframe)
        timeframe_expert.set_experts([rule_expert])
        pair_expert.set_experts([timeframe_expert])
        pair_expert.set_weights()
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
                init, new = df.iloc[max(split-1000, 0): split].values.T, df.iloc[split:].values
                mapping = {key: val for key, val in zip(df, init)}
                init_data.add(mapping, location=[timeframe, 'Init'])
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

    def trim_bad_experts(self, expert: experts.BaseExpert, **kwargs):
        if isinstance(expert, experts.TimeFrameExpert):
            expert._inner_experts = self.best_rule_experts(expert._inner_experts, **kwargs)
        else:
            for expert in expert._inner_experts:
                self.trim_bad_experts(expert, **kwargs)

    def best_rule_experts(self, candidates: list[experts.RuleExpert], *,
                                trashold: float = None,
                                nbest: int = None,
                                percent: 'float (0, 1)' = None) -> list[experts.RuleExpert]:
        nbest = nbest if percent is None else int(percent * len(candidates))
        candidates = [expert for expert in candidates if expert._estimated_ntrades > 5]
        candidates.sort(reverse=True, key=lambda x: x._estimated_profit)
        if trashold is not None:
            return [expert for expert in candidates if expert._estimated_profit > trashold]
        elif nbest is not None:
            return candidates[:nbest]

    def fit_weights(self, pair_trader: trader.PairTrader, epochs=10, population=10, nchildren=3):
        def estimate_trader(pair_trader: trader.PairTrader, *, ret_dict = None) -> float:
            profit, ntrades = self.simulate_pair_trader(pair_trader, ndays=90)
            ret_dict[hash(pair_trader)] = profit if ntrades >= min_trades else -999

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

        def parallel_estimation(traders: list[trader.PairTrader]):
            results = mp.Manager().dict()
            jobs = [mp.Process(target=estimate_trader, args=(trader,), kwargs={'ret_dict': results}) for trader in traders]
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            return [results[hash(trader)] for trader in traders]

        lr, decay = 1, .2
        min_trades = 10
        parents = [pair_trader.expert.get_weights()]

        for epoch in range(epochs):
            print(f'[ {epoch+1:>3} / {epochs:<3} ]')
            children = []
            for weights in parents:
                children += [change_weights(weights) for _ in range(nchildren)]
            parents += children
            traders = [construct_pair_traider_from_weights(weights) for weights in parents]

            estimations = parallel_estimation(traders)
            for pair_trader, profit in zip(traders, estimations):
                pair_trader.profit = profit
            traders = heapq.nlargest(population, traders, key=lambda x: x.profit)
            parents = [pair_trader.expert.get_weights() for pair_trader in traders]

            best = traders[0]
            print(f'profit: {best.profit:.2f} %')

        found_trader = trader.PairTrader('BTC/USDT')
        expert = config.deserialize_expert_from_json()
        expert.set_weights(best.expert.get_weights())
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

        ax2.plot(pair_trader.times, pair_trader._profits, linestyle=':')

        estimations = pair_trader.estimations
        ax3.plot(time[:len(estimations)], estimations)

        plt.show()



if __name__ == '__main__':
    trainer = Trainer()

    trainer.construct_system()

    pair_trader = trader.PairTrader('BTC/USDT')
    expert = config.deserialize_expert_from_json()
    expert.show()
    expert.set_weights()
    pair_trader.set_expert(expert)

    trainer.fit_weights(pair_trader)