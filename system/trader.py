from collections.abc import Mapping, Iterable

import numpy as np

import experts
import data



class BaseTrader:
    def __init__(self):
        pass

    def set_data(self, data: data.DataMaintainer):
        assert hasattr(self, 'expert'), 'PairExpert is not set'
        self.data = data
        self.expert.set_data(data)
        keys = list(self.data[self.min_timeframe, 'Init'].keys())
        self.data.drop('Init', recursively=True)
        for timeframe in self.timeframes:
            self.data.add({key: None for key in keys}, location=[timeframe])




class PairTrader(BaseTrader):
    def __init__(self, pair: str):
        self.pair = pair
        self.balance = self.initial_money = 100
        self.quantity = 0
        self.trashold = .2
        self.estimations = []
        self.trades = []
        self._profits = []
        self.profit = None
        self.time, self.times = 0, []
        self.commision = .00075

    def set_expert(self, expert: experts.PairExpert):
        self.expert = expert
        self.timeframes = [expert.timeframe for expert in self.expert._inner_experts]
        self.min_timeframe = self.timeframes[0] if len(self.timeframes) == 1 else 'error'

    def update(self, data: Mapping[str, Iterable]):
        """Update candlesticks data, update expert.

        Args:
            data: Dictionary that maps timeframe name to new candlestick
            (Example: '1h' -> [...])
        """

        for timeframe, data in data.items():
            self.data[timeframe].update(data)
        self.expert.update()

    def act(self):
        def stoploss(trade: tuple, price):
            trashold = .08
            action, _, trade_price = trade
            if action == 'buy' and price < trade_price * (1 - trashold):
                return True
            else:
                return False

        timeframe = self.min_timeframe
        price = self.data[timeframe, 'Close']
        time = self.data[timeframe, 'Close time']
        estimation = self.expert.estimate()
        self.estimations.append(estimation)
        if estimation > self.trashold and self.balance > 0: # buy
            self.quantity = (1 - self.commision) * self.balance / price
            self.times.append(time)
            self.trades.append(('buy', self.quantity, price))
            self.balance = 0
            self._profits.append(self._profits[-1] if self._profits else 0)
        elif self.quantity > 0 and (estimation < -self.trashold or stoploss(self.trades[-1], price)): # sell
            self.balance = (1 - self.commision) * self.quantity * price
            self.times.append(time)
            self.trades.append(('sell', self.quantity, price))
            self._profits.append(self.evaluate_profit())
            self.quantity = 0

    def show_evaluation(self):
        if self.quantity > 0:
            self.balance = self.trades[-1][2] * self.quantity
            self.quantity = 0
            self.trades.pop()
            self.times.pop()
            self._profits.pop()
        print(f'{len(self.trades)} trades made')
        print(f'profit : {self.evaluate_profit():.2f} %')

    def evaluate_profit(self):
        if self.quantity:
            balance = self.trades[-1][2] * self.quantity
        else:
            balance = self.balance
        return 100 * (balance - self.initial_money) / self.initial_money

    def fitness(self):
        profits = self._profits[1::2]
        if not profits:
            return float('-inf')
        else:
            mean = np.mean(profits)
            # return mean * len(diff)
            return (1 + mean / 100) ** (len(profits))