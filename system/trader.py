import time
from collections.abc import Mapping, Iterable

import experts



class BaseTrader:
    def __init__(self):
        pass

    def set_data(self, data):
        self.data = data



class PairTrader(BaseTrader):
    def __init__(self):
        self.balance = self.initial_money = 100
        self.quantity = 0
        self.trashold = .5
        self.trades = []
        self.profit = []
        self.time, self.times = 0, []

    def set_expert(self, expert: experts.PairExpert):
        self.expert = expert

    def update(self, data: Mapping[str, Iterable]):
        """Update candlesticks data, update expert.

        Args:
            data: Dictionary that maps timeframe name to new candlestick
            (Example: '1h' -> [...])
        """
        for timeframe, data in data.items():
            self.data[timeframe, 'History'].append(data)
            self.data[timeframe, 'History'].set_update_hash(time.time())
        self.expert.update()


    def act(self):
        estimation = self.expert.estimate()
        price = self.data['1h', 'History', 'Close'][-1]
        if estimation > self.trashold and self.balance > 0: # buy
            self.quantity = self.balance / price
            self.times.append(self.time)
            self.trades.append(('buy', self.quantity, price))
            self.balance = 0
        elif estimation < -self.trashold and self.quantity > 0: # sell
            self.balance = self.quantity * price
            self.times.append(self.time)
            self.trades.append(('sell', self.quantity, price))
            self.profit.append(self.evaluate_profit())
            self.quantity = 0
        self.time += 1


    def show_evaluation(self):
        if self.quantity > 0:
            self.balance = self.trades[-1][2] * self.quantity
            self.trades.pop()
            self.times.pop()
        print(f'{len(self.trades)} trades made')
        print(f'profit : {self.evaluate_profit():.2f} %')

    def evaluate_profit(self):
        return 100 * (self.balance - self.initial_money) / self.initial_money