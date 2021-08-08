import time
from collections.abc import Mapping, Iterable

import experts
import data



class BaseTrader:
    def __init__(self):
        pass

    def set_data(self, data: data.DataMaintainer):
        assert hasattr(self, 'expert'), 'PairExpert is not set'
        self.data = data
        self.expert.set_data(data)



class PairTrader(BaseTrader):
    def __init__(self, pair: str):
        self.pair = pair
        self.balance = self.initial_money = 100
        self.quantity = 0
        self.trashold = .2
        self.estimations = []
        self.trades = []
        self.profit = []
        self.time, self.times = 0, []
        self.commision = .00075

    def set_expert(self, expert: experts.PairExpert):
        self.expert = expert
        self.timeframes = [expert.timeframe for expert in self.expert._inner_experts]
        self.min_timeframe = self.timeframes[-1]

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
        timeframe =  self.min_timeframe
        price = self.data[timeframe, 'History', 'Close'][-1]
        time = self.data[timeframe, 'History', 'Close time'][-1]
        estimation = self.expert.estimate()
        self.estimations.append(estimation)
        if estimation > self.trashold and self.balance > 0: # buy
            self.quantity = (1 - self.commision) * self.balance / price
            self.times.append(time)
            self.trades.append(('buy', self.quantity, price))
            self.balance = 0
            self.profit.append(self.profit[-1] if self.profit else 0)
        elif estimation < -self.trashold and self.quantity > 0: # sell
            self.balance = (1 - self.commision) * self.quantity * price
            self.times.append(time)
            self.trades.append(('sell', self.quantity, price))
            self.profit.append(self.evaluate_profit())
            self.quantity = 0

    def show_evaluation(self):
        if self.quantity > 0:
            self.balance = self.trades[-1][2] * self.quantity
            self.quantity = 0
            self.trades.pop()
            self.times.pop()
            self.profit.pop()
        print(f'{len(self.trades)} trades made')
        print(f'profit : {self.evaluate_profit():.2f} %')

    def evaluate_profit(self):
        return 100 * (self.balance - self.initial_money) / self.initial_money