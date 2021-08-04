import matplotlib.pyplot as plt
import pandas as pd

import data
import trader
import indicators
import rules
import experts



class Simulation:
    def simulate(self, history, n):
        self.prepare_data(history, n)
        pair_trader = self.init_system()
        self.run(pair_trader)

        plot(self.new_history['Close'], pair_trader) 

    def prepare_data(self, history, n):
        self.pair = 'BTC/USDT'
        old_history, self.new_history = history.iloc[:-n], history.iloc[-n:-1]

        self.historical_data = data.DataMaintainer()
        self.historical_data.add(data=old_history.values.T, keys=list(history), location=[self.pair, '1h', 'History'])

    def init_system(self):
        pair_trader = trader.PairTrader()
        pair_trader.set_data(self.historical_data[self.pair])

        indicator_data = self.historical_data[self.pair, '1h']

        # inds = [indicators.RelativeStrengthIndexIndicator(indicator_data, 14)]
        # rule = rules.RelativeStrengthIndexTrasholdRule(lower=30, upper=70, patience=2)

        inds = [indicators.MovingAverageIndicator(indicator_data, 90),
                indicators.MovingAverageIndicator(indicator_data, 25)]
        rule = rules.MovingAverageCrossoverRule(patience=5)
        rule_exp = experts.RuleExpert(inds, rule)

        tf_exp = experts.TimeFrameExpert('1h')
        pair_exp = experts.PairExpert(*self.pair.split('/'))

        tf_exp.set_experts([rule_exp])
        pair_exp.set_experts([tf_exp])

        pair_trader.set_expert(pair_exp)
        return pair_trader

    def run(self, pair_trader):
        for idx, row in self.new_history.iterrows():
            pair_trader.update({'1h': row})
            pair_trader.act()
        return pair_trader.profit[-1] if pair_trader.profit else 0



def plot(close, pair_trader):
    n = len(close)
    pair_trader.show_evaluation()

    buy, sell = pair_trader.trades[::2], pair_trader.trades[1::2]
    buy_time, sell_time = pair_trader.times[::2], pair_trader.times[1::2]

    fig, axs = plt.subplots(nrows=2)
    fig.set_size_inches(19.5, 10.5)
    fig.set_tight_layout(True)

    ax1, ax2 = axs.reshape(-1)
    config_ax(ax1)

    ax1.plot(range(n), close, color='black')
    ax1.scatter(buy_time, [x[2] for x in buy], color='blue')
    ax1.scatter(sell_time, [x[2] for x in sell], color='red')

    config_ax(ax2)
    ax2.plot(sell_time, pair_trader.profit, linestyle=':')

    plt.show()


def config_plot(plt):
    plt.figure(figsize = (30, 5))
    plt.tight_layout()

def config_ax(ax):
    ax.margins(0, .1)
    ax.grid(True)


def load_history(filename):
    data_dir = './data/act_data/BTCUSDT/'
    return pd.read_csv(data_dir + filename)


def simulate():
    history = load_history('1h.csv')
    n = 24 * 365
    simulation = Simulation()
    simulation.simulate(history, n)