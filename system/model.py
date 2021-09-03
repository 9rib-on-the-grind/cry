from pprint import pprint
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import trainer
import experts
import trader



def get_inner_layers(repeat, inner, dropout, **kwargs):
    layers = []
    for _ in range(repeat):
        layers += [
            keras.layers.Dense(inner, activation='selu', use_bias=False),
            keras.layers.Dropout(dropout),
            keras.layers.BatchNormalization(),
        ]
    return layers

def construct_model_from_expert_config(hp=None, *, filename='expert.json'):
    def get_inputs(hierarchy):
        if hierarchy['name'] != 'RuleClassExpert':
            return [get_inputs(sub) for sub in hierarchy['inner experts']]
        else:
            return keras.Input(len(hierarchy['inner experts']))

    def build(hierarchy: dict, inputs: list, hp):
        name = hierarchy['name']

        if name == 'RuleClassExpert':
            dropout = .4
            inner = 35
            repeat = 3
            outer = 20

            prev = inputs
            for layer in get_inner_layers(repeat, inner, dropout):
                prev = layer(prev)
            outer = keras.layers.Dense(outer, activation='selu')(prev)
            aux = keras.layers.Dense(3, activation='softmax')(outer)

            return keras.Model(inputs=inputs, outputs=[outer, aux])

        elif name == 'TimeFrameExpert':
            submodels = [build(sub, inp, hp) for sub, inp in zip(hierarchy['inner experts'], inputs)]
            results = [subm(inp) for subm, inp in zip(submodels, inputs)]
            main_input, sub_aux = map(list, zip(*results))

            outer = 30
            concat = keras.layers.concatenate(main_input)
            outer = keras.layers.Dense(outer, activation='selu')(concat)
            outer = keras.layers.Dense(3, activation='softmax', use_bias=False)(outer)

            sub_aux = keras.layers.Average()(sub_aux)
            aux = keras.layers.Average()([sub_aux, outer]) # change by dense?

            return keras.Model(inputs=inputs, outputs=[outer, aux], name=hierarchy['parameters']['timeframe'])

        elif name == 'PairExpert':
            submodels = [build(sub, inp, hp) for sub, inp in zip(hierarchy['inner experts'], inputs)]
            results = [subm(inp) for subm, inp in zip(submodels, inputs)]
            main_input, sub_aux = map(list, zip(*results))

            stack = tf.stack(main_input, axis=1)

            mult = np.array([24, 1, 1/4]).reshape((-1, 1))
            outer = tf.math.multiply(stack, mult)
            outer = tf.math.reduce_sum(outer, axis=1)
            outer = tf.math.divide(outer, np.sum(mult))

            outputs = [outer] + sub_aux

            return keras.Model(inputs=inputs, outputs=outputs, name='main')

    hierarchy = json.load(open(filename, 'r'))
    inputs = get_inputs(hierarchy)
    model = build(hierarchy, inputs, hp)
    model.summary()
    lr = 1e-3
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=lr),
        loss='categorical_crossentropy',
        loss_weights=[1, 1, 1, 1],
    )

    return model


def get_conf(close, period, trashold=.8):
    close = pd.Series(close)
    mins = close[::-1].rolling(period, closed='both', min_periods=0).min()[::-1]
    maxs = close[::-1].rolling(period, closed='both', min_periods=0).max()[::-1]
    height = maxs - mins
    center = height / 2
    close = close - mins - center
    conf = close / center
    # conf = conf[::-1].ewm(span=2, min_periods=0).mean()[::-1]
    # conf = conf.ewm(span=3, min_periods=0).mean()
    conf = -conf

    buy = [1 if c > trashold else 0 for c in conf]
    sell = [1 if c < -trashold else 0 for c in conf]
    wait = [not b and not s for b, s in zip(buy, sell)]
    desicions = np.array([sell, wait, buy], dtype=int).T

    return (mins, maxs), desicions

def plot(close, true_conf=None, pred=None, pair_trader=None, **kw):

    fig, axs = plt.subplots(nrows=2)
    ax1, ax2 = axs.reshape(-1)
    for ax in (ax1, ax2):
        ax.grid(True)

    (mins, maxs), _ = get_conf(close, **kw)

    ax1.plot(close, color='black')
    ax1.plot(mins, color='blue', alpha=.2)
    ax1.plot(maxs, color='red', alpha=.2)

    if pair_trader is not None:
        buy_time, sell_time = pair_trader.times[::2], pair_trader.times[1::2]
        ax1.scatter(buy_time, [close[t] for t in buy_time], color='blue')
        ax1.scatter(sell_time, [close[t] for t in sell_time], color='red')

    if true_conf is not None:
        ax2.plot(true_conf, color='grey')
    if pred is not None:
        ax2.plot(pred)

    fig.tight_layout()
    plt.show()



np.set_printoptions(edgeitems=7)
np.core.arrayprint._line_width = 180

if __name__ == '__main__':

    history, signals = trainer.get_data()

    close = defaultdict(list)
    for upd in history:
        for timeframe, data in upd.items():
            close[timeframe].append(data[4]) # close price

    target = {}
    periods = {'4h': 6, '1h': 24, '15m': 4}
    for timeframe, data in close.items():
        _, desicions = get_conf(data, periods[timeframe])
        conf_iter = iter(desicions)
        rep = []
        for upd in history:
            if timeframe in upd:
                rep.append(next(conf_iter).tolist())
            else:
                rep.append(rep[-1])
        target[timeframe] = np.array(rep)

    _, desicions = get_conf(close['15m'], 4 * 24)
    target['main'] = desicions
    target = [target[key] for key in ['main', '4h', '15m', '1h']]






    model = construct_model_from_expert_config()
    model.summary()

    model.fit(
        x=signals,
        y=target,
        validation_split=.2,
        epochs=10,
        batch_size=32,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=8, verbose=1, restore_best_weights=True),
        ],
    )
    pred = model.predict(signals)[-1]

    from collections import Counter
    print(Counter(np.argmax(target[-1], axis=1) - 1))
    print(Counter(np.argmax(pred, axis=1) - 1))

    plt.plot(np.argmax(target[-1], axis=1) - 1, color='blue')
    plt.plot(np.argmax(pred, axis=1) - 1, color='red')
    plt.show()

    # pred = pred.reshape(-1)
    # # target = target[1].reshape(-1)

    # # plot(close['15m'], true_conf=target, pred=pred, period=24*4)

