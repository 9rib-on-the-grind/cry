from pprint import pprint

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json

import trainer
import experts



def get_inner_layers(repeat, inner, **kwargs):
    layers = []
    for _ in range(repeat):
        layers += [
            keras.layers.Dense(inner, activation='selu'),
            keras.layers.Dropout(.35),
            keras.layers.BatchNormalization(),
        ]
    return layers

def get_outer_layers(outer, **kwargs):
    layers = [
        keras.layers.Dense(outer, activation='selu'),
    ]
    return layers

def construct_model_from_expert_config(filename='expert.json'):
    def get_inputs(hierarchy):
        if 'inner experts' in hierarchy:
            return [get_inputs(sub) for sub in hierarchy['inner experts']]
        else:
            return keras.Input(1)

    def rec(hierarchy: dict, inputs: list):
        if 'inner experts' in hierarchy:
            submodels = [rec(sub, inp) for sub, inp in zip(hierarchy['inner experts'], inputs)]
            results = [subm(inp) for subm, inp in zip(submodels, inputs)]

            params = config[hierarchy['name']]
            concat = prev = keras.layers.concatenate(results)
            for layer in get_inner_layers(**params):
                prev = layer(prev)
            outer = keras.layers.Dense(params['outer'], activation='selu')(prev)
            if hierarchy['name'] == 'PairExpert':
                outer = keras.layers.Dense(1, activation='tanh')(outer)

            return keras.Model(inputs=inputs, outputs=outer)

        else:
            return keras.Model(inputs=inputs, outputs=inputs)

    hierarchy = json.load(open(filename, 'r'))
    inputs = get_inputs(hierarchy)
    return rec(hierarchy, inputs)



config = {
    'PairExpert': {'repeat': 0, 'inner': 3, 'outer': 5},
    'TimeFrameExpert': {'repeat': 0, 'inner': 15, 'outer': 20},
    'RuleClassExpert': {'repeat': 5, 'inner': 100, 'outer': 10},
}




def get_conf(close, period=24):
    close = pd.Series(close)
    mins = close[::-1].rolling(period, closed='right', min_periods=0).min()[::-1]
    maxs = close[::-1].rolling(period, closed='right', min_periods=0).max()[::-1]
    height = maxs - mins
    center = height / 2
    close = close - mins - center
    conf = close / center
    conf = conf.ewm(span=5, min_periods=0).mean()
    return (mins, maxs), conf

def plot(close, true_conf=None, pred=None):
    fig, axs = plt.subplots(nrows=2)
    ax1, ax2 = axs.reshape(-1)
    for ax in (ax1, ax2):
        ax.grid(True)

    (mins, maxs), _ = get_conf(close)

    ax1.plot(close, color='black')
    ax1.plot(mins, color='blue')
    ax1.plot(maxs, color='red')

    if true_conf is not None:
        ax2.plot(true_conf, color='grey')
    if pred is not None:
        ax2.plot(pred)

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':

    history, signals = trainer.get_data()
    close = [d[4] for d in [d['1h'] for d in history]]
    _, true_conf = get_conf(close, 24)

    model = construct_model_from_expert_config()
    
    model.compile(
        optimizer='rmsprop',
        loss='mse',
    )
    model.fit(
        x=signals,
        y=np.array(true_conf),
        validation_split=.2,
        epochs=30,
        batch_size=32,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=4, verbose=1, restore_best_weights=True),
        ],
        use_multiprocessing=True,
    )

    pred = model.predict(signals)
    print(pred)
    print(pred[0])
    print(np.min(pred), np.max(pred))
    plot(close, true_conf, pred)