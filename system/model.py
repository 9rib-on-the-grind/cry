from pprint import pprint

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import trainer
import experts
import config



def get_inner_layers(repeat, inner, **kwargs):
    layers = []
    for _ in range(repeat):
        layers += [
            keras.layers.Dense(inner, activation='selu'),
            keras.layers.Dropout(.4),
            keras.layers.BatchNormalization(),
        ]
    return layers

def get_outer_layers(outer, **kwargs):
    layers = [
        keras.layers.Dense(outer, activation='selu'),
    ]
    return layers



def construct_model_from_expert(expert: experts.BaseExpert, inputs=None):
    def get_inputs(shape):
        count, inner = shape
        if not inner:
            return [keras.Input(1) for _ in range(count)]
        else:
            return [get_inputs(inn) for inn in inner]


    inputs = inputs if inputs is not None else get_inputs(expert.get_shape())

    if hasattr(expert, '_inner_experts'):
        submodels = [construct_model_from_expert(exp, inp) for exp, inp in zip(expert._inner_experts, inputs)]
        results = [subm(inp) for subm, inp in zip(submodels, inputs)]

        params = config[expert.__class__]
        concat = prev = keras.layers.concatenate(results)
        for layer in get_inner_layers(**params):
            prev = layer(prev)
        outer = keras.layers.Dense(params['outer'], activation='selu')(prev)
        if isinstance(expert, experts.PairExpert):
            outer = keras.layers.Dense(1, activation='tanh')(outer)

        return keras.Model(inputs=inputs, outputs=outer, name=expert.get_model_name())
    else:
        return keras.Model(inputs, inputs, name=expert.get_model_name())



config = {
    experts.PairExpert: {'repeat': 0, 'inner': 3, 'outer': 5},
    experts.TimeFrameExpert: {'repeat': 5, 'inner': 15, 'outer': 20},
    experts.RuleClassExpert: {'repeat': 5, 'inner': 100, 'outer': 10},
}








def get_conf(close, repeat=24):
    close = pd.Series(close)
    mins = close[::-1].rolling(repeat, closed='right', min_periods=0).min()[::-1]
    maxs = close[::-1].rolling(repeat, closed='right', min_periods=0).max()[::-1]
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

    expert, data, signals = trainer.get_data()

    # t = time.time()
    model = construct_model_from_expert(expert)
    model.summary()
    expert.show()
    # print('model construction', time.time() - t)

    close = [d[4] for d in [d['1h'] for d in data]]
    _, true_conf = get_conf(close, 24)

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