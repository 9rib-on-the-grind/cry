from pprint import pprint

import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def construct_model_from_expert_config(hp, *, filename='expert.json'):
    def get_inputs(hierarchy):
        if 'inner experts' in hierarchy:
            return [get_inputs(sub) for sub in hierarchy['inner experts']]
        else:
            return keras.Input(1)

    def build(hierarchy: dict, inputs: list, hp):
        if 'inner experts' in hierarchy:
            submodels = [build(sub, inp, hp) for sub, inp in zip(hierarchy['inner experts'], inputs)]
            results = [subm(inp) for subm, inp in zip(submodels, inputs)]

            name = hierarchy['name']

            repeat = hp.Choice(name + '_repeat', values=[3, 5])
            inner = hp.Choice(name + '_inner', values=[100, 50])

            if len(results) > 1:
                concat = prev = keras.layers.concatenate(results)
            else:
                concat = prev = results[0]
            for layer in get_inner_layers(repeat, inner):
                prev = layer(prev)
            outer = keras.layers.Dense(10, activation='selu')(prev)
            if hierarchy['name'] == 'PairExpert':
                outer = keras.layers.Dense(1, activation='tanh')(outer)

            return keras.Model(inputs=inputs, outputs=outer)

        else:
            return keras.Model(inputs=inputs, outputs=inputs)

    hierarchy = json.load(open(filename, 'r'))
    inputs = get_inputs(hierarchy)
    model = build(hierarchy, inputs, hp)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
    )
    return model


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

    tuner = keras_tuner.RandomSearch(
        construct_model_from_expert_config,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld",
    )

    tuner.search(
        x=signals,
        y=np.array(true_conf),
        validation_split=.2,
        epochs=30,
        batch_size=32,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=4, verbose=1, restore_best_weights=True),
        ],
    )