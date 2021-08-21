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



def get_inner_layers(repeat, inner, dropout, **kwargs):
    layers = []
    for _ in range(repeat):
        layers += [
            keras.layers.Dense(inner, activation='selu'),
            keras.layers.Dropout(dropout),
            keras.layers.BatchNormalization(),
        ]
    return layers

def construct_model_from_expert_config(hp, *, filename='expert.json'):
    def get_inputs(hierarchy):
        if hierarchy['name'] != 'RuleClassExpert':
            return [get_inputs(sub) for sub in hierarchy['inner experts']]
        else:
            return keras.Input(len(hierarchy['inner experts']))

    def build(hierarchy: dict, inputs: list, hp):
        name = hierarchy['name']
        if name != 'RuleClassExpert':
            submodels = [build(sub, inp, hp) for sub, inp in zip(hierarchy['inner experts'], inputs)]
            results = [subm(inp) for subm, inp in zip(submodels, inputs)]
            main_input, aux_outs = map(list, zip(*results))
        else:
            main_input, aux_outs = [inputs], None

        if name == 'RuleClassExpert':
            dropout = hp.Float('1lvl_dropout', .2, .5, .05)
            inner = hp.Int('1lvl_inner', 30, 150, 10)
            repeat = hp.Int('1lvl_repeat', 0, 20)
            outer = hp.Int('1lvl_outer', 5, 30)
        elif name == 'TimeFrameExpert':
            dropout = hp.Float('2lvl_dropout', .2, .5, .05)
            inner = hp.Int('2lvl_inner', 50, 500, 50)
            repeat = hp.Int('2lvl_repeat', 0, 20)
            outer = hp.Int('2lvl_outer', 10, 100, 10)
        elif name == 'PairExpert':
            inner = 0
            repeat = 0
            dropout = 0
            outer = hp.Int('3lvl_outer', 5, 30)
        else:
            raise ValueError('No hyperparameters for ' + name)

        # main
        prev = (keras.layers.concatenate(main_input) if len(main_input) > 1 else main_input[0])
        for layer in get_inner_layers(repeat, inner, dropout):
            prev = layer(prev)
        outer = keras.layers.Dense(outer, activation='selu')(prev)
        if name == 'PairExpert':
            outer = keras.layers.Dense(1, activation='tanh', name='main_out')(outer)

        # aux
        if aux_outs is None:
            aux = keras.layers.Dense(1, activation='tanh')(outer)
        elif name != 'PairExpert':
            aux = keras.layers.Dense(1, activation='tanh')(outer)
            aux_avg = keras.layers.Average()(aux_outs) if len(aux_outs) > 1 else aux_outs[0]
            aux = keras.layers.Average()([aux_avg, aux])
        else:
            aux = keras.layers.Average()(aux_outs) if len(aux_outs) > 1 else aux_outs[0]
            aux = keras.layers.Layer(name='aux_out', trainable=False)(aux)

        return keras.Model(inputs=inputs, outputs=[outer, aux], name=hierarchy['unique name'])

    hierarchy = json.load(open(filename, 'r'))
    inputs = get_inputs(hierarchy)
    model = build(hierarchy, inputs, hp)
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    aux_rate = hp.Float('aux_rate', 0, .3, .05)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=lr),
        loss='mse',
        loss_weights=[1 - aux_rate, aux_rate],
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


    tuner = keras_tuner.BayesianOptimization(
        construct_model_from_expert_config,
        objective=keras_tuner.Objective("val_main_out_loss", direction="min"),
        max_trials=100,
        executions_per_trial=1,
        overwrite=True,
        directory="hyperparameters",
        project_name="cry",
    )

    conf = np.array(true_conf)
    tuner.search(
        x=signals,
        y=[conf, conf],
        validation_split=.2,
        epochs=100,
        batch_size=32,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=4, verbose=1, restore_best_weights=True),
        ],
    )
    tuner.results_summary()

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(
        x=signals,
        y=[conf, conf],
        validation_split=.2,
        epochs=1,
        batch_size=32,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=4, verbose=1, restore_best_weights=True),
        ],
    )
    pred = model.predict(signals)[0]
    plot(close, true_conf, pred)