from pprint import pprint

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import trainer
import experts
import config



class PairExpertLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layers = [
            keras.layers.Dense(1),
            keras.layers.Activation('tanh'),
        ]

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs



class TimeFrameExpertLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layers = [
            keras.layers.Dense(11),
            keras.layers.LeakyReLU(),

            keras.layers.Dense(10),
            # keras.layers.Activation('tanh'),
        ]

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs



class RuleClassExpertLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layers = [
            keras.layers.Dense(50),
            keras.layers.LeakyReLU(),

            keras.layers.Dense(10),
        ]

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def get_expert_layer(expert):
    table = {
        experts.PairExpert: PairExpertLayer,
        experts.TimeFrameExpert: TimeFrameExpertLayer,
        experts.RuleClassExpert: RuleClassExpertLayer,
    }
    return table[expert.__class__]()



def construct_model_from_expert(expert: experts.BaseExpert, inputs=None):
    def get_inputs(shape):
        count, inner = shape
        print(count, inner)
        if not inner:
            return [keras.Input(1) for _ in range(count)]
        else:
            return [get_inputs(inn) for inn in inner]

    inputs = inputs if inputs is not None else get_inputs(expert.get_shape())

    if hasattr(expert, '_inner_experts'):
        submodels = [construct_model_from_expert(exp, inp) for exp, inp in zip(expert._inner_experts, inputs)]
        results = [subm(inp) for subm, inp in zip(submodels, inputs)]

        concat = keras.layers.concatenate(results)
        expert_layer = get_expert_layer(expert)(concat)

        model = keras.Model(inputs=inputs, outputs=expert_layer, name=expert.get_model_name())
        return model

    else:
        return keras.Model(inputs, inputs, name=expert.get_model_name())


def get_conf(close, n=24):
    close = pd.Series(close)
    mins = close[::-1].rolling(n, closed='right', min_periods=0).min()[::-1]
    maxs = close[::-1].rolling(n, closed='right', min_periods=0).max()[::-1]
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

def to_lst(data):
    if not isinstance(data[0], list):
        return np.array(data)
    return [to_lst(d) for d in data]



if __name__ == '__main__':

    data, signals = trainer.get_signals()
    expert = config.deserialize_expert_from_json()
    expert.show(overview=False)
    expert.show()
    print(expert.get_shape())

    model = construct_model_from_expert(expert)
    model.summary()
    keras.utils.plot_model(model, "arch.png", expand_nested=True, rankdir='LR')


    data = [d['1h'] for d in data]
    close = [d[4] for d in data]

    signals = to_lst(signals.tolist())
    n = 24
    _, true_conf = get_conf(close, n)



    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=1e-3,
                                       momentum=.9,
                                       nesterov=True),
        loss='mse',
    )
    model.fit(
        x=signals,
        y=np.array(true_conf),
        validation_split=.2,
        epochs=100,
        batch_size=32,
        callbacks=[keras.callbacks.ReduceLROnPlateau()],
    )

    pred = model.predict(signals)
    print(np.min(pred), np.max(pred))
    plot(close, true_conf, pred)