from pprint import pprint

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import trainer
import experts
import config



class ExpertLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = keras.layers.Dense(10)
        self.layer2 = keras.layers.Dense(1)

    def call(self, inputs):
        res = self.layer1(inputs)
        return self.layer2(res)



def construct_model_from_expert(expert: experts.BaseExpert, inputs=None):
    def get_inputs(shape):
        n, *other = shape
        if not other:
            return [keras.Input(1) for _ in range(n)]
        else:
            return [get_inputs(other) for _ in range(n)]

    inputs = inputs or get_inputs(expert.get_shape())

    if hasattr(expert, '_inner_experts'):
        submodels = [construct_model_from_expert(exp, inp) for exp, inp in zip(expert._inner_experts, inputs)]
        results = [subm(inp) for subm, inp in zip(submodels, inputs)]

        concat = keras.layers.concatenate(results)
        expert_layer = ExpertLayer()(concat)

        model = keras.Model(inputs=inputs, outputs=expert_layer, name=expert.get_model_name())
        return model

    else:
        return keras.Model(inputs, inputs, name=expert.get_model_name())



if __name__ == '__main__':

    expert = config.deserialize_expert_from_json()
    expert.show()

    model = construct_model_from_expert(expert)
    model.summary()
    keras.utils.plot_model(model, "arch.png", expand_nested=True, rankdir='LR')