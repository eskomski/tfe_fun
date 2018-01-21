# Elliott Skomski (skomski.org)
# Deep neural network implemented with TensorFlow Eager Execution
# Model definitions

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

def glorot_init(shape):
    return np.random.uniform(-np.sqrt(6) / np.sqrt(shape[0] + shape[1]),
                             np.sqrt(6) / np.sqrt(shape[0] + shape[1]), size=shape)

class DNN(object):
    def __init__(self, input_shape=784, output_shape=10, layers=[100, 50],
            hidden_act=tf.nn.relu, output_act=tf.identity):
        self.hidden_act = hidden_act
        self.output_act = output_act

        # use small constant init for relu activation
        # saturating activations use zero init
        if self.hidden_act == tf.nn.relu:
            b_init = 0.1
        else:
            b_init = 0.0

        # input to hidden
        self.W_ih = tfe.Variable(glorot_init((input_shape, layers[0])), dtype=tf.float32)
        self.b_ih = tfe.Variable(np.full(layers[0], b_init), dtype=tf.float32)

        self.W_hh = []
        self.b_hh = []
        # hidden to hidden
        for i, l in enumerate(layers[1:]):
            self.W_hh.append(tfe.Variable(glorot_init((layers[i], l)), dtype=tf.float32))
            self.b_hh.append(tfe.Variable(np.full(l, b_init),dtype=tf.float32))

        # hidden to output
        self.W_ho = tfe.Variable(glorot_init((layers[-1], output_shape)), dtype=tf.float32)
        self.b_ho = tfe.Variable(np.full(output_shape, 0.0), dtype=tf.float32)

    def predict(self, inputs):
        # input to hidden
        result = self.hidden_act(tf.matmul(inputs, self.W_ih) + self.b_ih)

        # hidden to hidden
        for W, b in zip(self.W_hh, self.b_hh):
            result = self.hidden_act(tf.matmul(result, W) + b)

        # hidden to output
        return tf.matmul(result, self.W_ho) + self.b_ho

    def get_layer(self, inputs, layer):
        # input to hidden
        result = self.hidden_act(tf.matmul(inputs, self.W_ih) + self.b_ih)

        # hidden to hidden
        for i in range(layer):
            result = self.hidden_act(tf.matmul(result, self.W_hh[i]) + self.b_hh[i])

        return result

