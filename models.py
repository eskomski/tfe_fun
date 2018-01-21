# Elliott Skomski (skomski.org)
# Deep neural network implemented with TensorFlow Eager Execution
# Model definitions

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

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
        self.W_ih = tf.get_variable('W_ih', (input_shape, layers[0]),
                                    tf.float32, tf.glorot_uniform_initializer())
        self.b_ih = tf.get_variable('b_ih', (layers[0]),
                                    tf.float32, tf.constant_initializer(b_init))

        self.W_hh = []
        self.b_hh = []
        # hidden to hidden
        for i, l in enumerate(layers[1:]):
            self.W_hh.append(tf.get_variable('W_hh_%d' % i, (layers[i], l),
                                             tf.float32, tf.glorot_uniform_initializer()))
            self.b_hh.append(tf.get_variable('b_hh_%d' % i, l,
                                             tf.float32, tf.glorot_uniform_initializer()))

        # hidden to output
        self.W_ho = tf.get_variable('W_ho', (layers[-1], output_shape),
                                    tf.float32, tf.glorot_uniform_initializer())
        self.b_ho = tf.get_variable('b_ho', output_shape,
                                    tf.float32, tf.zeros_initializer())

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

class RNN(object):
    def __init__(self, output_shape=10, layers=[100],
            hidden_act=tf.nn.relu, output_act=tf.identity, cell_type=tf.nn.rnn_cell.LSTMCell):
        self.hidden_act = hidden_act
        self.output_act = output_act

        if len(layers) == 1:
            self.cell = cell_type(layers[0])
        else:
            multi_cell = [cell_type(l) for l in layers]
            self.cell = tf.nn.rnn_cell.MultiRNNCell(multi_cell)

        self.W_ho = tf.get_variable('W_ho', (layers[-1], output_shape),
                                    tf.float32, tf.glorot_uniform_initializer())

        self.b_ho = tf.get_variable('b_ho', output_shape,
                                    tf.float32, tf.zeros_initializer())

    def predict(self, inputs):
        hs, h_f = tf.nn.dynamic_rnn(self.cell, inputs, dtype=tf.float32)

        if type(h_f) == tuple:
            h_f = h_f[-1]

        if type(h_f) == tf.nn.rnn_cell.LSTMStateTuple:
            h_f = h_f.h

        return self.output_act(tf.matmul(h_f, self.W_ho) + self.b_ho)

