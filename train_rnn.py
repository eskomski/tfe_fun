# Elliott Skomski (skomski.org)
# Recurrent neural network implemented with TensorFlow Eager Execution
# Model training script
# Still using MNIST because why not?

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from models import RNN

np.random.seed(408)
tfe.enable_eager_execution()
tf.set_random_seed(408)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

rnn = RNN(layers=[128])

def loss(model, x, y):
    err = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.predict(x), labels=y)
    return tf.reduce_mean(err)

def acc(model, x, y):
    correct = tf.equal(tf.argmax(model.predict(x), 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

grad = tfe.implicit_gradients(loss)
opt = tf.train.AdamOptimizer(0.03)

for e in range(10):
    for b in range(mnist.train.num_examples // 128):
        batch_xs, batch_ys = mnist.train.next_batch(128)   # get batch
        batch_xs = batch_xs.reshape((128, 28, 28))
        opt.apply_gradients(grad(rnn, batch_xs, batch_ys)) # update weights
        if b % 32 == 0:
            print("epoch %d: acc=%.4f, mse=%.4f" %
                    (e+1, acc(rnn, batch_xs, batch_ys).numpy(), loss(rnn, batch_xs, batch_ys).numpy()))

    # eval on test at each epoch
    xs = mnist.test.images
    xs = xs.reshape((mnist.test.num_examples, 28, 28))
    ys = mnist.test.labels
    print("test set: acc=%.4f, mse=%.4f" %
            (acc(rnn, xs, ys).numpy(), loss(rnn, xs, ys).numpy()))

