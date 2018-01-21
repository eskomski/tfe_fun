# Elliott Skomski (skomski.org)
# Deep neural network implemented with TensorFlow Eager Execution
# Model training script

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from models import DNN

np.random.seed(408)
tfe.enable_eager_execution()
tf.set_random_seed(408)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def loss(model, x, y):
    err = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.predict(x), labels=y)
    return tf.reduce_mean(err)

def acc(model, x, y):
    correct = tf.equal(tf.argmax(model.predict(x), 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

dnn = DNN(layers=[512, 256, 128, 64])

grad = tfe.implicit_gradients(loss)
opt = tf.train.GradientDescentOptimizer(0.03)

for e in range(10):
    for b in range(mnist.train.num_examples // 128):
        batch_xs, batch_ys = mnist.train.next_batch(128)   # get batch
        opt.apply_gradients(grad(dnn, batch_xs, batch_ys)) # update weights

    # eval on test at each epoch
    xs = mnist.test.images
    ys = mnist.test.labels
    print("epoch %d: acc=%.4f, mse=%.4f" %
            (e+1, acc(dnn, xs, ys).numpy(), loss(dnn, xs, ys).numpy()))

