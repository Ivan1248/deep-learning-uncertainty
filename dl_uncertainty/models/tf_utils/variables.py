import numpy as np
import tensorflow as tf


def conv_weight_variable(ksize,
                         in_channels: int,
                         out_channels: int,
                         name='weights'):
    if type(ksize) is int:
        ksize = [ksize, ksize]
    shape = list(ksize) + [in_channels, out_channels]
    maxval = (6 / (ksize[0] * ksize[1] * in_channels + out_channels))**0.5
    initializer = tf.random_uniform_initializer(-maxval, maxval)
    return tf.get_variable(name, shape=shape, initializer=initializer)


def bias_variable(n: int, initial_value=0.05, name='biases'):
    return tf.get_variable(
        name, shape=[n], initializer=tf.constant_initializer(initial_value))
