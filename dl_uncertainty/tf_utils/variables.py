import numpy as np
import tensorflow as tf


def conv_weight_variable(ksize,
                         in_channels: int,
                         out_channels: int,
                         name='weights'):
    if type(ksize) is int:
        ksize = [ksize, ksize]
    shape = list(ksize) + [in_channels, out_channels]
    n = ksize[0] * ksize[1] * in_channels + out_channels
    initializer = tf.random_normal_initializer(stddev=np.sqrt(2 / n))
    var = tf.get_variable(name, shape=shape, initializer=initializer)
    print(var.name)
    return var

def vector_variable(n: int, initial_value, name):
    var = tf.get_variable(
        name, shape=[n], initializer=tf.constant_initializer(initial_value))
    print(var.name)
    return var

def bias_variable(n: int, initial_value=0.0, name='bias'):
    return vector_variable(n, initial_value, name)

