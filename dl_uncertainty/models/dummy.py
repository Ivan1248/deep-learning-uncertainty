import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # /*
from data import Dataset, MiniBatchReader

from .abstract_model import AbstractModel


class Dummy(AbstractModel):
    def __init__(self,
                 input_shape, ##
                 class_count, ##
                 batch_size=32,  #
                 conv_layer_count=4,
                 learning_rate_policy=1e-3,  #
                 training_log_period=1,  #
                 name='Dummy'):  #
        super().__init__(
            input_shape=input_shape,
            class_count=class_count,
            batch_size=batch_size,
            learning_rate_policy=learning_rate_policy,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        from tf_utils.layers import conv, max_pool, rescale_bilinear, avg_pool, bn_relu

        def layer_width(layer: int):  # number of channels (features per pixel)
            return min([4 * 4**(layer + 1), 32])

        input_shape = [None] + list(self.input_shape)
        output_shape = [None, self.class_count]

        # Input image and labels placeholders
        input = tf.placeholder(tf.float32, shape=input_shape)
        target = tf.placeholder(tf.float32, shape=output_shape)

        # Hidden layers
        h = input

        # Global pooling and softmax classification
        h = conv(h, 3, self.class_count)
        logits = tf.reduce_mean(h, axis=[1, 2])
        probs = tf.nn.softmax(logits)

        # Loss
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        loss = -tf.reduce_mean(target * tf.log(clipped_probs))

        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        # Dense predictions and labels
        preds, dense_labels = tf.argmax(probs, 1), tf.argmax(target, 1)

        # Other evaluation measures
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            probs=probs,
            loss=loss,
            training_step=training_step,
            evaluation={
                'accuracy': accuracy
            })