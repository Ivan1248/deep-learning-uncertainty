import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # /*
from data import Dataset, MiniBatchReader

from abstract_model import AbstractModel


class SemSegBaselineA(AbstractModel):
    def __init__(self,
                 input_shape,
                 class_count,
                 class0_unknown=False,
                 batch_size=32,
                 conv_layer_count=4,
                 learning_rate=1e-3,
                 training_log_period=1,
                 name='BaselineA'):
        self.conv_layer_count = conv_layer_count
        self.learning_rate = learning_rate
        self.completed_epoch_count = 0
        self.class0_unknown = class0_unknown
        super().__init__(
            input_shape=input_shape,
            class_count=class_count,
            batch_size=batch_size,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self):
        from tf_utils.layers import conv2d, max_pool, rescale_bilinear, avg_pool

        def layer_width(layer: int):  # number of channels (features per pixel)
            return min([4 * 4**(layer + 1), 64])

        input_shape = [None] + list(self.input_shape)
        output_shape = input_shape[:3] + [self.class_count]

        # Input image and labels placeholders
        input = tf.placeholder(tf.float32, shape=input_shape)
        target = tf.placeholder(tf.float32, shape=output_shape)

        # Downsampled input (to improve speed at the cost of accuracy)
        h = rescale_bilinear(input, 0.5)

        # Hidden layers
        h = conv2d(h, 3, layer_width(0))
        h = tf.nn.relu(h)
        for l in range(1, self.conv_layer_count):
            h = max_pool(h, 2)
            h = conv2d(h, 3, layer_width(l))
            h = tf.nn.relu(h)

        # Pixelwise softmax classification and label upscaling
        logits = conv2d(h, 1, self.class_count)
        probs = tf.nn.softmax(logits)
        probs = tf.image.resize_bilinear(probs, output_shape[1:3])

        # Loss
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        ts = lambda x: x[:, :, :, 1:] if self.class0_unknown else x
        cost = -tf.reduce_mean(ts(target) * tf.log(ts(clipped_probs)))

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_step = optimizer.minimize(cost)

        # Dense predictions and labels
        preds, dense_labels = tf.argmax(probs, 3), tf.argmax(target, 3)

        # Other evaluation measures
        self._n_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            probs=probs,
            loss=cost,
            training_step=training_step)

    def train(self,
              train_data: Dataset,
              validation_data: Dataset = None,
              epoch_count: int = 1):
        self._train(train_data, validation_data, epoch_count, {
            'accuracy': self._n_accuracy
        })

    def test(self, dataset):
        """ Override if extra fetches (maybe some evaluation measures) are needed """
        self._test(dataset, extra_fetches={'accuracy': self._n_accuracy})


def main(epoch_count=1):
    from data import Dataset, loaders
    from data.preparers import Iccv09Preparer
    import ioutils
    from ioutils import console

    print("Loading and deterministically shuffling data...")

    data_path = os.path.join(
        ioutils.path.find_ancestor(os.path.dirname(__file__), 'projects'), 'datasets/iccv09')
    ds = loaders.load_iccv09(data_path)
    ds.shuffle(order_determining_number=0.5)
    print("Splitting dataset...")
    ds_trainval, ds_test = ds.split(0, int(ds.size * 0.8))
    ds_train, ds_val = ds_trainval.split(0, int(ds_trainval.size * 0.8))
    print("Initializing model...")
    model = SemSegBaselineA(
        input_shape=ds.image_shape,
        class_count=ds.class_count,
        class0_unknown=True,
        batch_size=16,
        learning_rate=1e-3,
        name='BaselineA-bs16', 
        training_log_period=5)

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'd':
            viz.display(ds_val, lambda im: model.predict([im])[0])
        elif text == 'q':
            return True
        return False

    model.training_step_event_handler = handle_step

    #viz = visualization.Visualizer()
    print("Starting training and validation loop...")
    #model.test(ds_val)
    for i in range(epoch_count):
        model.train(ds_train, epoch_count=1)
        model.test(ds_val)
    model.save_state()


if __name__ == '__main__':
    main(epoch_count=200)

# "GTX 970" 43 times faster than "Pentium 2020M @ 2.40GHz × 2"
