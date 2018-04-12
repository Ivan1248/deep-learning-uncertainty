import tensorflow as tf
from tensorflow.python.framework import ops

from .abstract_model import AbstractModel


class SemSegBaselineA(AbstractModel):
    def __init__(self,
                 input_shape,
                 class_count,
                 class0_unknown=False,
                 batch_size=32,
                 conv_layer_count=4,
                 learning_rate=1e-3,
                 training_log_period=1,
                 name='SemSegBaselineA'):
        self.input_shape, self.class_count = input_shape, class_count
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
        from .tf_utils.layers import conv, max_pool, rescale_bilinear

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
        h = conv(h, 3, layer_width(0))
        h = tf.nn.relu(h)
        for l in range(1, self.conv_layer_count):
            h = max_pool(h, 2)
            h = conv(h, 3, layer_width(l))
            h = tf.nn.relu(h)

        # Pixelwise softmax classification and label upscaling
        logits = conv(h, 1, self.class_count)
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
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            output=preds,
            loss=cost,
            training_step=training_step,
            evaluation={'accuracy': accuracy},
            additional_outputs={'probs': probs,
                                'logits': logits},
            )