import tensorflow as tf

from ..data import Dataset, MiniBatchReader

from .abstract_model import AbstractModel


class BaselineA(AbstractModel):
    def __init__(self,
                 input_shape,
                 class_count,
                 class0_unknown=False,
                 batch_size=32,
                 conv_layer_count=4,
                 learning_rate_policy=1e-3,
                 training_log_period=1,
                 name='ClfBaselineA'):
        self.conv_layer_count = conv_layer_count
        self.completed_epoch_count = 0
        self.class0_unknown = class0_unknown
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

        convi = 0

        def conv_bn_relu(x, width):
            nonlocal convi
            x = conv(x, 3, width, bias=False, scope='conv' + str(convi))
            convi += 1
            #return bn_relu(x, is_training, scope='bnrelu' + str(convi))
            return tf.nn.relu(x)

        h = conv_bn_relu(h, layer_width(0))
        for l in range(1, self.conv_layer_count):
            h = max_pool(h, 2)
            h = conv_bn_relu(h, layer_width(l))

        # Global pooling and softmax classification
        h = conv(h, 1, self.class_count)
        logits = tf.reduce_mean(h, axis=[1, 2])
        probs = tf.nn.softmax(logits)

        # Loss
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        ts = lambda x: x[:, 1:] if self.class0_unknown else x
        loss = -tf.reduce_mean(ts(target) * tf.log(ts(clipped_probs)))

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