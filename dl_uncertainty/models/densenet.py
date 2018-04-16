import tensorflow as tf

from ..tf_utils import layers, regularization
from ..tf_utils.layers import conv, densenet

from .abstract_model import AbstractModel


class DenseNet(AbstractModel):

    def __init__(
            self,
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            class_count,
            group_lengths=[19, 19, 19],
            block_structure=layers.BlockStructure.densenet(),
            base_width=12,
            weight_decay=1e-4,
            batch_size=128,
            learning_rate_policy=1e-2,
            training_log_period=1,
            name='ResNet'):
        self.input_shape, self.class_count = input_shape, class_count
        self.completed_epoch_count = 0  # TODO: remove
        self.weight_decay = weight_decay
        # parameters to be forwarded to layers.resnet
        rpn = ['group_lengths', 'block_structure', 'base_width']
        self.densenet_params = {k: v for k, v in locals().items() if k in rpn}
        super().__init__(
            batch_size=batch_size,
            learning_rate_policy=learning_rate_policy,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')
        target = tf.placeholder(tf.int32, shape=[None], name='input')
        target_oh = tf.one_hot(indices=target, depth=self.class_count)

        # Hidden layers
        h = densenet(input, is_training=is_training, **self.densenet_params)

        # Global pooling and softmax classification
        h = tf.reduce_mean(h, axis=[1, 2], keepdims=True)
        logits = conv(h, 1, self.class_count)
        logits = tf.reshape(logits, [-1, self.class_count])
        probs = tf.nn.softmax(logits)

        # Loss and regularization
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=target_oh, logits=logits))

        w_vars = filter(lambda x: 'weights' in x.name, tf.global_variables())
        loss += self.weight_decay * regularization.l2_regularization(w_vars)

        # Optimization
        #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate * 5e-3)
        training_step = optimizer.minimize(loss)

        # Dense prediction
        output = tf.argmax(logits, 1, output_type=tf.int32)

        # Other evaluation measures
        accuracy = tf.reduce_mean(tf.cast(tf.equal(output, target), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            output=output,
            loss=loss,
            training_step=training_step,
            evaluation={'accuracy': accuracy},
            additional_outputs={'probs': probs, 'logits': logits},)
