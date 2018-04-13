import tensorflow as tf

from .abstract_model import AbstractModel
from .tf_utils import layers, regularization


class ResNet(AbstractModel):

    def __init__(
            self,
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            class_count,
            block_properties=layers.ResidualBlockConfig([3, 3]),
            group_lengths=[3, 3, 3],
            base_width=16,
            widening_factor=1,
            weight_decay=5e-4,
            batch_size=128,
            learning_rate_policy=1e-2,
            training_log_period=1,
            name='ResNet'):
        self.input_shape, self.class_count = input_shape, class_count
        self.completed_epoch_count = 0  # TODO: remove
        self.block_properties = block_properties
        self.group_lengths = group_lengths
        self.depth = 1 + sum(group_lengths) * len(block_properties.ksizes) + 1
        self.zagoruyko_depth = self.depth - 1 + len(group_lengths)
        self.base_width = base_width
        self.widening_factor = widening_factor
        self.weight_decay = weight_decay
        super().__init__(
            batch_size=batch_size,
            learning_rate_policy=learning_rate_policy,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        from .tf_utils.layers import conv, resnet

        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')
        target = tf.placeholder(tf.int32, shape=[None], name='input')
        target_oh = tf.one_hot(indices=target, depth=self.class_count)

        # Hidden layers
        h = resnet(
            input,
            is_training=is_training,
            base_width=self.base_width,
            widening_factor=self.widening_factor,
            group_lengths=self.group_lengths,
            block_properties=self.block_properties)

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
        optimizer = tf.train.AdamOptimizer(learning_rate*5e-3)
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
            additional_outputs={'probs': probs,
                                'logits': logits},)
