import tensorflow as tf

from ..tf_utils import layers, regularization, losses
from ..tf_utils.layers import conv, resnet

from .abstract_model import AbstractModel


class ResNetSS(AbstractModel):

    def __init__(
            self,
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            class_count,
            group_lengths=[3, 3, 3],
            block_structure=layers.BlockStructure.resnet(),
            base_width=None,
            weight_decay=5e-4,
            batch_size=128,
            learning_rate_policy=1e-2,
            training_log_period=1,
            name='ResNetSS'):
        self.input_shape, self.class_count = input_shape, class_count
        self.completed_epoch_count = 0  # TODO: remove
        self.depth = 1 + sum(group_lengths) * len(block_structure.ksizes) + 1
        self.zagoruyko_depth = self.depth - 1 + len(group_lengths)
        self.weight_decay = weight_decay

        base_width = base_width
        rpn = ['group_lengths', 'block_structure', 'base_width']
        self.resnet_params = {k: v for k, v in locals().items() if k in rpn}

        super().__init__(
            batch_size=batch_size,
            learning_rate_policy=learning_rate_policy,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')
        target = tf.placeholder(tf.int32, shape=input_shape[:-1], name='target')
        target_oh = tf.one_hot(target, self.class_count)

        # Hidden layers
        h = resnet(
            input,
            **self.resnet_params,
            bn_params={'is_training': is_training},
            dropout_params={
                **layers.default_arg(layers.resnet, 'dropout_params'),
                'is_training': is_training
            })

        # Global pooling and softmax classification
        logits = conv(h, 1, self.class_count)
        logits = tf.image.resize_bilinear(logits, input_shape[1:3])
        probs = tf.nn.softmax(logits)

        # Loss and regularization
        loss = losses.cross_entropy_loss(logits, target_oh)
        #loss = -tf.reduce_mean(target_oh * tf.log(probs))

        w_vars = filter(lambda x: 'weights' in x.name, tf.global_variables())
        loss += self.weight_decay * regularization.l2_regularization(w_vars)

        # Optimization
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        #optimizer = tf.train.AdamOptimizer(learning_rate * 5e-3)
        training_step = optimizer.minimize(loss)

        # Dense prediction
        output = tf.argmax(logits, -1, output_type=tf.int32)

        # Other evaluation measures
        accuracy = tf.reduce_mean(tf.cast(tf.equal(output, target), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            output=output,
            loss=loss,
            training_step=training_step,
            evaluation={'accuracy': accuracy},
            additional_outputs={'probs': probs, 'logits': logits})
