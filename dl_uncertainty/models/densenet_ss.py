import tensorflow as tf

from ..tf_utils import layers, regularization
from ..tf_utils.layers import conv, densenet
from ..tf_utils.evaluation import evaluate_semantic_segmentation

from .abstract_model import AbstractModel


class DenseNet(AbstractModel):

    def __init__(
            self,
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            class_count,
            group_lengths=[6, 12, 24, 16],  # 121 [6, 12, 32, 32]-169
            block_structure=layers.BlockStructure.densenet(),
            base_width=12,
            weight_decay=1e-4,
            batch_size=128,
            learning_rate_policy=5e-4,
            base_learning_rate=5e-4,
            training_log_period=20,
            name='DenseNetSS'):
        self.input_shape, self.class_count = input_shape, class_count
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
        target = tf.placeholder(tf.int32, shape=input_shape[:-1], name='target')

        # Hidden layers
        h = densenet(
            input,
            **self.densenet_params,
            bn_params={'is_training': is_training},
            dropout_params={
                **layers.default_arg(layers.resnet, 'dropout_params'),
                'is_training': is_training
            }, 
            large_input=False)

        # Global pooling and softmax classification
        logits = conv(h, 1, self.class_count, bias=False)
        logits = tf.image.resize_bilinear(logits, input_shape[1:3])
        probs = tf.nn.softmax(logits)

        # Loss and regularization
        loss = losses.cross_entropy_loss(logits, target)

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
            additional_outputs={'probs': probs,
                                'logits': logits},)
