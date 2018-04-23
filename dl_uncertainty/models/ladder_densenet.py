import tensorflow as tf

from ..tf_utils import layers, regularization, losses, evaluation
from ..tf_utils.layers import conv, densenet

from .abstract_model import AbstractModel


class LadderDenseNet(AbstractModel):

    def __init__(
            self,
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            class_count,
            epoch_count,  # for learning rate decay
            group_lengths=[6, 12, 24, 16],  # 121 [6, 12, 32, 32]-169
            weight_decay=1e-4,
            batch_size=4,
            base_learning_rate=5e-4,
            training_log_period=1,
            name='DenseNet'):
        self.input_shape, self.class_count = input_shape, class_count
        self.completed_epoch_count = 0  # TODO: remove
        self.weight_decay = weight_decay
        self.epoch_count = epoch_count
        self.base_learning_rate = base_learning_rate
        self.group_lengths=group_lengths
        # parameters to be forwarded to layers.resnet
        super().__init__(
            batch_size=batch_size,
            learning_rate_policy=None,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')
        target = tf.placeholder(tf.int32, shape=input_shape[:-1], name='input')
        target_oh = tf.one_hot(target, self.class_count)

        # Hidden layers
        bn_params = {'is_training': is_training}
        pre_logits, pre_logits_aux = layers.ladder_densenet(
            input,
            group_lengths=self.group_lengths,
            bn_params=bn_params,
            dropout_params={
                **layers.default_arg(layers.resnet, 'dropout_params'),
                'is_training': is_training
            })

        # Logits, auxiliary logits, softmax
        logits, logits_aux = layers.ladder_densenet_logits(
            pre_logits, pre_logits_aux, input_shape, self.class_count,
            bn_params)
        probs = tf.nn.softmax(logits)

        # Loss and regularization
        weight_logits_pairs = [(0.7, logits), (0.3, logits_aux)]
        loss_emp = sum(w * losses.cross_entropy_loss(l, target_oh)
                       for w, l in weight_logits_pairs)
        w_vars = filter(lambda x: 'weights' in x.name, tf.global_variables())
        loss_reg = self.weight_decay * regularization.l2_regularization(w_vars)
        loss = loss_emp + loss_reg

        # Optimization
        learning_rate = tf.train.polynomial_decay(
            self.base_learning_rate,
            global_step=epoch,
            decay_steps=self.epoch_count,
            end_learning_rate=0,
            power=1.5)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        # Dense prediction
        output = tf.argmax(logits, axis=-1, output_type=tf.int32)

        # Other evaluation measures
        accuracy = tf.reduce_mean(tf.cast(tf.equal(output, target), tf.float32))
        miou = evaluation.mean_iou(target, output)

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            output=output,
            loss=loss,
            training_step=training_step,
            evaluation={'accuracy': accuracy, 'mIoU': miou},
            additional_outputs={'probs': probs, 'logits': logits},)
