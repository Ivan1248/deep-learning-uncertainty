import tensorflow as tf

from .tf_utils import layers, losses
from .modeldef import TrainingComponent

# Inference

# Head


def global_pool_affine_logits_func(class_count):

    def f(h, **k):
        h = tf.reduce_mean(h, axis=[1, 2], keep_dims=True)
        h = layers.conv(h, 1, class_count, bias=True, name='conv_logits')
        return tf.reshape(h, [-1, class_count])

    return f


def semseg_affine_logits_func(class_count, shape=None):

    def f(h, **k):
        h = layers.conv(h, 1, class_count, bias=True, name='conv_logits')
        if shape is None:
            return h
        return tf.image.resize_bilinear(h, shape[:2])

    return f


def semseg_probs_func(shape=None):

    def f(h):
        h = tf.nn.softmax(h)
        if shape is None:
            return h
        return tf.image.resize_bilinear(h, shape[:2])

    return f


class Layers:

    @staticmethod
    def dummy(x, **k):
        # k is a dictionary containing graph nodes and/or parameters
        y = x
        return y, {}  # the main output, and optional auxiliary outputs

    @staticmethod
    def sequence(layer_func_sequence):

        def total_func(x, **k):
            k = {**k}
            for f in layer_func_sequence:
                x, additional = f(x, **k)
                k.update(additional)
            return x, k

        return total_func

    class InputToFeatures:
        # http://ethereon.github.io/netscope/#/editor
        # https://github.com/shicai/DenseNet-Caffe
        # https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt
        # https://github.com/binLearning/caffe_toolkit
        # https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

        @staticmethod
        def resnet(base_width,
                   group_lengths,
                   block_structure,
                   dim_change,
                   cifar_root_block,
                   width_factor=1,
                   dropout_rate=0):
            p = [
                'base_width', 'width_factor', 'group_lengths',
                'block_structure', 'dim_change'
            ]
            params = {k: v for k, v in locals().items() if k in p}

            def input_to_features(x, is_training, pretrained_lr_factor, **k):
                x = layers.resnet_root_block(
                    x, base_width=base_width, cifar_type=cifar_root_block)
                features = layers.resnet_middle(
                    x,
                    **params,
                    bn_params={'is_training': is_training},
                    dropout_params={
                        'rate': dropout_rate,
                        'is_training': is_training
                    },
                    pretrained_lr_factor=pretrained_lr_factor)
                return features, {'features': features}

            return input_to_features

        @staticmethod
        def densenet(base_width, group_lengths, block_structure,
                     cifar_root_block, dropout_rate):
            p = ['base_width', 'group_lengths', 'block_structure']
            params = {k: v for k, v in locals().items() if k in p}

            def input_to_features(x, is_training, pretrained_lr_factor, **k):
                x = layers.densenet_root_block(
                    x, base_width=base_width, cifar_type=cifar_root_block)
                features = layers.densenet_middle(
                    x,
                    **params,
                    bn_params={'is_training': is_training},
                    dropout_params={
                        'rate': dropout_rate,
                        'is_training': is_training
                    },
                    pretrained_lr_factor=pretrained_lr_factor)
                return features, {'features': features}

            return input_to_features

        @staticmethod
        def ladder_densenet(base_width=32,
                            group_lengths=[6, 12, 24, 16],
                            block_structure=layers.BlockStructure.densenet(),
                            dropout_rate=0,
                            cifar_root_block=False):
            p = [
                'base_width', 'group_lengths', 'block_structure',
                'cifar_root_block'
            ]
            params = {k: v for k, v in locals().items() if k in p}

            def input_to_features(x, is_training, pretrained_lr_factor, **k):
                features = layers.ladder_densenet(
                    x,
                    **params,
                    bn_params={'is_training': is_training},
                    dropout_params={
                        'rate': dropout_rate,
                        'is_training': is_training
                    },
                    pretrained_lr_factor=pretrained_lr_factor)
                return features, {'features': features}

            return input_to_features

    class FeaturesToLogits:

        @classmethod
        def standard_generic(cls, problem_id, class_count, shape):
            if problem_id == 'clf':
                return cls.standard_classification(class_count)
            elif problem_id == 'semseg':
                return cls.standard_segmentation(class_count, class_count)

        @staticmethod
        def standard_classification(class_count):  # TODO: dropout

            def f(h, **k):
                h = tf.reduce_mean(
                    h, axis=[1, 2], keep_dims=True)  # global pooling
                h = layers.conv(
                    h, 1, class_count, bias=True, name='conv_logits')
                logits = tf.reshape(h, [-1, class_count])
                return logits, {'logits': logits}

            return f

        @staticmethod
        def standard_segmentation(class_count, shape=None):  # TODO: dropout

            def f(h, **k):
                logits = layers.conv(
                    h, 1, class_count, bias=True, name='conv_logits')
                if shape:
                    logits = tf.image.resize_bilinear(logits, shape[:2])
                return logits, {'logits': logits}

            return f

        @staticmethod
        def ladder_densenet(class_count, shape):  # TODO: dropout

            def features_to_logits(all_pre_logits, is_training, **k):
                pre_logits, pre_logits_aux = all_pre_logits
                all_logits = layers.ladder_densenet_logits(
                    pre_logits,
                    pre_logits_aux,
                    image_shape=shape[0:2],
                    class_count=class_count,
                    bn_params={'is_training': is_training})
                return all_logits, {
                    'all_logits': all_logits,
                    'logits': all_logits[0]
                }

            return features_to_logits

    class LogitsToOutput:

        @staticmethod
        def standard():

            def logits_to_output(x, **k):
                output = tf.argmax(x, axis=-1, output_type=tf.int32)
                return output, {'probs': tf.nn.softmax(x), 'output': output}

            return logits_to_output

    class InputToOutput:

        @staticmethod
        def standard_classification(input_to_features,
                                    features_to_logits=None,
                                    class_count=None):
            assert features_to_logits or class_count
            features_to_logits = features_to_logits or \
                Layers.FeaturesToLogits.standard_classification(class_count)
            return Layers.sequence([
                input_to_features, features_to_logits,
                Layers.LogitsToOutput.standard()
            ])

        @staticmethod
        def standard_segmentation(input_to_features,
                                  features_to_logits=None,
                                  class_count=None,
                                  output_shape=None):
            assert features_to_logits or class_count and output_shape
            features_to_logits = features_to_logits or \
                Layers.FeaturesToLogits.standard_segmentation(class_count, output_shape)
            return Layers.InputToOutput.standard_classification(
                input_to_features, features_to_logits, class_count)


# Training


def standard_loss(problem_id):
    if problem_id in ['clf', 'semseg']:
        return (losses.cross_entropy_loss, ['logits', 'label'])
    elif problem_id == 'regr':
        return (losses.mean_squared_error, ['output', 'label'])
    else:
        assert False


class TrainingComponents:

    @staticmethod
    def standard(
            batch_size,
            loss,
            weight_decay,
            optimizer,  # function lr -> optimizer(lr)
            learning_rate_policy,  # (epoch -> lr(epoch)) or float or dict
            pretrained_lr_factor=1,
            training_post_step=None):
        a = ['batch_size', 'optimizer', 'weight_decay', 'training_post_step', \
             'pretrained_lr_factor']
        args = {k: v for k, v in locals().items() if k in a}

        if type(loss) is str:
            loss = standard_loss(problem_id=loss)

        if type(learning_rate_policy) is dict:
            lrp = learning_rate_policy
            learning_rate_policy = lambda ep: \
                tf.train.piecewise_constant(ep, **lrp)
        elif type(learning_rate_policy) in [int, float]:
            learning_rate_policy = lambda ep: \
                tf.constant(learning_rate_policy)

        return TrainingComponent(
            loss=loss, learning_rate_policy=learning_rate_policy, **args)

    @staticmethod
    def ladder_densenet(epoch_count, base_learning_rate, batch_size,
                        weight_decay, pretrained_lr_factor):

        def learning_rate_policy(epoch):
            return tf.train.polynomial_decay(
                base_learning_rate,
                global_step=epoch - 1,  # epochs start at 1
                decay_steps=epoch_count,
                end_learning_rate=0,
                power=1.5)

        return TrainingComponents.standard(
            batch_size=batch_size,
            loss='semseg',
            weight_decay=weight_decay,
            optimizer=tf.train.AdamOptimizer,
            learning_rate_policy=learning_rate_policy,
            pretrained_lr_factor=pretrained_lr_factor,
            training_post_step=None)
