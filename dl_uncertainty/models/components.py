import tensorflow as tf

from .tf_utils import layers, losses
from .modeldef import TrainingComponent
from ..utils.collections import UnoverwritableDict

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
    def dummy(x, **c):
        # k is a dictionary containing graph nodes and/or parameters
        y = x
        return y, {}  # the main output, and optional auxiliary outputs

    @staticmethod
    def pull_node(node):

        def total_func(_, **c):
            if type(node) in [list, tuple]:
                return type(node)(c[n] for n in node), {}

            return c[node], {}

        return total_func

    @staticmethod
    def sequence(layer_funcs):

        def total_func(x, **c):
            c_local = UnoverwritableDict(c)
            for f in layer_funcs:
                x, additional = f(x, **c_local)
                c_local.update(additional)
            return x, {k: v for k, v in c_local.items() if k not in c}

        return total_func

    class Features:
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

            def input_to_features(x, is_training, dropout_active,
                                  pretrained_lr_factor, **k):
                x = layers.resnet_root_block(
                    x, base_width=base_width, cifar_type=cifar_root_block)
                features = layers.resnet_middle(
                    x,
                    **params,
                    bn_params={'is_training': is_training},
                    dropout_params={
                        'rate': dropout_rate,
                        'active': dropout_active
                    },
                    pretrained_lr_factor=pretrained_lr_factor)
                return features, {'features': features}

            return input_to_features

        @staticmethod
        def densenet(base_width, group_lengths, block_structure,
                     cifar_root_block, dropout_rate):
            p = ['base_width', 'group_lengths', 'block_structure']
            params = {k: v for k, v in locals().items() if k in p}

            def input_to_features(x, is_training, dropout_active,
                                  pretrained_lr_factor, **k):
                x = layers.densenet_root_block(
                    x, base_width=base_width, cifar_type=cifar_root_block)
                features = layers.densenet_middle(
                    x,
                    **params,
                    bn_params={'is_training': is_training},
                    dropout_params={
                        'rate': dropout_rate,
                        'active': dropout_active
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

            def input_to_features(x, is_training, dropout_active,
                                  pretrained_lr_factor, **k):
                features = layers.ladder_densenet(
                    x,
                    **params,
                    bn_params={'is_training': is_training},
                    dropout_params={
                        'rate': dropout_rate,
                        'active': dropout_active
                    },
                    pretrained_lr_factor=pretrained_lr_factor)
                return features, {'features': features}

            return input_to_features

    class Logits:

        @staticmethod
        def classification(class_count, logits_name='logits'):

            def f(h, **c):
                h = tf.reduce_mean(
                    h, axis=[1, 2], keep_dims=True)  # global pooling
                h = layers.conv(
                    h, 1, class_count, bias=True, name='logits/conv')
                logits = tf.reshape(h, [-1, class_count])
                return logits, {logits_name: logits}

            return f

        @staticmethod
        def segmentation(class_count, resize=True, logits_name='logits'):

            def f(h, **c):
                logits = layers.conv(
                    h, 1, class_count, bias=True, name='logits/conv')
                if resize:
                    logits = tf.image.resize_bilinear(logits, c['image_shape'])
                return logits, {logits_name: logits}

            return f

        @staticmethod
        def segmentation_multi(class_count, resize=True, shape=None):

            def f(all_pre_logits, **c):
                # Accepts multiple feature tensors as the first parameter and
                # context **k. Returns (resized) logits for each feature tensor.
                all_logits = []
                for i, x in enumerate(all_pre_logits):
                    logits = layers.conv(
                        x, 1, class_count, bias=True, name=f'logits{i}/conv')
                    if resize:
                        logits = tf.image.resize_bilinear(
                            logits, c['image_shape'])
                    all_logits.append(logits)
                return all_logits, {
                    'logits': all_logits[0],
                    'all_logits': all_logits
                }

            return f

        @staticmethod
        def segmentation_gaussian(class_count, shape=None, sample_count=50):

            def f(h, **c):
                logits = layers.conv(
                    h, 1, class_count * 2, bias=True, name='logits/conv')
                if shape:
                    logits = tf.image.resize_bilinear(logits, shape)

                logits_mean = logits[..., :class_count]  # NHWC
                logits_logvar = logits[..., class_count:]  # NHWC
                logits_var = tf.exp(logits_logvar)
                logits_std = tf.sqrt(logits_var)
                # TODO: try log(1+exp(logits[..., class_count:])) for logits_std

                samp_shape = tf.concatenate(
                    [sample_count, tf.shape(logits_logvar)])
                noise = tf.random.normal(samp_shape)  # nNHWC
                logits_samples = logits_mean + noise * logits_std  # nNHWC

                return logits_samples, {
                    'logits_mean': logits_mean,
                    'logits_logvar': logits_logvar,
                    'logits_var': logits_var,
                    'logits_std': logits_std,
                    'logits_samples': logits_samples
                }

            return f

    class Output:

        @staticmethod
        def argmax_and_probs_from_logits(output_name='output',
                                         probs_name='probs'):

            def logits_to_output(x, **c):
                output = tf.argmax(x, axis=-1, output_type=tf.int32)
                probs = tf.nn.softmax(x)
                return output, {output_name: output, probs_name: probs}

            return logits_to_output

        @staticmethod
        def argmax_and_probs_from_gaussian_logits(
                logits_samples_name='logits_samples',
                output_name='output',
                probs_name='probs'):

            def f(_, **c):
                logits_samples = c[logits_samples_name]
                probs_samples = tf.nn.softmax(logits_samples)
                probs = tf.reduce_mean(probs_samples, 0)
                output = tf.argmax(probs, -1)
                return probs, {output_name: output, probs_name: probs}

            return f

    class ProbsUncertainty:

        @staticmethod
        def entropy(probs_name='probs'):

            def f(_, **c):
                p = c[probs_name]
                pe = tf.reduce_sum(-p * tf.log(p), axis=-1)
                return pe, {f'{probs_name}_entropy': pe}

            return f

    class Loss:

        @staticmethod
        def cross_entropy_loss():
            return lambda logits, label, **k: losses.cross_entropy_loss(logits, label)

        @staticmethod
        def cross_entropy_loss_gaussian_logits_temporary(sample_count=50):

            def loss(_, logits_samples, label, **c):
                probs = tf.reduce_mean(tf.nn.softmax(logits_samples), 0)
                loss = -probs * tf.log(probs)

            return loss

        @staticmethod
        def cross_entropy_loss_gaussian_logits():

            def loss(_, logits_samples, label, **c):
                sample_count = logits_samples.shape[0]
                class_count = logits_samples.shape[-1]
                logits_samples = tf.reshape(logits_samples,
                                            [sample_count, -1, class_count])
                label = tf.reshape(label, [1, -1])
                logsumexp_logits_samples = \
                    tf.log(tf.reduce_sum(tf.exp(logits_samples), -1))
                true_class_logits = tf.gather_nd(logits_samples, label)
                loss = tf.exp(true_class_logits - logsumexp_logits_samples)
                loss = tf.reduce_mean(loss, axis=0)
                loss = tf.reduce_mean(tf.log(loss))

            return loss

        @staticmethod
        def cross_entropy_loss_multi(weights):

            def loss(all_logits, label, **c):
                if type(all_logits) not in [list, tuple]:
                    all_logits = [all_logits]
                assert len(all_logits) == len(weights)
                return sum(w * losses.cross_entropy_loss(lg, label)
                           for w, lg in zip(weights, all_logits))

            return loss


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
                        weight_decay, pretrained_lr_factor, loss_weights):

        def learning_rate_policy(epoch):
            return tf.train.polynomial_decay(
                base_learning_rate,
                global_step=epoch - 1,  # epochs start at 1
                decay_steps=epoch_count,
                end_learning_rate=0,
                power=1.5)

        return TrainingComponents.standard(
            batch_size=batch_size,
            loss=(Layers.Loss.cross_entropy_loss_multi(loss_weights),
                  ['all_logits', 'label']),
            weight_decay=weight_decay,
            optimizer=tf.train.AdamOptimizer,
            learning_rate_policy=learning_rate_policy,
            pretrained_lr_factor=pretrained_lr_factor,
            training_post_step=None)
