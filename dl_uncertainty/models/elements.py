import tensorflow as tf
from .tf_utils import layers, losses

# Head


def global_pool_affine_logits_func(class_count):

    def f(h, **k):
        h = tf.reduce_mean(h, axis=[1, 2], keep_dims=True)
        h = layers.conv(h, 1, class_count, bias=True, name='conv_logits')
        return tf.reshape(h, [-1, class_count])

    return f


def semseg_affine_logits_func(class_count):

    def f(h, **k):
        return layers.conv(h, 1, class_count, bias=True, name='conv_logits')

    return f


def semseg_affine_resized_logits_func(class_count, spatial_shape):

    def f(h, **k):
        h = layers.conv(h, 1, class_count, bias=True, name='conv_logits')
        return tf.image.resize_bilinear(h, spatial_shape)

    return f


def semseg_affine_resized_probs_func(spatial_shape):

    def f(h):
        h = tf.nn.softmax(h)
        return tf.image.resize_bilinear(h, spatial_shape)

    return f


# Whole network


def clf_input_to_output_func(input_to_features,
                             features_to_logits=None,
                             logits_to_probs=None,
                             class_count=None):
    """
    Creates a function that maps the network input to a single output tensor.
    :param input_to_features: a function that returns 1 or more tensors.
    :param features_to_logits: a function that returns 1 or more tensors. argmax
        of the first of the returned values is taken as the classification 
        output. Default: global pooling followed by a 1x1 convolution.
    :param logits_to_probs: a function that maps 1 logits tensor to 1 probs 
        tensor. Default: tf.nn.softmax.
    :param class_count: needs to be given if features_to_logits is None.
    """
    assert features_to_logits or class_count
    if features_to_logits is None:
        features_to_logits = global_pool_affine_logits_func(class_count)
    logits_to_probs = logits_to_probs or tf.nn.softmax

    def input_to_output(input, is_training, pretrained_lr_factor):
        features = input_to_features(
            input,
            is_training=is_training,
            pretrained_lr_factor=pretrained_lr_factor)
        additional_outputs = {'features': features}

        all_logits = features_to_logits(features, is_training=is_training)
        if type(all_logits) not in [list, tuple]:
            all_logits = [all_logits]
        all_probs = list(map(logits_to_probs, all_logits))

        output = tf.argmax(all_logits[0], -1, output_type=tf.int32)
        additional_outputs.update({
            'logits': all_logits[0],
            'probs': all_probs[0],
            'all_logits': all_logits,
            'all_probs': all_probs,
        })
        return output, additional_outputs

    return input_to_output


def semseg_input_to_output_func(input_shape,
                                input_to_features,
                                features_to_logits=None,
                                logits_to_probs=None,
                                class_count=None,
                                resize_probs=False):
    assert features_to_logits or class_count
    if features_to_logits is None:
        if resize_probs:
            features_to_logits = semseg_affine_logits_func(class_count)
            logits_to_probs = logits_to_probs or \
                              semseg_affine_resized_probs_func(input_shape[0:2])
        else:
            features_to_logits = semseg_affine_resized_logits_func(
                class_count, input_shape[0:2])
    return clf_input_to_output_func(input_to_features, features_to_logits,
                                    logits_to_probs)


# Loss


def standard_loss(problem_id):
    if problem_id in ['clf', 'semseg']:
        return (losses.cross_entropy_loss, ['logits', 'label'])
    elif problem_id == 'regr':
        return (losses.mean_squared_error, ['output', 'label'])
    else:
        assert False