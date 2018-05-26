import tensorflow as tf
import numpy as np


def multiclass_hinge_loss(labels_oh, logits, delta=1):
    # TODO: optimize
    #labels = tf.cast(tf.argmax(labels_oh, 1), tf.int32)
    #p = tf.dynamic_partition(logits, labels, labels_oh.shape[1])
    #s_true, s_other = p[0], tf.concat(p[1:], axis=1)
    s_true = tf.reduce_sum(logits * labels_oh, axis=1, keepdims=True)
    r = (logits - s_true + delta) * (1 - labels_oh)
    return tf.reduce_mean(tf.reduce_sum(tf.nn.relu(r), axis=1))


def cross_entropy_loss(logits, labels, reduce_mean=True):
    class_count = logits.shape[-1].value
    labels_oh = tf.one_hot(labels, class_count)  # NHWC -> MC, M=N*H*W
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels_oh)  # M
    nonignored_labels = tf.cast(tf.greater_equal(labels, 0), tf.float32)
    loss = loss * nonignored_labels  # NOTE: IMPORTANT
    label_count = tf.reduce_sum(nonignored_labels)
    if reduce_mean:
        return tf.reduce_sum(loss) / label_count
    return loss * (tf.size(labels) / label_count)  # N


def weighted_cross_entropy_loss(logits, labels):
    class_count = logits.shape[-1].value
    labels = tf.reshape(labels, [-1])
    onehot_labels = tf.one_hot(labels, class_count)
    logits = tf.reshape(logits, [-1, class_count])
    xent = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=onehot_labels)
    num_labels = tf.reduce_sum(onehot_labels)
    class_weights = tf.ones([class_count])
    class_weights = tf.concat([[0], class_weights], axis=0)
    weights = tf.gather(class_weights, labels + 1)
    xent = tf.multiply(weights, xent)
    xent = tf.reduce_sum(xent) / num_labels
    return xent


def cross_entropy_loss_p(probs, labels, reduce_mean=True):  # TODO: fix
    class_count = probs.shape[-1].value
    logits_oh = tf.one_hot(labels, class_count)
    loss = -logits_oh * tf.log(probs)
    label_count = tf.reduce_sum(logits_oh)  # TODO: make more efficient
    if reduce_mean:
        return tf.reduce_sum(loss) / label_count  # 1
    return loss * (tf.size(labels) / label_count)  # N


def class_distribution_weighted_cross_entropy_loss(
        logits, labels, class_distribution=None, eps=1e-3,
        reduce_mean=True):  # TODO: fix
    class_count = logits.shape[-1].value
    if len(logits.shape) > 2:
        logits = tf.reshape(logits, [-1, class_count])
        labels = tf.reshape(labels, [-1])
    labels_oh = tf.one_hot(labels, class_count)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels_oh)
    if class_distribution is None:
        class_distribution = tf.reduce_sum(labels_oh, axis=0) + \
            eps * tf.cast(tf.shape(loss)[0], tf.float32)  # C
        class_distribution /= tf.reduce_sum(class_distribution)  # C
    weights = tf.concat([[0], tf.reciprocal(class_distribution)], axis=0)
    loss = loss * tf.gather(weights, labels + 1)  # gather
    label_count = tf.reduce_sum(labels_oh)
    if reduce_mean:
        return tf.reduce_sum(loss) / label_count
    loss = tf.reshape(loss, labels.shape)
    return loss * (tf.size(labels) / label_count)


def gaussian_logit_expected_cross_entropy_loss(logits_means,
                                               logit_log_variances,
                                               labels,
                                               sample_count=50,
                                               reduce_mean=True):  # TODO: fix
    """
    NOTE: NOT EQUIVALENT to eq. 11 or eq. 12 in https://arxiv.org/abs/1703.04977
    :param logits: Tensor. N[HW]C
    :param logit_log_variances: Tensor. N[HW]1
    :param labels: Tensor. N[HW]C
    """
    stddevs = tf.exp(0.5 * logit_log_variances)

    def sample_loss(*args):  # args are ignored
        logits = tf.random_normal(
            shape=logits_means.shape, mean=logits_means, stddev=stddevs)
        return cross_entropy_loss(logits, labels, reduce_mean=reduce_mean)

    map_hack = tf.constant(0, shape=[sample_count], dtype=logits_means.dtype)
    loss_samples = tf.map_fn(sample_loss, map_hack)

    return tf.reduce_mean(loss_samples, axis=0)


def gaussian_logit_expected_probs_cross_entropy_loss(logits_means,
                                                     logit_log_variances,
                                                     labels,
                                                     sample_count=50,
                                                     reduce_mean=True,
                                                     eps=1e-5):  # TODO: fix
    """
    NOTE: EQUIVALENT to eq. 11 or eq. 12 in https://arxiv.org/abs/1703.04977
    TODO: check numerical stability and make it more stable as in eq. 12
    :param logits: Tensor. N[HW]C
    :param logit_log_variances: Tensor. N[HW]1
    :param labels: Tensor. N[HW]C
    """
    stddevs = tf.exp(0.5 * logit_log_variances)  # exp(1+log(x))) ?

    def sample_probs(*args):  # args are ignored
        logits = tf.random_normal(
            shape=logits_means.shape, mean=logits_means, stddev=stddevs)
        return tf.nn.softmax(logits)

    map_hack = tf.constant(0, shape=[sample_count], dtype=logits_means.dtype)
    probs_samples = tf.map_fn(sample_probs, map_hack)
    expected_probs = tf.reduce_mean(probs_samples)

    class_count = logits_means.shape[-1].value
    labels_oh = tf.one_hot(labels, class_count)
    axis = None or reduce_mean and np.arange(1, len(labels_oh.shape))
    return -tf.reduce_mean(tf.log(expected_probs + eps) * labels_oh, axis=axis)


def temperature_uncertainty_cross_entropy_loss(logits, log_temperatures,
                                               labels):  # TODO: fix
    """
    https://arxiv.org/abs/1705.07115
    :param logits: Tensor. N[HW]C
    :param log_temperatures: Tensor. N[HW]1
    :param labels: Tensor. N[HW]C
    """
    class_count = logits.shape[-1].value
    labels_oh = tf.one_hot(labels, class_count)
    valid_labels = tf.reduce_sum(labels_oh, axis=-1)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels_oh)
    loss /= tf.exp(log_temperatures)
    loss -= tf.log(tf.reduce_sum(tf.exp(logits), axis=-1)) * valid_labels

    label_count = tf.reduce_sum(valid_labels)
    return loss * (tf.size(labels) / label_count)


def mean_squared_error(output, label):
    return tf.reduce_mean((output - label)**2)
