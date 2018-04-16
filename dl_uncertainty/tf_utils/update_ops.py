import tensorflow as tf


def _multi_update(vars, update):
    with tf.control_dependencies(list(map(update, vars))):
        return tf.constant(0)


def ortho_retraction_step(weight_vars, retraction_rate):

    def update(w):
        m = tf.reshape(w, (-1, w.shape[3].value))
        mTm = tf.matmul(m, m, transpose_a=True)
        m = (1 + retraction_rate) * m - tf.matmul(m, (retraction_rate * mTm))
        return tf.assign(w, tf.reshape(m, w.shape))

    return _multi_update(weight_vars, update)


def weight_decay_step(weight_vars, decay_rate):
    """ decay_rate = lambda*learning_rate """
    return _multi_update(weight_vars,
                         lambda w: tf.assign(w, (1 - decay_rate) * w))


def normalize_kernels(weight_vars, widthwise=True):
    """  """
    def update(w):
        m = tf.reshape(w, (-1, w.shape[3].value))
        magnitudes = tf.sqrt(tf.reduce_sum(m**2, 1))
        m = m/magnitudes
        return tf.assign(w, tf.reshape(m, w.shape))

    return _multi_update(weight_vars, update)
