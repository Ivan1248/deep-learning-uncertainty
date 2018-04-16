import tensorflow as tf


def l2_regularization(weight_vars):
    return tf.reduce_sum(list(map(tf.nn.l2_loss, weight_vars)))


def orthogonality_penalty(weight_vars, ord='fro'):
    """
    A loss that penalizes non-orthogonal matrices with an operator norm 
    (defined with ord) of (weight_vars.T @ weight_vars - I). The norm is squared
    in case of Frobenius or 2 norm.
    :param weights: a list of convolutional kernel weights with shape 
        [ksize, ksize, in_channels, out_channels]
    :param ord: operator norm. see ord parameter in 
        https://www.tensorflow.org/api_docs/python/tf/norm
    """
    def get_loss(w):
        I = tf.eye(w.shape[-1].value)
        m = tf.reshape(w, (-1, w.shape[-1].value))
        d = None
        d = tf.norm(tf.matmul(m, m, transpose_a=True) - I, ord)
        if ord in ['fro', 2]:
            d=d**2
        return tf.reduce_sum(d)

    return tf.add_n(list(map(get_loss, weight_vars)))


def normality_penalty(weight_vars, ord='fro'):
    """
    A loss that penalizes weight matrices with non-normalized rows with an 
    operator norm (defined with ord) of (weight_vars.T @ weight_vars - I). 
    The norm is squared in case of Frobenius or 2 norm.
    :param weights: a list of convolutional kernel weights with shape 
        [ksize, ksize, in_channels, out_channels]
    :param ord: operator norm. see ord parameter in 
        https://www.tensorflow.org/api_docs/python/tf/norm
    """
    return None
