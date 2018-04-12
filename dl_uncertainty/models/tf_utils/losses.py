import tensorflow as tf

def multiclass_hinge_loss(labels_oh, logits, delta=1):
    # TODO: optimize
    #labels = tf.cast(tf.argmax(labels_oh, 1), tf.int32)
    #p = tf.dynamic_partition(logits, labels, labels_oh.shape[1])
    #s_true, s_other = p[0], tf.concat(p[1:], axis=1)
    s_true = tf.reduce_sum(logits * labels_oh, axis=1, keep_dims=True)
    r = (logits - s_true + delta) * (1 - labels_oh)
    return tf.reduce_mean(tf.reduce_sum(tf.nn.relu(r), axis=1))
