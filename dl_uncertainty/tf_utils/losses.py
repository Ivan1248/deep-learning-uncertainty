import tensorflow as tf


def multiclass_hinge_loss(labels_oh, logits, delta=1):
    # TODO: optimize
    #labels = tf.cast(tf.argmax(labels_oh, 1), tf.int32)
    #p = tf.dynamic_partition(logits, labels, labels_oh.shape[1])
    #s_true, s_other = p[0], tf.concat(p[1:], axis=1)
    s_true = tf.reduce_sum(logits * labels_oh, axis=1, keepdims=True)
    r = (logits - s_true + delta) * (1 - labels_oh)
    return tf.reduce_mean(tf.reduce_sum(tf.nn.relu(r), axis=1))


def cross_entropy_loss(logits, labels_oh):
    class_count = logits.shape[-1].value
    if len(logits.shape) > 2:
        logits = tf.reshape(logits, [-1, class_count])
        labels_oh = tf.reshape(labels_oh, [-1, class_count])
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels_oh)
    label_count = tf.reduce_sum(labels_oh)
    loss = tf.reduce_sum(loss) / label_count
    return loss