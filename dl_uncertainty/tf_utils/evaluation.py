import tensorflow as tf


def mean_iou(labels, predictions):
    if len(labels.shape) > 1:
        labels = tf.reshape(labels, [-1])
        predictions = tf.reshape(predictions, [-1])
    mean_iou, update = tf.metrics.mean_iou(labels, predictions, name="moiu")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="miou")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    with tf.control_dependencies([running_vars_initializer]):
        return tf.identity(mean_iou)

