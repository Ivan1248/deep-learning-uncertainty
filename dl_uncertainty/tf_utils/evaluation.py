import tensorflow as tf


def mean_iou(labels, predictions, class_count):
    if len(labels.shape) > 1:
        labels = tf.reshape(labels, [-1])
        predictions = tf.reshape(predictions, [-1])
    mean_iou, update = tf.metrics.mean_iou(labels, predictions, class_count, name="mIoU")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mIoU")
    for _ in range(15):
        print(running_vars)
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    with tf.control_dependencies([running_vars_initializer]):
        return tf.identity(mean_iou)

