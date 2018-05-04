import tensorflow as tf


def accuracy(labels, predictions):
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def evaluate_semantic_segmentation(labels,
                                   predictions,
                                   class_count,
                                   returns=['m_p', 'm_r', 'm_f1', 'm_iou']):
    # pixel-level, classwise: NHWC
    labels_oh = tf.one_hot(labels, class_count, dtype=tf.float32)
    predictions_oh = tf.one_hot(predictions, class_count, dtype=tf.float32)
    labels_oh_neg = 1 - labels_oh
    predictions_oh_neg = 1 - predictions_oh

    tp = predictions_oh * labels_oh
    fp = predictions_oh * labels_oh_neg
    fn = predictions_oh_neg * labels_oh
    tn = predictions_oh_neg * labels_oh_neg

    not_ignored = tf.cast(tf.not_equal(labels, -1), dtype=tf.float32)  # NHW
    not_ignored = tf.expand_dims(not_ignored, -1)  #NHW1
    tp, fp, fn, tn = [x * not_ignored for x in [tp, fp, fn, tn]]

    # image-level, classwise: NC
    tp, fp, fn, tn = [tf.reduce_sum(x, axis=[1, 2]) for x in [tp, fp, fn, tn]]
    tp = tp
    pos = tp + fp
    true = tp + tn

    #zero = tf.zeros(tp.shape, tf.float32)  # float32
    tp_nonzero = tf.not_equal(tp, 0)

    p = tf.where(tp_nonzero, tp / pos, tp)
    r = tf.where(tp_nonzero, tp / true, tp)
    f1 = tf.where(tp_nonzero, 2 * p * r / (p + r), tp)
    iou = tf.where(tp_nonzero, tp / (pos + fn), tp)

    # image-level: N
    # TODO: weight by not_ignored_proportion?
    #not_ignored_proportion = tf.reduce_mean(not_ignored, axis=[1, 2, 3])  #N
    m_p, m_r, m_f1, m_iou = [
        tf.reduce_mean(x, axis=[-1]) for x in [p, r, f1, iou]
    ]

    # batch-level
    m_p, m_r, m_f1, m_iou = [tf.reduce_mean(x) for x in [m_p, m_r, m_f1, m_iou]]
    local_vars = locals()
    return [local_vars[name] for name in returns]
