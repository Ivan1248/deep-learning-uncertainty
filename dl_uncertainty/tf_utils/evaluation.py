import tensorflow as tf


def accuracy(labels, predictions):
    labels = tf.reshape(labels, [-1])
    predictions = tf.reshape(predictions, [-1])
    not_ignored = tf.cast(tf.not_equal(labels, -1), dtype=tf.float32)
    equal = tf.cast(tf.equal(labels, predictions), dtype=tf.float32)
    return tf.reduce_sum(equal) / tf.reduce_sum(not_ignored)


def multiclass_scores(labels,
                      predictions,
                      class_count,
                      returns=['mP', 'mR', 'mF1', 'mIoU']):
    # gives incorrect values
    # pixel-level: NHW -> N*H*W =: P
    labels = tf.reshape(labels, [-1])
    predictions = tf.reshape(labels, [-1])

    # pixel-level, classwise: P -> PC
    labels_oh = tf.one_hot(labels, class_count, dtype=tf.float32)  # PC
    predictions_oh = tf.one_hot(predictions, class_count, dtype=tf.float32)
    labels_oh_neg = 1 - labels_oh
    predictions_oh_neg = 1 - predictions_oh

    tp = predictions_oh * labels_oh
    fp = predictions_oh * labels_oh_neg
    fn = predictions_oh_neg * labels_oh
    tn = predictions_oh_neg * labels_oh_neg
    not_ignored = tf.cast(tf.not_equal(labels, -1), dtype=tf.float32)  # P
    not_ignored = tf.expand_dims(not_ignored, -1)  #P1
    tp, fp, fn, tn = [x * not_ignored for x in [tp, fp, fn, tn]]  # PC

    # classwise: PC -> C
    tp, fp, fn, tn = [tf.reduce_sum(x, axis=0) for x in [tp, fp, fn, tn]]  # C
    pred_pos = tp + fp
    real_pos = tp + fn
    
    p = tp / pred_pos
    r = tp / real_pos
    f1 = 2 * p * r / (p + r)
    iou = tp / (tp + fp + fn)
    tp_nonzero = tf.not_equal(tp, 0)  #
    p, r, f1, iou = [tf.where(tp_nonzero, x, tp) for x in [p, r, f1, iou]]

    # mean: 1
    mP, mR, mF1, mIoU = [tf.reduce_mean(x) for x in [p, r, f1, iou]]

    local_vars = locals()
    return [local_vars[name] for name in returns]