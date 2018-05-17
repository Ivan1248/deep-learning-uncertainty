import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import threading


class AccumulatingEvaluator:

    def accumulate(self, target, prediction):
        return self.accumulate_batch(*(np.expand_dims(x, -1)
                                       for x in [target, prediction]))

    def accumulate_batch(self, targets, predictions):
        pass

    def evaluate(self):
        pass

    def reset(self):
        pass


class DummyAccumulatingEvaluator(AccumulatingEvaluator):

    def accumulate(self, target, prediction):
        return self.accumulate_batch(*(np.expand_dims(x, -1)
                                       for x in [target, prediction]))

    def accumulate_batch(self, targets, predictions):
        pass

    def evaluate(self):
        return []

    def reset():
        pass


class NumPyClassificationEvaluator(AccumulatingEvaluator):

    def __init__(self, class_count):
        self.class_count = class_count
        self.cm = np.zeros([class_count] * 2)
        self.labels = np.arange(class_count)
        self.active = False

    def accumulate_batch(self, targets: np.ndarray, predictions):
        cm = confusion_matrix(
            targets.flatten(), predictions.flatten(), labels=self.labels)
        self.cm += cm

    def evaluate(self, returns=['A', 'mP', 'mR', 'mF1', 'mIoU']):
        # Computes macro-averaged classification evaluation metrics based on the
        # accumulated confusion matrix and clears the confusion matrix.
        assert self.active
        tp = np.diag(self.cm)
        actual_pos = self.cm.sum(axis=1)
        pos = self.cm.sum(axis=0)
        fp = pos - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            P = tp / pos
            R = tp / actual_pos
            F1 = 2 * P * R / (P + R)
            IoU = tp / (actual_pos + fp)
            P, R, F1, IoU = map(np.nan_to_num, [P, R, F1, IoU])  # 0 where tp=0
        mP, mR, mF1, mIoU = map(np.mean, [P, R, F1, IoU])
        A = tp.sum() / pos.sum()
        loc = locals()
        return [(x, loc[x]) for x in returns]

    def reset(self):
        self.cm.fill(0)


class ClassificationEvaluator(AccumulatingEvaluator):

    def __init__(self, class_count):
        self.class_count = class_count

    def __call__(self):
        self.cm = tf.get_variable(
            "cm",
            shape=[self.class_count + 1] * 2,
            trainable=False,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            dtype=tf.int32)
        return self

    def accumulate_batch(self, targets: np.ndarray, predictions):
        cm_batch = tf.confusion_matrix(
            tf.reshape(targets, [-1]) + 1,
            tf.reshape(predictions, [-1]) + 1,
            num_classes=self.class_count + 1)
        with tf.control_dependencies([cm_batch]):
            return tf.assign_add(self.cm, cm_batch)

    def evaluate(self, returns=['A', 'mIoU']):  # 'mP', 'mR', 'mF1',
        # Computes macro-averaged classification evaluation metrics based on the
        # accumulated confusion matrix and clears the confusion matrix.
        cm = self.cm[1:, 1:]
        tp = tf.diag_part(cm)
        actual_pos = tf.reduce_sum(cm, axis=1)  #  tp+fn
        pred_pos = tf.reduce_sum(cm, axis=0)  # tp+fp
        fp = pred_pos - tp
        tn = tf.reduce_sum(tp) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            P = tp / pred_pos
            R = tp / actual_pos  # TPR, sensitivity
            F1 = 2 * P * R / (P + R)
            IoU = tp / (actual_pos + fp)
            TNR = fp / (fp + tn)  # for binary classification
            tp_nonzero = tf.not_equal(tp, 0)  #
            P, R, F1, IoU, TNR = [
                tf.where(tp_nonzero, x, tf.cast(tp, tf.float64))
                for x in [P, R, F1, IoU, TNR]
            ]
        mP, mR, mF1, mIoU = map(tf.reduce_mean, [P, R, F1, IoU])
        A = tf.reduce_sum(tp) / tf.reduce_sum(pred_pos)
        loc = locals()
        return dict([(x, loc[x]) for x in returns])

    def reset(self):
        return tf.assign(self.cm, tf.zeros_like(self.cm))
