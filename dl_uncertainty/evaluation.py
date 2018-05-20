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

    def evaluate(self, returns=['A', 'mP', 'mR', 'mF1',
                                'mIoU']):  # 'mP', 'mR', 'mF1'
        # Computes macro-averaged classification evaluation metrics based on the
        # accumulated confusion matrix and clears the confusion matrix.
        cm = self.cm[1:, 1:]
        actual_pos = tf.reduce_sum(cm, axis=1)  #  tp+fn
        pred_pos = tf.reduce_sum(cm, axis=0)  # tp+fp
        tp = tf.diag_part(cm)
        fp = pred_pos - tp

        P = tp / pred_pos
        R = tp / actual_pos  # TPR, sensitivity
        F1 = 2 * P * R / (P + R)
        IoU = tp / (actual_pos + fp)
        tp_nonzero = tf.not_equal(tp, 0)  #
        tp_float = tf.cast(tp, tf.float64)
        P, R, F1, IoU = [
            tf.where(tp_nonzero, x, tp_float) for x in [P, R, F1, IoU]
        ]

        #tn = tf.reduce_sum(tp) - tp
        #TNR = fp / (fp + tn)  # for binary classification
        #TNR = tf.where(tf.not_equal(fp, 0), TNR, tf.cast(fp, tf.float64))

        mP, mR, mF1, mIoU = map(tf.reduce_mean, [P, R, F1, IoU])
        A = tf.reduce_sum(tp) / tf.reduce_sum(pred_pos)
        name_to_tensor = locals()
        return dict([(x, name_to_tensor[x]) for x in returns])

    def reset(self):
        return tf.assign(self.cm, tf.zeros_like(self.cm))


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


def compute_errors(conf_mat, name="name", verbose=True):
    # from Ivan Krešo
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFN = conf_mat.sum(0)
    TPFP = conf_mat.sum(1)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    if verbose:
        print(name + ' errors:')
    for i in range(num_classes):
        TP = conf_mat[i, i]
        class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0

        class_name = f"class{i}"
        if verbose:
            print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    if verbose:
        print(name + ' IoU mean class accuracy - TP / (TP+FN+FP) = %.2f %%' %
              avg_class_iou)
        print(name +
              ' mean class recall - TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print(name + ' mean class precision - TP / (TP+FP) = %.2f %%' %
              avg_class_precision)
        print(name + ' pixel accuracy = %.2f %%' % avg_pixel_acc)
    return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size
