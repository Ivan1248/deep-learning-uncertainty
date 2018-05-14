import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_similarity_score

from _context import dl_uncertainty
from dl_uncertainty.evaluation import NumPyClassificationEvaluator

#### begin from https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_semantic_segmentation.py
import six


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 0
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class**2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion


def calc_semantic_segmentation_iou(confusion):
    iou_denominator = (
        confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou


def eval_semantic_segmentation(pred_labels, gt_labels):
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

    return {
        'iou': iou,
        'miou': np.nanmean(iou),
        'pixel_accuracy': pixel_accuracy,
        'class_accuracy': class_accuracy,
        'mean_class_accuracy': np.nanmean(class_accuracy)
    }


#### end

class_count = 20
labels = np.arange(class_count)
evaluator = ClassificationEvaluator(class_count)

N = 10000
targ = np.random.random_integers(low=0, high=class_count - 1, size=(10000))
pred = np.random.random_integers(low=0, high=class_count - 1, size=(10000))

for i in range(N):
    if np.random.rand() < 0.8 and pred[i] <= class_count // 2:
        targ[i] = pred[i]
    if np.random.rand() < 0.5:
        targ[i] = -1

mask = targ >= 0

evaluator.accumulate_batch(targ, pred)

results = evaluator.evaluate()
results = dict(results)


name_to_func = {
    'A': lambda t, p, **k: accuracy_score(t[mask], p[mask]),
    'mP': lambda t, p, **k: precision_score(t[mask], p[mask], **k),
    'mR': lambda t, p, **k: recall_score(t[mask], p[mask], **k),
    'mF1': lambda t, p, **k: f1_score(t[mask], p[mask], **k),
    'mIoU': lambda t, p, **k: eval_semantic_segmentation(np.reshape(p, [1,1,-1]), np.reshape(t, [1,1,-1]))['miou'],
}

for name, metric in name_to_func.items():
    m = lambda t, p: metric(t, p, average='macro', labels=labels)
    r, m = results[name], m(targ, pred)
    print(f"{name}: {r} == {m}?")
    assert np.isclose(r, m, rtol=1e-8), f"{name}: {r} != {m}"
