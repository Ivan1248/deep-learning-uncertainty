import argparse
import datetime

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, aupr
import matplotlib.pyplot as plt

from _context import dl_uncertainty

from dl_uncertainty.data import DataLoader
from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.utils.visualization import view_predictions
from dl_uncertainty.processing.shape import fill_to_shape

# Use "--trainval" only for training on "trainval" and testing "test".
# CUDA_VISIBLE_DEVICES=0 python train_ood.py
# CUDA_VISIBLE_DEVICES=1 python train_ood.py
# CUDA_VISIBLE_DEVICES=2 python train_ood.py
#   cifar wrn 28 10
#   cifar dn 100 12
#   cifar rn 34 8
#   cityscapes dn 121 32  /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/dn-121-32-pretrained-e30/2018-05-16-1623/Model
#   mozgalo rn 50 64
#   mozgalo rn 18 64

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('saved_path', type=str)
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--mcdropout', action='store_true')
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--test_on_training_set', action='store_true')
args = parser.parse_args()
print(args)

# Cached dataset with normalized inputs

print("Setting up data loading...")
ds_id_to_dataset = {
    ds_id: data_utils.get_cached_dataset_with_normalized_inputs(
        args.ds, trainval_test=args.trainval)[1]
    for ds_id in ['cifar', 'mozgalo', 'camvid']
}

ds = ds_id_to_ds[args.ds]
del ds_id_to_ds[args.ds]

shape = ds[0][0].shape
for k, v in ds_id_to_ds:
    ds_id_to_ds[k] = v.map(lambda x: fill_to_shape(x, shape), 0)

ds_train, ds_test = data_utils.get_cached_dataset_with_normalized_inputs(
    args.ds, trainval_test=args.trainval)

# Model

print("Initializing model and loading state...")
model = model_utils.get_model(
    net_name=args.net,
    ds_train=ds_train,
    depth=args.depth,
    width=args.width,  # width factor for WRN, base_width for others
    epoch_count=args.epochs,
    dropout=args.dropout or args.mcdropout,
    pretrained=args.pretrained)
model.load_state(args.saved_path)


def get_misclassification_example(xy):
    x, y = xy
    logits, output = model.predict(x, outputs=['logits', 'output'])
    return logits, int(output != y)


def to_logit_out_distribution_label(xy):
    logits = model.predict(xy[0], outputs=['logits'])
    return logits, 1


def to_logit_in_distribution_label(xy):
    logits = model.predict(xy[0], outputs=['logits'])
    return logits, 0


ds_id_to_logits_ds = {
    k: v.map(get_misclassification_example)
    for k, v in ds_id_to_ds.items()
}

logits_ds_test = ds_test.map(get_misclassification_example)

max_logits_ds_test = np.array([y for x, y in logits_ds_test])

ds_id_to_max_logits_ds = {
    k: v.map(lambda xy: (np.max(xy[0]), xy[1]))
    for k, v in ds_id_to_logits_ds.items()
}


def max_logit_ood_classifier(threshold):
    return lambda x: np.max(x) < threshold


max_logits_range = (np.min(max_logits_ds_test), np.max(max_logits_ds_test))
thresholds = np.linspace(*max_logits_range, 50)

y_true_in, y_score_in = zip(* [(x, y) for x, y in max_logits_ds_test.items()])

for ds_id, ds in ds_id_to_max_logits_ds.items():
    y_true_out, y_score_out = zip(* [(x, y) for x, y in ds.items()])
    y_true = y_true_in + y_true_out
    y_score = y_score_in + y_score_out

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    plt.plot(fpr, tpr)
