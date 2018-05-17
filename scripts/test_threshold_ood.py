import argparse
import datetime

import numpy as np

from _context import dl_uncertainty

from dl_uncertainty.data import DataLoader
from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.visualization import view_semantic_segmentation

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, aupr

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
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('saved_path', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--test_on_training_set', action='store_true')
args = parser.parse_args()
print(args)

# Cached dataset with normalized inputs

print("Setting up data loading...")
datasets = {id: data_utils.get_cached_dataset_with_normalized_inputs(
    args.ds, trainval_test=args.trainval) for id in ['cifar', 'mozgalo', '']}

ds_train, ds_test = data_utils.get_cached_dataset_with_normalized_inputs(
    args.ds, trainval_test=args.trainval)

# Model

print("Initializing model and loading state...")
model = model_utils.get_model(
    net_name=args.net, ds_train=ds_train, depth=args.depth, width=args.width)
model.load_state(args.saved_path)


def get_misclassification_example(xy):
    x, y = xy
    logits, output = model.predict(x, outputs=['logits', 'output'])
    return logits, int(output == y)


ds_test_logits = ds_test.map(get_misclassification_example)

def roc(ds, thresholds):
    pass