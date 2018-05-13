import os
import argparse
import datetime

import numpy as np

from _context import dl_uncertainty

from dl_uncertainty import data_utils, visualization
from dl_uncertainty.processing.data_augmentation import random_fliplr, augment_cifar

# python view_dataset.py
#   voc2012 test
#   cityscapes val
#   mozgalo train

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('part', type=str)
parser.add_argument('--augmentation', action='store_false')
args = parser.parse_args()

# Dataset

ds_train, ds_val = data_utils.get_dataset(args.ds, trainval_test=False)
ds_trainval, ds_test = data_utils.get_dataset(args.ds, trainval_test=True)
ds_trainval = ds_train.join(ds_val)
ds = {
    'train': ds_train,
    'val': ds_val,
    'trainval': ds_train.join(ds_val),
    'test': ds_test
}[args.part]

problem_id = ds.info['problem_id']

visualization.view_semantic_segmentation(ds)