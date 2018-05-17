import argparse

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
parser.add_argument('--augment', action='store_true')
args = parser.parse_args()

ds_train, ds_val = data_utils.get_dataset(args.ds, trainval_test=False)
ds_trainval, ds_test = data_utils.get_dataset(args.ds, trainval_test=True)
ds = {
    'train': ds_train,
    'val': ds_val,
    'trainval': ds_train + ds_val,
    'test': ds_test
}[args.part]

if args.augment:
    ds = ds.map(data_utils.get_augmentation_func(ds))

problem_id = ds.info['problem_id']

visualization.view_semantic_segmentation(ds, infer=lambda x: ds[0][1])