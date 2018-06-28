import argparse

import numpy as np

from _context import dl_uncertainty

from dl_uncertainty.data import datasets
from dl_uncertainty import data_utils
from dl_uncertainty.utils.visualization import view_predictions
from dl_uncertainty.processing.data_augmentation import random_fliplr, augment_cifar

# python view_dataset.py
#   voc2012 test
#   cityscapes val
#   mozgalo train

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('part', type=str)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--permute', action='store_true')
args = parser.parse_args()

if args.ds == 'wilddash':
    names = ['val', 'bench']
    datasets = data_utils.get_cached_dataset_with_normalized_inputs(args.ds)
    datasets = dict(zip(names, datasets))
    ds = datasets[args.part]
elif args.part.startswith('test'):
    ds = data_utils.get_cached_dataset_with_normalized_inputs(
        args.ds, trainval_test=True)[1]
else:
    ds_train, ds_val = data_utils.get_cached_dataset_with_normalized_inputs(
        args.ds, trainval_test=False)
    ds = {
        'train': ds_train,
        'val': ds_val,
        'trainval': ds_train + ds_val,
    }[args.part]

if 'class_count' not in ds.info:
    ds.info['class_count'] = 2

if args.augment:
    ds = ds.map(data_utils.get_augmentation_func(ds))

if args.permute:
    ds = ds.permute()

problem_id = ds.info['problem_id']

view_predictions(ds, infer=lambda x: ds[0][1])