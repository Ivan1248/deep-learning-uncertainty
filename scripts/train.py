import os
import argparse
import datetime

import numpy as np

from _context import dl_uncertainty

from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.processing.data_augmentation import random_fliplr_with_label, augment_cifar
from dl_uncertainty import parameter_loading

# Use "--trainval" only for training on "trainval" and testing "test".
# CUDA_VISIBLE_DEVICES=0 python train.py
# CUDA_VISIBLE_DEVICES=1 python train.py
# CUDA_VISIBLE_DEVICES=2 python train.py
#   cifar wrn 28 10 --epochs 200 --trainval
#   cifar dn 100 12 --epochs 300 --trainval
#   cifar rn 34 8   --epochs 200 --trainval
#   cityscapes dn 121 32  --pretrained --epochs 30
#   cityscapes rn 50 64   --pretrained --epochs 30
#   cityscapes ldn 121 32 --pretrained --epochs 30
#   mozgalo rn 50 64 --pretrained --epochs 10 --trainval
#   mozgalo rn 18 64 --epochs 15 --trainval

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')  # for wrn
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
args = parser.parse_args()
print(args)

assert not args.dropout, "Not implemented"

# Cached dataset with normalized inputs

print("Setting up data loading...")
ds_train, ds_test = data_utils.get_cached_dataset_with_normalized_inputs(
    args.ds, trainval_test=args.trainval)

# Model

print("Initializing model...")
model = model_utils.get_model(
    net_name=args.net,
    ds_train=ds_train,
    depth=args.depth,
    width=args.width,  # width factor for WRN, base_width for others
    epoch_count=args.epochs,
    pretrained=args.pretrained)

# Training

print("Starting training and validation loop...")
if args.ds in ['cifar']:
    jitter = lambda xy: (augment_cifar(xy[0]), xy[0])
elif ds_train.info['problem_id'] == 'semseg':
    jitter = random_fliplr_with_label
else:
    jitter = lambda x: x

training.train(
    model,
    ds_train,
    ds_test,  # 25
    jitter=jitter,
    epoch_count=args.epochs,
    data_loading_worker_count=4)

# Saving

print("Saving...")
model_utils.save_trained_model(
    model,
    ds_id=ds_train.info['id'] + ('-trainval' if args.trainval else '-train'),
    net_name=f"{args.net}-{args.depth}-{args.width}",
    epoch_count=args.epochs,
    dropout=args.dropout,
    pretrained=args.pretrained)