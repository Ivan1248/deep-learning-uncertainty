import os
import argparse
import datetime

import numpy as np

from _context import dl_uncertainty

from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.processing.data_augmentation import random_fliplr, augment_cifar
from dl_uncertainty import parameter_loading

# Use "--test" only for training on "trainval" and testing "test".
# CUDA_VISIBLE_DEVICES=0 python train.py
#   cifar10 wrn 28 10 --epochs 200 --test
#   cifar10 dn 100 12 --epochs 300 --test
#   cifar10 rn 34 8   --epochs 200 --test
#   cityscapes dn 121 32  --pretrained --epochs 30
#   cityscapes rn 50 64   --pretrained --epochs 30
#   cityscapes ldn 121 32 --pretrained --epochs 30
#   mozgalo rn 50 64 --pretrained --epochs 10 --test

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--nodropout', action='store_true')  # for wrn
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
args = parser.parse_args()
print(args)

# Dataset

print("Setting up data loading...")
ds_train, ds_test = data_utils.get_dataset(args.ds, args.test)
problem_id = ds_train.info['problem_id']

# Input normalization and data caching

cache_dir = f"{dirs.CACHE}"

print("Setting up data preprocessing...")
normalizer = data_utils.LazyNormalizer(ds_train, cache_dir)
ds_train = ds_train.map(normalizer.normalize, 0)
ds_test = ds_test.map(normalizer.normalize, 0)

print("Setting up data caching on HDD...")
ds_train = ds_train.cache_hdd_only(cache_dir)
ds_test = ds_test.cache_hdd_only(cache_dir)

# Model

print("Initializing model...")
model = model_utils.get_model(
    net_name=args.net,
    problem_id=problem_id,
    epoch_count=args.epochs,
    ds_id=args.ds,
    ds_train=ds_train,
    depth=args.depth,
    width=args.width,  # width factor for WRN, base_width for others
    pretrained=args.pretrained)

if args.pretrained:
    print("Loading pretrained parameters...")
    if args.net == 'rn' and args.depth == 50:
        names_to_params = parameter_loading.get_resnet_parameters_from_checkpoint_file(
            f'{dirs.PRETRAINED}/resnetv2_50/resnet_v2_50.ckpt')
    if args.net == 'dn' and args.depth == 121:
        names_to_params = parameter_loading.get_densenet_parameters_from_checkpoint_file(
            f'{dirs.PRETRAINED}/densenet_121/tf-densenet121.ckpt')
    model.load_parameters(names_to_params)

# Training

print("Starting training and validation loop...")
training.train(
    model,
    ds_train,
    ds_test,
    input_jitter={'cifar': augment_cifar}.get(args.ds, random_fliplr),
    epoch_count=args.epochs,
    data_loading_worker_count=4)

# Saving

print("Saving...")
train_set_name = 'trainval' if args.test else 'train'
name = f'{args.net}-{args.depth}-{args.width}' + \
       ('-nd' if args.nodropout else '')
model.save_state(f'{dirs.SAVED_NETS}/{args.ds}-{train_set_name}/' +
                 f'{name}-e{args.epochs}/' +
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
