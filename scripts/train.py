import os
import argparse
import datetime

import numpy as np

from _context import dl_uncertainty

from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.processing.data_augmentation import random_fliplr_with_label, augment_cifar
from dl_uncertainty import parameter_loading
""" 
Use "--trainval" only for training on "trainval" and testing on "test".
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=1 python train.py
CUDA_VISIBLE_DEVICES=2 python train.py
  cifar wrn 28 10 --epochs 200 --trainval
  cifar dn 100 12 --epochs 300 --trainval
  cifar rn  34  8 --epochs 200 --trainval
  cifar rn 164 16 --epochs 200 --trainval
  voc2012 rn 50 64 --pretrained --epochs 30 --trainval
  cityscapes dn  121 32 --epochs 30 --pretrained
  cityscapes rn   50 64 --epochs 30 --pretrained
  cityscapes ldn 121 32 --epochs 30 --pretrained
  cityscapes ldn 121 32 --epochs 140 --pretrained --randomcrop --mcdropout
  camvid ldn 121 32 --epochs 30 --trainval --pretrained
  mozgalo rn 50 64 --pretrained --epochs 15 --trainval
  mozgalo rn 18 64 --epochs 12 --trainval
CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 30 --trainval --mcdropout --pretrained --frac 8; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 30 --trainval --mcdropout --pretrained --frac 4; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 30 --trainval --mcdropout --pretrained --frac 2

CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 120 --trainval --mcdropout --pretrained --frac 8; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 60 --trainval --mcdropout --pretrained --frac 8; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 30 --trainval --mcdropout --pretrained --frac 8; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 120 --trainval --mcdropout --pretrained --frac 4; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 60 --trainval --mcdropout --pretrained --frac 4; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 30 --trainval --mcdropout --pretrained --frac 4; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 120 --trainval --mcdropout --pretrained --frac 2; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 60 --trainval --mcdropout --pretrained --frac 2; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 30 --trainval --mcdropout --pretrained; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 120 --trainval --mcdropout --pretrained; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 60 --trainval --mcdropout --pretrained; CUDA_VISIBLE_DEVICES=0 python train.py camvid ldn 121 32 --epochs 30 --trainval --mcdropout --pretrained;


"""

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--mcdropout', action='store_true')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--randomcrop', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
parser.add_argument('--frac', nargs='?', default=None, type=int)
parser.add_argument('--name_addition', required=False, default="", type=str)
args = parser.parse_args()
print(args)

# Cached dataset with normalized inputs

print("Setting up data loading...")
ds_train, ds_test = data_utils.get_cached_dataset_with_normalized_inputs(
    args.ds, trainval_test=args.trainval)

if args.frac:
    ds_train = ds_train.permute(53).split(1/args.frac)[0]

# Model

print("Initializing model...")
model = model_utils.get_model(
    net_name=args.net,
    ds_train=ds_train,
    depth=args.depth,
    width=args.width,  # width factor for WRN, base_width for others
    epoch_count=args.epochs,
    dropout=args.dropout or args.mcdropout,
    pretrained=args.pretrained)

# Training

print("Starting training and validation loop...")
jitter = data_utils.get_augmentation_func(ds_train)
train_kwargs = {}
if args.randomcrop:
    from dl_uncertainty.processing.data_augmentation import random_crop_with_label
    shape = [ds_train[0][0].shape[0]] * 2
    jitter = lambda xy: random_crop_with_label(xy, shape)
    train_kwargs['jitter_name'] = "random_crop"
training.train(
    model,
    ds_train,
    ds_test,  # 25
    epoch_count=args.epochs,
    jitter=jitter,
    mc_dropout=args.mcdropout,
    data_loading_worker_count=4,
    **train_kwargs)

# Saving

print("Saving...")
name_addition = ""
if args.dropout or args.mcdropout:
    name_addition += "-dropout"
if args.randomcrop:
    name_addition += "-randomcrop"
if args.frac:
    name_addition += f"-frac{args.frac}"
if args.name_addition:
    name_addition += f"-{args.name_addition}"
model_utils.save_trained_model(
    model,
    ds_id=ds_train.info['id'] + ('-trainval' if args.trainval else '-train'),
    net_name=f"{args.net}-{args.depth}-{args.width}{name_addition}",
    epoch_count=args.epochs,
    dropout=args.dropout,
    pretrained=args.pretrained)