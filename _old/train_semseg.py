import sys
import argparse
import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from _context import dl_uncertainty

from dl_uncertainty import dirs
from dl_uncertainty.data import DataLoader
from dl_uncertainty.data.datasets import ICCV09Dataset, VOC2012SegmentationDataset, CityscapesSegmentationDataset
from dl_uncertainty.data_utils import get_input_mean_std
from dl_uncertainty.models import Model, ModelDef, InferenceComponent, TrainingComponent
from dl_uncertainty.models import InferenceComponents, TrainingComponents, EvaluationMetrics
from dl_uncertainty.models import BlockStructure
from dl_uncertainty.model_utils import StandardInferenceComponents
from dl_uncertainty.training import train
from dl_uncertainty.processing.data_augmentation import random_fliplr
from dl_uncertainty.ioutils import RangeProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--nodropout', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
args = parser.parse_args()
print(args)

# Dataset

print("Setting up data loading...")

if args.ds == 'cityscapes':
    ds_path = dirs.DATASETS + '/cityscapes'
    load = lambda s: CityscapesSegmentationDataset(ds_path, s, \
        downsampling_factor=2, remove_hood=True)
    ds_train, ds_test = map(load, ['train', 'val'])
    if args.test:
        ds_train = ds_train.join(ds_test)
        ds_test = load('test')
elif args.ds == 'voc2012':
    ds_path = dirs.DATASETS + '/VOC2012'
    load = lambda s: VOC2012SegmentationDataset(ds_path, s)
    if args.test:
        ds_train, ds_test = map(load, ['trainval', 'test'])
    else:
        ds_train, ds_test = map(load, ['train', 'val'])
elif args.ds == 'iccv09':
    if args.test:
        assert False, "Test set not defined"
    ds_path = dirs.DATASETS + '/iccv09'
    ds_train = ICCV09Dataset(dirs.DATASETS + '/iccv09')
    ds_train, ds_test = ds_train.permute().split(0.8)

# Cache size (must be computed before normalization because we want to cache normalized images)

Gi = 1024**3
cache_mem = 14 * Gi
example_mem = (ds_train[0][0].nbytes * 4 + ds_train[0][1].nbytes)
cache_size = cache_mem // example_mem
print(f"Cache size = {cache_size} examples ({cache_mem / Gi} GiB)")

cache_dir = f"{dirs.DATASETS}/_CACHE"

# Input normalization


def normalize(x):  # TODO: fix multithreading problem
    if normalize.mean is None:  # lazy for disk caching purposes
        print("Computing dataset statistics")
        normalize.mean, normalize.std = get_input_mean_std(tqdm(normalize.ds))
    return ((x - normalize.mean) / normalize.std).astype(np.float32)


normalize.ds, normalize.mean, normalize.std = ds_train, None, None

ds_train = ds_train.map(normalize, 0)
ds_test = ds_test.map(normalize, 0)

# Data caching (HDD, RAM)

ds_train = ds_train.cache_hdd_examplewise(cache_dir).cache(cache_size)
ds_test = ds_test.cache_hdd_examplewise(cache_dir)
if cache_size > len(ds_train):
    ds_test = ds_test.cache(cache_size - len(ds_train))

# Model

print("Initializing model...")

resnet_learning_rate_policy = {
    'boundaries': [int(i * args.epochs / 200 + 0.5) for i in [60, 120, 160]],
    'values': [5e-4 * 0.2**i for i in range(4)]
}
densenet_learning_rate_policy = {
    'boundaries': [int(i * args.epochs / 100 + 0.5) for i in [50, 75]],
    'values': [5e-4 * 0.1**i for i in range(3)]
}

if args.net in ['rn', 'dn']:
    tc = TrainingComponent(
        batch_size=4,
        weight_decay={'dn': 1e-4,
                      'rn': 5e-4}[args.net],
        loss='auto',
        optimizer=lambda lr: tf.train.AdamOptimizer(lr),
        learning_rate_policy={
            'dn': densenet_learning_rate_policy,
            'rn': resnet_learning_rate_policy
        }[args.net])
elif args.net == 'ldn':
    tc = TrainingComponents.ladder_densenet(
        epoch_count=args.epochs, base_learning_rate=5e-4, batch_size=4)

ic_args = {
    'input_shape': ds_train[0][0].shape,
    'class_count': ds_train.info['class_count'],
}

if args.net != 'ldn':
    ic_args['problem'] = 'semseg'

if args.net in ['dn', 'ldn']:
    densenet_group_lengths = {
        121: [6, 12, 24, 16],
        169: [6, 12, 32, 32]
    }[args.depth]

if args.net == 'ldn':
    print(f'Ladder-DenseNet-{args.depth}')
    ic = InferenceComponents.ladder_densenet(
        **ic_args,
        base_width=32,
        group_lengths=densenet_group_lengths,
        block_structure=BlockStructure.densenet(dropout_locations=[]))
elif args.net == 'wrn':
    ic = StandardInferenceComponents.wide_resnet(
        **ic_args,
        depth=args.depth,
        width_factor=args.width,
        dropout_locations=[] if args.nodropout else [0])
elif args.net == 'rn':
    ic = StandardInferenceComponents.resnet(
        **ic_args,
        depth=args.depth,
        base_width=args.base_width,
        dropout_locations=[])
elif args.net == 'dn':
    ic = StandardInferenceComponents.densenet(
        **ic_args,
        depth=args.depth,
        base_width=args.base_width,
        dropout_locations=[])
else:
    assert False, f"invalid model name: {args.net}"

model = Model(
    modeldef=ModelDef(
        inference_component=ic,
        training_component=tc,
        evaluation_metrics=[
            EvaluationMetrics.accuracy,
            EvaluationMetrics.semantic_segementation(
                ds_train.info['class_count'])
        ]),
    training_log_period=40,
    name="Model")

# Training

print("Starting training and validation loop...")

train(
    model,
    ds_train,
    ds_test,
    input_jitter=random_fliplr,
    epoch_count=args.epochs)

# Saving

print("Saving...")

train_set_name = 'trainval' if args.test else 'train'
name = f'{args.net}-{args.depth}-{args.width}' + \
       ('-nd' if args.nodropout else '')
model.save_state(f'{dirs.SAVED_NETS}/{args.ds}-{train_set_name}/' +
                 f'{name}-e{args.epochs}/' +
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
