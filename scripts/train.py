import sys
import os
import argparse
import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from _context import dl_uncertainty

from dl_uncertainty import dirs
from dl_uncertainty.data import DataLoader
from dl_uncertainty.data.datasets import ICCV09Dataset, VOC2012SegmentationDataset, CityscapesSegmentationDataset
from dl_uncertainty.data.datasets import Cifar10Dataset, MozgaloRobustVisionChallengeDataset
from dl_uncertainty.data_utils import get_input_mean_std
from dl_uncertainty.models import Model, ModelDef, InferenceComponent, TrainingComponent
from dl_uncertainty.models import InferenceComponents, TrainingComponents, EvaluationMetrics
from dl_uncertainty.models import BlockStructure
from dl_uncertainty.model_utils import StandardInferenceComponents
from dl_uncertainty.training import train
from dl_uncertainty.processing.data_augmentation import random_fliplr
from dl_uncertainty.processing.data_augmentation import augment_cifar

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
if args.ds in ['cifar10', 'svhn', 'mozgalo']:
    problem = 'clf'
    if args.ds == 'cifar10':
        ds_path = dirs.DATASETS + '/cifar-10-batches-py'
        ds_train = Cifar10Dataset(ds_path, 'train')
        if args.test:
            ds_test = Cifar10Dataset(ds_path, 'test')
        else:
            ds_train, ds_test = ds_train.permute().split(0.8)
    if args.ds == 'mozgalo':
        mozgalo_path = dirs.DATASETS + '/mozgalo_robust_ml_challenge'
        ds_train = MozgaloRobustVisionChallengeDataset(mozgalo_path, 'train')
        ds_train = ds_train.permute()
        ds_train, ds_test = ds_train.split(0.8)
        if not args.test:
            ds_train, ds_test = ds_train.split(0.8)
elif args.ds in ['cityscapes', 'voc2012', 'iccv09']:
    problem = 'semseg'
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


def get_cache_size(mem_B):
    img, lab = ds_train[0]
    img = img.astype(np.float32)
    example_mem = (img.nbytes + np.array(lab).nbytes)
    return mem_B // example_mem


Gi = 1024**3
cache_mem = 14 * Gi
cache_size = get_cache_size(cache_mem)
print(f"Cache size = {cache_size} examples ({cache_mem / Gi} GiB)")

# Input preprocessing and caching


class Normalizer:
    ds, mean, std = ds_train, None, None

    @classmethod
    def normalize(cls, x):  # TODO: fix multithreading problem
        if cls.mean is None:  # lazy
            print("Computing dataset statistics")
            cls.mean, cls.std = get_input_mean_std(tqdm(cls.ds))
        return ((x - cls.mean) / cls.std).astype(np.float32)


ds_train = ds_train.map(Normalizer.normalize, 0)
ds_test = ds_test.map(Normalizer.normalize, 0)

# Caching
cache_dir = f"{dirs.DATASETS}/{os.path.basename(dirs.DATASETS)}_cache"


def cache(ds, cache_size):  # Data caching (HDD, RAM)
    if cache_size > len(ds):
        ds = ds.cache_hdd(cache_dir)
    else:
        ds = ds.cache_hdd_examplewise(cache_dir).cache(cache_size)
    return ds


ds_train = cache(ds_train, cache_size)
cache_size_left = cache_size - len(ds_train)
if cache_size_left > 0:
    ds_test = cache(ds_test, cache_size_left)

# If normalized data is not already on disk, this will trigger normalization
# statistics computation. Normalization statistics need to be computed before
# parallelized DataLoader is used.
ds_train[0]

# Model

print("Initializing model...")

base_learning_rate = {'clf': 1e-1, 'semseg': 5e-4}[problem]
resnet_learning_rate_policy = {
    'boundaries': [int(i * args.epochs / 200 + 0.5) for i in [60, 120, 160]],
    'values': [base_learning_rate * 0.2**i for i in range(4)]
}
densenet_learning_rate_policy = {
    'boundaries': [int(i * args.epochs / 100 + 0.5) for i in [50, 75]],
    'values': [base_learning_rate * 0.1**i for i in range(3)]
}

if problem == 'clf':
    tc = TrainingComponent(
        batch_size=64 if args.net == 'dn' else 128,
        weight_decay=1e-4 if args.net == 'dn' else 5e-4,
        loss='auto',
        optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
        learning_rate_policy=densenet_learning_rate_policy
        if args.net == 'dn' else resnet_learning_rate_policy)
elif problem == 'semseg':
    if args.net in ['rn', 'dn', 'wrn']:
        tc = TrainingComponent(
            batch_size=4,
            weight_decay={'dn': 1e-4,
                          'rn': 5e-4,
                          'wrn': 5e-4}[args.net],
            loss='auto',
            optimizer=lambda lr: tf.train.AdamOptimizer(lr),
            learning_rate_policy={
                'dn': densenet_learning_rate_policy,
                'rn': resnet_learning_rate_policy,
                'wrn': resnet_learning_rate_policy,
            }[args.net])
    elif args.net == 'ldn':
        tc = TrainingComponents.ladder_densenet(
            epoch_count=args.epochs, base_learning_rate=5e-4, batch_size=4)

ic_args = {
    'input_shape': ds_train[0][0].shape,
    'class_count': ds_train.info['class_count'],
    'problem': problem,
}

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

evaluation_metrics = [EvaluationMetrics.accuracy]
if problem == 'semseg':
    evaluation_metrics.append(
        EvaluationMetrics.semantic_segementation(ds_train.info['class_count']))

model = Model(
    modeldef=ModelDef(ic, tc, evaluation_metrics),
    training_log_period=80,
    name="Model")

# Training

print("Starting training and validation loop...")

train(
    model,
    ds_train,
    ds_test,
    input_jitter=random_fliplr if problem == 'semseg' else augment_cifar,
    epoch_count=args.epochs,
    data_loading_worker_count=8)

# Saving

print("Saving...")

train_set_name = 'trainval' if args.test else 'train'
name = f'{args.net}-{args.depth}-{args.width}' + \
       ('-nd' if args.nodropout else '')
model.save_state(f'{dirs.SAVED_NETS}/{args.ds}-{train_set_name}/' +
                 f'{name}-e{args.epochs}/' +
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
