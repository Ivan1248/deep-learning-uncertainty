# python train_cifar.py wrn 28 10 --test --epochs 200
# python train_cifar.py dn 100 12 --test --epochs 300

import sys
import argparse
import datetime

import tensorflow as tf

from _context import dl_uncertainty

from dl_uncertainty import dirs
from dl_uncertainty.data import DataLoader, ConcatDataset
from dl_uncertainty.data.datasets import Cifar10Dataset
from dl_uncertainty.data_utils import get_input_mean_std
from dl_uncertainty.models import Model, ModelDef, InferenceComponent, TrainingComponent, InferenceComponents, EvaluationMetrics
from dl_uncertainty.standard_models import densenet, BlockStructure
from dl_uncertainty.training import train
from dl_uncertainty.processing.data_augmentation import augment_cifar

parser = argparse.ArgumentParser()
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--nodropout', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
args = parser.parse_args()
print(args)

print("Loading and preparing data...")
cifar_path = dirs.DATASETS + '/cifar-10-batches-py'
ds_train = Cifar10Dataset(cifar_path, 'train')
if args.test:
    ds_test = Cifar10Dataset(cifar_path, 'test')
else:
    ds_train, ds_test = ds_train.shuffle().split(0.8)

mean, std = get_input_mean_std(ds_train)
normalize = lambda x: (x - mean) / std
ds_train = ds_train.map(normalize, 0).cache_all()
ds_test = ds_test.map(normalize, 0).cache_all()

print("Initializing model...")

resnet_learning_rate_policy = {
    'boundaries': [int(i * args.epochs / 200 + 0.5) for i in [60, 120, 160]],
    'values': [1e-1 * 0.2**i for i in range(4)]
}
densenet_learning_rate_policy = {
    'boundaries': [int(i * args.epochs / 100 + 0.5) for i in [50, 75]],
    'values': [1e-1 * 0.1**i for i in range(3)]
}

tc = TrainingComponent(
    batch_size=64 if args.net == 'dn' else 128,
    weight_decay=1e-4 if args.net == 'dn' else 5e-4,
    loss='auto',
    optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
    learning_rate_policy=densenet_learning_rate_policy
    if args.net == 'dn' else resnet_learning_rate_policy)

ic_args = {
    'input_shape': ds_train[0][0].shape,
    'class_count': ds_train.class_count,
}

if args.net == 'wrn':
    print(f'WRN-{args.depth}-{args.width}')
    zagoruyko_depth = args.depth
    group_count, ksizes = 3, [3, 3]
    blocks_per_group = (zagoruyko_depth - 4) // (group_count * len(ksizes))
    ic = InferenceComponents.resnet(
        **ic_args,
        base_width=args.width * 16,
        group_lengths=[blocks_per_group] * group_count,
        block_structure=BlockStructure.resnet(
            ksizes=ksizes, dropout_locations=[] if args.nodropout else [0]),
        dim_change='proj')
elif args.net == 'rn':
    print(f'ResNet-{args.depth}-{args.width}')
    group_lengths, ksizes, width_factors = {
        34: ([3, 4, 6, 3], [3, 3], [1, 1]),
        50: ([3, 4, 6, 3], [1, 3, 1], [1, 1, 4]),
    }[args.depth]
    ic = InferenceComponents.resnet(
        **ic_args,
        base_width=args.width,
        group_lengths=group_lengths,
        block_structure=BlockStructure.resnet(
            ksizes=ksizes, dropout_locations=[], width_factors=width_factors),
        dim_change='id')
elif args.net == 'dn':
    print(f'DenseNet-{args.depth}-{args.width}')
    group_count, ksizes = 3, [1, 3]
    assert (args.depth - group_count - 1) % 3 == 0, \
        f"invalid depth: (depth-group_count-1) must be divisible by 3"
    blocks_per_group = (args.depth - 5) // (group_count * len(ksizes))
    ic = InferenceComponents.densenet(
        **ic_args,
        base_width=args.width,
        group_lengths=[blocks_per_group] * group_count,
        block_structure=BlockStructure.densenet(
            ksizes=ksizes, dropout_locations=[] if args.nodropout else [0]))
else:
    assert False, f"invalid model name: {args.net}"

model = Model(
    modeldef=ModelDef(
        inference_component=ic,
        training_component=tc,
        evaluation_metrics=[EvaluationMetrics.accuracy]),
    training_log_period=40,
    name="Model")

print("Starting training and validation loop...")
train(
    model,
    ds_train,
    ds_test,
    input_jitter=augment_cifar,
    epoch_count=args.epochs)

print("Saving model...")
train_set_name = 'trainval' if args.test else 'train'
name = f'{args.net}-{args.depth}-{args.width}' + \
       ('-nd' if args.nodropout else '')
model.save_state(f'{dirs.SAVED_NETS}/cifar-{train_set_name}/{name}/'
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
