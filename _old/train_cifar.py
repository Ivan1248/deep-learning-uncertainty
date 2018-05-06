# python train_cifar.py wrn 28 10 --test --epochs 200
# python train_cifar.py dn 100 12 --test --epochs 300

import sys
import argparse
import datetime

import tensorflow as tf

from _context import dl_uncertainty

from dl_uncertainty import dirs
from dl_uncertainty.data import DataLoader
from dl_uncertainty.data.datasets import Cifar10Dataset
from dl_uncertainty.data_utils import get_input_mean_std
from dl_uncertainty.models import Model, ModelDef, InferenceComponent, TrainingComponent, EvaluationMetrics
from dl_uncertainty.model_utils import StandardInferenceComponents
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
    ds_train, ds_test = ds_train.permute().split(0.8)

mean, std = get_input_mean_std(ds_train)
normalize = lambda x: (x - mean) / std
ds_train = ds_train.map(normalize, 0).cache()
ds_test = ds_test.map(normalize, 0).cache()

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
model.save_state(
    f'{dirs.SAVED_NETS}/cifar-{train_set_name}/{name}-e{args.epochs}/'
    f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
