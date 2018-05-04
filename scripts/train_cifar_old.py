# python train_cifar.py wrn 28 10 --test --epochs 200
# python train_cifar.py dn 100 12 --test --epochs 300

import sys
import argparse
import datetime

from _context import dl_uncertainty

from dl_uncertainty.data import OldDataset
from dl_uncertainty.data_utils import Cifar10Loader
from dl_uncertainty.models import ResNet, DenseNet
from dl_uncertainty.standard_models import densenet, BlockStructure
from dl_uncertainty.training import train_cifar_old
from dl_uncertainty.processing.preprocessing import get_normalization_statistics
from dl_uncertainty import dirs

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
ds_train = Cifar10Loader.load('train')
if args.test:
    ds_test = Cifar10Loader.load('test')
else:
    ds_train.shuffle()
    ds_train, ds_test = ds_train.split(0, int(ds_train.size * 0.8))

# resnets
resnet_initial_learning_rate = 1e-1
resnet_learning_rate_policy = {
    'boundaries': [int(i * args.epochs / 200 + 0.5) for i in [60, 120, 160]],
    'values': [resnet_initial_learning_rate * 0.2**i for i in range(4)]
}

print("Initializing model...")
if args.net == 'wrn':
    print(f'WRN-{args.depth}-{args.width}')
    zagoruyko_depth = args.depth
    group_count = 3
    ksizes = [3, 3]
    blocks_per_group = (zagoruyko_depth - 4) // (group_count * len(ksizes))
    print(f"group count: {group_count}, blocks per group: {blocks_per_group}")
    group_lengths = [blocks_per_group] * group_count
    model = ResNet(
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        batch_size=128,
        learning_rate_policy=resnet_learning_rate_policy,
        block_structure=BlockStructure.resnet(
            ksizes=ksizes, dropout_locations=[] if args.nodropout else [0]),
        group_lengths=group_lengths,
        base_width=args.width * 16,
        dim_change='proj',
        weight_decay=5e-4,
        training_log_period=60)
    assert zagoruyko_depth == model.zagoruyko_depth, "invalid depth (zagoruyko_depth={}!={}=model.zagoruyko_depth)".format(
        zagoruyko_depth, model.zagoruyko_depth)
elif args.net == 'rn':
    print(f'ResNet-{args.depth}-{args.width}')
    if args.depth == 34:
        group_lengths = [3, 4, 6, 3]
        ksizes = [3, 3]
        width_factors = [1, 1]
    if args.depth == 50:
        group_lengths = [3, 4, 6, 3]
        ksizes = [1, 3, 1]
        width_factors = [1, 1, 4]
    print(args.depth)
    model = ResNet(
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        batch_size=128,
        learning_rate_policy=resnet_learning_rate_policy,
        block_structure=BlockStructure.resnet(
            ksizes=ksizes, dropout_locations=[], width_factors=width_factors),
        group_lengths=group_lengths,
        base_width=args.width,
        weight_decay=5e-4,
        training_log_period=39)
elif args.net == 'dn':
    print(f'DenseNet-{args.depth}-{args.width}')
    initial_learning_rate = 1e-1
    depth = args.depth
    group_count = 3
    ksizes = [1, 3]
    blocks_per_group = (depth - 5) // (group_count * len(ksizes))
    dm = (depth - group_count - 1) % 3
    assert dm == 0, f"invalid depth ((depth-group_count-1) mod 3 = {dm} must be divisible by 3)"
    print(f"group count: {group_count}, blocks per group: {blocks_per_group}")
    model = DenseNet(
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        batch_size=64,
        learning_rate_policy={
            'boundaries': [int(i * args.epochs / 100 + 0.5) for i in [50, 75]],
            'values': [initial_learning_rate * 0.1**i for i in range(3)]
        },
        block_structure=BlockStructure.densenet(
            ksizes=ksizes, dropout_locations=[] if args.nodropout else [0]),
        base_width=args.width,
        group_lengths=[blocks_per_group] * group_count,
        weight_decay=1e-4,
        training_log_period=39 * 2)
else:
    assert False, f"invalid model name: {args.net}"

model.input_mean, model.input_stddev = get_normalization_statistics(
    ds_train.images)

if False:
    print("Setting up progress bars...")
    model.training_log_period = 1e9
    from tqdm import tqdm, trange
    import builtins
    old_range = builtins.range

    def progress_range(*args, **kwargs):
        return tqdm(old_range(*args, **kwargs), leave=True)

    builtins.range = progress_range

print("Starting training and validation loop...")
train_cifar_old(model, ds_train, ds_test, epoch_count=args.epochs)

print("Saving model...")
traning_set = 'trainval' if args.test else 'train'
name = f'{args.net}-{args.depth}-{args.width}' + ('-nd'
                                                  if args.nodropout else '')
model.save_state(f'{dirs.SAVED_NETS}/cifar-{traning_set}/{name}/'
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
