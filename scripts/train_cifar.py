# python train_cifar.py wrn 28 10 --test --epochs 200
# python train_cifar.py dn 100 12 --test --epochs 300

import sys
import argparse
import datetime

from _context import dl_uncertainty

from dl_uncertainty.data import Dataset
from dl_uncertainty.data_utils import Cifar10Loader
from dl_uncertainty.standard_models import resnet, densenet, BlockStructure
from dl_uncertainty.training import train_cifar
from dl_uncertainty import dirs

parser = argparse.ArgumentParser()
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
args = parser.parse_args()
print(args)

print("Loading and preparing data...")
if args.test:
    ds_train = Dataset.join(*Cifar10Loader.load_train_val())
    ds_test = Cifar10Loader.load_test()
else:
    ds_train, ds_test = Cifar10Loader.load_train_val()

print("Initializing model...")
if args.net == 'wrn':
    print(f'WRN-{args.depth}-{args.width}')
    zagoruyko_depth=args.depth
    group_count = 3
    ksizes=[3,3]
    blocks_per_group = (zagoruyko_depth - 4) // (group_count * len(ksizes))
    print(
        f"group count: {group_count}, blocks per group: {blocks_per_group}")
    group_lengths = [blocks_per_group] * group_count
    model = resnet(
        group_lengths=group_lengths,
        block_structure=BlockStructure.resnet(
            ksizes=ksizes, dropout_locations=[0]),
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        base_width=args.width*16)
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
    model = resnet(
        group_lengths=group_lengths,
        block_structure=BlockStructure.resnet(
            ksizes=ksizes, dropout_locations=[], width_factors=width_factors),
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        base_width=args.width)
elif args.net == 'dn':
    print(f'DenseNet-{args.depth}-{args.width}')
    model = densenet(
        depth=args.depth,
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        base_width=args.width)
else:
    print(args.net)

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
train_cifar(model, ds_train, ds_test, epoch_count=args.epochs)

print("Saving model...")
traning_set = 'train-val' if args.test else 'train'
model.save_state(f'{dirs.SAVED_NETS}/' +
                 f'{args.net}-{args.depth}-{args.width}-{traning_set}/' +
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
