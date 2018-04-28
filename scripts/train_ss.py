# python train_cifar.py wrn 28 10 --test --epochs 200
# python train_cifar.py dn 100 12 --test --epochs 300

import sys
import argparse
import datetime

from _context import dl_uncertainty

from dl_uncertainty.data import Dataset
from dl_uncertainty.data_utils import VOC2012SegmentationLoader, CityscapesSegmentation, ICCV09Loader
from dl_uncertainty.training import train_semantic_segmentation
from dl_uncertainty.models import LadderDenseNet, ResNetSS, BlockStructure
from dl_uncertainty.processing.preprocessing import get_normalization_statistics
from dl_uncertainty import dirs

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
args = parser.parse_args()
print(args)

print("Loading and preparing data...")
if args.ds == 'cityscapes':
    load = lambda s: CityscapesSegmentation.load(s, downsampling_factor=2)  # TODO 2
    ds_train = load('train')
    if args.test:
        ds_train = ds_train.join(load('val'))
        ds_test = load('test')
    else:
        ds_test = load('val')
elif args.ds == 'voc2012':
    ds_train = VOC2012SegmentationLoader.load('train')
    if args.test:
        ds_test = VOC2012SegmentationLoader.load('val')
    else:
        ds_train.shuffle(random_seed=53)
        ds_train, ds_test = ds_train.split(0, int(ds_train.size * 0.8))
elif args.ds == 'iccv09':
    if args.test:
        assert False, "Test set not defined"
    ds_train = ICCV09Loader.load()
    ds_train.shuffle(random_seed=53)
    ds_train, ds_test = ds_train.split(0, int(ds_train.size * 0.8))

print("Initializing model...")

# resnets
resnet_initial_learning_rate = 1e-4  #1e-3
resnet_learning_rate_policy = {
    'boundaries': [int(i * args.epochs / 200 + 0.5) for i in [60, 120, 160]],
    'values': [resnet_initial_learning_rate * 0.2**i for i in range(4)]
}

if args.net == 'ldn':
    net_name = f'Ladder-DenseNet-{args.depth}'
    print(net_name)
    group_lengths = {121: [6, 12, 24, 16], 
                     169: [6, 12, 32, 32]}[args.depth]
    model = LadderDenseNet(
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        epoch_count=args.epochs,
        group_lengths=group_lengths,
        base_learning_rate=5e-4,  # 5e-4
        training_log_period=40)
elif args.net == 'rn':
    net_name = f'ResNet-{args.depth}-{args.width}'
    print(net_name)
    if args.depth == 34:
        group_lengths = [3, 4, 6, 3]
        ksizes = [3, 3]
        width_factors = [1, 1]
    if args.depth == 50:
        group_lengths = [3, 4, 6, 3]
        ksizes = [1, 3, 1]
        width_factors = [1, 1, 4]
    print(args.depth)
    model = ResNetSS(
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        batch_size=4,
        learning_rate_policy=resnet_learning_rate_policy,
        block_structure=BlockStructure.resnet(
            ksizes=ksizes, dropout_locations=[], width_factors=width_factors),
        group_lengths=group_lengths,
        base_width=args.width,
        weight_decay=5e-4,
        training_log_period=39)
else:
    assert False, f"invalid model name: {args.net}"

if False:
    print("Setting up progress bars...")
    model.training_log_period = 1e9
    from tqdm import tqdm, trange
    import builtins
    old_range = builtins.range

    def progress_range(*args, **kwargs):
        return tqdm(old_range(*args, **kwargs), leave=True)

    builtins.range = progress_range

model.input_mean, model.input_stddev = get_normalization_statistics(
    ds_train.images)

print("Starting training and validation loop...")
train_semantic_segmentation(model, ds_train, ds_test, epoch_count=args.epochs)

print("Saving model...")
traning_set = 'trainval' if args.test else 'train'
model.save_state(f'{dirs.SAVED_NETS}/voc2012/' +
                 f'{net_name}-{traning_set}-{args.epochs}/' +
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
