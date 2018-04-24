# python train_cifar.py wrn 28 10 --test --epochs 200
# python train_cifar.py dn 100 12 --test --epochs 300

import sys
import argparse
import datetime

from _context import dl_uncertainty

from dl_uncertainty.data import Dataset
from dl_uncertainty.data_utils import VOC2012SegmentationLoader
from dl_uncertainty.standard_models import resnet, densenet, BlockStructure
from dl_uncertainty.training import train_semantic_segmentation
from dl_uncertainty.models import LadderDenseNet
from dl_uncertainty import dirs

parser = argparse.ArgumentParser()
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
#parser.add_argument('depth', type=int)
#parser.add_argument('width', type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
args = parser.parse_args()
print(args)

print("Loading and preparing data...")
if args.test:
    ds_train = VOC2012SegmentationLoader.load('train')
    ds_test = VOC2012SegmentationLoader.load('val')
else:
    ds_train = VOC2012SegmentationLoader.load('train')
    ds_train.shuffle(random_seed=53)
    ds_train, ds_test = ds_train.split(0, int(ds_train.size * 0.8))

print("Initializing model...")
if args.net == 'ldn':
    net_name = f'Ladder-DenseNet'
    print(net_name)
    model = LadderDenseNet(
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        epoch_count=args.epochs)
else:
    assert False, "invalid model name"

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
train_semantic_segmentation(model, ds_train, ds_test, epoch_count=args.epochs)

print("Saving model...")
traning_set = 'train-val' if args.test else 'train'
model.save_state(f'{dirs.SAVED_NETS}/' + f'{net_name}-{traning_set}/' +
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
