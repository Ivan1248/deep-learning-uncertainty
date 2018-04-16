import sys
import argparse
import datetime

from _context import dl_uncertainty

from dl_uncertainty.data import Dataset
from dl_uncertainty.data_utils import Cifar10Loader
from dl_uncertainty.standard_models import resnet, densenet
from dl_uncertainty.training import train
from dl_uncertainty import dirs

parser = argparse.ArgumentParser()
parser.add_argument('net', type=str)  # 'wrn' or 'dn'
parser.add_argument('depth_k', nargs=2, type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--epochs', nargs='?', const=200, type=int)
args = parser.parse_args()
print(args)

depth, k = args.depth_k

print("Loading and preparing data...")
if args.test:
    ds_train = Dataset.join(*Cifar10Loader.load_train_val())
    ds_test = Cifar10Loader.load_test()
else:
    ds_train, ds_test = Cifar10Loader.load_train_val()

print("Initializing model...")
if args.net == 'wrn':
    print(f'WRN-{depth}-{k}')
    model = resnet(
        zagoruyko_depth=depth,
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        widening_factor=k)
elif args.net == 'dn':
    print(f'Densenet-{depth}-{k}')
    model = densenet(
        depth=depth,
        input_shape=ds_train.input_shape,
        class_count=ds_train.class_count,
        base_width=k)
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
train(model, ds_train, ds_test, epoch_count=200)

print("Saving model...")
traning_set = 'train-val' if args.test else 'train'
model.save_state(f'{dirs.SAVED_NETS}/{args.net}-{depth}-{k}-{traning_set}/' +
                 f'{datetime.datetime.now():%Y-%m-%d-%H%M}')
