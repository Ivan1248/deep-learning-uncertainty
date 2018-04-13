import sys
import datetime

from _context import dl_uncertainty

from dl_uncertainty.models import ResidualBlockConfig, ResNet
from dl_uncertainty.data_utils import Cifar10Loader
from dl_uncertainty.standard_nets import get_wrn
from dl_uncertainty.training import train
from dl_uncertainty import dirs

dimargs = sys.argv[1:]
if len(dimargs) not in [0, 2]:
    print("usage: train-wrn.py [<Zagoruyko-depth> <widening-factor>]")
zaggydepth, k = (16, 4) if len(dimargs) == 0 else map(int, dimargs)

print("Loading and preparing data...")
ds_train, ds_val = Cifar10Loader.load_train_val()

print("Initializing model...")
model = get_wrn(
    zagoruyko_depth=zaggydepth,
    input_shape=ds_train.input_shape,
    class_count=ds_train.class_count,
    widening_factor=k)

print("Starting training and validation loop...")
train(model, ds_train, ds_val, epoch_count=200)

print("Saving model...")
model.save_state(dirs.SAVED_NETS + '/wrn-%d-%d--' % (zaggydepth, k) +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
