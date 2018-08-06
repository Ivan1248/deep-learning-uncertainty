import argparse
import datetime

import numpy as np
from tqdm import tqdm

from _context import dl_uncertainty

from dl_uncertainty.data import DataLoader
from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.utils.visualization import view_predictions, view_predictions_2
from dl_uncertainty.models.tf_utils.adversarial_examples import fgsm, fgm
from dl_uncertainty.models.adversarial_example_generator import AdversarialExampleGenerator
"""
Use "--trainval" only for training on "trainval" and testing "test".
CUDA_VISIBLE_DEVICES=0 python test_inverse_ae.py
CUDA_VISIBLE_DEVICES=1 python test_inverse_ae.py
CUDA_VISIBLE_DEVICES=2 python test_inverse_ae.py
 --mcdropout --trainval --uncertainties
 cifar wrn 28 10 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/wrn-28-10-e200/2018-08-02-0841/Model --trainval
 cifar dn 100 12 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-08-02-2016/Model --trainval

 mozgalo rn 50 64
 mozgalo rn 18 64
"""

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('saved_path', type=str)
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--mcdropout', action='store_true')
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--test_on_training_set', action='store_true')
parser.add_argument('--test_dataset', default="", type=str)
args = parser.parse_args()
print(args)

# Cached dataset with normalized inputs

print("Setting up data loading...")
ds_train, ds_test = data_utils.get_cached_dataset_with_normalized_inputs(
    args.ds, trainval_test=args.trainval)

model_ds = ds_train

if args.test_dataset != "":
    ds_train, ds_test = data_utils.get_cached_dataset_with_normalized_inputs(
        args.test_dataset, trainval_test=args.trainval)

# Model

print("Initializing model and loading state...")
model = model_utils.get_model(
    net_name=args.net,
    ds_train=model_ds,
    depth=args.depth,
    width=args.width,  # width factor for WRN, base_width for others
    epoch_count=1,
    dropout=args.dropout or args.mcdropout)
model.load_state(args.saved_path)

aegen = AdversarialExampleGenerator(model)
aegen_targ = AdversarialExampleGenerator(model, targeted=True)

#ds_train = ds_train.split(0.005)[0]
ds_test = ds_test.split(0.05)[0]
model.test(DataLoader(ds_test, model.batch_size), "test set")

eps = 0.05


def perturb_ds(ds, eps):
    return ds.map(lambda x: aegen.perturb(x, eps=eps, single_input=True),
                  0).cache()


def perturb_ds_targeted(ds, eps, y):
    return ds.map(
        lambda x: aegen_targ.perturb(x, y=y, eps=eps, single_input=True),
        0).cache()


def perturb_ds_targeted_true(ds, eps):
    return ds.map(
        lambda x: (aegen_targ.perturb(x[0], y=x[1], eps=eps, single_input=True), x[1])
    ).cache()


def perturb_ds_targeted_maxprob(ds, eps):
    return ds.map(
        lambda x: (aegen_targ.perturb_most_confident(x[0], ys=np.arange(10), eps=eps, single_input=True), x[1])
    ).cache()


ds_adv = perturb_ds(ds_test, eps)
model.test(DataLoader(ds_adv, model.batch_size), "test adv")

model.test(
    DataLoader(perturb_ds_targeted_true(ds_adv, -eps), model.batch_size),
    "test adv-invadv-true")

#model.test(
#    DataLoader(perturb_ds(ds_adv, eps), model.batch_size), "test adv-adv")

model.test(
    DataLoader(perturb_ds_targeted_true(ds_test, -eps), model.batch_size),
    "test invadv-true")

d = perturb_ds_targeted_maxprob(ds_adv, -eps)
model.test(DataLoader(d, model.batch_size), "test invadv-maxprob")

print("")

for i in range(0):
    model.test(
        DataLoader(perturb_ds_targeted(ds_adv, -eps, y=i), model.batch_size),
        f"test adv-invadv-{i}")
