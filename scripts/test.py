import argparse
import datetime

import numpy as np

from _context import dl_uncertainty

from dl_uncertainty.data import DataLoader
from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.utils.visualization import view_predictions, view_predictions_2
"""
Use "--trainval" only for training on "trainval" and testing "test".
CUDA_VISIBLE_DEVICES=0 python test.py
CUDA_VISIBLE_DEVICES=1 python test.py
CUDA_VISIBLE_DEVICES=2 python test.py
 cifar wrn 28 10 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/wrn-28-10-e200/2018-05-28-0956/Model
 cifar wrn 28 10 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/wrn-28-10-dropout-e200/2018-05-29-1708/Model --mcdropout
 cifar dn 100 12 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-05-28-0121/Model
 cifar rn 34 8
 cityscapes dn 121 32  /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/dn-121-32-pretrained-e30/2018-05-18-0010/Model
 cityscapes rn 50 64
 cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-pretrained-e30/2018-05-18-2129/Model
 cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-dropout-pretrained-e80/2018-05-24-0403/Model --mcdropout
 cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-dropout-pretrained-e80/2018-05-24-0403/Model --mcdropout --test_dataset wilddash
 dropout0.1
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-pretrained-e30/2018-06-24-1051/Model --mcdropout
 dropout0.1 1/2
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac2-pretrained-e30/2018-06-22-1256/Model --mcdropout
 dropout0.1 1/4
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac4-pretrained-e30/2018-06-22-1226/Model --mcdropout
 dropout0.1 1/8
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac8-pretrained-e30/2018-06-22-1205/Model --mcdropout
 dropout0.1 1/2
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac2-pretrained-e30/2018-06-24-1509/Model --mcdropout
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
parser.add_argument('--view', action='store_true')
parser.add_argument('--view2', action='store_true')
parser.add_argument('--hard', action='store_true')  # display hard exampels
parser.add_argument('--save', action='store_true')
args = parser.parse_args()
print(args)

assert not args.hard or args.hard and args.view

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

if args.view or args.hard or args.view2:
    ds_disp = ds_train if args.test_on_training_set else ds_test
    ds_disp = ds_disp.permute()

    if args.hard:
        ds_disp = training.get_hard_examples(model, ds_disp)

    def predict(x):
        out_names = ['output', 'probs', 'probs_entropy']
        output, probs, probs_entropy = model.predict(
            x, single_input=True, outputs=out_names)
        if args.mcdropout:
            mc_output, mc_probs, mc_probs_entropy, mc_probs_mi = model.predict(
                x,
                single_input=True,
                mc_dropout=args.mcdropout,
                outputs=out_names + ['probs_mi']  #,'pred_logits_var']
            )
            epistemic = mc_probs_mi
            aleatoric = mc_probs_entropy - mc_probs_mi
            uncertainties = [
                probs_entropy, mc_probs_entropy, aleatoric, epistemic * 20
            ]
            maxent = np.log(mc_probs.shape[-1])
            for a in uncertainties:
                a.flat[0] = maxent
                a.flat[1] = 0
            return [output, mc_output] + uncertainties
        return [output, probs_entropy]

    save_dir = f'{dirs.CACHE}/viewer/{args.ds}-{args.net}-{args.depth}-{args.width}'
    if args.test_dataset != "":
        save_dir += f'-{args.test_dataset}'
    if args.dropout or args.mcdropout:
        save_dir += "-dropout"
    view = view_predictions_2 if args.view2 else view_predictions
    view(ds_disp, predict, save_dir=save_dir if args.save else None)
else:
    model.test(
        DataLoader(ds_train, model.batch_size),
        "training set",
        mc_dropout=args.mcdropout)
    model.test(
        DataLoader(ds_test, model.batch_size),
        "test set",
        mc_dropout=args.mcdropout)
