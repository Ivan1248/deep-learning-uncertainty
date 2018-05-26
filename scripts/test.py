import argparse
import datetime

import numpy as np

from _context import dl_uncertainty

from dl_uncertainty.data import DataLoader
from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.utils.visualization import view_predictions, view_predictions_2

# Use "--trainval" only for training on "trainval" and testing "test".
# CUDA_VISIBLE_DEVICES=0 python test.py
# CUDA_VISIBLE_DEVICES=1 python test.py
# CUDA_VISIBLE_DEVICES=2 python test.py
#   cifar wrn 28 10
#   cifar dn 100 12
#   cifar rn 34 8
#   cityscapes dn 121 32  /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/dn-121-32-pretrained-e30/2018-05-18-0010/Model
#   cityscapes rn 50 64
#   cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-pretrained-e30/2018-05-18-2129/Model
#   cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-dropout-pretrained-e80/2018-05-24-0403/Model --mcdropout
#   cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-dropout-pretrained-e80/2018-05-24-0403/Model --mcdropout --test_dataset wilddash
#   camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-train/ldn-121-32-mc_dropout-pretrained-e30/2018-05-23-1320/Model
#   mozgalo rn 50 64
#   mozgalo rn 18 64

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
    if args.hard:
        ds_disp = training.get_hard_examples(model, ds_disp)

    def predict(x):
        out_names = ['output', 'probs_entropy']
        outs = model.predict(
            x,
            single_input=True,
            mc_dropout=args.mcdropout,
            outputs=out_names + (['pred_logits_var'] if args.mcdropout else []))
        if args.mcdropout:
            van = model.predict(x, single_input=True, outputs=out_names)
            mine, maxe = np.min([outs[1], van[1]]), np.max([outs[1], van[1]])
            outs[1].flat[0], van[1].flat[0] = [mine] * 2
            outs[1].flat[1], van[1].flat[1] = [maxe] * 2
            return [van[0], outs[0], van[1], outs[1], outs[2], outs[1]*outs[2]]
        return outs

    save_dir = f'{dirs.CACHE}/viewer/{args.ds}-{args.net}-{args.depth}-{args.width}'
    if args.dropout or args.mcdropout:
        save_dir += "-dropout"
    if args.test_dataset != "":
        save_dir += f"-{args.test_dataset}"

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
