import argparse
import datetime

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from skimage.transform import resize

from _context import dl_uncertainty

from dl_uncertainty.data import DataLoader, datasets
from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.utils.visualization import view_predictions
from dl_uncertainty.processing.shape import fill_to_shape
from dl_uncertainty.processing.data_augmentation import random_crop

# Use "--trainval" only for training on "trainval" and testing "test".
# CUDA_VISIBLE_DEVICES=0 python test_threshold_ood.py
# CUDA_VISIBLE_DEVICES=1 python test_threshold_ood.py
# CUDA_VISIBLE_DEVICES=2 python test_threshold_ood.py
#   cifar wrn 28 10 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/wrn-28-10/2018-04-28-1926/Model
#   cifar dn 100 12 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-05-28-0121/Model
#   cifar rn 34 8
#   cityscapes dn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/dn-121-32-pretrained-e30/2018-05-16-1623/Model
#   mozgalo rn 50 64
#   mozgalo rn 18 64
#   mozgalo dn 100 12 /home/igrubisic/projects/dl-uncertainty/data/nets/mozgalo-trainval/dn-100-12-e10/2018-05-30-0144/Model
#   mozgalooodtrain dn 100 12 /home/igrubisic/projects/dl-uncertainty/data/nets/mozgalo-trainval/dn-100-12-e10/2018-05-30-1857/Model

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('saved_path', type=str)
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--mcdropout', action='store_true')
parser.add_argument('--test_on_training_set', action='store_true')
args = parser.parse_args()
print(args)


def myresize(x, shape):
    x_min, x_max = np.min(x), np.max(x)
    s = x_max - x_min
    x = (x - x_min) / s
    x = resize(x, shape, order=1, clip=False)
    return s * x + x_min


# Cached dataset with normalized inputs

print("Setting up data loading...")
test_dataset_ids = ['cifar', 'tinyimagenet', 'isun', 'mozgalo', 'cityscapes']
if args.ds == 'mozgalooodtrain':
    test_dataset_ids.remove('mozgalo')
    test_dataset_ids.append('mozgaloood')
    test_dataset_ids.append('mozgalooodtrain')
ds_id_to_ds = {
    ds_id: data_utils.get_cached_dataset_with_normalized_inputs(
        ds_id, trainval_test=True)[1].permute().subset(np.arange(150))
    for ds_id in test_dataset_ids
}

shape = ds_id_to_ds[args.ds][0][0].shape
size = len(ds_id_to_ds[args.ds])
for ds_id, ds in list(ds_id_to_ds.items()):
    if ds_id == args.ds:
        continue
    if args.ds == 'cifar' and ds_id in ['tinyimagenet', 'cityscapes']:
        ds_id_to_ds[ds_id + '-crop'] = \
            ds.map(lambda d: random_crop(d, shape[:2]), 0, func_name='crop')
    ds_id_to_ds[ds_id + '-res'] = \
        ds.map(lambda d: myresize(d, shape[:2]), 0, func_name='resize')
    #ds_id_to_ds[ds_id + '-plus'] = \
    #    ds.map(lambda d: myresize(d, shape[:2])*3+3, 0, func_name='resize')
    del ds_id_to_ds[ds_id]
ds_id_to_ds['gaussian'] = \
    datasets.WhiteNoiseDataset(shape, size=size).map(lambda x: (x, -1))
ds_id_to_ds['uniform'] = datasets.WhiteNoiseDataset(
    shape, size=size, uniform=True).map(lambda x: (x, -1))

# Model

print("Initializing model and loading state...")
model = model_utils.get_model(
    net_name=args.net,
    ds_train=ds_id_to_ds[args.ds],
    depth=args.depth,
    width=args.width,
    epoch_count=1,
    dropout=args.dropout or args.mcdropout)
model.load_state(args.saved_path)


def to_logits(xy, label=0):
    x, _ = xy
    logits, output = model.predict(
        x, single_input=True, outputs=['logits', 'output'])
    return logits, label  #int(output != y)


ds_id_to_logits_ds = {
    k: v.map(
        lambda x: to_logits(x, label=int(k == args.ds)),
        func_name='to_logits').cache()
    for k, v in ds_id_to_ds.items()
}

ds_id_to_max_logits_ds = {
    k: v.map(lambda xy: (np.max(xy[0]), xy[1]))
    for k, v in ds_id_to_logits_ds.items()
}

ds_id_to_sum_logits_ds = {
    k: v.map(lambda xy: (np.sum(xy[0]), xy[1]))
    for k, v in ds_id_to_logits_ds.items()
}

max_logits_ds = ds_id_to_max_logits_ds[args.ds]

y_score_in, y_true_in = zip(* [(x, y) for x, y in max_logits_ds])

# Max-logits-sum-logits distributions

fig, axes = plt.subplots(
    2, (len(ds_id_to_max_logits_ds) + 1) // 2,
    figsize=(40, 16),
    sharex=True,
    sharey=True)

ds_ids = ds_id_to_max_logits_ds.keys()
for ds_id, ax in zip(ds_ids, axes.flat):
    max_logits = [d[0] for d in ds_id_to_max_logits_ds[ds_id]]
    sum_logits = [d[0] for d in ds_id_to_sum_logits_ds[ds_id]]
    ax.set_title(ds_id)
    ax.scatter(max_logits, sum_logits, alpha=0.5, s=2, edgecolors='none')
for ax in axes[1, :]:
    ax.set_xlabel('max logits')
for ax in axes[:, 0]:
    ax.set_ylabel('sum logits')
plt.show()

# Max-logits distributions, evaluation curves

fig, axes = plt.subplots(
    4,
    len(ds_id_to_max_logits_ds),
    figsize=(40, 16),
    sharex='row',
    sharey='row')

for i, (ds_id, ds) in enumerate(ds_id_to_max_logits_ds.items()):
    y_score_out, y_true_out = zip(* [(x, y) for x, y in ds])
    y_true = np.array(y_true_in + y_true_out)
    y_score = np.array(y_score_in + y_score_out)

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)

    ax = axes[0, i]
    ax.set_title(ds_id)
    r = (min(y_score), max(y_score))
    ax.hist(y_score_in, bins=40, range=r, alpha=0.5)
    ax.hist(y_score_out, bins=40, range=r, alpha=0.5)

    ax = axes[1, i]
    ax.plot(fpr, fpr, linestyle='--')
    ax.plot(fpr, tpr, label=f'{auc(fpr, tpr):.3f}')
    tpr95idx = np.argmin(np.abs(tpr - 0.95))
    f = fpr[tpr95idx]
    t = tpr[tpr95idx]
    det_err = ((1 - t) + f) / 2
    ax.plot([f], [t], label=f'FPR@95%={f:.3f}')
    ax.plot([f], [t], label=f'd_err={det_err:.3f}')
    ax.set_xlabel('FPR')
    if i == 0:
        ax.set_ylabel('TPR')
    ax.legend()

    ax = axes[2, i]
    p, r, thresholds = precision_recall_curve(
        y_true=y_true, probas_pred=y_score)
    pi = np.copy(p)
    for j, v in enumerate(p):
        pi[j] = np.max(p[:j + 1])
    ax.plot(r, p, label=f'{auc(r, p):.3f}')
    ax.plot(r, pi, label=f'{auc(r, pi):.3f}')
    ax.set_xlabel('R_in')
    if i == 0:
        ax.set_ylabel('P_in')
    ax.legend()

    ax = axes[3, i]
    p, r, thresholds = precision_recall_curve(
        y_true=1 - y_true, probas_pred=-y_score)
    pi = np.copy(p)
    for j, v in enumerate(p):
        pi[j] = np.max(p[:j + 1])
    ax.plot(r, p, label=f'{auc(r, p):.3f}')
    ax.plot(r, pi, label=f'{auc(r, pi):.3f}')
    ax.set_xlabel('R_in')
    if i == 0:
        ax.set_ylabel('P_in')
    ax.legend()
#fig.tight_layout(pad=0.1)
plt.tight_layout(pad=0.5)
plt.xticks(fontsize=8)
plt.xticks(fontsize=8)
plt.show()
