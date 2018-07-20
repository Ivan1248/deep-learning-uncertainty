import argparse
import datetime

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from skimage import transform
from tqdm import tqdm
import pandas as pd

from _context import dl_uncertainty

from dl_uncertainty.data import DataLoader, datasets
from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.processing.shape import fill_to_shape
from dl_uncertainty.processing.data_augmentation import random_crop
from dl_uncertainty.models.odin import Odin

from functools import lru_cache

#from dl_uncertainty.utils.figstyle import figsize
figsize = lambda: (15, 10)
#oldfigsize = figsize
#figsize = lambda: oldfigsize(10)
(figw, figh) = figsize()

# Use "--trainval" only for training on "trainval" and testing "test".
# CUDA_VISIBLE_DEVICES=0 python test_threshold_ood.py
# CUDA_VISIBLE_DEVICES=1 python test_threshold_ood.py
# CUDA_VISIBLE_DEVICES=2 python test_threshold_ood.py
#   cifar wrn 28 10 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/wrn-28-10-e200/2018-06-22-1412/Model
#     A = 0.9571
#   cifar dn 100 12 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-05-28-0121/Model
#     A = 0.9481
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
parser.add_argument('--perturb', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--histograms', action='store_true')
parser.add_argument('--plot2d', action='store_true')
parser.add_argument('--noperturb', action='store_true')
parser.add_argument('--notemp', action='store_true')
parser.add_argument('--dssize', default=10000, type=int)
parser.add_argument('--misclassified', action='store_true')
args = parser.parse_args()
print(args)

# Helper functions


def map_to_dict(f, keys):
    return {k: f(k) for k in keys}


def map_dict_v(f, d):
    return {k: f(v) for k, v in d.items()}


def resize(x, shape):
    x_min, x_max = np.min(x), np.max(x)
    s = x_max - x_min
    x = (x - x_min) / s  # values need to be in [0,1]
    x = transform.resize(x, shape, order=1, clip=False, mode='constant')
    x = np.float32(x)
    return s * x + x_min  # restore value scaling


# Cached dataset with normalized inputs

print("Setting up data loading...")
get_ds = data_utils.get_cached_dataset_with_normalized_inputs

ds_id_to_ds = dict()

if args.misclassified:
    ds_id_to_ds[args.ds] = get_ds(args.ds, trainval_test=True)[1].cache()
else:
    if args.ds == 'mozgalooodtrain':
        for ds_id in ['mozgalo', 'mozgaloood', 'mozgalooodtrain']:
            ds_id_to_ds[ds_id] = get_ds(ds_id, trainval_test=True)[1]

    for ds_id in ['cifar', 'tinyimagenet', 'lsun']:
        ds_id_to_ds[ds_id] = get_ds(ds_id, trainval_test=True)[1]
    isun_trainval, isun_test = get_ds('isun', trainval_test=True)
    ds_id_to_ds['isun'] = isun_trainval.join(isun_test)

    # val sets
    for ds_id, ds in ds_id_to_ds.items():
        print(ds_id, len(ds))  # print dataset length

    # prepare cropped and resized datasets
    shape = ds_id_to_ds[args.ds][0][0].shape
    for ds_id, ds in list(ds_id_to_ds.items()):
        if ds_id == args.ds:
            continue
        if args.ds == 'cifar' and ds_id in [
                'tinyimagenet', 'cityscapes', 'lsun'
        ]:
            ds_id_to_ds[ds_id + '-c'] = \
                ds.map(lambda d: random_crop(d, shape[:2]), 0, func_name='c')
        ds_id_to_ds[ds_id + '-r'] = \
            ds.map(lambda d: resize(d, shape[:2]), 0, func_name='r')
        del ds_id_to_ds[ds_id]
        #ds_id_to_ds[ds_id + '-plus'] = \
        #    ds.map(lambda d: resize(d, shape[:2])*3+3, 0, func_name='resize')

    # add noise datasets
    ds_id_to_ds['gaussian'] = \
        datasets.WhiteNoiseDataset(shape, size=args.dssize).map(lambda x: (x, -1))
    ds_id_to_ds['uniform'] = datasets.WhiteNoiseDataset(
        shape, size=args.dssize, uniform=True).map(lambda x: (x, -1))

    ds_id_to_ds = map_dict_v(
        lambda v: v.permute().subset(np.arange(min(len(v), args.dssize))),
        ds_id_to_ds)
    ds_id_to_ds = map_dict_v(lambda v: v.cache(), ds_id_to_ds)


def translate_ds_name(name):
    return {
        'cifar-miscl': r'CIFAR-10-M',
        'cifar': r'CIFAR-10',
        'tinyimagenet-c': r'TinyImageNet-C',
        'tinyimagenet-r': r'TinyImageNet-R',
        'lsun-c': r'LSUN-C',
        'lsun-r': r'LSUN-R',
        'isun-r': r'iSUN',
        'gaussian': r'Gaussiov šum',
        'uniform': r'Uniformni šum',
    }[name]


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

temp = 0 if args.notemp else 1000
odin = Odin(model, temp=temp)

# Missclassification

if args.misclassified:
    ds = ds_id_to_ds[args.ds]
    corr = [model.predict(x, single_input=True) == y for x, y in tqdm(ds)]
    correct = []
    misclassified = []
    for i, c in enumerate(corr):
        if c:
            correct += [i]
        else:
            misclassified += [i]
    ds_id_to_ds['cifar'] = ds.subset(correct)
    ds_id_to_ds['cifar-miscl'] = ds.subset(misclassified)

# Logits

if args.plot2d:

    def to_logits(xy, label=0):
        x, _ = xy
        logits, output = model.predict(
            x, single_input=True, outputs=['logits', 'output'])
        return logits, label  # int(output != y)


    ds_id_to_logits_ds = {
        k: v.map(
            lambda x: to_logits(x, label=int(k == args.ds)),
            func_name='to_logits') #.cache()
        for k, v in ds_id_to_ds.items()
    }

    ds_id_to_max_logits_ds = map_dict_v(
        lambda v: v.map(lambda xy: (np.max(xy[0]), xy[1])), ds_id_to_logits_ds)
    ds_id_to_sum_logits_ds = map_dict_v(
        lambda v: v.map(lambda xy: (np.sum(xy[0]), xy[1])), ds_id_to_logits_ds)

    max_logits_ds = ds_id_to_max_logits_ds[args.ds]

    y_score_in, y_true_in = zip(* [(x, y) for x, y in max_logits_ds])


def evaluate(scores_in, scores_out):
    y_true = np.concatenate(
        [[1] * len(scores_in), [0] * len(scores_out)], axis=0)
    y_score = np.concatenate([scores_in, scores_out], axis=0)

    fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_score)

    tpr95idx = np.argmin(np.abs(tpr - 0.95))
    if tpr[tpr95idx] < 0.95:
        fpr95_indices = [tpr95idx, min(len(tpr) - 1, tpr95idx + 1)]
    else:
        fpr95_indices = [max(0, tpr95idx - 1), tpr95idx]
    fpr95_weights = np.array(
        [tpr[fpr95_indices[1]] - 0.95, 0.95 - tpr[fpr95_indices[0]]])
    if np.sum(fpr95_weights) < 1e-5:
        fpr95_weights = np.array([0, 1])
    else:
        fpr95_weights /= np.sum(fpr95_weights)
    fpr95t = fpr[fpr95_indices].dot(fpr95_weights)  # tpr95 ~= 0.95
    tpr95t = 0.95

    det_err95t = ((1 - tpr95t) + fpr95t) / 2

    p_in, r_in, thr = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    pi_in = np.array([np.max(p_in[:j + 1]) for j, _ in enumerate(p_in)])

    p_out, r_out, thr = precision_recall_curve(
        y_true=1 - y_true, probas_pred=-y_score)
    pi_out = np.copy(p_out)
    pi_out = np.array([np.max(p_out[:j + 1]) for j, _ in enumerate(p_out)])

    return {
        'fpr': fpr,
        'tpr': tpr,
        'auroc': auc(fpr, tpr),
        'fpr95t': fpr95t,
        'det_err95t': det_err95t,
        'p_in': p_in,
        'r_in': r_in,
        'pi_in': pi_in,
        'aupr_in': auc(r_in, p_in),
        'aupir_in': auc(r_in, pi_in),
        'p_out': p_out,
        'r_out': r_out,
        'pi_out': pi_out,
        'aupr_out': auc(r_out, p_out),
        'aupir_out': auc(r_out, pi_out),
    }


# Max-logits-sum-logits distributions plots

if args.plot2d:
    fig, axes = plt.subplots(
        2, (len(ds_id_to_max_logits_ds) + 1) // 2,
        figsize=(figw, figh),
        sharex=True,
        sharey=True)

    ds_ids = ds_id_to_max_logits_ds.keys()
    for ds_id, ax in zip(ds_ids, axes.flat):
        max_logits = [d[0] for d in ds_id_to_max_logits_ds[ds_id]]
        sum_logits = [d[0] for d in ds_id_to_sum_logits_ds[ds_id]]
        ax.set_title(translate_ds_name(ds_id))
        ax.scatter(max_logits, sum_logits, alpha=0.5, s=1, edgecolors='none')
    for ax in axes[1, :]:
        ax.set_xlabel(r'$\max_i s_i$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\sum_i s_i$')
    plt.show()

# Max-logits distributions, evaluation curves plots


def plot_histograms(ax, y_score_in, y_score_out):
    y_score = np.concatenate([y_score_in, y_score_out], axis=0)
    hist_range = (min(y_score_in), max(y_score))
    ax.hist(y_score_in, bins=40, range=hist_range, alpha=0.5)
    ax.hist(y_score_out, bins=40, range=hist_range, alpha=0.5)


def plot_roc(ax, fpr, tpr, fpr95t, det_err=None):
    ax.plot(fpr, fpr, linestyle='--')
    ax.plot(fpr, tpr, label=f'{auc(fpr, tpr):.3f}')
    ax.plot([fpr95t], [0.95], label=f'FPR95={fpr95t:.3f}')
    if det_err is not None:
        ax.plot([fpr95t], [0.95], label=f'de={det_err:.3f}')
    ax.set_xlabel('FPR')
    if i == 0:
        ax.set_ylabel('TPR')
    ax.set_ylim(bottom=0)
    ax.legend()


def plot_pr(ax, p, p_interp, r, aupr, aupir, name):
    ax.plot(r, p, label=f'{aupr:.3f}')
    ax.plot(r, p_interp, label=f'{aupir:.3f}')
    ax.set_xlabel(f'$R_{name}$')
    if i == 0:
        ax.set_ylabel(f'$P_{name}$')
    ax.set_ylim(bottom=0)
    ax.legend()



methods = [
    ('max-probs', 1, False),
    ('max-probs', 1000, False),
    ('max-probs', 1000, True),
    ('max-logits', 1, False),
    ('max-logits', 1000, True),
]

method_to_evaluations = dict()
for method, temp, perturb in methods:
    odin.temp = temp

    full_method_name = f"{method}{temp}"
    if perturb:
        full_method_name += '-p'
    method_to_evaluations[full_method_name] = dict()

    if args.plot:
        fig, axes = plt.subplots(
            4,
            len(ds_id_to_ds),
            figsize=(figw, figh),
            sharex='row',
            sharey='row')

    ds_in = ds_id_to_ds[args.ds]

    print(f"Evaluating: {full_method_name}")
    for i, (ds_out_id, ds_out) in enumerate(tqdm(ds_id_to_ds.items())):
        ds_out = ds_id_to_ds[ds_out_id]
        ds_out_val, ds_out = ds_out.split(0.1)

        eps = 0

        if perturb and ds_out_id != args.ds:
            epsilons, perfs = odin.fit_epsilon(
                ds_in, ds_out_val,
                lambda si, so: 1 - evaluate(si, so)['fpr95t'])
            print(ds_out_id)
            print(perfs)
            print(perfs)
            eps = epsilons[np.argmax(perfs)]

        scores_name = method[method.index('-') + 1:]
        y_score_in = odin.get_scores(ds_in, scores_name=scores_name)
        y_score_out = odin.get_scores(ds_out, scores_name=scores_name)

        # evaluation
        ev = evaluate(y_score_in, y_score_out)
        method_to_evaluations[full_method_name][ds_out_id] = {
            'fpr95t': ev['fpr95t'],
            'auroc': ev['auroc'],
            'aupr_in': ev['aupr_in'],
            'aupr_out': ev['aupr_out'],
            'eps': eps,
        }


        # histograms
        if args.plot or args.histograms:
            plot_histograms(axes[0, i], y_score_in, y_score_out)

        if not args.plot:
            continue
        axes[0, i].set_title(f"{ds_out_id}{odin.epsilon}")
        # in-distribution ROC, AUROC, FPR@95%TPR
        plot_roc(axes[1, i], ev['fpr'], ev['tpr'], ev['fpr95t'])
        # in-distribution P-R, AUPR
        plot_pr(axes[2, i], ev['p_in'], ev['pi_in'], ev['r_in'], ev['aupr_in'],
                ev['aupir_in'], 'in')
        # out-distribution P-R, AUPR
        plot_pr(axes[3, i], ev['p_out'], ev['pi_out'], ev['r_out'],
                ev['aupr_out'], ev['aupir_out'], 'out')

    if args.plot:
        plt.xticks(fontsize=8)
        plt.xticks(fontsize=8)
        #fig.tight_layout(pad=0.1)
        plt.tight_layout(pad=0.5)
        plt.show()

m2e = method_to_evaluations
evaluations = dict()
methods = []
for method_name, ds2evs in m2e.items():
    methods += [method_name]
    for ds_out_id, evs in ds2evs.items():
        if ds_out_id not in evaluations:
            evaluations[ds_out_id] = dict()
        for ev_name, ev_val in evs.items():
            if ev_name not in evaluations[ds_out_id]:
                evaluations[ds_out_id][ev_name] = []
            evaluations[ds_out_id][ev_name] += [ev_val]


def rank(ev_name, ev_vals):
    arr = np.array(ev_vals)
    if ev_name not in ['fpr95t']:
        arr *= -1
    return np.argsort(arr)


def translate_evaluation_measure_name(name):
    return {
        'fpr95t': r'$\mathit{FPR}_{R=0.95}/\%$',
        'auroc': r'$\mathit{AUROC}/\%$',
        'aupr_in': r'$\mathit{AP}/\%$',
        'aupr_out': r'$\mathit{AP}_\text{n}/\%$',
        'eps': r'$\epsilon/10^{-3}$',
    }[name]


for ds_out_id in list(evaluations.keys()):
    evs = evaluations[ds_out_id]
    for ev_name in list(evs.keys()):
        if ev_name == 'eps':
            ev_strings = [f"{x*1000:.1f}" for x in evs[ev_name]]
        else:
            ev_strings = [f"{x*100:.1f}" for x in evs[ev_name]]
            r = rank(ev_name, evs[ev_name])
            for i, e in enumerate(evs[ev_name]):
                if np.abs(e - evs[ev_name][r[0]]) < 0.003:
                    ev_strings[i] = r"\mathbf{" + ev_strings[i] + r"}"
            #ev_strings[r[1]] = r"\mathbf{" + ev_strings[r[1]] + r"}"
        evs[ev_name] = '$' + r'\;'.join(
            [r'\hnc{' + e + '}' for e in ev_strings]) + '$'
        new_name = translate_evaluation_measure_name(ev_name)
        evs[new_name] = evs[ev_name]
        del evs[ev_name]
    new_name = translate_ds_name(ds_out_id)
    evaluations[new_name] = evaluations[ds_out_id]
    del evaluations[ds_out_id]

pd.set_option('display.max_colwidth', -1)

df = pd.DataFrame.from_dict(evaluations, orient='index')


def table():
    print(f"T={temp}, net='{args.saved_path}''")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begingroup")
    print(r"\newcommand\hnc[1]{\phantom{\mathbf{00.0}}\mathllap{#1}}")
    print(
        pd.DataFrame.from_dict(evaluations, orient='index').to_latex(
            escape=False))
    print(r"\endgroup")
    print(r"}")


print(f"T={temp}, net='{args.saved_path}''")
import pdb
pdb.set_trace()

import os.path
file_path = os.path.dirname(args.saved_path)
with open(file_path + "/ood.log", mode='w') as fs:
    fs.write(df)
    fs.write('\n')
    fs.write(df.to_latex(escape=False))
    fs.flush()

# print(pd.DataFrame.from_dict(evaluations, orient='index').to_latex(escape=False))