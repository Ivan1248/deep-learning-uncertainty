import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from _context import dl_uncertainty

from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.data import DataLoader, Dataset
from dl_uncertainty.processing.data_augmentation import random_fliplr, augment_cifar
from dl_uncertainty import parameter_loading

print("Loading and preparing data")
_, ds = data_utils.get_dataset('cifar', trainval_test=True)
ds = ds.permute().split(0, 4096)[0]
ds.name = 'CIFAR-10-test[:4096]'

ds_ud = ds.map(np.flipud, 0)
ds_rand = Dataset(data=np.random.randn(ds[0][0].shape), name='rand')

dsid_to_ds = {'CIFAR-10': ds, 'CIFAR-10-flipud': ds_ud, 'random': ds_rand}

print("Loading model")

model = model_utils.get_model(
    net_name='wrn',
    depth=28,
    width=10,
    problem='cls',
    epoch_count=1,
    ds_id='cifar',
    ds_train=ds)

saved_path = dirs.SAVED_NETS + '/wrn-28-10-t--2018-01-23-19-13/ResNet'  # vanilla
model.load_state(saved_path)

print("Printing logit biases")
with model._graph.as_default():
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    logit_biases_var = list(
        filter(lambda x: x.name == 'conv_logits/bias:0', vars))[0]
    logit_biases = model._sess.run(logit_biases_var)
    print("Logit biases:", logit_biases)
    print("Logit biases sum:", logit_biases.sum())
    print("Logit biases mean:", logit_biases.mean())

print("Collecting logit statistics")


def get_logit_value_statistics(ds, biases=0, stddevs=1):
    cost, ev = model.test(ds)
    values = {'logits': [], 'max': [], 'nonmax': [], 'sorted': []}

    for images, labels in DataLoader(ds, batch_size=model.batch_size):
        logitses = (model._run(model.nodes.logits, images) - biases) / stddevs
        for logits in logitses:
            sorted_logits = np.sort(logits)[::-1]
            values['all'].append(logits)
            values['sorted'].append(sorted_logits)
            values['max'].append(sorted_logits[0])
            values['nonmax'].extend(sorted_logits[1:])
    return values


# THESE ARE NOT USED FOR PLOTS NOW
values_in = get_logit_value_statistics(ds)
#values_ud = get_logit_value_statistics(ds_ud)
#values_rand = get_logit_value_statistics(ds_rand)

logits_matrix = np.stack(values_in['all'])
logit_means = np.average(logits_matrix, axis=0)
logit_stds = np.std(logits_matrix, axis=0)

print("Logit means:", logit_means)
print("Logit stddevs:", logit_stds)

x = np.arange(logit_means.shape[0])
plt.title("Logit values")
plt.scatter(x, logit_means, label='mean')
plt.scatter(x, logit_biases, label='bias')
plt.xlabel('index')
plt.ylabel('value')
points = np.array([[ind[1], v] for ind, v in np.ndenumerate(logits_matrix)])
print(points)
plt.scatter(
    points[:, 0], points[:, 1], alpha=0.1, s=3, edgecolors='none', color='gray')
plt.legend()
plt.show()

plt.title("Logit values - standardized")
plt.xlabel('index')
plt.ylabel('value')
points = np.array(
    [[ind[1], v]
     for ind, v in np.ndenumerate((logits_matrix - logit_means) / logit_stds)])
print(points)
plt.scatter(
    points[:, 0], points[:, 1], alpha=0.1, s=3, edgecolors='none', color='gray')
plt.legend()
plt.show()

# THESE ARE USED FOR PLOTS
dsid_to_values = {
    dsid: get_logit_value_statistics(dsid_to_ds[ds], logit_means, logit_stds)
    for dsid, ds in dsid_to_ds.items()
}

print("Generating plots")
r = (np.min(np.stack([v['nonmax'] for v in dsid_to_values.values()])),
     np.min(np.stack([v['max'] for v in dsid_to_values.values()])))


def hist(x, label):
    plt.hist(x, bins=40, range=r, alpha=0.5, label=label)


for valkind, titlekind in [('max', 'Max'), ('nonmax', 'Non-max'), ('Any',
                                                                   'all')]:
    for dsid, values in dsid_to_values.items():
        hist(values[valkind], dsid)
        hist(values[valkind], dsid)
    plt.title(f"[{titlekind} logit] distribution")
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()

for label, values in dsid_to_values.items():
    plt.scatter(
        [sl[0]
         for sl in values['sorted']], [np.sum(sl) for sl in values['sorted']],
        alpha=0.5,
        s=1,
        edgecolors='none',
        label=label)
plt.xlabel('max(logits)')
plt.ylabel('sum(logits)')
plt.legend()
plt.show()

for label, values in dsid_to_values.items():
    plt.scatter(
        [sl[0] for sl in values['sorted']], [sl[1] for sl in values['sorted']],
        alpha=0.5,
        s=2,
        edgecolors='none',
        label=label)
plt.xlabel('max(logits)')
plt.ylabel('sorted_logits[1]')
plt.legend()
plt.show()

for label, values in dsid_to_values.items():
    plt.scatter(
        [sl[0] for sl in values['sorted']],
        [sl[1] + sl[2] + sl[3] for sl in values['sorted']],
        alpha=0.5,
        s=2,
        edgecolors='none',
        label=label)
plt.xlabel('max(logits)')
plt.ylabel('sum(sorted_logits[1:3])')
plt.legend()
plt.show()
