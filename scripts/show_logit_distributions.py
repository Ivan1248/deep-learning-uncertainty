import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_utils import Cifar10Loader, Dataset, MiniBatchReader
import visualization
from visualization import compose, Viewer
import dirs
from training import train
import standard_resnets


print("Loading and preparing data")
ds_test = Cifar10Loader.load_test()
ds_test = ds_test.split(0, 4096)[0]
images_outof = list(map(np.flipud, ds_test.images))
ds_test_outof = Dataset(images_outof, ds_test.labels, ds_test.class_count)


print("Loading model")
zaggydepth, k = (28, 10)
from standard_resnets import get_wrn
model = standard_resnets.get_wrn(
    zaggydepth,
    k,
    ds_test.image_shape,
    ds_test.class_count)
saved_path = dirs.SAVED_MODELS + '/wrn-28-10-t--2018-01-23-19-13/ResNet'  # vanilla
model.load_state(saved_path)


print("Printing logit biases")
with model._graph.as_default():
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    logit_biases_var = list(filter(lambda x: x.name=='Conv/biases:0', vars))[0]
    logit_biases = model._sess.run(logit_biases_var)
    print("Logit biases:", logit_biases)
    print("Logit biases sum:", np.sum(logit_biases))
    print("Logit biases mean:", np.average(logit_biases))


print("Collecting logit statistics")
def get_logit_value_statistics(ds, biases=0, stdevs=1):
    cost, ev = model.test(ds)
    values ={'logits':[], 'max':[],'nonmax':[], 'sorted':[], 'sums':[]}

    mbr = MiniBatchReader(ds,128)

    for _ in range(len(ds)//128):
        ims, labs = mbr.get_next_batch()
        logitses = (model._run(model.nodes.logits, ims) -biases)/stdevs
        for logits in logitses:
            sorted_logits = np.sort(logits)[::-1]
            values['max'].append(sorted_logits[0])
            values['nonmax'].extend(sorted_logits[1:])
            values['logits'].append(logits)
            values['sorted'].append(sorted_logits)
            values['sums'].append(np.sum(logits))
    return values

# THESE ARE NOT USED FOR PLOTS NOW
values_in = get_logit_value_statistics(ds_test)
values_outof = get_logit_value_statistics(ds_test_outof)

logits_matrix = np.stack(values_in['logits'])
logit_means = np.average(logits_matrix, axis=0)
logit_stds = np.std(logits_matrix, axis=0)

print("Logit means:", logit_means)
print("Logit stdevs:", logit_stds)

x = np.arange(logit_means.shape[0])
plt.title("Logit values") 
plt.scatter(x, logit_means, label='mean')
plt.scatter(x, logit_biases, label='bias')
plt.xlabel('index')
plt.ylabel('value')
points = np.array([[ind[1], v] for ind, v in np.ndenumerate(logits_matrix)])
print(points)
plt.scatter(points[:,0], points[:,1], alpha=0.1, s=3, edgecolors='none', color='gray')
plt.legend()
plt.show()

plt.title("Logit values - standardized") 
plt.xlabel('index')
plt.ylabel('value')
points = np.array([[ind[1], v] for ind, v in np.ndenumerate((logits_matrix-logit_means)/logit_stds)])
print(points)
plt.scatter(points[:,0], points[:,1], alpha=0.1, s=3, edgecolors='none', color='gray')
plt.legend()
plt.show()

# THESE ARE USED FOR PLOTS
values_in = get_logit_value_statistics(ds_test, logit_means, logit_stds)
values_outof = get_logit_value_statistics(ds_test_outof, logit_means, logit_stds)

print("Generating plots")
r = (np.min(values_in['nonmax']+values_outof['nonmax']), np.max(values_in['max']+values_outof['max']))

plt.hist(values_in['max'], bins=40, range=r, alpha=0.5, label='CIFAR-10-test') 
plt.hist(values_outof['max'], bins=40, range=r, alpha=0.5, label='CIFAR-10-test-ud')
plt.title("[Max logit] distribution") 
plt.legend()
plt.show()

plt.hist(values_in['nonmax'], bins=40, range=r, alpha=0.5, label='CIFAR-10-test') 
plt.hist(values_outof['nonmax'], bins=40, range=r, alpha=0.5, label='CIFAR-10-test-ud')
plt.title("[Non-max logit sum] distribution") 
plt.legend()
plt.show()

plt.hist(np.concatenate(values_in['sorted']), bins=40, range=r, alpha=0.5, label='CIFAR-10-test') 
plt.hist(np.concatenate(values_outof['sorted']), bins=40, range=r, alpha=0.5, label='CIFAR-10-test-UD')
plt.title("[Any logit] distribution") 
plt.xlabel('value')
plt.ylabel('frequency')
plt.legend()
plt.show()

for values, label in [(values_in, 'CIFAR-10-test'), (values_outof, 'CIFAR-10-test-UD')]:
    plt.scatter(
        [sl[0] for sl in values['sorted']],
        [np.sum(sl) for sl in values['sorted']], 
        alpha=0.5, s=1, edgecolors='none', label=label)
plt.xlabel('max(logits)')
plt.ylabel('sum(logits)')
plt.legend()
plt.show()

for values, label in [(values_in, 'CIFAR-10-test'), (values_outof, 'CIFAR-10-test-UD')]:
    plt.scatter(
        [sl[0] for sl in values['sorted']],
        [sl[1] for sl in values['sorted']], 
        alpha=0.5, s=2, edgecolors='none', label=label) 
plt.xlabel('max(logits)')
plt.ylabel('sorted_logits[1]')
plt.legend()
plt.show()

for values, label in [(values_in, 'CIFAR-10-test'), (values_outof, 'CIFAR-10-test-UD')]:
    plt.scatter(
        [sl[0] for sl in values['sorted']],
        [sl[1]+sl[2]+sl[3] for sl in values['sorted']], 
        alpha=0.5, s=2, edgecolors='none', label=label) 
plt.xlabel('max(logits)')
plt.ylabel('sum(sorted_logits[1:3])')
plt.legend()
plt.show()

