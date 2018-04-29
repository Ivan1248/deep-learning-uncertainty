import tensorflow as tf

from .ioutils import console
from .data import Dataset
from .models import AbstractModel
from . import dirs
from .processing.data_augmentation import augment_cifar, random_fliplr
from .visualization import view_semantic_segmentation


def train_cifar(model: AbstractModel,
                ds_train: Dataset,
                ds_val: Dataset,
                epoch_count=200):

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'q':
            return True
        elif text == 's':
            writer = tf.summary.FileWriter(dirs.LOGS, graph=model._sess.graph)
        return False

    model.training_step_event_handler = handle_step

    model.test(ds_val)
    ds_train_part = ds_train[:ds_val.size * 2]
    for i in range(epoch_count):
        ds_train.shuffle()
        prepr_ds_train = Dataset(
            list(map(augment_cifar, ds_train.images)), ds_train.labels,
            ds_train.class_count)
        model.train(prepr_ds_train, epoch_count=1)
        model.test(ds_val, 'validation data')
        model.test(ds_train_part, 'training data subset')


def train_semantic_segmentation(model: AbstractModel,
                                ds_train: Dataset,
                                ds_val: Dataset,
                                epoch_count=200):

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'q':
            return True
        elif text == 's':
            writer = tf.summary.FileWriter(dirs.LOGS, graph=model._sess.graph)
        elif text == 'd':
            view_semantic_segmentation(ds_val, lambda x: model.predict([x])[0])
        return False

    model.training_step_event_handler = handle_step

    model.test(ds_val)
    ds_train_part = ds_train[:ds_val.size * 2]
    for i in range(epoch_count):
        prepr_ds_train = Dataset(
            list(map(random_fliplr, ds_train.images)), ds_train.labels,
            ds_train.class_count)
        model.train(prepr_ds_train, epoch_count=1)  # shuffled here
        model.test(ds_val, 'validation data')
        model.test(ds_train_part, 'training data subset')