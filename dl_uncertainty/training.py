import tensorflow as tf

from .ioutils import console
from .data import Dataset
from .models import AbstractModel
from . import dirs
from .processing.data_augmentation import augment_cifar


def train(model: AbstractModel,
          ds_train: Dataset,
          ds_val: Dataset,
          epoch_count=200):

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'q':
            return True
        if text == 's':
            writer = tf.summary.FileWriter(dirs.LOGS, graph=model._sess.graph)
        return False

    model.training_step_event_handler = handle_step

    model.test(ds_val)
    ds_train_part = ds_train[:ds_val.size*2]
    for i in range(epoch_count):
        prepr_ds_train = Dataset(
            list(map(augment_cifar, ds_train.images)), ds_train.labels,
            ds_train.class_count)
        model.train(prepr_ds_train, epoch_count=1)
        model.test(ds_val, 'validation data')
        model.test(ds_train_part, 'training data subset')
