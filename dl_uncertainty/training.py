import tensorflow as tf
import numpy as np
from tqdm import tqdm

from .ioutils import console
from .data import Dataset, DataLoader
from .models import Model
from . import dirs
from .visualization import view_semantic_segmentation, plot_curves_from_log_lines


def get_hard_examples(model, ds):
    print("Looking for hard examples...")
    dl = DataLoader(ds, batch_size=model.batch_size, drop_last=False)
    predictions = np.concatenate([model.predict(ims) for ims, _ in tqdm(dl)])
    labels = np.concatenate([labs for _, labs in dl])
    hard_mask = predictions != labels
    if len(labels.shape) > 1:
        hard_mask = hard_mask.mean(axis=np.arange(1, len(labels.shape)))
    hard_indices = np.concatenate(np.argwhere(hard_mask))
    for i in hard_indices:
        print(i)
    return ds.subset(hard_indices)


def train(model: Model,
          ds_train: Dataset,
          ds_val: Dataset,
          jitter=None,
          epoch_count=200,
          data_loading_worker_count=0):

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'q':
            return True
        elif text == 's':
            writer = tf.summary.FileWriter(dirs.LOGS, graph=model._sess.graph)
        elif text == 'd':
            view_semantic_segmentation(ds_val, lambda x: model.predict([x])[0])
        elif text == 'dt':
            view_semantic_segmentation(ds_train,
                                       lambda x: model.predict([x])[0])
        elif text == 'dc':
            plot_curves_from_log_lines(model.log)
        elif text == 'h':
            view_semantic_segmentation(
                get_hard_examples(model, ds_val),
                lambda x: model.predict([x])[0])
        return False

    model.training_step_event_handler = handle_step

    ds_train_part, _ = ds_train.permute() \
                               .split(min(0.2, len(ds_val) / len(ds_train)))
    ds_train = ds_train.map(jitter)

    ds_train_loader, ds_val_loader, ds_train_part_loader = [
        DataLoader(
            ds,
            batch_size=model.batch_size,
            shuffle=True,
            num_workers=data_loading_worker_count,
            drop_last=True) for ds in [ds_train, ds_val, ds_train_part]
    ]

    print(f"Starting training ({epoch_count} epochs)...")
    model.test(ds_val_loader)
    for i in range(epoch_count):
        model.train(ds_train_loader, epoch_count=1)
        model.test(ds_val_loader, 'validation data')
        model.test(ds_train_part_loader, 'training data subset')