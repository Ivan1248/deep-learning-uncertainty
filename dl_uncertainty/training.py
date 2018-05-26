import tensorflow as tf
import numpy as np
from tqdm import tqdm

from .ioutils import console
from .data import Dataset, DataLoader
from .models import Model
from . import dirs
from .utils.parsing import parse_log
from .utils.visualization import view_predictions, plot_curves


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
          epoch_count,
          mc_dropout=False,
          jitter=None,
          jitter_name="jitter",
          data_loading_worker_count=0):

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)

        def predict(x, mcd):
            return model.predict(
                x,
                single_input=True,
                mc_dropout=mcd,
                outputs=['output', 'probs_entropy'])

        def view(ds, mcd):
            assert not mcd or mcd and mc_dropout
            view_predictions(ds, lambda x: predict(x, mcd))

        if text is None:
            return
        if text == 'q':
            return True
        elif text[0] == 'd':
            ds = ds_train if 't' in text else ds_val
            if 'h' in text:
                ds = get_hard_examples(model, ds)
            mcd = 'mc' in text
            view(ds, mcd)
        elif text[0] == 'p':
            subset = 'train' if 't' in text else 'val'
            plot_curves(parse_log(model.log, subset))

    model.training_step_event_handler = handle_step

    ds_train_part, _ = ds_train.permute() \
                               .split(min(0.2, len(ds_val) / len(ds_train)))

    ds_train = ds_train.map(jitter, func_name=jitter_name)

    ds_train_loader, ds_val_loader, ds_train_part_loader = [
        DataLoader(
            ds,
            batch_size=model.batch_size,
            shuffle=True,
            num_workers=data_loading_worker_count,
            drop_last=True) for ds in [ds_train, ds_val, ds_train_part]
    ]

    print(f"Starting training ({epoch_count} epochs)...")
    model.test(ds_val_loader, 'validation data')
    for _ in range(epoch_count):
        model.train(ds_train_loader, epoch_count=1)
        model.test(ds_val_loader, 'validation data')
        model.test(ds_train_part_loader, 'training data subset')
    model.test(ds_val_loader, 'validation data', mc_dropout=True)
    model.test(ds_train_part_loader, 'training data subset', mc_dropout=True)