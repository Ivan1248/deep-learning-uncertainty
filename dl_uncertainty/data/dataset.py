import numpy as np
import random

class Dataset:
    def __init__(self, images, labels, class_count: int, random_seed=51):
        self._images = np.ascontiguousarray(np.array(images))
        self._labels = np.ascontiguousarray(np.array(labels))
        self._class_count = class_count
        self._rand = np.random.RandomState(random_seed)
        self._indices = np.arange(len(self._images))

    def __len__(self):
        return len(self._images)

    def __getitem__(self, key):
        if isinstance(key, int):  # int
            return self._images[key], self._labels[key]
        return Dataset(
            self._images.__getitem__(key),
            self._labels.__getitem__(key), self.class_count)

    @property
    def size(self) -> int:
        return len(self._images)

    @property
    def input_shape(self):
        return self._images[0].shape

    @property
    def class_count(self):
        return self._class_count

    def set_random_seed(self, seed:int):
        self._rand.seed(seed)

    def shuffle(self, random_seed=None):
        indices = np.arange(self._images.shape[0])
        self._rand.shuffle(indices)
        arrs = [self._images, self._labels, self._indices]
        arrs = [np.ascontiguousarray(arr[indices]) for arr in arrs]
        self._images, self._labels, self._indices = arrs

    def unshuffle(self, random_seed=None):
        indices = np.zeros(len(self))
        for i, k in enumerate(self._indices):
            indices[k] = i
        arrs = [self._images, self._labels, self._indices]
        arrs = [np.ascontiguousarray(arr[indices]) for arr in arrs]
        self._images, self._labels, self._indices = arrs

    @staticmethod
    def join(ds1, ds2):
        assert (ds1.class_count == ds2.class_count)
        return Dataset(
            np.concatenate([ds1.images, ds2.images]),
            np.concatenate([ds1.labels, ds2.labels]), ds1.class_count)

    def split(self, start, end):
        """ Splits the dataset into two datasets. """
        return self[start:end], Dataset.join(self[:start], self[end:])

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def unpack(self):
        return self._images, self._labels


def save_dataset(path: str):
    if not path.endswith('.npy'):
        path += '.npy'
    np.load(path)

def load_dataset(path: str) -> Dataset:
    if not path.endswith('.npy'):
        path += '.npy'
    return np.load(path)[()]


class MiniBatchReader:
    def __init__(self, dataset: Dataset, batch_size: int):
        self.current_batch_number = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_batches = dataset.size // batch_size

    def reset(self, shuffle: bool = False):
        if shuffle:
            self.dataset.shuffle()
        self.current_batch_number = 0

    def get_next_batch(self):
        """ Return the next `batch_size` image-label pairs. """
        end = self.current_batch_number + self.batch_size
        if end > self.dataset.size:  # Finished epoch
            return None
        else:
            start = self.current_batch_number
        self.current_batch_number = end
        return self.dataset[start:end].unpack()

    def get_generator(self):
        b = self.get_next_batch()
        while b is not None:
            b = self.get_next_batch()
            yield b