from .processing import preprocessing as pp
from .data import datasets, Dataset, MiniBatchReader
from . import dirs

class Cifar10Loader(object):
    mean, std = None, None
    data_path = dirs.DATASETS + '/cifar-10-batches-py'

    @classmethod
    def load_train_val(cls, normalize=True):
        ds = datasets.load_cifar10(cls.data_path, 'train')
        if normalize:
            cls.mean, cls.std = pp.get_normalization_statistics(ds.images)
            ds = Dataset(
                pp.normalize(ds.images, cls.mean, cls.std), ds.labels,
                ds.class_count)
        ds.shuffle()
        ds_train, ds_val = ds.split(0, int(ds.size * 0.8))
        return ds_train, ds_val

    @classmethod
    def load_test(cls,
                  normalize=True,
                  use_test_set_normalization_statistics=False):
        if cls.mean is None:
            cls.load_train_val()
        ds = datasets.load_cifar10(cls.data_path, 'test')
        if not normalize:
            return ds
        mean, std = None, None
        if use_test_set_normalization_statistics:
            mean, std = pp.get_normalization_statistics(ds.images)
        else:
            if cls.mean is None:
                cls.load_train_val()
            mean, std = cls.mean, cls.std
        return Dataset(
            pp.normalize(ds.images, mean, std), ds.labels, ds.class_count)