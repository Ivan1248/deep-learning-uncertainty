import numpy as np
import multiprocessing
from tqdm import tqdm
import os

from . import dirs
from .data import datasets

# Datasets

def get_dataset(name, trainval_test=False):
    if name == 'cifar10':
        ds_path = dirs.DATASETS + '/cifar-10-batches-py'
        ds_train = datasets.Cifar10Dataset(ds_path, 'train')
        if trainval_test:
            ds_test = datasets.Cifar10Dataset(ds_path, 'test')
        else:
            ds_train, ds_test = ds_train.permute().split(0.8)
    elif name == 'mozgalorvc':
        mozgalo_path = dirs.DATASETS + '/mozgalo_robust_ml_challenge'
        ds_train = datasets.MozgaloRVCDataset(
            mozgalo_path, remove_bottom_half=True)
        ds_train, ds_test = ds_train.permute().split(0.8)
        if not trainval_test:
            ds_train, ds_test = ds_train.split(0.8)
    elif name == 'cityscapes':
        ds_path = dirs.DATASETS + '/cityscapes'
        load = lambda s: datasets.CityscapesSegmentationDataset(ds_path, s, \
            downsampling_factor=2, remove_hood=True)
        ds_train, ds_test = map(load, ['train', 'val'])
        if trainval_test:
            ds_train = ds_train.join(ds_test)
            ds_test = load('test')
    elif name == 'camvid':
        ds_path = dirs.DATASETS + '/CamVid'
        load = lambda s: datasets.CamVidDataset(ds_path, s)
        ds_train, ds_test = map(load, ['train', 'val'])
        if trainval_test:
            ds_train = ds_train.join(ds_test)
            ds_test = load('test')
    elif name == 'voc2012':
        ds_path = dirs.DATASETS + '/VOC2012'
        load = lambda s: datasets.VOC2012SegmentationDataset(ds_path, s)
        if trainval_test:
            ds_train, ds_test = map(load, ['trainval', 'test'])
        else:
            ds_train, ds_test = map(load, ['train', 'val'])
    elif name == 'iccv09':
        if trainval_test:
            assert False, "Test set not defined"
        ds_path = dirs.DATASETS + '/iccv09'
        ds_train = datasets.ICCV09Dataset(dirs.DATASETS + '/iccv09')
        ds_train, ds_test = ds_train.permute().split(0.8)
    else:
        assert False, f"Invalid dataset name: {name}"
    return ds_train, ds_test

# Normalization


def get_input_mean_std(dataset):
    ms = np.array([(x.mean((0, 1)), x.std((0, 1))) for x, y in dataset])
    m, s = ms.mean(0)
    return m, s


class LazyNormalizer:

    def __init__(self, ds):
        self.ds = ds
        self.mean, self.std = get_input_mean_std([ds[0], ds[1]])
        self.initialized = multiprocessing.Value('i', 0)
        self.mean = multiprocessing.Array('f', self.mean)
        self.std = multiprocessing.Array('f', self.std)

    def _initialize(self):
        print(f"Computing dataset statistics for {self.ds.name}")
        self.mean.value, self.std.value = get_input_mean_std(tqdm(self.ds))

    def normalize(self, x):
        with self.initialized.get_lock():
            if not self.initialized.value:  # lazy
                self._initialize()
                self.initialized.value = True
        return ((x - self.mean) / self.std).astype(np.float32)


# Caching


def example_size(example):
    # assuming img will be float32 after normalization
    img, lab = example
    return img.astype(np.float32).nbytes + np.array(lab).nbytes


class CacheSpaceAssigner:

    def __init__(self, cache_dir, max_cache_size):
        self.cache_max = max_cache_size
        self.cache_left = max_cache_size
        self.cache_dir = cache_dir

    def cache(self, ds):  # caching (HDD, RAM)
        if self.cache_left >= len(ds):
            self.cache_left -= len(ds)
            return ds.cache_hdd(self.cache_dir)
        elif self.cache_left == 0:
            return ds.cache_hdd_only(self.cache_dir)
        else:
            ds1, ds2 = ds.split(self.cache_left / len(ds))
            self.cache_left = 0
            ds1 = ds1.cache_hdd(self.cache_dir)
            ds2 = ds2.cache_hdd_only(self.cache_dir)
            return ds1.join(ds2)

    def cache_ram_only(self, ds):  # caching (HDD, RAM)
        assert self.cache_left >= len(ds), \
            f"Data doesn't fit in cache space ({len(ds)}>{self.cache_left})"
        return ds.cache()

    @property
    def cache_used(self):
        return self.cache_max - self.cache_left    