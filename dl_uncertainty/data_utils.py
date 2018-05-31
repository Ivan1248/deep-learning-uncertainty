import os
import numpy as np
import multiprocessing
from tqdm import tqdm
import pickle

from . import dirs
from .data import datasets
from .processing.data_augmentation import random_fliplr_with_label, augment_cifar

# Datasets


def get_dataset(name, trainval_test=False):
    if name == 'cifar':
        ds_path = dirs.DATASETS + '/cifar-10-batches-py'
        ds_train = datasets.Cifar10Dataset(ds_path, 'train')
        if trainval_test:
            ds_test = datasets.Cifar10Dataset(ds_path, 'test')
        else:
            ds_train, ds_test = ds_train.permute().split(0.8)
    elif name == 'tinyimagenet':
        ds_path = dirs.DATASETS + '/tiny-imagenet-200'
        ds_train = datasets.TinyImageNet(ds_path, 'train')
        ds_test = datasets.TinyImageNet(ds_path, 'val')
        if trainval_test:
            ds_train = ds_train + ds_test
            ds_test = datasets.TinyImageNet(ds_path, 'test')
    elif name == 'svhn':
        ds_path = dirs.DATASETS + '/tiny-imagenet-200'
        ds_train = datasets.TinyImageNet(ds_path, 'train')
        ds_test = datasets.TinyImageNet(ds_path, 'val')
        if trainval_test:
            ds_train = ds_train + ds_test
            ds_test = datasets.TinyImageNet(ds_path, 'test')
    elif name.startswith('mozgalo'):
        mozgalo_path = dirs.DATASETS + '/mozgalo_robust_ml_challenge'
        ds_train = datasets.MozgaloRVCDataset(
            mozgalo_path, remove_bottom_proportion=0.5, downsampling_factor=4)
        if name.startswith('mozgaloood'):
            test_labels = [0, 6, 12, 18, 24]
            filt_ood_test = lambda xy: xy[1] in test_labels
            filt_ood_train = lambda xy: xy[1] not in test_labels
            if name[7:] == 'ood':
                return ds_train.filter(filt_ood_train), \
                       ds_train.filter(filt_ood_test)
            else:
                if name[7:] == 'oodtest':
                    filt = filt_ood_test
                elif name[7:] == 'oodtrain':
                    filt = filt_ood_train
                ds_train = ds_train.filter(filt)
        ds_train, ds_test = ds_train.permute().split(0.8)
        if not trainval_test:
            ds_train, ds_test = ds_train.split(0.8)
    elif name == 'cityscapes':
        ds_path = dirs.DATASETS + '/cityscapes'
        load = lambda s: datasets.CityscapesSegmentationDataset(ds_path, s, \
            downsampling_factor=2, remove_hood=True)
        ds_train, ds_test = map(load, ['train', 'val'])
        if trainval_test:
            ds_train = ds_train + ds_test
            ds_test = load('test')
    elif name == 'wilddash':
        assert not trainval_test
        ds_path = dirs.DATASETS + '/wilddash'
        load = lambda s: datasets.WildDashSegmentationDataset(ds_path, s, downsampling_factor=2)
        return tuple(map(load, ['val', 'bench']))
    elif name == 'camvid':
        ds_path = dirs.DATASETS + '/CamVid'
        load = lambda s: datasets.CamVidDataset(ds_path, s)
        ds_train, ds_test = map(load, ['train', 'val'])
        if trainval_test:
            ds_train = ds_train + ds_test
            ds_test = load('test')
    elif name == 'voc2012':
        ds_path = dirs.DATASETS + '/VOC2012'
        load = lambda s: datasets.VOC2012SegmentationDataset(ds_path, s)
        if trainval_test:
            ds_train, ds_test = map(load, ['trainval', 'test'])
        else:
            ds_train, ds_test = map(load, ['train', 'val'])
    elif name == 'iccv09':
        ds_train = datasets.ICCV09Dataset(dirs.DATASETS + '/iccv09')
        ds_train, ds_test = ds_train.permute().split(0.8)
        if not trainval_test:
            ds_train, ds_test = ds_train.split(0.8)
    elif name == 'isun':
        ds_path = dirs.DATASETS + '/iSUN'
        load = lambda s: datasets.ISUNDataset(ds_path, s)
        ds_train, ds_test = map(load, ['training', 'validation'])
        if trainval_test:
            ds_train = ds_train + ds_test
            ds_test = load('testing')
    else:
        assert False, f"Invalid dataset name: {name}"
    return ds_train, ds_test


# Normalization


def get_input_mean_std(dataset):
    ms = np.array([(x.mean((0, 1)), x.std((0, 1))) for x, y in dataset])
    m, s = ms.mean(0)
    return m, s


class LazyNormalizer:

    def __init__(self, ds, cache_dir=None):
        self.ds = ds
        self.mean, self.std = get_input_mean_std([ds[0], ds[1]])
        self.initialized = multiprocessing.Value('i', 0)
        self.mean = multiprocessing.Array('f', self.mean)
        self.std = multiprocessing.Array('f', self.std)
        self.cache_dir = f"{cache_dir}/lazy-normalizer-cache/"
        self.cache_path = f"{self.cache_dir}/{ds.name}.p"

    def _initialize(self):
        mean_std = None
        if os.path.exists(self.cache_path):
            try:
                print(f"Loading dataset statistics for {self.ds.name}")
                with open(self.cache_path, 'rb') as cache_file:
                    mean_std = pickle.load(cache_file)
            except:
                os.remove(self.cache_path)
        if mean_std is None:
            print(f"Computing dataset statistics for {self.ds.name}")
            mean_std = get_input_mean_std(tqdm(self.ds))
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(f"{self.cache_path}", 'wb') as cache_file:
                pickle.dump(mean_std, cache_file, protocol=4)
        self.mean.value, self.std.value = mean_std

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
            return ds1 + ds2

    def cache_ram_only(self, ds):  # caching (HDD, RAM)
        assert self.cache_left >= len(ds), \
            f"Data doesn't fit in cache space ({len(ds)}>{self.cache_left})"
        return ds.cache()

    @property
    def cache_used(self):
        return self.cache_max - self.cache_left


# Cached dataset with normalized inputs


def get_cached_dataset_with_normalized_inputs(name, trainval_test=False):
    ds_train, ds_test = get_dataset(name, trainval_test)
    dss = (ds_train, ds_test)
    cache_dir = f"{dirs.CACHE}"
    print("Setting up data preprocessing...")
    normalizer = LazyNormalizer(ds_train, cache_dir)
    dss = map(lambda ds: ds.map(normalizer.normalize, 0), dss)
    print("Setting up data caching on HDD...")
    dss = map(lambda ds: ds.cache_hdd_only(cache_dir), dss)
    return tuple(dss)


# Augmentation


def get_augmentation_func(dataset):
    if dataset.info['id'] in ['cifar', 'tinyimagenet']:
        return lambda xy: (augment_cifar(xy[0]), xy[1])
    elif dataset.info['problem_id'] == 'semseg':
        return random_fliplr_with_label
    else:
        return lambda x: x
