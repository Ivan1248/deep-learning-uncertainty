import numpy as np
import multiprocessing
from tqdm import tqdm
import os

# Normalization


def get_input_mean_std(dataset):
    ms = np.array([(x.mean((0, 1)), x.std((0, 1))) for x, y in dataset])
    m, s = ms.mean(0)
    return m, s


class LazyNormalizer:

    def __init__(self, ds):
        self.ds = ds
        self.mean, self.std = None, None  # TODO: make variables shared between processes
        self._lock = multiprocessing.Lock()
        self._initialization_access_count = 0

    def _initialize(self):
        print(f"Computing dataset statistics for {self.ds.name}")
        self.mean, self.std = get_input_mean_std(tqdm(self.ds))
        self._initialization_access_count += 1
        assert self._initialization_access_count == 1

    def normalize(self, x):
        with self._lock:
            if self.mean is None:  # lazy
                self._initialize()
        return ((x - self.mean) / self.std).astype(np.float32)


# Caching


def example_size(example):
    # assuming img will be float32 after normalization
    img, lab = example
    return img.astype(np.float32).nbytes + np.array(lab).nbytes


class CacheAssigner:

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
