import os
import pickle
import pickle

import numpy as np
import torch.utils.data
from torch.utils.data.dataset import ConcatDataset
from functools import lru_cache
import itertools
from tqdm import tqdm, trange

# Dataset


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, info, name):
        self.name = name
        self.data = data
        self.info = info

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _print(self, *args, **kwargs):
        print(f"{self.name}:", *args, **kwargs)

    def id(self):
        return IdDataset(self)

    def test(self, callback):
        return TestDataset(self, callback)

    def map(self, func=lambda x: x, component=None):
        if component is not None:
            f = func  # to avoid lambda capturing itself
            func = lambda x: tuple(
                    f(c) if i == component else c for i, c in enumerate(x))
        return MappedDataset(self, func)

    def cache(self, max_cache_size=np.inf):
        # caches the dataset in RAM (partially or wholly)
        return CachedDataset(self, max_cache_size)

    def cache_lru(self, max_cache_size=1000):
        # caches the dataset in RAM (partially or wholly) keepeing only the last
        # used examples
        if max_cache_size > len(self):
            return self.cache()
        return LRUCachedDataset(self, max_cache_size)

    def cache_hdd(self, directory, buffer_size=100):
        # caches the whole dataset in RAM and on HDD as a single file
        return HDDCachedDataset(self, directory, buffer_size)

    def cache_hdd_examplewise(self, directory):
        # caches the whole dataset on disk without keeping it in RAM
        return ExamplewiseHDDCachedDataset(self, directory)

    def permute(self, seed=53):
        indices = np.random.RandomState(seed=seed).permutation(len(self))
        ds = SubDataset(self, indices)
        ds.name = self.name + "-permute"
        return ds

    def subset(self, indices):
        return SubDataset(self, indices)

    def split(self, proportion: float):
        indices = np.arange(len(self))
        s = int(proportion * len(self) + 0.5)
        dss = SubDataset(self, indices[:s]), SubDataset(self, indices[s:])
        dss[0].name += f"_0_to_{s-1}"
        dss[1].name += f"_{s}_to_end"
        return dss

    def join(self, other):
        if type(other) is not list:
            other = [other]
        name = f"concat[{self.name}," + ",".join(x.name for x in other) + "]"
        return Dataset(ConcatDataset([self] + other), self.info, name)


class IdDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.name = dataset.name + "-id"
        self.info = self.dataset.info

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class TestDataset(Dataset):

    def __init__(self, dataset, callback):
        self.dataset = dataset
        self.name = dataset.name + "-id"
        self.info = self.dataset.info
        self.callback = callback

    def __getitem__(self, idx):
        self.callback(idx)
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class MappedDataset(Dataset):

    def __init__(self, dataset, func=lambda x: x):
        self.name = dataset.name + "-map"
        self.dataset = dataset
        self.info = self.dataset.info
        self._func = func

    def __getitem__(self, idx):
        return self._func(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class CachedDataset(Dataset):

    def __init__(self, dataset, max_cache_size=np.inf):
        cache_size = min(len(dataset), max_cache_size)
        self.cached_all = cache_size == len(dataset)
        if self.cached_all:
            self.name = dataset.name + "-cache"
            self._print("Caching whole dataset...")
            self.data = [x for x in tqdm(dataset)]
        else:
            self.name = dataset.name + f"-cache_0_to_{cache_size-1}"
            self._print(
                f"Caching {cache_size}/{len(dataset)} of the dataset...")
            self.data = [dataset[i] for i in trange(cache_size)]
            self.dataset = dataset
        self.info = dataset.info

    def __getitem__(self, idx):
        cache_hit = self.cached_all or idx < len(self.data)
        return (self.data if cache_hit else self.dataset)[idx]

    def __len__(self):
        return len(self.data if self.cached_all else self.dataset)


class LRUCachedDataset(Dataset):

    def __init__(self, dataset, max_cache_size=1000):
        self.name = dataset.name + "-cache_lru"
        self.dataset = dataset
        self.info = self.dataset.info

        @lru_cache(maxsize=max_cache_size)
        def _cached_get(i):
            return self.dataset[i]

        self._cached_get = _cached_get

    def __getitem__(self, idx):
        return self._cached_get(idx)

    def __len__(self):
        return len(self.dataset)


class HDDCachedDataset(Dataset):

    def __init__(self, dataset, cache_dir, buffer_size=100):
        self.cache_dir = cache_dir
        self.name = dataset.name + "-cache_hdd"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = f"{cache_dir}/{self.name}.p"
        data = None
        if os.path.exists(cache_path):
            try:
                self._print("Loading dataset cache from hdd...")
                with open(cache_path, 'rb') as f:
                    n = (len(dataset) - 1) // buffer_size + 1  # ceil
                    chunks = [pickle.load(f) for i in trange(n)]
                    data = list(itertools.chain(*chunks))
            except Exception as ex:                
                self._print(ex)
                self._print("Removing invalid hdd-cached dataset...")
                os.remove(cache_path)
        if data is None:
            self._print(f"Caching whole dataset...")
            data = [x for x in tqdm(dataset)]

            def get_chunks(data):
                chunk = []
                for x in data:
                    chunk.append(x)
                    if len(chunk) >= buffer_size:
                        yield chunk
                        chunk = []
                yield chunk

            self._print("Saving dataset cache to hdd...")
            with open(cache_path, 'wb') as f:
                #pickle.dump(data, cache_file, protocol=4)
                # examples pickled individually because of memory constraints
                for x in tqdm(get_chunks(data)):
                    pickle.dump(x, f, protocol=4)
        self.data, self.info = data, dataset.info

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class ExamplewiseHDDCachedDataset(Dataset):

    def __init__(self, dataset, cache_dir):
        self.dataset = dataset
        self.info = dataset.info
        self.name = dataset.name + "-cache_hdd_examplewise"
        self.cache_dir = f"{cache_dir}/{self.name}"
        os.makedirs(self.cache_dir, exist_ok=True)

    def __getitem__(self, idx):
        cache_path = f"{self.cache_dir}/{idx}.p"
        example = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as cache_file:
                    example = pickle.load(cache_file)
            except:
                os.remove(cache_path)
        if example is None:
            example = self.dataset[idx]
            with open(f"{cache_path}", 'wb') as cache_file:
                pickle.dump(example, cache_file, protocol=4)
        return example

    def __len__(self):
        return len(self.dataset)


class SubDataset(Dataset):

    def __init__(self, dataset, indices):
        self.name = dataset.name + "-sub"
        self.dataset = dataset
        self.indices = indices
        self.info = self.dataset.info

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


# DataLoader


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 num_workers=0,
                 drop_last=False):

        collate_fn = lambda batch: tuple(map(np.array, zip(*batch)))

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn)