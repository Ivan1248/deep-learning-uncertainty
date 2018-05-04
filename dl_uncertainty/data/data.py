import os
import pickle

import numpy as np
import torch.utils.data
from torch.utils.data.dataset import ConcatDataset
from functools import lru_cache
from tqdm import tqdm

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

    def map(self, func=lambda x: x, component=None):
        if component is not None:
            f = func
            func = lambda x: tuple(
                    f(c) if i == component else c for i, c in enumerate(x))
        return MappedDataset(self, func)

    def cache(self, max_cache_size=1000):
        if max_cache_size > len(self):
            return self.cache_all()
        return CachedDataset(self, max_cache_size)

    def cache_all(self):
        return CompletelyCachedDataset(self)

    def cache_all_disk(self, directory):
        return CompletelyCachedDataset(self, directory)

    def subset(self, indices):
        return SubDataset(self, indices)

    def shuffle(self):
        indices = np.random.permutation(len(self))
        ds = SubDataset(self, indices)
        ds.name += "shuffled"
        return ds

    def split(self, proportion: float):
        indices = np.arange(len(self))
        s = int(proportion * len(self) + 0.5)
        dss = SubDataset(self, indices[:s]), SubDataset(self, indices[s:])
        dss[0].name += f"0to{s}"
        dss[1].name += f"{s}toend"
        return dss

    def join(self, other):
        if type(other) is not list:
            other = [other]
        name = f"concat[{self.name}," + ",".join(x.name for x in other) + "]"
        return Dataset(ConcatDataset([self] + other), self.info, name)


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

    def __init__(self, dataset, max_cache_size=1000):
        self.name = dataset.name + "-cache"
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


class CompletelyCachedDataset(Dataset):

    def __init__(self, dataset, cache_dir=None):
        print("Loading whole dataset into memory...")
        self.cache_dir = cache_dir
        if cache_dir is not None:
            self.name = dataset.name + "-diskfullcache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = f"{cache_dir}/{self.name}.cache"
            cache = None
            if os.path.exists(cache_path):
                try:
                    print("Loading dataset cache from disk...")
                    cache = pickle.load(open(cache_path, 'rb'))
                except:
                    print("Removing invalid disk-cached dataset...")
                    os.remove(cache_path)
            if cache is not None:
                self.data, self.info = cache
            else:
                self.data, self.info = [x for x in tqdm(dataset)], dataset.info
            if cache is None:
                print("Saving dataset cache to disk...")
                pickle.dump(
                    (self.data, self.info),
                    open(f"{cache_path}", 'wb'),
                    protocol=4)
        else:
            self.name = dataset.name + "-fullcache"
            self.data, self.info = [x for x in tqdm(dataset)], dataset.info

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


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
                 shuffle=True,
                 num_workers=0,
                 drop_last=True):

        collate_fn = lambda batch: tuple(map(np.array, zip(*batch)))

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn)