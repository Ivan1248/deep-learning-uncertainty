import os

import numpy as np

from .processing import preprocessing as pp
from .data import dataset_loaders, Dataset, MiniBatchReader, tfrecords
from . import dirs


class DatasetLoader:

    def __init__(self, data_path, loader, subsets, default_statistics_subset):
        self.mean, self.std = None, None
        self.data_path = data_path
        self.loader = loader
        self.subsets = subsets
        self.statistics_subset = default_statistics_subset

    def change_statistics_subset(self, subset):
        self.default_statistics_subset = subset
        self.mean, self.std = None, None

    def get_generator(self, subset='all', **kwargs):
        assert subset in self.subsets
        return self.loader(self.data_path, subset, **kwargs)

    def load(self, subset='all', normalize=False, **kwargs):
        assert not normalize, "Normalization is done in the model"

        def normalize_dataset(ds):
            images = pp.normalize(ds.images, self.mean, self.std)
            return Dataset(images, ds.labels, ds.class_count)

        dsgen = self.get_generator(subset=subset, **kwargs)
        print(f"Loading '{subset}' subset")
        ds = dsgen.to_dataset()
        if normalize:
            if subset == self.statistics_subset:
                if self.mean is None:
                    print("Computing data normalization statistics " +
                          f"based on '{self.statistics_subset}' subset")
                    self.mean, self.std = pp.get_normalization_statistics(
                        ds.images)
            else:  # learn statistics
                self.load(self.statistics_subset, normalize=True)
            ds = normalize_dataset(ds)
        return ds

    def get_tfrecords_list(self, subset, **kwargs):
        # returns a list of tfrecords file names
        tfrecords_path = f"{self.data_path}-tfrecords/{subset}"
        if not os.path.exists(tfrecords_path):
            os.makedirs(tfrecords_path)
            dsgen = self.get_generator(subset=subset, **kwargs)
            tfrecords.prepare_dataset(dsgen, tfrecords_path)
        return os.listdir(tfrecords_path)


Cifar10Loader = DatasetLoader(
    data_path=dirs.DATASETS + '/cifar-10-batches-py',
    loader=dataset_loaders.load_cifar10,
    subsets=['train', 'test'],
    default_statistics_subset='train')

CityscapesSegmentation = DatasetLoader(
    data_path=dirs.DATASETS + '/cityscapes',
    loader=dataset_loaders.load_cityscapes_segmentation,
    subsets=['train', 'val', 'trainval'],
    default_statistics_subset='train')

VOC2012SegmentationLoader = DatasetLoader(
    data_path=dirs.DATASETS + '/VOC2012',
    loader=dataset_loaders.load_voc2012_segmentation,
    subsets=['train', 'val', 'trainval'],
    default_statistics_subset='train')

ICCV09Loader = DatasetLoader(
    data_path=dirs.DATASETS + '/iccv09',
    loader=lambda data_path, subset: dataset_loaders.load_iccv09(data_path),
    subsets=['all'],
    default_statistics_subset='all')


def get_input_mean_std(dataset):
    ms = np.array([(x.mean((0, 1)), x.std((0, 1))) for x, y in dataset])
    m, s = ms.mean(0)
    return m, s
