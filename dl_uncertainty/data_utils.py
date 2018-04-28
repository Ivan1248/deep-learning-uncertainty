from .processing import preprocessing as pp
from .data import datasets, Dataset, MiniBatchReader
from . import dirs


class ImageDatasetLoader:

    def __init__(self, data_path, loader, subsets, default_statistics_subset):
        self.mean, self.std = None, None
        self.data_path = data_path
        self.loader = loader
        self.subsets = subsets
        self.statistics_subset = default_statistics_subset

    def change_statistics_subset(self, subset):
        self.default_statistics_subset = subset
        self.mean, self.std = None, None

    def load(self, subset='all', normalize=False, **kwargs):
        assert not normalize, "Normalization is done in the model"

        def normalize_dataset(ds):
            images = pp.normalize(ds.images, self.mean, self.std)
            return Dataset(images, ds.labels, ds.class_count)

        assert subset in self.subsets
        print(f"Loading '{subset}' subset")
        ds = self.loader(self.data_path, subset, **kwargs)
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


Cifar10Loader = ImageDatasetLoader(
    data_path=dirs.DATASETS + '/cifar-10-batches-py',
    loader=datasets.load_cifar10,
    subsets=['train', 'test'],
    default_statistics_subset='train')

CityscapesSegmentation = ImageDatasetLoader(
    data_path=dirs.DATASETS + '/cityscapes',
    loader=datasets.load_cityscapes_segmentation,
    subsets=['train', 'val', 'trainval'],
    default_statistics_subset='train')

VOC2012SegmentationLoader = ImageDatasetLoader(
    data_path=dirs.DATASETS + '/VOC2012',
    loader=datasets.load_voc2012_segmentation,
    subsets=['train', 'val', 'trainval'],
    default_statistics_subset='train')

ICCV09Loader = ImageDatasetLoader(
    data_path=dirs.DATASETS + '/iccv09',
    loader=lambda dp, ss: datasets.load_iccv09(dp),
    subsets=['all'],
    default_statistics_subset='all')
