import os, glob
import pickle

import PIL.Image as pimg
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import torch.utils.data
from functools import lru_cache

from .data import Dataset

from ..processing.shape import pad_to_shape, crop
from ..ioutils import file


def _load_image(path):
    return np.array(pimg.open(path))


# Datasets


class Cifar10Dataset(Dataset):

    def __init__(self, data_dir, subset='train'):
        assert subset in ['train', 'test']

        def unpickle(file):
            with open(file, 'rb') as f:
                return pickle.load(f, encoding='latin1')

        h, w, ch = 32, 32, 3
        if subset == 'train':
            train_x = np.ndarray((0, h * w * ch), dtype=np.float32)
            train_y = []
            for i in range(1, 6):
                subset = unpickle(os.path.join(data_dir, 'data_batch_%d' % i))
                train_x = np.vstack((train_x, subset['data']))
                train_y += subset['labels']
            train_x = train_x.reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
            train_y = np.array(train_y, dtype=np.int32)
            self.x, self.y = train_x, train_y
        elif subset == 'test':
            subset = unpickle(os.path.join(data_dir, 'test_batch'))
            test_x = subset['data'].reshape((-1, ch, h, w)).transpose(
                0, 2, 3, 1).astype(np.float32)
            test_y = np.array(subset['labels'], dtype=np.int32)
            self.x, self.y = test_x, test_y
        else:
            raise ValueError("The value of subset must be in {'train','test'}.")
        self.info = {'class_count': 10}
        self.name = f"Cifar10Dataset-{subset}"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class SVHNDataset(Dataset):

    def __init__(self, data_dir, subset='train'):
        assert subset in ['train', 'test']
        import scipy.io as sio
        data = sio.loadmat(subset + '_32x32.mat')
        self.x, self.y = data['X'], np.remainder(data['y'], 10)
        self.info = {'class_count': 10}
        self.name = f"SVHNDataset-{subset}"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class ICCV09DiskDataset(Dataset):

    def __init__(self, data_dir):  # TODO subset
        self._shape = [240, 320]
        self._images_dir = f'{data_dir}/images'
        self._labels_dir = f'{data_dir}/labels'
        self._image_list = [x[:-4] for x in os.listdir(self._images_dir)]

        self.info = {'class_count': 8}
        self.name = "ICCV09Dataset"

    def _get_image(self, i):
        img = _load_image(f"{self._images_dir}/{self._image_list[i]}.jpg")
        return pad_to_shape(crop(img, self._shape), self._shape)

    def _get_label(self, i):
        path = f"{self._labels_dir}/{self._image_list[i]}.regions.txt"
        label = np.loadtxt(path, dtype=np.int8)
        return pad_to_shape(crop(label, self._shape), self._shape, value=-1)

    def __getitem__(self, idx):
        return self._get_image(idx), self._get_label(idx)

    def __len__(self):
        return len(self._image_list)


class VOC2012SegmentationDiskDataset(Dataset):

    def __init__(self, data_dir, subset='train'):
        assert subset in ['train', 'val', 'trainval', 'test']
        sets_dir = f'{data_dir}/ImageSets/Segmentation'
        self._images_dir = f'{data_dir}/JPEGImages'
        self._labels_dir = f'{data_dir}/SegmentationClass'
        self._image_list = file.read_all_lines(f'{sets_dir}/{subset}.txt')
        self.info = {'class_count': 21}
        self.name = f"VOC2012Segmentation-{subset}"

    def _get_image(self, i):
        img = _load_image(f"{self._images_dir}/{self._image_list[i]}.jpg")
        return pad_to_shape(img, [500] * 2)

    def _get_label(self, i):
        label = _load_image(f"{self._labels_dir}/{self._image_list[i]}.png")
        label = label.astype(np.int8)
        return pad_to_shape(label, [500] * 2, value=-1)  # -1 ok?

    def __getitem__(self, idx):
        return self._get_image(idx), self._get_label(idx)

    def __len__(self):
        return len(self._image_list)


class CityscapesSegmentationDiskDataset(Dataset):

    def __init__(self,
                 data_dir,
                 subset='train',
                 downsampling_factor=1,
                 remove_hood=False):
        assert subset in ['train', 'val', 'test']
        assert downsampling_factor >= 1

        self._downsampling_factor = downsampling_factor
        self._shape = np.array([1024, 2048]) // downsampling_factor
        self._remove_hood = remove_hood

        IMG_SUFFIX = "_leftImg8bit.png"
        LAB_SUFFIX = "_gtFine_labelIds.png"
        from .cityscapes_labels import labels as cslabels
        self._id_to_label = [(l.id, l.trainId) for l in cslabels]

        self._images_dir = f'{data_dir}/left/leftImg8bit/{subset}'
        self._labels_dir = f'{data_dir}/fine_annotations/{subset}'
        self._image_list = [
            os.path.relpath(x, start=self._images_dir)
            for x in glob.glob(self._images_dir + '/*/*')
        ]
        self._label_list = [
            x[:-len(IMG_SUFFIX)] + LAB_SUFFIX for x in self._image_list
        ]
        self.info = {'class_count': 19}
        self.name = f"CityscapesSegmentation-{subset}"

        if downsampling_factor > 1:
            self.name += f"-downsampled{downsampling_factor}x"
        if remove_hood:
            self.name += f"-removedhood"

    def _get_image(self, i):
        img = pimg.open(f"{self._images_dir}/{self._image_list[i]}")
        if self._downsampling_factor > 1:
            img = img.resize(self._shape[::-1], pimg.BILINEAR)
        img = np.array(img, dtype=np.uint8)
        if self._remove_hood:
            img = img[:, :self._shape[0] * 7 // 8, :]
        return img

    def _get_label(self, i):
        lab = pimg.open(f"{self._labels_dir}/{self._label_list[i]}")
        if self._downsampling_factor > 1:
            lab = lab.resize(self._shape[::-1], pimg.NEAREST)
        lab = np.array(lab, dtype=np.int8)
        for id, lb in self._id_to_label:
            lab[lab == id] = lb
        if self._remove_hood:
            lab = lab[:self._shape[0] * 7 // 8, :]
        return lab

    def __getitem__(self, idx):
        return self._get_image(idx), self._get_label(idx)

    def __len__(self):
        return len(self._image_list)


iccv09_classes = [
    'sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain',
    'foreground object'
]

voc2012_classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
