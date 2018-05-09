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


# Classification


class SVHNDataset(Dataset):

    def __init__(self, data_dir, subset='train'):
        assert subset in ['train', 'test']
        import scipy.io as sio
        data = sio.loadmat(subset + '_32x32.mat')
        self.x, self.y = data['X'], np.remainder(data['y'], 10)
        self.info = {'class_count': 10}
        self.name = f"SVHN-{subset}"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


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
                ds = unpickle(os.path.join(data_dir, 'data_batch_%d' % i))
                train_x = np.vstack((train_x, ds['data']))
                train_y += ds['labels']
            train_x = train_x.reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
            train_y = np.array(train_y, dtype=np.int32)
            self.x, self.y = train_x, train_y
        elif subset == 'test':
            ds = unpickle(os.path.join(data_dir, 'test_batch'))
            test_x = ds['data'].reshape((-1, ch, h, w)) \
                               .transpose(0, 2, 3, 1).astype(np.float32)
            test_y = np.array(ds['labels'], dtype=np.int32)
            self.x, self.y = test_x, test_y
        else:
            raise ValueError("The value of subset must be in {'train','test'}.")
        self.info = {'class_count': 10}
        self.name = f"Cifar10-{subset}"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class MozgaloRVCDataset(Dataset):

    def __init__(self, data_dir, subset='train', remove_bottom_half=False):
        assert subset in ['train']

        self._shape = [1104, 600]
        self._remove_bottom = remove_bottom_half
        train_dir = f"{data_dir}/train"
        class_names = sorted(map(os.path.basename, os.listdir(train_dir)))
        self._subset_dir = f"{data_dir}/{subset}"
        subset_class_names = sorted(
            map(os.path.basename, os.listdir(self._subset_dir)))
        self._image_list = [
            os.path.relpath(x, start=self._subset_dir)
            for x in glob.glob(self._subset_dir + '/*/*')
        ]

        assert len(class_names) == 25
        self.info = {'class_count': 25, 'class_names': class_names}
        self.name = f"MozgaloRVC-{subset}"
        if remove_bottom_half:
            self.name += "-remove_bottom_half"

    def __getitem__(self, idx):
        example_name = self._image_list[idx]
        lab_str = os.path.dirname(example_name)
        img = _load_image(f"{self._subset_dir}/{example_name}")
        lab = self.info['class_names'].index(lab_str)
        img = pad_to_shape(crop(img, self._shape), self._shape)
        if len(img.shape) == 2:  # greyscale -> rgb
            img = np.dstack([img] * 3)
        if self._remove_bottom:
            img = img[:self._shape[0] // 2, :, :]
        return img, lab

    def __len__(self):
        return len(self._image_list)


# Semantic segmentation


class CamVidDataset(Dataset):
    # https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

    def __init__(self, data_dir, subset='train'):
        assert subset in ['train', 'val', 'test']

        lines = file.read_all_lines(f'{data_dir}/{subset}.txt')
        self._img_lab_list = [[
            f"{data_dir}/{p.replace('/SegNet/CamVid/', '')}"
            for p in line.split()
        ] for line in lines]

        self.info = {
            'class_count':
                11,
            'class_names': [
                'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree',
                'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
            ],
            'class_colors': [
                (128, 128, 128),
                (128, 0, 0),
                (192, 192, 128),
                (128, 64, 128),
                (60, 40, 222),
                (128, 128, 0),
                (192, 128, 128),
                (64, 64, 128),
                (64, 0, 128),
                (64, 64, 0),
                (0, 128, 192),
            ]
        }
        self.name = f"CamVid-{subset}"

    def __getitem__(self, idx):
        img, lab = map(_load_image, self._img_lab_list[i])
        lab = lab.astype(np.int8)
        lab[lab == 11] = -1
        return img, lab

    def __len__(self):
        return len(self._img_lab_list)


class CityscapesSegmentationDataset(Dataset):

    def __init__(self,
                 data_dir,
                 subset='train',
                 downsampling_factor=1,
                 remove_hood=False):
        assert subset in ['train', 'val', 'test']
        assert downsampling_factor >= 1
        from .cityscapes_labels import labels as cslabels

        self._downsampling_factor = downsampling_factor
        self._shape = np.array([1024, 2048]) // downsampling_factor
        self._remove_hood = remove_hood

        IMG_SUFFIX = "_leftImg8bit.png"
        LAB_SUFFIX = "_gtFine_labelIds.png"
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

        self.info = {
            'class_count': 19,
            'class_names': [l.name for l in cslabels if l.trainId >= 0],
            'class_colors': [l.color for l in cslabels if l.trainId >= 0],
        }
        self.name = f"CityscapesSegmentation-{subset}"

        if downsampling_factor > 1:
            self.name += f"-downsampled{downsampling_factor}x"
        if remove_hood:
            self.name += f"-removedhood"

    def __getitem__(self, idx):
        img = pimg.open(f"{self._images_dir}/{self._image_list[idx]}")
        lab = pimg.open(f"{self._labels_dir}/{self._label_list[idx]}")
        if self._downsampling_factor > 1:
            img = img.resize(self._shape[::-1], pimg.BILINEAR)
            lab = lab.resize(self._shape[::-1], pimg.NEAREST)
        img = np.array(img, dtype=np.uint8)
        lab = np.array(lab, dtype=np.int8)
        for id, lb in self._id_to_label:
            lab[lab == id] = lb
        if self._remove_hood:
            img = img[:self._shape[0] * 7 // 8, :, :]
            lab = lab[:self._shape[0] * 7 // 8, :]
        return img, lab

    def __len__(self):
        return len(self._image_list)


class ICCV09Dataset(Dataset):

    def __init__(self, data_dir):  # TODO subset
        self._shape = [240, 320]
        self._images_dir = f'{data_dir}/images'
        self._labels_dir = f'{data_dir}/labels'
        self._image_list = [x[:-4] for x in os.listdir(self._images_dir)]

        self.info = dict()
        self.info['class_names'] = [
            'sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain',
            'foreground object'
        ]
        self.info['class_count'] = len(self.info['class_names'])  # 8
        self.name = "ICCV09"

    def __getitem__(self, idx):
        name = self._image_list[idx]
        img = _load_image(f"{self._images_dir}/{name}.jpg")
        lab = np.loadtxt(
            f"{self._labels_dir}/{name}.regions.txt", dtype=np.int8)
        img = pad_to_shape(crop(img, self._shape), self._shape)
        lab = pad_to_shape(crop(lab, self._shape), self._shape, value=-1)
        return img, lab

    def __len__(self):
        return len(self._image_list)


class VOC2012SegmentationDataset(Dataset):

    def __init__(self, data_dir, subset='train'):
        assert subset in ['train', 'val', 'trainval', 'test']
        sets_dir = f'{data_dir}/ImageSets/Segmentation'
        self._images_dir = f'{data_dir}/JPEGImages'
        self._labels_dir = f'{data_dir}/SegmentationClass'
        self._image_list = file.read_all_lines(f'{sets_dir}/{subset}.txt')
        self.info = dict()
        self.info['class_names'] = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor'
        ]
        self.info['class_count'] = len(self.info['class_names'])  # 21
        self.name = f"VOC2012Segmentation-{subset}"

    def __getitem__(self, idx):
        name = self._image_list[i]
        img = _load_image(f"{self._images_dir}/{name}.jpg")
        lab = _load_image(f"{self._labels_dir}/{name}.png").astype(np.int8)
        img = pad_to_shape(img, [500] * 2)
        lab = pad_to_shape(lab, [500] * 2, value=-1)  # -1 ok?
        return img, lab

    def __len__(self):
        return len(self._image_list)