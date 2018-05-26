import os, glob
import pickle

import PIL.Image as pimg
import numpy as np
from skimage.transform import resize
import torch.utils.data
from functools import lru_cache

from .data import Dataset

from ..processing.shape import pad_to_shape, crop
from ..ioutils import file

# Helper functions


def _load_image(path):
    return np.array(pimg.open(path))


# Artificial datasets


class WhiteNoiseDataset(Dataset):

    def __init__(self, example_shape, size, uniform=False, seed=None):
        self._shape = example_shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.random_integers(low=0, high=100, size=(size))
        self.name = 'white_noise'
        if uniform:
            self.name += 'uniform'
        self.info = {'id': 'noise'}

    def __getitem__(self, idx):
        self._rand.seed(self._seeds[idx])
        return self._rand.randn(self._shape)

    def __len__(self):
        return len(self._seeds)


# Classification


class SVHNDataset(Dataset):

    def __init__(self, data_dir, subset='train'):
        assert subset in ['train', 'test']
        import scipy.io as sio
        data = sio.loadmat(subset + '_32x32.mat')
        self.x, self.y = data['X'], np.remainder(data['y'], 10)
        self.info = {'id': 'svhn', 'class_count': 10, 'problem_id': 'clf'}
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
        self.info = {'id': 'cifar', 'class_count': 10, 'problem_id': 'clf'}
        self.name = f"Cifar10-{subset}"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class MozgaloRVCDataset(Dataset):

    def __init__(self,
                 data_dir,
                 subset='train',
                 remove_bottom_proportion=0.0,
                 downsampling_factor=1):
        assert subset in ['train']

        self._shape = [1104, 600]
        self._remove_bottom_proportion = remove_bottom_proportion
        self._downsampling_factor = downsampling_factor
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
        self.info = {
            'id': 'mozgalo',
            'class_count': 25,
            'class_names': class_names,
            'problem_id': 'clf'
        }
        self.name = f"MozgaloRVC-{subset}"
        if remove_bottom_proportion:
            self.name += f"-remove_bottom_{remove_bottom_proportion:0.2f}"
        if downsampling_factor > 1:
            self.name += f"-downsample_{downsampling_factor}x"

    def __getitem__(self, idx):
        example_name = self._image_list[idx]
        lab_str = os.path.dirname(example_name)
        img = _load_image(f"{self._subset_dir}/{example_name}")
        lab = self.info['class_names'].index(lab_str)
        img = pad_to_shape(crop(img, self._shape), self._shape)
        if len(img.shape) == 2:  # greyscale -> rgb
            img = np.dstack([img] * 3)
        if self._remove_bottom_proportion > 0:
            a = int(img.shape[0] * (1 - self._remove_bottom_proportion))
            img = img[:a, :, :]
        if self._downsampling_factor > 1:
            h, w = (round(x / self._downsampling_factor) for x in img.shape[:2])
            img = resize(img, (h, w))  #, anti_aliasing=True)
        return img, lab

    def __len__(self):
        return len(self._image_list)


class TinyImageNet(Dataset):

    def __init__(self, data_dir, subset='train'):
        assert subset in ['train', 'val', 'test']

        with open(f"{data_dir}/wnids.txt") as fs:
            class_names = [l.strip() for l in fs.readlines()]
        subset_dir = f"{data_dir}/{subset}"

        self._examples = []

        if subset == 'train':
            for i, class_name in enumerate(class_names):
                images_dir = f"{subset_dir}/{class_name}/images"
                for im in os.listdir(images_dir):
                    self._examples.append((f"{images_dir}/{im}", i))
        elif subset == 'val':
            with open(f"{subset_dir}/val_annotations.txt") as fs:
                im_labs = [l.split()[:2] for l in fs.readlines()]
                images_dir = f"{subset_dir}/images"
                for im, lab in im_labs:
                    lab = class_names.index(lab)
                    self._examples.append((f"{images_dir}/{im}", lab))
        elif subset == 'test':
            images_dir = f"{subset_dir}/images"
            self._examples = [(f"{images_dir}/{im}", -1)
                              for im in os.listdir(images_dir)]

        self.info = {
            'id': 'tinyimagenet',
            'class_count': 200,
            'class_names': class_names,
            'problem_id': 'clf'
        }
        self.name = f"TinyImageNet-{subset}"

    def _load_image(self, path):
        im = _load_image(path)
        if len(im.shape) == 2:
            im = np.reshape(np.repeat(im, 3, axis=-1), list(im.shape) + [3])
        return im

    def __getitem__(self, idx):
        img_path, lab = self._examples[idx]
        return self._load_image(img_path), lab

    def __len__(self):
        return len(self._examples)


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
            'id':
                'camvid',
            'problem_id':
                'semseg',
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
        img, lab = map(_load_image, self._img_lab_list[idx])
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
        assert subset in ['train', 'val', 'test']  # 'test' labels are invalid
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
            'id': 'cityscapes',
            'problem_id': 'semseg',
            'class_count': 19,
            'class_names': [l.name for l in cslabels if l.trainId >= 0],
            'class_colors': [l.color for l in cslabels if l.trainId >= 0],
        }
        self.name = f"CityscapesSegmentation-{subset}"

        if downsampling_factor > 1:
            self.name += f"-downsample_{downsampling_factor}x"
        if remove_hood:
            self.name += f"-remove_hood"

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


class WildDashSegmentationDataset(Dataset):

    def __init__(self, data_dir, subset='val', downsampling_factor=1):
        assert subset in ['val', 'bench', 'both']  # 'test' labels are invalid
        assert downsampling_factor >= 1
        from .cityscapes_labels import labels as cslabels

        self._subset = subset

        self._downsampling_factor = downsampling_factor
        self._shape = np.array([1070, 1920]) // downsampling_factor

        self._IMG_SUFFIX = "0.png"
        self._LAB_SUFFIX = "0_labelIds.png"
        self._id_to_label = [(l.id, l.trainId) for l in cslabels]

        self._images_dir = f'{data_dir}/wd_{subset}_01'
        self._image_names = [
            os.path.relpath(x, start=self._images_dir)[:-5]
            for x in glob.glob(self._images_dir + f'/*{self._IMG_SUFFIX}')
        ]
        class_count = 19
        self.info = {
            'id': 'wilddash',
            'problem_id': 'semseg',
            'class_count': class_count,
            'class_names': [l.name for l in cslabels if l.trainId >= 0],
            'class_colors': [l.color for l in cslabels if l.trainId >= 0],
        }
        self.name = f"WildDashSegmentation-{subset}"

        self._blank_label = np.full(list(self._shape), -1, dtype=np.int8)

        if downsampling_factor > 1:
            self.name += f"-downsample_{downsampling_factor}x"

    def __getitem__(self, idx):
        path_prefix = f"{self._images_dir}/{self._image_names[idx]}"
        img = pimg.open(f"{path_prefix}{self._IMG_SUFFIX}")
        if self._downsampling_factor > 1:
            img = img.resize(self._shape[::-1], pimg.BILINEAR)
        img = np.array(img, dtype=np.uint8)
        if len(img.shape) == 2:
            shape = list(img.shape) + [3]
            img = np.reshape(np.repeat(img, 3, axis=-1), shape)
        if img.shape[-1] > 3:
            img = img[:, :, :3]

        if self._subset == 'bench':
            lab = self._blank_label
        else:
            lab = pimg.open(f"{path_prefix}{self._LAB_SUFFIX}")
            if self._downsampling_factor > 1:
                lab = lab.resize(self._shape[::-1], pimg.NEAREST)
            lab = np.array(lab, dtype=np.int8)

        for id, lb in self._id_to_label:
            lab[lab == id] = lb

        return img, lab

    def __len__(self):
        return len(self._image_names)


class ICCV09Dataset(Dataset):

    def __init__(self, data_dir):  # TODO subset
        self._shape = [240, 320]
        self._images_dir = f'{data_dir}/images'
        self._labels_dir = f'{data_dir}/labels'
        self._image_list = [x[:-4] for x in os.listdir(self._images_dir)]

        self.info = {
            'id':
                'iccv09',
            'problem_id':
                'semseg',
            'class_count':
                8,
            'class_names': [
                'sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain',
                'foreground object'
            ]
        }
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
        self.info = {
            'id':
                'voc2012',
            'problem_id':
                'semseg',
            'class_count':
                21,
            'class_names': [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor'
            ]
        }
        self.name = f"VOC2012Segmentation-{subset}"

    def __getitem__(self, idx):
        name = self._image_list[idx]
        img = _load_image(f"{self._images_dir}/{name}.jpg")
        lab = _load_image(f"{self._labels_dir}/{name}.png").astype(np.int8)
        img = pad_to_shape(img, [500] * 2)
        lab = pad_to_shape(lab, [500] * 2, value=-1)  # -1 ok?
        return img, lab

    def __len__(self):
        return len(self._image_list)
