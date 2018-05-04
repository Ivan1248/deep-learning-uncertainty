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


def load_cifar10(data_dir, subset='train'):
    # https://dlunizg.github.io/lab2/
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
        return DatasetGenerator(train_x, train_y, 10, train_x.shape[0])
    elif subset == 'test':
        subset = unpickle(os.path.join(data_dir, 'test_batch'))
        test_x = subset['data'].reshape(
            (-1, ch, h, w)).transpose(0, 2, 3, 1).astype(np.float32)
        test_y = np.array(subset['labels'], dtype=np.int32)
        return DatasetGenerator(test_x, test_y, 10, test_x.shape[0])
    else:
        raise ValueError("The value of subset must be in {'train','test'}.")


def load_svhn(data_dir, subset='train'):
    # 0..9 instead of 10,1..9
    import scipy.io as sio
    data = sio.loadmat(subset + '_32x32.mat')
    xs = data['X']
    return DatasetGenerator(xs, np.remainder(data['y'], 10), 10, xs.shape[0])


def load_voc2012_segmentation(data_dir, subset='trainval'):  # TODO
    assert subset in ['train', 'val', 'trainval']

    sets_dir = f'{data_dir}/ImageSets/Segmentation'
    images_dir = f'{data_dir}/JPEGImages'
    labels_dir = f'{data_dir}/SegmentationClass'
    #image_list = [x[:-4] for x in os.listdir(labels_dir)]
    image_list = file.read_all_lines(f'{sets_dir}/{subset}.txt')

    def get_image(name):
        img = _load_image(f"{images_dir}/{name}.jpg")
        return pad_to_shape(img, [500] * 2)

    def get_label(name):
        label = _load_image(f"{labels_dir}/{name}.png")
        label = label.astype(np.int8)
        return pad_to_shape(label, [500] * 2, value=-1)  # -1 ok?

    images = map(get_image, image_list)
    labels = map(get_label, image_list)
    return DatasetGenerator(
        images, labels, class_count=21, size=len(image_list))


def load_cityscapes_segmentation(data_dir,
                                 subset='train',
                                 downsampling_factor=1):
    assert subset in ['train', 'val', 'test']
    assert downsampling_factor >= 1
    """
    prepared_ds_dir = f"{data_dir}.prepared"
    prepared_ds_path = f"{prepared_ds_dir}/{subset}-{downsampling_factor}"
    if os.path.exists(prepared_ds_path):
        try:
            return pickle.load(open(prepared_ds_path, 'rb'))
        except:
            print("Removing invalid prepared datased")
            os.remove(prepared_ds_path)
    """

    if downsampling_factor > 1:
        shape = np.array([2018, 1024]) // downsampling_factor

    IMAGE_SUFFIX = "_leftImg8bit.png"
    LABEL_SUFFIX = "_gtFine_labelIds.png"
    from .cityscapes_labels import labels as cslabels
    id_to_label = [(l.id, l.trainId) for l in cslabels]

    images_dir = f'{data_dir}/left/leftImg8bit/{subset}'
    labels_dir = f'{data_dir}/fine_annotations/{subset}'
    image_list = glob.glob(images_dir + '/*/*')
    image_list = [os.path.relpath(x, start=images_dir) for x in image_list]
    label_list = [x[:-len(IMAGE_SUFFIX)] + LABEL_SUFFIX for x in image_list]

    def get_image(name):
        img = pimg.open(f"{images_dir}/{name}")
        if downsampling_factor > 1:
            img = img.resize(shape, pimg.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def get_label(name):
        lab = pimg.open(f"{labels_dir}/{name}")
        if downsampling_factor > 1:
            lab = lab.resize(shape, pimg.NEAREST)
        lab = np.array(lab, dtype=np.int8)
        for id, lb in id_to_label:
            lab[lab == id] = lb
        return lab

    """
    ds = Dataset(images, labels, class_count=19)

    os.makedirs(prepared_ds_dir, exist_ok=True)
    pickle.dump(ds, open(f"{prepared_ds_path}", 'wb'), protocol=4)
    return ds
    """

    images = map(get_image, tqdm(image_list))
    labels = map(get_label, label_list)
    return DatasetGenerator(
        images, labels, class_count=19, size=len(image_list))


def load_iccv09(data_dir):  # TODO subset
    shape = [240, 320]
    images_dir = f'{data_dir}/images'
    labels_dir = f'{data_dir}/labels'
    image_list = [x[:-4] for x in os.listdir(images_dir)]

    def get_image(name):
        img = _load_image(f"{images_dir}/{name}.jpg")
        return pad_to_shape(crop(img, shape), shape)

    def get_label(name):
        label = np.loadtxt(f"{labels_dir}/{name}.regions.txt", dtype=np.int8)
        return pad_to_shape(crop(label, shape), shape, value=-1)

    images = map(get_image, tqdm(image_list))
    labels = map(get_label, image_list)
    return DatasetGenerator(
        images, labels, class_count=19, size=len(image_list))


iccv09_classes = [
    'sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain',
    'foreground object'
]

voc2012_classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

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
        assert subset in ['train', 'val', 'trainval']
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
        self._shape = np.array([2018, 1024]) // downsampling_factor
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
            img = img.resize(self._shape, pimg.BILINEAR)
        img = np.array(img, dtype=np.uint8)
        if self._remove_hood:
            img = img[:, :self._shape[1] * 7 // 8, :]
        return img

    def _get_label(self, i):
        lab = pimg.open(f"{self._labels_dir}/{self._label_list[i]}")
        if self._downsampling_factor > 1:
            lab = lab.resize(self._shape, pimg.NEAREST)
        lab = np.array(lab, dtype=np.int8)
        for id, lb in self._id_to_label:
            lab[lab == id] = lb
        if self._remove_hood:
            lab = lab[:, :self._shape[1] * 7 // 8]
        return lab

    def __getitem__(self, idx):
        return self._get_image(idx), self._get_label(idx)

    def __len__(self):
        return len(self._image_list)