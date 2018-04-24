import os
import pickle

import PIL.Image as pimg
import numpy as np

from .dataset import Dataset

from ..processing.shape import pad_to_shape
from ..ioutils import file


def load_cifar10(data_dir, subset='train'):
    # https://dlunizg.github.io/lab2/
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
        return Dataset(train_x, train_y, 10)
    elif subset == 'test':
        subset = unpickle(os.path.join(data_dir, 'test_batch'))
        test_x = subset['data'].reshape(
            (-1, ch, h, w)).transpose(0, 2, 3, 1).astype(np.float32)
        test_y = np.array(subset['labels'], dtype=np.int32)
        return Dataset(test_x, test_y, 10)
    else:
        raise ValueError("The value of subset must be in {'train','test'}.")


def load_svhn(data_dir, subset='train'):
    # 0..9 instead of 10,1..9
    import scipy.io as sio
    data = sio.loadmat(subset + '_32x32.mat')
    return Dataset(data['X'], np.remainder(data['y'], 10), 10)


def load_voc2012_segmentation(data_dir, subset='trainval'): # TODO subset
    sets_dir = f'{data_dir}/ImageSets/Segmentation'
    images_dir = f'{data_dir}/JPEGImages'
    labels_dir = f'{data_dir}/SegmentationClass'
    #image_list = [x[:-4] for x in os.listdir(labels_dir)]
    image_list = file.read_all_lines(f'{sets_dir}/{subset}.txt')  

    def load_image(path):
        return np.array(pimg.open(path))

    def get_image(name):
        img = load_image(f"{images_dir}/{name}.jpg")
        return pad_to_shape(img, [500] * 2)

    def get_labels(name):
        label = load_image(f"{labels_dir}/{name}.png")
        label = label.astype(np.int8)
        return pad_to_shape(label, [500] * 2, value=-1)  # -1 ok?

    images = list(map(get_image, image_list))
    labels = list(map(get_labels, image_list))
    return Dataset(images, labels, class_count=21)


voc2012_classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
