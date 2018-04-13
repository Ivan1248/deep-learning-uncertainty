import os
import pickle

import numpy as np

from .dataset import Dataset


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