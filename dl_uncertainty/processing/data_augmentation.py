import numpy as np
from .shape import pad, crop


def random_crop(im, shape):
    d = [im.shape[i] - shape[i] + 1 for i in [0, 1]]
    d = list(map(np.random.randint, d))
    return im[d[0]:d[0] + shape[0], d[1]:d[1] + shape[1]]

def random_fliplr(im):
    return np.fliplr(im) if np.random.rand() < .5 else im

def augment_cifar(im):
    im = pad(im, 4)
    im = random_crop(im, [32,32])
    return random_fliplr(im)

