import skimage.util, skimage.transform
import numpy as np


def crop(image: np.ndarray, maxshape):
    d = [max(0, image.shape[i] - maxshape[i]) for i in [0, 1]]
    if (np.sum(d) == 0):
        return image
    return image[(d[0] + 1) // 2:image.shape[0] - (d[0] // 2), (d[1] + 1) // 2:
                 image.shape[1] - (d[1] // 2)]


def pad(image: np.ndarray, padding: int):
    pad_width = [[padding] * 2] * 2
    if len(image.shape) == 3: pad_width.append((0, 0))
    return np.pad(image, pad_width, mode='constant')


def pad_to_shape(image: np.ndarray, shape):
    d = [shape[i] - image.shape[i] for i in [0, 1]]
    pad_width = [((d[0] + 1) // 2, d[0] // 2), ((d[1] + 1) // 2, d[1] // 2)]
    if len(image.shape) == 3: pad_width.append((0, 0))
    return np.pad(image, pad_width, mode='constant')


def adjust_shape(image: np.ndarray, shape):
    return pad_to_shape(crop(image, shape), shape)


def resize(image: np.ndarray, shape):
    return skimage.transform.resize(image, shape, anti_aliasing=True)
