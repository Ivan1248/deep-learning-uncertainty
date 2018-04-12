import os
from typing import List

import skimage.io
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
from ioutils import filesystem, file

IMAGE_EXT = '.png'
LABELING_EXT = '.lab'
IMAGES_DIR = 'images'
LABELS_DIR = 'labels'


class DataSetInfo:
    def __init__(self, class_count):
        self.class_count = class_count

def get_images_dir(data_dir_path: str) -> str:
    return os.path.join(data_dir_path, IMAGES_DIR)


def get_labels_dir(data_dir_path: str) -> str:
    return os.path.join(data_dir_path, LABELS_DIR)


def save_labeling(labeling: np.ndarray, dir_path: str, name: str):
    np.save(os.path.join(dir_path, name + LABELING_EXT), labeling)


def load_labeling(labeling_path: str) -> np.ndarray:
    return np.load(labeling_path)


def save_image(image: np.ndarray, dir_path: str, name: str):
    skimage.io.imsave(os.path.join(dir_path, name + IMAGE_EXT), image)


def load_image(image_path: str) -> np.ndarray:
    return skimage.io.imread(image_path)


def save_info(data_dir_path: str, class_count: int):
    with open(os.path.join(data_dir_path, 'info.cfg'), mode='w') as fs:
        fs.write(str(class_count))
        fs.flush()


def load_info(data_dir_path: str) -> DataSetInfo:
    return DataSetInfo(int(file.read_all_text(os.path.join(data_dir_path, 'info.cfg'))))


def _load_all(dir_path: str, loading_function) -> List[np.ndarray]:
    paths = filesystem.get_files(dir_path)
    paths.sort()
    return [loading_function(p) for p in paths]


def load_images(data_dir_path: str) -> List[np.ndarray]:
    return _load_all(get_images_dir(data_dir_path), load_image)


def load_labels(data_dir_path: str) -> List[np.ndarray]:
    return _load_all(get_labels_dir(data_dir_path), load_labeling)
