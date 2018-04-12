import os.path
import skimage.io
import numpy as np

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # semseg/*
from processing.shape import adjust_shape
from ioutils import path, directory

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # data/*
from dataset_dir import save_image, save_labeling, save_info, get_images_dir, get_labels_dir
from preparers.abstract_preparer import AbstractPreparer

class Iccv09Preparer(AbstractPreparer):
    IN_IMAGE_EXT = '.jpg'
    IN_LABELING_EXT = '.regions.txt'
    SHAPE = (240, 320)

    @staticmethod
    def prepare(data_path: str, shape=SHAPE):
        out_data_path = data_path + '.prepared'
        if os.path.exists(out_data_path):
            print("Dataset already prepared. Delete " + out_data_path + " if you would like it to be prepared again.")
            return out_data_path
        os.makedirs(out_data_path)

        in_images_path, in_labels_path = (os.path.join(data_path, d) for d in ('images', 'labels'))
        out_images_path = get_images_dir(out_data_path)
        out_labels_path = get_labels_dir(out_data_path)
        os.makedirs(out_images_path)
        os.makedirs(out_labels_path)

        names = [path.get_file_name_without_extension(p) for p in
                 directory.get_files(in_images_path)]

        for name in names:
            in_image_path = os.path.join(in_images_path, name + Iccv09Preparer.IN_IMAGE_EXT)
            image = adjust_shape(skimage.io.imread(in_image_path), Iccv09Preparer.SHAPE)
            save_image(image, out_images_path, name)

            in_labeling_path = os.path.join(in_labels_path, name + Iccv09Preparer.IN_LABELING_EXT)
            labeling = adjust_shape(Iccv09Preparer._load_and_convert_labeling(in_labeling_path), Iccv09Preparer.SHAPE)
            save_labeling(labeling, out_labels_path, name)

        save_info(out_data_path, class_count=9)

        return out_data_path

    @staticmethod
    def _load_and_convert_labeling(file_path):
        """
        Creates a matrix representation of labels.
        0-valued bytes represent pixels representing unknown/nothing/background.
        1-10-valued bytes represent pixels belonging to classes listed in README.
        """
        # TODO check unknown/nothing
        return np.loadtxt(file_path, dtype=np.uint8)+1
