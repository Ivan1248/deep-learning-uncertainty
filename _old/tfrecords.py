import os
import itertools

import tensorflow as tf

from . import DatasetGenerator


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _byte_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_tfrecord(images, labels, save_dir, name):
    height, width, channel_count, *_ = images[0].shape
    filename = os.path.join(save_dir, name + '.tfrecords')
    with tf.python_io.TFRecordWriter(filename) as writer:
        for image, label in zip(images, labels):
            image_str = image.tostring()
            label_str = label.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'channels': _int64_feature(channel_count),
                    'image': _byte_feature(image_str),
                    'label': _byte_feature(label_str),
                }))
            writer.write(example.SerializeToString())


def prepare_dataset_parts(dsgen: DatasetGenerator, save_dir, shard_size=200):
    full_part_count = dsgen.size // shard_size
    last_shard_size = dsgen.size - full_part_count * shard_size
    part_count = full_part_count + int(last_shard_size > 0)

    def create_tfrecord(i, size):
        images = itertools.islice(dsgen.inputs, size)
        labels = itertools.islice(dsgen.inputs, size)
        _create_tfrecord(images, labels, save_dir, f'{i}of{part_count}')

    for i in range(full_part_count):
        create_tfrecord(i, shard_size)
    if last_shard_size > 0:
        create_tfrecord(part_count - 1, last_shard_size)


def prepare_dataset(dsgen: DatasetGenerator, directory):
    for i, img, lab in enumerate(zip(dsgen.inputs, dsgen.labels)):
        _create_tfrecord([img], [lab], directory, f'{i}of{dsgen.size}')

def parse_tfrecord(example, dense_labels=False):
    features = tf.parse_single_example(
        example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['channels'], tf.int32)
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.int8)

    image_shape = tf.parallel_stack([height, width, channels])
    image = tf.reshape(image, image_shape)
    if dense_labels:
        label_shape = tf.parallel_stack([height, width])
        label = tf.reshape(label, shape=label_shape)

    return image, label



def load_tf_dataset(save_dir):

    def parser(record):
        keys_to_features = {
            'height': tf.FixedLenFeature((), tf.int64, default_value=0),
            'width': tf.FixedLenFeature((), tf.int64, default_value=0),
            'channels': tf.FixedLenFeature((), tf.int64, default_value=0),
            'image': tf.FixedLenFeature((), tf.string, default_value=''),
            'label': tf.FixedLenFeature((), tf.string, default_value=''),
        }

        items_to_handlers = {
            'image':
                tfexample_decoder.Image(
                    image_key='image', format_key='image/format', channels=3),
            'height':
                tfexample_decoder.Tensor('height'),
            'width':
                tfexample_decoder.Tensor('width'),
            'label':
                tfexample_decoder.Image(
                    image_key='image/segmentation/class/encoded',
                    format_key='image/segmentation/class/format',
                    channels=1),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.image.decode_jpeg(parsed["image_data"])
        image = tf.reshape(image, [299, 299, 1])
        label = tf.cast(parsed["label"], tf.int32)

        return {"image_data": image, "date_time": parsed["date_time"]}, label

    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)


def get_dataset(dir, dataset_dir):
    file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Specify how the TF-Examples are decoded.
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }
    items_to_handlers = {
        'image':
            tfexample_decoder.Image(
                image_key='image/encoded',
                format_key='image/format',
                channels=3),
        'image_name':
            tfexample_decoder.Tensor('image/filename'),
        'height':
            tfexample_decoder.Tensor('image/height'),
        'width':
            tfexample_decoder.Tensor('image/width'),
        'labels_class':
            tfexample_decoder.Image(
                image_key='image/segmentation/class/encoded',
                format_key='image/segmentation/class/format',
                channels=1),
    }

    decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                 items_to_handlers)

    return dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=splits_to_sizes[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        ignore_label=ignore_label,
        num_classes=num_classes,
        name=dataset_name,
        multi_label=True)
