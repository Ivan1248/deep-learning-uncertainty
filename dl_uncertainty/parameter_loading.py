import time
import tensorflow as tf
import argparse
import os, re
import numpy as np
import h5py

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

# model depth can be 50, 101 or 152
MODEL_DEPTH = 50
#DATA_MEAN = [103.939, 116.779, 123.68]
DATA_MEAN = np.load('imagenet_mean.npy')
MODEL_PATH = '/home/kivan/datasets/pretrained/resnet/ResNet' + str(
    MODEL_DEPTH) + '.npy'
DATA_PATH = '/home/kivan/datasets/imagenet/ILSVRC2015/numpy/val_data.hdf5'


def normalize_input(rgb):
    return rgb - DATA_MEAN


def build(image, labels, is_training):
    weight_decay = 1e-4
    bn_params = {
        # Decay for the moving averages.
        'decay': 0.9,
        'center': True,
        'scale': True,
        # epsilon to prevent 0s in variance.
        'epsilon': 1e-5,
        # None to force the updates
        'updates_collections': None,
        'is_training': is_training,
    }
    init_func = layers.variance_scaling_initializer(mode='FAN_OUT')

    def shortcut(net, num_maps_out, stride):
        num_maps_in = net.get_shape().as_list()[-1]
        if num_maps_in != num_maps_out:
            return layers.convolution2d(
                net,
                num_maps_out,
                kernel_size=1,
                stride=stride,
                activation_fn=None,
                scope='convshortcut')
        return net

    def bottleneck(net, num_maps, stride):
        net = tf.nn.relu(net)
        bottom_net = net
        with arg_scope(
            [layers.convolution2d],
                padding='SAME',
                activation_fn=tf.nn.relu,
                normalizer_fn=layers.batch_norm,
                normalizer_params=bn_params,
                weights_initializer=init_func,
                weights_regularizer=layers.l2_regularizer(weight_decay)):

            net = layers.convolution2d(
                net, num_maps, kernel_size=1, stride=stride, scope='conv1')
            net = layers.convolution2d(
                net, num_maps, kernel_size=3, scope='conv2')
            net = layers.convolution2d(
                net,
                num_maps * 4,
                kernel_size=1,
                activation_fn=None,
                scope='conv3')
            return net + shortcut(bottom_net, num_maps * 4, stride)

    def layer(net, name, num_maps, num_layers, stride):
        with tf.variable_scope(name):
            for i in range(num_layers):
                with tf.variable_scope('block{}'.format(i)):
                    s = stride if i == 0 else 1
                    net = bottleneck(net, num_maps, s)
            return net

    config_map = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    config = config_map[MODEL_DEPTH]

    image = normalize_input(image)
    #image = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]])
    #net = layers.convolution2d(image, 64, 7, stride=2, padding='VALID',
    net = layers.convolution2d(
        image,
        64,
        7,
        stride=2,
        padding='SAME',
        activation_fn=None,
        weights_initializer=init_func,
        normalizer_fn=layers.batch_norm,
        normalizer_params=bn_params,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        scope='conv0')
    net = layers.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool0')
    net = layer(net, 'group0', 64, config[0], 1)
    net = layer(net, 'group1', 128, config[1], 2)
    net = layer(net, 'group2', 256, config[2], 2)
    net = layer(net, 'group3', 512, config[3], 2)
    net = tf.nn.relu(net)
    in_size = net.get_shape().as_list()[1:3]
    net = layers.avg_pool2d(net, kernel_size=in_size, scope='global_avg_pool')
    net = layers.flatten(net, scope='flatten')
    logits = layers.fully_connected(
        net, 1000, activation_fn=None, scope='fc1000')
    return logits


def name_conversion(caffe_layer_name):
    NAME_MAP = {
        'bn_conv1/beta': 'conv0/BatchNorm/beta:0',
        'bn_conv1/gamma': 'conv0/BatchNorm/gamma:0',
        'bn_conv1/mean/EMA': 'conv0/BatchNorm/moving_mean:0',
        'bn_conv1/variance/EMA': 'conv0/BatchNorm/moving_variance:0',
        'conv1/W': 'conv0/weights:0',
        'conv1/b': 'conv0/biases:0',
        'fc1000/W': 'fc1000/weights:0',
        'fc1000/b': 'fc1000/biases:0'
    }
    if caffe_layer_name in NAME_MAP:
        return NAME_MAP[caffe_layer_name]

    s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
    if s is None:
        s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
        layer_block_part1 = s.group(3)
        layer_block_part2 = s.group(4)
        assert layer_block_part1 in ['a', 'b']
        layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)
    else:
        layer_block = ord(s.group(3)) - ord('a')
    layer_type = s.group(1)
    layer_group = s.group(2)

    layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
    assert layer_branch in [1, 2]
    if layer_branch == 2:
        layer_id = re.search('_branch[0-9]([a-z])/', caffe_layer_name).group(1)
        layer_id = ord(layer_id) - ord('a') + 1

    type_dict = {'res': 'conv', 'bn': 'BatchNorm'}
    name_map = {
        '/W': '/weights:0',
        '/b': '/biases:0',
        '/beta': '/beta:0',
        '/gamma': '/gamma:0',
        '/mean/EMA': '/moving_mean:0',
        '/variance/EMA': '/moving_variance:0'
    }

    tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
    if tf_name in name_map:
        tf_name = name_map[tf_name]
    if layer_type == 'res':
        layer_type = type_dict[layer_type] + \
            (str(layer_id) if layer_branch == 2 else 'shortcut')
    elif layer_branch == 2:
        layer_type = 'conv' + str(layer_id) + '/' + type_dict[layer_type]
    elif layer_branch == 1:
        layer_type = 'convshortcut/' + type_dict[layer_type]
    tf_name = 'group{}/block{}/{}'.format(
        int(layer_group) - 2, layer_block, layer_type) + tf_name
    return tf_name


def create_init_op(params):
    variables = tf.contrib.framework.get_variables()
    init_map = {}
    for var in variables:
        name = var.name
        if name in params:
            init_map[var.name] = params[name]
            del params[name]
        else:
            print(var.name, ' --> init not found!')
            raise ValueError('Init not found')
    print('Unused pretrained params:')
    print(list(params.keys()))
    init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
    return init_op, init_feed


def evaluate():
    params = np.load(MODEL_PATH, encoding='latin1').item()
    resnet_param = {}
    for k, v in params.items():
        newname = name_conversion(k)
        resnet_param[newname] = v

    img_size = 224
    image = tf.placeholder(tf.float32, [None, img_size, img_size, 3], 'input')
    labels = tf.placeholder(tf.int32, [None], 'label')
    logits = build(image, labels, is_training=False)
    all_vars = tf.contrib.framework.get_variables()
    for v in all_vars:
        print(v.name)
    init_op, init_feed = create_init_op(resnet_param)

    sess = tf.Session()
    sess.run(init_op, feed_dict=init_feed)

    batch_size = 100

    h5f = h5py.File(DATA_PATH, 'r')
    data_x = h5f['data_x'][()]
    print(data_x.shape)
    data_y = h5f['data_y'][()]
    h5f.close()

    N = data_x.shape[0]
    assert N % batch_size == 0
    num_batches = N // batch_size

    top5_error = tf.nn.in_top_k(logits, labels, 5)
    top5_wrong = 0
    cnt_wrong = 0
    for i in range(num_batches):
        offset = i * batch_size
        batch_x = data_x[offset:offset + batch_size, ...]
        batch_y = data_y[offset:offset + batch_size, ...]
        start_time = time.time()
        logits_val, top5 = sess.run(
            [logits, top5_error], feed_dict={image: batch_x,
                                             labels: batch_y})
        duration = time.time() - start_time
        num_examples_per_step = batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        top5_wrong += (top5 == 0).sum()
        yp = logits_val.argmax(1).astype(np.int32)
        cnt_wrong += (yp != batch_y).sum()
        if i % 10 == 0:
            print(
                '[%d / %d] top1error = %.2f - top5error = %.2f (%.1f examples/sec; %.3f sec/batch)'
                % (i, num_batches, cnt_wrong / ((i + 1) * batch_size) * 100,
                   top5_wrong / ((i + 1) * batch_size) * 100, examples_per_sec,
                   sec_per_batch))
    print(cnt_wrong / N)
    print(top5_wrong / N)


if __name__ == '__main__':
    evaluate()
