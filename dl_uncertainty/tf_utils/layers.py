import inspect

import functools

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from .variables import conv_weight_variable, bias_variable

var_scope = variable_scope.variable_scope


def scoped(fun):

    @functools.wraps(fun)  # preserves the signature (needed for default_arg)
    def wrapper(*args, reuse: bool = None, name: str = None, **kwargs):
        #if reuse == -1:  # TODO: change reuse default to tf.AUTO_REUSE
        #    reuse = tf.get_variable_scope().reuse
        a = list(args) + list(kwargs.values())
        tensors = [v for v in a if type(v) is tf.Tensor]
        with var_scope(
                name, default_name=fun.__name__, values=tensors, reuse=reuse):
            return fun(*args, **kwargs)

    return wrapper


def default_arg(func, param):
    return inspect.signature(func).parameters[param].default


def default_args(func, params):
    return dict([(p, inspect.signature(func).parameters[p].default)
                 for p in params])


# Slicing, joining, copying


def repeat(x, n, axis=0):
    """
    Repeats subarrays in axis. The returned array has shape such that
    shape[i] = x.shape[i] if i != axis, shape[axis] = n*x.shape[axis].
    Based on https://github.com/tensorflow/tensorflow/issues/8246
    :param repeats: list. Numbers of repeats for each dimension    
    """
    repeats = [1] * len(x.shape)
    repeats[axis] = n
    xe = tf.expand_dims(x, -1)
    xt = tf.tile(xe, multiples=[1] + repeats)
    return tf.reshape(xt, tf.shape(x) * repeats)


def repeat_batch(x, n, expand_dims=True):
    multiples = [n] + [1] * (len(x.shape) - 1)
    if expand_dims:  # [N,H,W,C]->[n,N,H,W,C] else: [N,H,W,C]->[nN,H,W,C]
        x = tf.expand_dims(x, -1)
        multiples += [1]
    return tf.tile(x, multiples)


def join_batches(x):  # [n,N,H,W,C]->[nN,H,W,C]
    s = tf.shape(x)
    shape = tf.concat([[s[0] * s[1]], s[2:]], 0)
    return tf.reshape(x, shape)


def split_batch(x, n: int):  # [nN,H,W,C]->[n,N,H,W,C]
    s = tf.shape(x)
    shape = tf.concat([[n, tf.floordiv(s[0], n)], s[1:]], 0)
    return tf.reshape(x, shape)


# Linear, affine


@scoped
def add_biases(x, return_params=False):
    b = bias_variable(x.shape[-1].value)
    h = x + b
    return (h, b) if return_params else h


# Convolution


@scoped
def conv(x, ksize, width, stride=1, dilation=1, padding='SAME', bias=True):
    """
    A wrapper for tf.nn.conv2d.
    :param x: 4D input "NHWC" tensor 
    :param ksize: int or tuple of 2 ints representing spatial dimensions 
    :param width: number of output channels
    :param stride: int (or float in case of transposed convolution)
    :param dilation: dilation of the kernel
    :param padding: string. "SAME" or "VALID" 
    :param bias: add biases
    """
    h = tf.nn.convolution(
        x,
        filter=conv_weight_variable(ksize, x.shape[-1].value, width),
        strides=[stride] * 2,
        dilation_rate=[dilation] * 2,
        padding=padding)
    if bias:
        h += bias_variable(width)
    print(x.shape[1:], h.shape[1:], h.name)
    return h


@scoped
def separable_conv(x,
                   ksize,
                   width,
                   channel_multiplier=1,
                   stride=1,
                   padding='SAME',
                   bias=True):
    """
    A wrapper for tf.nn.conv2d.
    :param x: 4D input "NHWC" tensor 
    :param ksize: int or tuple of 2 ints representing spatial dimensions 
    :param width: number of output channels
    :param stride: int (or float in case of transposed convolution)
    :param padding: string. "SAME" or "VALID" 
    :param bias: add biases
    """
    in_width = x.shape[-1].value
    h = tf.nn.separable_conv2d(
        x,
        depthwise_filter=conv_weight_variable(
            ksize, in_width, channel_multiplier, name="wweights"),
        pointwise_filter=conv_weight_variable(
            (1, 1), in_width * channel_multiplier, width, name="pweights"),
        strides=[1] + [stride] * 2 + [1],
        padding=padding)
    if bias:
        h += bias_variable(width)
    return h


# Pooling


def max_pool(x, stride, ksize=None, padding='SAME'):
    if ksize is None:
        ksize = stride
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1],
                          padding)


def avg_pool(x, stride, ksize=None, padding='SAME'):
    if ksize is None:
        ksize = stride
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1],
                          padding)


# Rescaling


def _get_rescaled_shape(x, factor):
    return (np.array([d.value
                      for d in x.shape[1:3]]) * factor + 0.5).astype(np.int)


def rescale_nearest_neighbor(x, factor):
    shape = _get_rescaled_shape(x, factor)
    return x if factor == 1 else tf.image.resize_nearest_neighbor(x, shape)


def rescale_bilinear(x, factor):
    shape = _get_rescaled_shape(x, factor)
    return x if factor == 1 else tf.image.resize_bilinear(x, shape)


# Special


@scoped
def batch_normalization(x,
                        mode,
                        offset=True,
                        scale=True,
                        decay=0.95,
                        var_epsilon=1e-8,
                        scope: str = None):
    """
    Batch normalization with scaling that normalizes over all but the last
    dimension of x.
    :param x: input tensor
    :param mode: Tensor or str. 'moving' or 'fixed' (or 'accumulating' -- not 
        implemented)
    :param decay: exponential moving average decay
    """
    moving = tf.equal(mode, 'moving')

    ema = tf.train.ExponentialMovingAverage(decay)

    m, v = tf.nn.moments(x, axes=list(range(len(x.shape) - 1)), name='moments')

    def m_v_with_update():
        nonlocal m, v
        with tf.control_dependencies([ema.apply([m, v])]):
            return tf.identity(m), tf.identity(v)

    mean, var = tf.cond(moving, m_v_with_update,
                        lambda: (ema.average(m), ema.average(v)))

    offset = 0.0 if not offset else tf.get_variable(
        'offset',
        shape=[x.shape[-1].value],
        initializer=tf.constant_initializer(0.0))
    scale = 1.0 if not scale else tf.get_variable(
        'scale',
        shape=[x.shape[-1].value],
        initializer=tf.constant_initializer(1.0))

    return tf.nn.batch_normalization(x, mean, var, offset, scale, var_epsilon)


def dropout(x, on, rate, feature_map_wise=False, **kwargs):
    if feature_map_wise:  # TODO: remove True
        kwargs = {**kwargs, 'noise_shape': [x.shape[0], 1, 1, x.shape[3]]}
    return tf.layers.dropout(x, rate, training=on, **kwargs)


@scoped
def convex_combination(x, r, scope: str = None):
    # TODO: improve initialization and constraining
    a = tf.get_variable('a', [1], initializer=tf.constant_initializer(0.99))
    constrain_a = tf.assign(a, tf.clip_by_value(a, 0, 1))

    def a_with_update():
        with tf.control_dependencies([constrain_a]):
            return tf.identity(a)

    a = a_with_update()  # TODO: use constraint in var_scope instead
    return a * x + (1 - a) * r


@scoped
def sample1(fn, x, sample_count: int):
    res = [fn(x)]
    tf.get_variable_scope().reuse_variables()
    res += [fn(x) for _ in range(sample_count - 1)]
    return tf.stack(res)


def _sample(fn, x, n: int, examplewise=False):
    if examplewise:
        xr = repeat(x, n, axis=0)  # [N,H,W,C]->[Nn,H,W,C]
        yr = fn(xr)  # [Nn,H',W',C']
        return split_batch(yr, tf.shape(x)[0])  # [N,n,H',W',C']
    else:
        xr = repeat_batch(x, n, expand_dims=False)  # [nN,H,W,C]
        yr = fn(xr)  # [nN,H',W',C']
        return split_batch(yr, n)  # [n,N,H',W',C']


@scoped
def sample(fn,
           x,
           n: int,
           examplewise=False,
           max_batch_size=None,
           backprop=True):
    """
    Sampling of a stochastic operation. 
    :param fn: function. Stochastic operation to be sampled.
    :param x: Tensor. Input.
    :param n: number of samples; must be smaller than `max_batch_size`.
    :param examplewise: samples of elements side to side instead of batches. If
        True, output has shape `[N,n,...]` and `[n,N,...]` otherwise, where 
        `N = x.shape[0]` is the batch size.
    :param max_batch_size: int. maximum batch size to be let through `op`.
        Defaults to `n`.
    :param backprop: True enables support for back propagation.
    """
    max_batch_size = max_batch_size or n
    assert n <= max_batch_size  # TODO: case when n > max_batch_size
    examples_per_iteration = max(1, max_batch_size // n)
    ys = tf.map_fn(
        fn=lambda x: _sample(fn, x, n),  # [H,W,C]->[n,H,W,C]
        elems=x,  # [N,H,W,C]
        parallel_iterations=examples_per_iteration,
        back_prop=backprop)  # [N,n,H,W,C]
    if examplewise:  # [N,n,H,W,C]
        return ys
    else:  # [n,N,H,W,C]
        t = [1, 0] + list(range(2, len(ys.shape)))
        return tf.transpose(ys, t)  # [n,N,H,W,C]


# Blocks


def _complement_bn_params(bn_params, is_training=None):
    if 'mode' not in bn_params:
        assert is_training is not None
        return {
            'mode':
                tf.cond(is_training, lambda: tf.constant('moving'),
                        lambda: tf.constant('fixed')), **bn_params
        }
    return bn_params


def _complement_dropout_params(dropout_params, is_training=None):
    if 'mode' not in dropout_params:
        assert is_training is not None
        return {
            'on': is_training, **default_arg(residual_block, 'dropout_params'),
            **dropout_params
        }
    return dropout_params


@scoped
def bn_relu(x, bn_params=dict(), is_training=None):
    bn_params = _complement_bn_params(bn_params, is_training)
    x = batch_normalization(x, **bn_params)
    return tf.nn.relu(x)


@scoped
def bn_relu_conv(x, bn_params=dict(), conv_params=dict(), is_training=None):
    """ Convolutions are unbiased """
    bn_params = _complement_bn_params(bn_params, is_training)
    x = batch_normalization(x, **bn_params)
    x = tf.nn.relu(x)
    return conv(x, **conv_params, bias=False)


@scoped
def identity_mapping(x, stride, width):
    x_width = x.shape[-1].value
    assert width >= x_width
    if stride > 1:
        x = avg_pool(x, stride, padding='VALID')
    if width > x_width:
        x = tf.pad(x, 3 * [[0, 0]] + [[0, width - x_width]])
    return x


class BlockStructure:

    def __init__(self, ksizes, width_factors, dropout_locations):
        """
        :param ksizes: list of ints or int pairs. kernel sizes of convolutional 
            layers
        :param dropout_locations: indexes of convolutional layers after which
            dropout is to be applied
        :param width_factors: int or list[int]
        """
        self.ksizes = ksizes
        self.width_factors = width_factors
        if type(width_factors) is int:
            self.width_factors = [width_factors for _ in ksizes]
        elif len(ksizes) != len(width_factors):
            raise ValueError(
                "width_factors must be an integer or a list of equal length as ksizes"
            )
        self.dropout_locations = dropout_locations

    @classmethod
    def resnet(cls, ksizes=[3, 3], width_factors=1, dropout_locations=[0]):
        args = {k: v for k, v in locals().items() if k != 'cls'}
        return BlockStructure(**args)

    @classmethod
    def densenet(cls,
                 ksizes=[1, 3],
                 width_factors=[4, 1],
                 dropout_locations=[0, 1]):
        args = {k: v for k, v in locals().items() if k != 'cls'}
        return BlockStructure(**args)


@scoped
def block(x,
          is_training,
          structure=BlockStructure.resnet(),
          stride=1,
          base_width=16,
          bn_params=dict(),
          dropout_params=dict()):
    bn_params = _complement_bn_params(bn_params, is_training)
    dropout_params = _complement_dropout_params(dropout_params, is_training)
    s = structure
    for i, (ksize, wf) in enumerate(zip(s.ksizes, s.width_factors)):
        x = bn_relu_conv(
            x,
            bn_params=bn_params,
            conv_params={
                'ksize': ksize, 'stride': stride if i == 0 else 1,
                'width': base_width * wf
            },
            is_training=is_training,
            name=f'bn_relu_conv{i}')
        if i in s.dropout_locations:
            x = dropout(x, **dropout_params)
    return x


@scoped
def residual_block(x,
                   is_training,
                   structure=BlockStructure.resnet(),
                   stride=1,
                   width=16,
                   dim_change='id',
                   bn_params=dict(),
                   dropout_params={'rate': 0.3}):
    """
    A generic ResNet full pre-activation residual block.
    :param x: input tensor
    :param is_training: Tensor or bool. training indicator for dropout if `on` 
        parameter is not in `dropout_params`, and batch normalization if `mode` 
        parameter is not in `bn_params`
    :param structure: a BlockStructure instance
    :param width: number of output channels
    :param bn_params: parameters for `batch_normalization`
    :param dropout_params: parameters for `dropout`
    """

    def _skip_connection(x, stride, width):
        x_width = x.shape[-1].value
        assert width >= x_width
        if width > x_width or stride > 1:
            if dim_change == 'id':
                x = identity_mapping(x, stride, width)
            elif dim_change == 'nin':  # zagoruyko, TODO: bn_relu can be shared
                x = bn_relu_conv(
                    x,
                    bn_params=bn_params,
                    conv_params={'ksize': 1, 'stride': stride, 'width': width},
                    is_training=is_training,
                    name='skip')
        return x

    r = block(
        x,
        is_training,
        structure,
        stride=stride,
        base_width=width,
        bn_params=bn_params,
        dropout_params=dropout_params)
    s = _skip_connection(x, stride, width)
    return s + r


@scoped
def densenet_transition(x,
                        compression_factor: float = 0.5,
                        bn_params=dict(),
                        is_training=None):
    """ Densenet transition layer without dropoout """
    width = int(x.shape[-1].value * compression_factor)
    x = bn_relu_conv(
        x,
        bn_params=bn_params,
        conv_params={'ksize': 1, 'width': width},
        is_training=is_training)
    x = avg_pool(x, stride=2)
    return x


@scoped
def dense_block(x,
                is_training,
                length=6,
                block_structure=BlockStructure.densenet(),
                base_width=12,
                bn_params=dict(),
                dropout_params={'rate': 0.2}):
    """
    A generic dense block.
    :param x: input tensor
    :param is_training: Tensor or bool. training indicator for dropout if `on` 
        parameter is not in `dropout_params`, and batch normalization if `mode` 
        parameter is not in `bn_params`
    :param size: number of elementary blocks
    :param block_structure: a BlockStructure instance
    :param base_width: number of output channels of an elementary block
    :param bn_params: parameters for `batch_normalization`
    :param dropout_params: parameters for `dropout`
    """
    for i in range(length):
        a = block(
            x,
            is_training,
            block_structure,
            base_width=base_width,
            bn_params=bn_params,
            dropout_params=dropout_params,
            name=f'block{i}')
        x = tf.concat([x, a], axis=3)
    return x


# Components


@scoped
def resnet(x,
           is_training,
           base_width=16,
           group_lengths=[2] * 3,
           block_structure=BlockStructure.resnet(),
           dim_change='id',
           bn_params=dict(),
           dropout_params={'rate': 0.3}):
    """
    A pre-activation resnet without the final global pooling and classification
    layers.
    :param x: input tensor
    :param is_training: Tensor. training indicator
    :param base_width: number of output channels of layers in the first group
    :param group_lengths: (N) numbers of blocks per group (width)
    :param block_structure: a BlockStructure instance
    :param bn_params: parameters for `batch_normalization`
    :param dropout_params: parameters for `dropout`
    """
    x = conv(x, 3, base_width, bias=False)  # cifar
    for i, length in enumerate(group_lengths):
        group_width = 2**i * base_width
        with tf.variable_scope(f'group{i}'):
            for j in range(length):
                x = residual_block(
                    x,
                    is_training=is_training,
                    structure=block_structure,
                    stride=1 + int(i > 0 and j == 0),
                    width=group_width,
                    dim_change=dim_change,
                    bn_params=bn_params,
                    dropout_params=dropout_params,
                    name=f'block{j}')
    return bn_relu(x, bn_params, is_training=is_training)


@scoped
def densenet(x,
             is_training,
             base_width=12,
             group_lengths=[19] * 3,
             block_structure=BlockStructure.densenet(),
             compression_factor=0.5,
             bn_params=dict(),
             dropout_params={'rate': 0.2}):
    """
    A densenet without the final global pooling and classification layers.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization
    :param base_width: number of output channels of the first layer
    :param group_lengths: numbers of elementary blocks per dense block
    :param block_structure: a BlockStructure instance
    :param compression_factor: float from 0 to 1. transition layer compression
    :param bn_params: parameters for `batch_normalization`
    :param dropout_params: parameters for `dropout`
    """
    x = conv(x, 3, 2 * base_width, bias=False)  # cifar
    for i, length in enumerate(group_lengths):
        if i > 0:
            x = densenet_transition(
                x,
                compression_factor=compression_factor,
                bn_params=bn_params,
                is_training=is_training,
                name=f'transition{i-1}')
        x = dense_block(
            x,
            is_training=is_training,
            length=length,
            block_structure=block_structure,
            base_width=base_width,
            bn_params=bn_params,
            dropout_params=dropout_params,
            name=f'dense_block{i}')
    return x