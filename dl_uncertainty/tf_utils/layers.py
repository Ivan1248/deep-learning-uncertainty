# pylint: disable=unexpected-keyword-arg

import inspect

import functools

import numpy as np
import tensorflow as tf

from .variables import conv_weight_variable, bias_variable


def scoped(fun):

    @functools.wraps(fun)  # preserves the signature (needed for default_arg)
    def wrapper(*args, reuse: bool = None, name: str = None, **kwargs):
        #if reuse == -1:  # TODO: change reuse default to tf.AUTO_REUSE
        #    reuse = tf.get_variable_scope().reuse
        a = list(args) + list(kwargs.values())
        tensors = [v for v in a if type(v) is tf.Tensor]
        with tf.variable_scope(
                name, default_name=fun.__name__, values=tensors, reuse=reuse):
            return fun(*args, **kwargs)

    return wrapper


def default_arg(func, param):
    return inspect.signature(func).parameters[param].default


def default_args(func, params):
    return dict([(p, inspect.signature(func).parameters[p].default)
                 for p in params])


def filter_dict(d: dict, keys):
    if callable(keys):
        return {k: v for k, v in d.items() if keys(k)}
    return {k: v for k, v in d.items() if k in keys}


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
def conv(x, ksize, width, stride=1, dilation=1, padding='SAME', bias=False):
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
                   bias=False):
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


def _get_resized_shape(x, factor):
    return (np.array([d.value
                      for d in x.shape[1:3]]) * factor + 0.5).astype(np.int)


def rescale_nearest_neighbor(x, factor):
    shape = _get_resized_shape(x, factor)
    return x if factor == 1 else tf.image.resize_nearest_neighbor(x, shape)


def resize_bilinear(x, factor):
    shape = _get_resized_shape(x, factor)
    return x if factor == 1 else tf.image.resize_bilinear(x, shape)


# Special


@scoped
def batch_normalization(
        x,
        mode=None,
        is_training=None,  # has no effect if mode is deffind
        offset=True,
        scale=True,
        decay=0.95,
        var_epsilon=1e-8,
        scope: str = None):
    """
    Batch normalization with scaling that normalizes over all but the last
    dimension of x.
    :param x: input tensor.
    :param mode: Tensor or str. 'moving' or 'fixed' (TODO or 'accumulation' -- not 
        implemented).
    :param is_training: Tensor or bool. Has no effect if mode is defined,
        otherwise mode is 'moving' if True and 'fixed' if False.
    :param offset: Tensor or bool. Add learned offset to the output.
    :param scale: Tensor or bool. Add learned scaling to the output.
    :param decay: Exponential moving average decay.
    """
    moving = tf.equal(mode, 'moving') if mode is not None else is_training
    assert moving is not None

    ema = tf.train.ExponentialMovingAverage(decay)
    m, v = tf.nn.moments(x, axes=list(range(len(x.shape) - 1)), name='moments')

    def m_v_with_update():
        nonlocal m, v
        with tf.control_dependencies([ema.apply([m, v])]):
            return tf.identity(m), tf.identity(v)

    mean, var = tf.cond(moving, m_v_with_update,
                        lambda: (ema.average(m), ema.average(v)))

    def get_var(name, value):
        return tf.get_variable(
            name,
            shape=[x.shape[-1].value],
            initializer=tf.constant_initializer(value))

    offset = 0.0 if not offset else get_var('offset', 0.0)
    scale = 1.0 if not scale else get_var('scale', 1.0)

    return tf.nn.batch_normalization(x, mean, var, offset, scale, var_epsilon)


def dropout(x,
            rate,
            active=None,
            is_training=None,
            feature_map_wise=False,
            **kwargs):
    active = active or is_training
    assert active is not None
    if feature_map_wise:  # TODO: remove True
        kwargs = {**kwargs, 'noise_shape': [x.shape[0], 1, 1, x.shape[3]]}
    return tf.layers.dropout(x, rate, **kwargs, training=active)


@scoped
def convex_combination(x, r, scope: str = None):
    # TODO: improve initialization and constraining
    a = tf.get_variable('a', [1], initializer=tf.constant_initializer(0.99))
    constrain_a = tf.assign(a, tf.clip_by_value(a, 0, 1))

    def a_with_update():
        with tf.control_dependencies([constrain_a]):
            return tf.identity(a)

    a = a_with_update()  # TODO: use constraint in tf.variable_scope instead
    return a * x + (1 - a) * r


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
def sample(fn, x, n: int, examplewise=False, max_batch_size=None,
           backprop=True, use_tf_loop=True):
    """
    Sampling of a stochastic operation. NOTE: unfortunately doesn't work if 
    there are variables defined inside fn.
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
    if use_tf_loop:
        examples_per_iteration = max(1, max_batch_size // n)
        ys = tf.map_fn(
            fn=lambda x: _sample(fn, x, n),  # [H,W,C]->[n,H,W,C]
            elems=x,  # [N,H,W,C]
            parallel_iterations=examples_per_iteration,
            back_prop=backprop)  # [N,n,H,W,C]
    else:
        ys = [fn(x)]
        tf.get_tf.variable_scope().reuse_variables()
        ys += [fn(x) for _ in range(sample_count - 1)]
        ys = tf.stack(ys)  # [N,n,H,W,C]
    if examplewise:  # [N,n,H,W,C]
        return ys
    else:  # [n,N,H,W,C]
        t = [1, 0] + list(range(2, len(ys.shape)))
        return tf.transpose(ys, t)  # [n,N,H,W,C]


# Blocks


@scoped
def bn_relu(x, bn_params=dict()):
    x = batch_normalization(x, **bn_params)
    return tf.nn.relu(x)


@scoped
def identity_mapping(x, stride, width):
    x_width = x.shape[-1].value
    assert width >= x_width
    if stride > 1:
        x = avg_pool(x, stride, padding='SAME')
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
        return BlockStructure(**filter_dict(locals(), lambda k: k != 'cls'))

    @classmethod
    def densenet(cls,
                 ksizes=[1, 3],
                 width_factors=[4, 1],
                 dropout_locations=[0, 1]):
        return BlockStructure(**filter_dict(locals(), lambda k: k != 'cls'))


@scoped
def block(x,
          structure=BlockStructure.resnet(),
          stride=1,
          base_width=16,
          bn_params=dict(),
          dropout_params=dict()):
    s = structure
    for i, (ksize, wf) in enumerate(zip(s.ksizes, s.width_factors)):
        x = bn_relu(x, bn_params, name=f'bn_relu{i}')
        x = conv(
            x,
            ksize,
            base_width * wf,
            stride=stride if i == 0 else 1,
            name=f'conv{i}')
        if i in s.dropout_locations:
            x = dropout(x, **dropout_params)
    return x


@scoped
def residual_block(x,
                   structure=BlockStructure.resnet(),
                   stride=1,
                   width=16,
                   dim_change='id',
                   bn_params=dict(),
                   dropout_params={'rate': 0.3}):
    """
    A generic ResNet full pre-activation residual block.
    :param x: input tensor
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
                x = bn_relu(x, bn_params)
                x = conv(x, 1, width, stride=stride, name='conv_skip')
        return x

    r = block(
        x,
        structure,
        stride=stride,
        base_width=width,
        bn_params=bn_params,
        dropout_params=dropout_params)
    s = _skip_connection(x, stride, r.shape[-1])
    assert s.shape[-1] == r.shape[-1]
    return s + r


@scoped
def densenet_transition(x, compression: float = 0.5, bn_params=dict()):
    """ Densenet transition layer without dropoout """
    width = int(x.shape[-1].value * compression)
    x = bn_relu(x, bn_params)
    x = conv(x, 1, width)
    x = avg_pool(x, stride=2)
    return x


@scoped
def dense_block(x,
                length=6,
                block_structure=BlockStructure.densenet(),
                base_width=12,
                bn_params=dict(),
                dropout_params={'rate': 0.2}):
    """
    A generic dense block.
    :param x: input tensor
    :param size: number of elementary blocks
    :param block_structure: a BlockStructure instance
    :param base_width: number of output channels of an elementary block
    :param bn_params: parameters for `batch_normalization`
    :param dropout_params: parameters for `dropout`
    """
    for i in range(length):
        a = block(
            x,
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
           base_width=16,
           group_lengths=[2] * 3,
           block_structure=BlockStructure.resnet(),
           dim_change='id',
           bn_params=dict(),
           dropout_params={'rate': 0.3},
           small_input=None):
    """
    A pre-activation resnet without the final global pooling and classification
    layers.
    :param x: input tensor
    :param base_width: number of output channels of layers in the first group
    :param group_lengths: (N) numbers of blocks per group (width)
    :param block_structure: a BlockStructure instance
    :param bn_params: parameters for `batch_normalization`
    :param dropout_params: parameters for `dropout`
    """
    small_input = small_input or (min(x.shape[i].value for i in [1, 2]) < 128)
    if small_input:
        x = conv(x, 3, base_width)  # cifar
    else:
        x = conv(x, 7, base_width, stride=2)
        x = max_pool(x, stride=2, ksize=3)
    for i, length in enumerate(group_lengths):
        group_width = 2**i * base_width
        with tf.variable_scope(f'group{i}'):
            for j in range(length):
                x = residual_block(
                    x,
                    structure=block_structure,
                    stride=1 + int(i > 0 and j == 0),
                    width=group_width,
                    dim_change=dim_change,
                    bn_params=bn_params,
                    dropout_params=dropout_params,
                    name=f'block{j}')
    return bn_relu(x, bn_params)


@scoped
def densenet(
        x,
        base_width=12,
        group_lengths=[19] * 3,  # dense block sizes
        block_structure=BlockStructure.densenet(),
        compression=0.5,
        bn_params=dict(),
        dropout_params={'rate': 0.3},
        small_input=None):
    """
    A densenet without the final global pooling and classification layers.
    :param x: input tensor
    :param base_width: number of output channels of the first layer
    :param group_lengths: numbers of elementary blocks per dense block
    :param block_structure: a BlockStructure instance
    :param compression: float from 0 to 1. transition layer compression
    :param bn_params: parameters for `batch_normalization`
    :param dropout_params: parameters for `dropout`
    """
    small_input = small_input or (min(x.shape[i].value for i in [1, 2]) < 128)
    if small_input:
        x = conv(x, 3, 2 * base_width, name='conv_in')  # cifar
    else:
        x = conv(x, 7, base_width, stride=2, name='conv_in')
        x = max_pool(x, stride=2, ksize=3)
    x = conv(x, 3, 2 * base_width)  # cifar
    for i, length in enumerate(group_lengths):
        if i > 0:
            x = densenet_transition(
                x, compression, bn_params=bn_params, name=f'transition{i-1}')
        x = dense_block(
            x,
            length=length,
            block_structure=block_structure,
            base_width=base_width,
            bn_params=bn_params,
            dropout_params=dropout_params,
            name=f'dense_block{i}')
    return bn_relu(x, bn_params)


@scoped
def ladder_densenet(
        x,
        base_width=32,
        group_lengths=[6, 12, 24, 16],  # 121 [6, 12, 32, 32]-169
        block_structure=BlockStructure.densenet(dropout_locations=[1]),
        compression=0.5,
        upsampling_block_width=128,
        bn_params=dict(),
        dropout_params={'rate': 0.2}):
    """
    A densenet without the final global pooling and classification layers.
    :param x: input tensor
    :param base_width: number of output channels of the first layer
    :param group_lengths: numbers of elementary blocks per dense block
    :param block_structure: a BlockStructure instance
    :param compression: float from 0 to 1. transition layer compression
    :param bn_params: parameters for `batch_normalization`
    :param dropout_params: parameters for `dropout`
    """
    db_params = filter_dict(locals(), \
        ['block_structure', 'base_width', 'bn_params', 'dropout_params'])
    tr_params = filter_dict(locals(), ['compression', 'bn_params'])

    def _split_dense_block(x):
        i = len(group_lengths) - 1
        l = group_lengths[i]
        x = dense_block(x, length=l // 2, **db_params, name=f'dblock{i}a')
        skip = x
        x = avg_pool(x, 2)
        x = dense_block(x, length=l - l // 2, **db_params, name=f'dblock{i}b')
        return x, skip

    @scoped
    def blend_up(x, skip):
        x = tf.image.resize_bilinear(x, [d.value for d in skip.shape[1:3]])
        skip = bn_relu(skip, bn_params)  # TODO name
        skip = conv(skip, 1, x.shape[-1].value, name=f'conv_bottleneck{i}')
        x = tf.concat([x, skip], axis=3)
        x = bn_relu(x, bn_params)
        x = conv(x, 3, upsampling_block_width, name=f'conv_blend{i}')
        return x

    with tf.variable_scope('layer_in'):
        x = conv(x, 7, 2 * base_width, stride=2)
        x = bn_relu(x, bn_params)
        x = max_pool(x, 2)

    skips = []
    for i, length in enumerate(group_lengths[:-1]):
        x = dense_block(x, length=length, **db_params, name=f'dblock{i}')
        skips.append(x)
        x = densenet_transition(x, **tr_params, name=f'transition{i}')
    x, skip = _split_dense_block(x)
    skips.append(x)

    with tf.variable_scope('head'):
        x = bn_relu(x, bn_params, name='bn_relu_bottleneck')
        x = conv(x, 1, 512, name='conv_bottleneck')
        x = bn_relu(x, bn_params, name='bn_relu_context')
        x = conv(x, 3, 128, dilation=2, name='conv_context')

        pre_logits_aux = x
        for i, skip in reversed(list(enumerate(skips))):
            x = blend_up(x, skip, name=f'blend{i}')
        pre_logits = x

        return pre_logits, pre_logits_aux


def ladder_densenet_logits(pre_logits, pre_logits_aux, image_shape, class_count,
                           bn_params):
    logitses = []
    for i, x in enumerate([pre_logits, pre_logits_aux]):
        with tf.variable_scope(f'logits{i}'):
            x = bn_relu(x, bn_params)  # no bn in the original code
            x = conv(x, 1, class_count)
            logits = tf.image.resize_bilinear(x, image_shape)
            logitses.append(logits)
    return logitses[0], logitses[1]
