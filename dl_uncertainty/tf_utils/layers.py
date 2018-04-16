import inspect

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from .variables import conv_weight_variable, bias_variable

var_scope = variable_scope.variable_scope


def default_arg(func, param):
    return inspect.signature(func).parameters[param].default


def default_args(func, params):
    return dict([(p, inspect.signature(func).parameters[p].default)
                 for p in params])


# Linear, affine


def add_biases(x, return_params=False, scope='add_biases'):
    with tf.variable_scope(scope):
        b = bias_variable(x.shape[-1].value)
        h = x + b
        return (h, b) if return_params else h


# Convolution


def conv(x,
         ksize,
         width,
         stride=1,
         dilation=1,
         padding='SAME',
         bias=True,
         scope: str = None):
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
    with var_scope(scope, 'Conv', [x]):
        h = tf.nn.convolution(
            x,
            filter=conv_weight_variable(ksize, x.shape[-1].value, width),
            strides=[stride] * 2,
            dilation_rate=[dilation] * 2,
            padding=padding)
        if bias:
            h += bias_variable(width)
        print(scope, x.shape[1:], h.shape[1:])
        return h


def separable_conv(x,
                   ksize,
                   width,
                   channel_multiplier=1,
                   stride=1,
                   padding='SAME',
                   bias=True,
                   scope: str = None):
    """
    A wrapper for tf.nn.conv2d.
    :param x: 4D input "NHWC" tensor 
    :param ksize: int or tuple of 2 ints representing spatial dimensions 
    :param width: number of output channels
    :param stride: int (or float in case of transposed convolution)
    :param padding: string. "SAME" or "VALID" 
    :param bias: add biases
    """
    with var_scope(scope, 'Conv', [x]):
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

    with var_scope(scope, 'BN', [x]):
        ema = tf.train.ExponentialMovingAverage(decay)

        m, v = tf.nn.moments(
            x, axes=list(range(len(x.shape) - 1)), name='moments')

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

        return tf.nn.batch_normalization(x, mean, var, offset, scale,
                                         var_epsilon)


def dropout(x, on, rate, is_training=None, **kwargs):
    return tf.layers.dropout(x, rate, training=on, **kwargs)


def convex_combination(x, r, scope: str = None):
    # TODO: improve initialization and constraining
    with var_scope(scope, 'ConvexCombination', [x]):
        a = tf.get_variable('a', [1], initializer=tf.constant_initializer(0.99))
        constrain_a = tf.assign(a, tf.clip_by_value(a, 0, 1))

        def a_with_update():
            with tf.control_dependencies([constrain_a]):
                return tf.identity(a)

        a = a_with_update()  # TODO: use constraint in var_scope instead
        return a * x + (1 - a) * r


def identity_mapping(x, stride, width):
    x_width = x.shape[-1].value
    assert width >= x_width
    if stride > 1:
        x = avg_pool(x, stride, padding='VALID')
    if width > x_width:
        x = tf.pad(x, 3 * [[0, 0]] + [[0, width - x_width]])
    return x


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


def bn_relu(x, bn_params=dict(), is_training=None, scope: str = None):
    bn_params = _complement_bn_params(bn_params, is_training)
    with var_scope(scope, 'BNReLU', [x]):
        x = batch_normalization(x, **bn_params)
        return tf.nn.relu(x)


def bn_relu_conv(x,
                 bn_params=dict(),
                 conv_params=dict(),
                 is_training=None,
                 scope: str = None):
    """ Convolutions are unbiased """
    bn_params = _complement_bn_params(bn_params, is_training)
    with var_scope(scope, 'BNReLUConv', [x]):
        x = batch_normalization(x, **bn_params)
        x = tf.nn.relu(x)
        return conv(x, **conv_params, bias=False)


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
        self.width_factors=width_factors
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

def elementary_block(x,
                     is_training,
                     structure=BlockStructure.resnet(),
                     stride=1,
                     base_width=16,
                     bn_params=dict(),
                     dropout_params=dict(),
                     scope: str = None):
    bn_params = _complement_bn_params(bn_params, is_training)
    dropout_params = _complement_dropout_params(dropout_params, is_training)
    s = structure
    with var_scope(scope, 'Block' + '-'.join(map(str, s.ksizes)), [x]):
        for i, (ksize, wf) in enumerate(zip(s.ksizes, s.width_factors)):
            x = bn_relu_conv(
                x,
                bn_params=bn_params,
                conv_params={
                    'ksize': ksize, 'stride': stride if i == 0 else 1,
                    'width': base_width * wf
                },
                is_training=is_training,
                scope=f'bn_relu_conv{i}')
            if i in s.dropout_locations:
                x = dropout(x, **dropout_params)
        return x


def residual_block(x,
                   is_training,
                   structure=BlockStructure.resnet(),
                   stride=1,
                   width=16,
                   dim_change='id',
                   bn_params=dict(),
                   dropout_params={'rate': 0.3},
                   scope: str = None):
    """
    A generic ResNet full pre-activation residual block.
    :param x: input tensor
    :param is_training: Tensor or bool. training indicator for dropout if `on` 
        parameter is not in `dropout_params`, and batch normalization if `mode` 
        parameter is not in `bn_params`
    :param structure: a BlockStructure instance
    :param width: number of output channels
    :param bn_params: batch normalization parameters
    :param dropout_params: dropout parameters
    """

    def _skip_connection(x, stride, width):
        x_width = x.shape[-1].value
        assert width >= x_width
        if width > x_width or stride > 1:
            if dim_change == 'id':
                x = identity_mapping(x, stride, width)
            elif dim_change == 'nin':
                x = bn_relu_conv(
                    x,
                    bn_params=bn_params,
                    conv_params={'ksize': 1, 'stride': stride, 'width': width},
                    is_training=is_training,
                    scope='skip')
        return x

    with var_scope(scope, 'ResidualBlock', [x]):
        r = elementary_block(
            x,
            is_training,
            structure,
            stride=stride,
            base_width=width,
            bn_params=bn_params,
            dropout_params=dropout_params)
        s = _skip_connection(x, stride, width)
        #print(scope, s.shape, r.shape)
        return s + r


def densenet_transition(x,
                        compression_factor: float = 0.5,
                        bn_params=dict(),
                        is_training=None,
                        scope: str = None):
    """ Densenet transition layer without dropoout """
    with var_scope(scope, 'Transition', [x]):
        width = int(x.shape[-1].value * compression_factor)
        x = bn_relu_conv(
            x,
            bn_params=bn_params,
            conv_params={'ksize': 1, 'width': width},
            is_training=is_training,
            scope='conv_compression')
        x = avg_pool(x, stride=2)
        return x


def dense_block(x,
                is_training,
                length=6,
                block_structure=BlockStructure.densenet(),
                base_width=12,
                bn_params=dict(),
                dropout_params={'rate': 0.2},
                scope: str = None):
    """
    A generic dense block.
    :param x: input tensor
    :param is_training: Tensor or bool. training indicator for dropout if `on` 
        parameter is not in `dropout_params`, and batch normalization if `mode` 
        parameter is not in `bn_params`
    :param size: number of elementary blocks
    :param block_structure: a BlockStructure instance
    :param base_width: number of output channels of an elementary block
    :param bn_params: batch normalization parameters
    :param dropout_params: dropout parameters
    """
    with var_scope(scope, 'ResidualBlock', [x]):
        for i in range(length):
            a = elementary_block(
                x,
                is_training,
                block_structure,
                base_width=base_width,
                bn_params=bn_params,
                dropout_params=dropout_params,
                scope=f'block{i}')
            x = tf.concat([x, a], axis=3, name=f'concat{i}')
            #print(scope, x.shape)
        return x


# Components


def resnet(x,
           is_training,
           base_width=16,
           widening_factor=1,
           group_lengths=[2, 2, 2],
           block_structure=BlockStructure.resnet(),
           dim_change='id',
           bn_params=dict(),
           dropout_params={'rate': 0.3},
           scope: str = None):
    """
    A pre-activation resnet without the final global pooling and classification
    layers.
    :param x: input tensor
    :param is_training: Tensor. training indicator
    :param base_width: number of output channels of the first layer
    :param widening_factor: (k) block widths are proportional to 
        base_width*widening_factor
    :param group_lengths: (N) numbers of blocks per group (width)
    :param block_structure: a BlockStructure instance
    :param bn_params: batch normalization parameters
    :param dropout_params: dropout parameters
    """

    def _bn_relu(x, id):
        return bn_relu(
            x, bn_params, is_training=is_training, scope='BNReLU' + id)

    with var_scope(scope, 'ResNet', [x]):
        x = conv(x, 3, base_width, bias=False)
        for i, length in enumerate(group_lengths):
            group_width = 2**i * base_width * widening_factor
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
                        scope=f'block{j}')
        return _bn_relu(x, '1')


def densenet(x,
             is_training,
             group_lengths=[38, 38, 38],
             block_structure=BlockStructure.densenet(),
             base_width=12,
             compression_factor=0.5,
             bn_params=dict(),
             dropout_params={'rate': 0.2},
             scope: str = None):
    """
    A densenet without the final global pooling and classification layers.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization
    :param base_width: number of output channels of the first layer
    :param widening_factor: (k) block widths are proportional to 
        base_width*widening_factor
    :param group_lengths: (N) numbers of blocks per group (width)
    :param block_properties: kernel sizes of convolutional layers in a block
    :param bn_decay: batch normalization exponential moving average decay
    """

    def _bn_relu(x, id):
        return bn_relu(
            x, bn_params, is_training=is_training, scope='BNReLU' + id)

    with var_scope(scope, 'DenseNet', [x]):
        x = conv(x, ksize=7, width=2 * base_width, stride=2, bias=False)
        x = _bn_relu(x, 'in')
        x = max_pool(x, stride=2, ksize=3)
        for i, length in enumerate(group_lengths):
            if i > 0:
                x = densenet_transition(
                    x,
                    compression_factor=compression_factor,
                    bn_params=bn_params,
                    is_training=is_training,
                    scope=f'transition{i-1}')
            x = dense_block(
                x,
                is_training=is_training,
                length=length,
                block_structure=block_structure,
                base_width=base_width,
                bn_params=bn_params,
                dropout_params=dropout_params,
                scope=f'block{i}')
        return x