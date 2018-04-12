import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from .variables import conv_weight_variable, bias_variable

var_scope = variable_scope.variable_scope


def default_arg(func, param):
    import inspect
    return inspect.signature(func).parameters[param].default


# Linear, affine


def add_biases(x, return_params=False, reuse=False, scope='add_biases'):
    with tf.variable_scope(scope, reuse=reuse):
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
         reuse: bool = None,
         scope: str = None):
    '''
    A wrapper for tf.nn.conv2d.
    :param x: 4D input "NHWC" tensor 
    :param ksize: int or tuple of 2 ints representing spatial dimensions 
    :param width: number of output channels
    :param stride: int (or float in case of transposed convolution)
    :param dilation: dilation of the kernel
    :param padding: string. "SAME" or "VALID" 
    :param bias: add biases
    '''
    with var_scope(scope, 'Conv', [x], reuse=reuse):
        h = tf.nn.convolution(
            x,
            filter=conv_weight_variable(ksize, x.shape[-1].value, width),
            strides=[stride] * 2,
            dilation_rate=[dilation] * 2,
            padding=padding)
        if bias:
            h += bias_variable(width)
        return h


def separable_conv(x,
                   ksize,
                   width,
                   channel_multiplier=1,
                   stride=1,
                   padding='SAME',
                   bias=True,
                   reuse: bool = None,
                   scope: str = None):
    '''
    A wrapper for tf.nn.conv2d.
    :param x: 4D input "NHWC" tensor 
    :param ksize: int or tuple of 2 ints representing spatial dimensions 
    :param width: number of output channels
    :param stride: int (or float in case of transposed convolution)
    :param padding: string. "SAME" or "VALID" 
    :param bias: add biases
    '''
    with var_scope(scope, 'Conv', [x], reuse=reuse):
        in_width = x.shape[-1].value
        h = tf.nn.separable_conv2d(
            x,
            depthwise_filter=conv_weight_variable(
                ksize, in_width, channel_multiplier, name="wweights"),
            pointwise_filter=conv_weight_variable(
                (1, 1), in_width*channel_multiplier, width, name="pweights"),
            strides=[1] + [stride] * 2 + [1],
            padding=padding)
        if bias:
            h += bias_variable(width)
        return h


def conv_transp(
        x,
        ksize,
        output_shape,  # TODO: width+output_stride
        stride=1,
        padding='SAME',
        bias=True,
        reuse: bool = None,
        scope: str = None):
    '''
    A wrapper for tf.nn.tf.nn.conv2d_transpose.
    :param x: 4D input "NHWC" tensor 
    :param ksize: int or tuple of 2 ints representing spatial dimensions 
    :param width: number of output channels TODO
    :param output_shape: tuple of 2 ints
    :param padding: string. "SAME" or "VALID" 
    :param bias: add biases
    '''
    with var_scope(scope, 'ConvT', [x], reuse=reuse):
        h = tf.nn.conv2d_transpose(
            x,
            filter=conv_weight_variable(ksize, x.shape[-1].value, width),
            output_shape=output_shape,
            strides=[stride] * 2,
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
                        is_training,
                        offset_scale=(True, True),
                        decay=0.95,
                        var_epsilon=1e-8,
                        reuse: bool = None,
                        scope: str = None):
    '''
    Batch normalization that normalizes over all but the last dimension of x.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization
    :param decay: exponential moving average decay
    '''
    with var_scope(scope, 'BN', [x], reuse=reuse):
        ema = tf.train.ExponentialMovingAverage(decay)

        m, v = tf.nn.moments(
            x, axes=list(range(len(x.shape) - 1)), name='moments')

        def m_v_with_update():
            with tf.control_dependencies([ema.apply([m, v])]):
                return tf.identity(m), tf.identity(v)

        mean, var = tf.cond(is_training, m_v_with_update,
                            lambda: (ema.average(m), ema.average(v)))

        offset = 0.0 if not offset_scale[0] else tf.get_variable(
            'offset',
            shape=[x.shape[-1].value],
            initializer=tf.constant_initializer(0.0))
        scale = 1.0 if not offset_scale[1] else tf.get_variable(
            'scale',
            shape=[x.shape[-1].value],
            initializer=tf.constant_initializer(1.0))

        return tf.nn.batch_normalization(x, mean, var, offset, scale,
                                         var_epsilon)


# Blocks


def bn_relu(x,
            is_training,
            offset_scale=(True, True),
            decay=default_arg(batch_normalization, 'decay'),
            reuse: bool = None,
            scope: str = None):
    with var_scope(scope, 'BNReLU', [x], reuse=reuse):
        x = batch_normalization(
            x, is_training, offset_scale=offset_scale, decay=decay)
        return tf.nn.relu(x)


def convex_combination(x, r, is_training, reuse: bool = None,
                       scope: str = None):
    # TODO: improve initialization and constraining
    with var_scope(scope, 'ConvexCombination', [x], reuse=reuse):
        a = tf.get_variable('a', [1], initializer=tf.constant_initializer(0.99))
        constrain_a = tf.assign(a, tf.clip_by_value(a, 0, 1))

        def a_with_update():
            with tf.control_dependencies([constrain_a]):
                return tf.identity(a)

        a = tf.cond(is_training, a_with_update, lambda: a)
        #a = tf.Print(a,[a, a.name])
        return a * x + (1 - a) * r


class ResidualBlockConfig:

    def __init__(self,
                 ksizes=[3, 3],
                 dropout_locations=[],
                 dropout_rate=0.3,
                 aggregation='sum',
                 dim_increase='id'):
        '''
        A ResNet full pre-activation residual block with a (padded) identity 
        shortcut.
        :param ksizes: list of ints or int pairs. kernel sizes of convolutional 
            layers
        :param dropout_locations: indexes of convolutional layers after which
            dropout is to be applied
        :param dropout_rate: int
        :param aggregation: string. 'sum' for sum as in ResNets, 'convex' for a 
            convex combination as in Parseval networks
        :param dim_increase: string. 'id' for identity with zero padding or 
            'conv1' for projection with a 1×1 convolution with stride 2. See 
            section 3.3. in https://arxiv.org/abs/1512.03385.
        '''
        self.ksizes = ksizes
        self.dropout_locations = dropout_locations
        self.dropout_rate = dropout_rate
        if dim_increase not in ['id', 'conv1']:
            raise ValueError("dim_increase must be 'id' or 'conv1'")
        self.dim_increase = dim_increase
        if aggregation not in ['sum', 'convex']:
            raise ValueError("aggregation must be 'sum' or 'convex'")
        self.aggregation = aggregation


def residual_block(x,
                   is_training,
                   properties=ResidualBlockConfig([3, 3]),
                   width=16,
                   stride=1,
                   bn_offset_scale=(True, True),
                   bn_decay=default_arg(batch_normalization, 'decay'),
                   reuse: bool = None,
                   scope: str = None):
    '''
    A generic ResNet full pre-activation residual block.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization    
    :param properties: a ResidualBlockConfig instance
    :param width: number of output channels
    :param bn_decay: batch normalization exponential moving average decay
    :param first_block: needs to be set to True if this is the first residual 
        block of the whole ResNet, the first activation is omitted
    '''

    def _bn_relu(x, id):
        return bn_relu(
            x,
            is_training,
            offset_scale=bn_offset_scale,
            decay=bn_decay,
            reuse=reuse,
            scope='bn_relu' + id)

    def _conv(x, ksize, stride, id):
        return conv(
            x, ksize, width, stride, bias=False, reuse=reuse, scope='conv' + id)

    x_width = x.shape[-1].value
    assert width >= x_width
    with var_scope(scope, 'ResBlock', [x], reuse=reuse):
        xb = _bn_relu(x, 'in')  # <- note this
        # residual
        r = xb
        for i, ksize in enumerate(properties.ksizes):
            if i > 0:
                r = _bn_relu(r, str(i))
            r = _conv(r, ksize, stride if i == 0 else 1, str(i))
            if i in properties.dropout_locations:
                r = tf.layers.dropout(
                    r, properties.dropout_rate, training=is_training)
        # dimension increase
        if width > x_width or stride > 1:
            if properties.dim_increase == 'id':
                if stride > 1:
                    x = avg_pool(x, stride, padding='VALID')
                x = tf.pad(x, 3 * [[0, 0]] + [[0, width - x_width]])
            elif properties.dim_increase == 'conv1':
                x = _conv(xb, 1, stride, 'skip')
        print(scope, x.shape, r.shape)
        # aggregation
        if properties.aggregation == 'sum':
            return x + r
        elif properties.aggregation == 'convex':
            return convex_combination(
                x, r, is_training, scope='convex_combin', reuse=reuse)


# Components


def resnet(x,
           is_training,
           base_width=16,
           widening_factor=1,
           group_lengths=[2, 2, 2],
           block_properties=default_arg(residual_block, 'properties'),
           bn_offset_scale=(True, True),
           bn_decay=default_arg(batch_normalization, 'decay'),
           reuse: bool = None,
           scope: str = None):
    '''
    A pre-activation resnet without the final global pooling and 
    classification layers.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization
    :param base_width: number of output channels of the first layer
    :param widening_factor: (k) block widths are proportional to 
        base_width*widening_factor
    :param group_lengths: (N) numbers of blocks per group (width)
    :param block_properties: kernel sizes of convolutional layers in a block
    :param bn_decay: batch normalization exponential moving average decay
    '''

    def _bn_relu(h, id):
        return bn_relu(
            h,
            is_training,
            offset_scale=bn_offset_scale,
            decay=bn_decay,
            reuse=reuse,
            scope='BNReLU' + id)

    with var_scope(scope, 'ResNet', [x], reuse=reuse):
        h = conv(x, 3, base_width, bias=False)
        for i, length in enumerate(group_lengths):
            group_width = 2**i * base_width * widening_factor
            with tf.variable_scope('group' + str(i), reuse=False):
                for j in range(length):
                    h = residual_block(
                        h,
                        is_training=is_training,
                        properties=block_properties,
                        width=group_width,
                        stride=1 + int(i > 0 and j == 0),
                        bn_offset_scale=bn_offset_scale,
                        bn_decay=bn_decay,
                        reuse=reuse,
                        scope='block' + str(j))
        return _bn_relu(h, '1')