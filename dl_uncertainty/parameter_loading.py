import numpy as np
from tensorflow.python import pywrap_tensorflow
import re


def get_parameters_from_checkpoint_file(path, param_name_filter=lambda x: True):
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return {
        key: reader.get_tensor(key)
        for key in sorted(var_to_shape_map) if param_name_filter(key)
    }


def translate_variable_names(names: list, simple_replacements,
                             generic_replacements):

    def replace_simple(string, dictionary):
        for k, v in dictionary.items():
            string = string.replace(k, v)
        return string

    def replace_generic(string, dictionary):
        for k, v in dictionary.items():
            in_pattern = k
            out_pattern = v[0]
            f = v[1]

            def replace(match):
                g1 = match.group(1)
                return replace_simple(out_pattern, {'{}': f(g1)})

            string = re.sub(k, replace, string)
        return string

    names = map(lambda s: replace_simple(s, simple_replacements), names)
    names = map(lambda s: replace_generic(s, generic_replacements), names)
    return list(names)


def get_resnet_parameters_from_checkpoint_file(checkpoint_file_path):
    # Checkpoint files can be downloaded here:
    # https://github.com/tensorflow/models/tree/master/research/slim
    drop_out_words = ['oving', 'Momentum', 'global_step', 'logits', 'bias']
    filters = [lambda x: not any(map(lambda d: d in x, drop_out_words))]

    def filter_param_name(param_name):
        return all(map(lambda f: f(param_name), filters))

    param_name_to_array = get_parameters_from_checkpoint_file(
        checkpoint_file_path, filter_param_name)

    simple_replacements = {  # by priority
        'resnet_v2_50/conv1': 'resnet_root_block/conv',
        'resnet_v2_50/logits': 'conv_logits',
        'resnet_v2_50': 'resnet_middle',
        'unit_1/bottleneck_v2/preact': 'rb0/bn_relu/bn',
        'bottleneck_v2/preact': 'block/bn_relu0/bn',
        'bottleneck_v2/shortcut': 'conv_skip',
        'postnorm': 'post_bn_relu/bn',
        'weights': 'weights:0',
        'biases': 'bias:0',
        'beta': 'offset:0',
        'gamma': 'scale:0',
    }
    id = lambda x: x
    sub1 = lambda x: str(int(x) - 1)
    generic_replacements = {
        r'block(\d+)': ['group{}', sub1],
        r'unit_(\d+)': ['rb{}', sub1],
        r'bottleneck_v2\/conv(\d+)\/BatchNorm': ['block/bn_relu{}/bn', id],
        r'bottleneck_v2\/conv(\d+)': ['block/conv{}', sub1],
    }

    new_names = translate_variable_names(
        param_name_to_array.keys(), simple_replacements, generic_replacements)

    name_to_new_name = dict(zip(param_name_to_array.keys(), new_names))
    return {
        name_to_new_name[n]: param_name_to_array[n]
        for n in param_name_to_array.keys()
    }


def get_densenet_parameters_from_checkpoint_file(checkpoint_file_path):
    # Checkpoint files can be downloaded here:
    # https://github.com/pudae/tensorflow-densenet
    drop_out_words = [
        'oving', 'Momentum', 'global_step', 'logits', 'bias',
        'densenet121/BatchNorm'
    ]
    filters = [lambda x: not any(map(lambda d: d in x, drop_out_words))]

    def filter_param_name(param_name):
        return all(map(lambda f: f(param_name), filters))

    param_name_to_array = get_parameters_from_checkpoint_file(
        checkpoint_file_path, filter_param_name)

    simple_replacements = {  # by priority
        'densenet121/conv1': 'densenet_root_block/conv',
        'densenet121/logits': 'conv_logits',
        'densenet121': 'densenet_middle',
        'final_block/BatchNorm': 'post_bn_relu/bn',
        'weights': 'weights:0',
        'biases': 'bias:0',
        'beta': 'offset:0',
        'gamma': 'scale:0',
    }
    id = lambda x: x
    sub1 = lambda x: str(int(x) - 1)
    generic_replacements = {
        r'dense_block(\d+)': ['db{}', sub1],
        r'conv_block(\d+)': ['block{}', sub1],
        r'x(\d+)\/BatchNorm': ['bn_relu{}/bn', sub1],
        r'x(\d+)\/Conv': ['conv{}', sub1],
        r'transition_block(\d+)\/blk\/BatchNorm': [
            'transition{}/bn_relu/bn', sub1
        ],
        r'transition_block(\d+)\/blk\/Conv': ['transition{}/conv', sub1],
    }

    new_names = translate_variable_names(
        param_name_to_array.keys(), simple_replacements, generic_replacements)

    name_to_new_name = dict(zip(param_name_to_array.keys(), new_names))
    return {
        name_to_new_name[n]: param_name_to_array[n]
        for n in param_name_to_array.keys()
    }


def test():
    from . import dirs

    param_name_to_array = get_resnet_parameters_from_checkpoint_file(
        f'{dirs.PRETRAINED}/resnetv2_50/resnet_v2_50.ckpt')

    #for n in param_name_to_array.keys():
    #    print(n, param_name_to_array[n])


    param_name_to_array = get_densenet_parameters_from_checkpoint_file(
        f'{dirs.PRETRAINED}/densenet_121/tf-densenet121.ckpt')

    for n in param_name_to_array.keys():
        #print(n, param_name_to_array[n])
        print(n)