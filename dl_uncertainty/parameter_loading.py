import numpy as np
from tensorflow.python import pywrap_tensorflow
import re


def print_tensors_in_checkpoint_file(file_name,
                                     tensor_name,
                                     all_tensors,
                                     all_tensor_names=False):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
        file_name: Name of the checkpoint file.
        tensor_name: Name of the tensor in the checkpoint file to print.
        all_tensors: Boolean indicating whether to print all tensors.
        all_tensor_names: Boolean indicating whether to print all tensor names.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors or all_tensor_names:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print(key)
                if all_tensors:
                    print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        if ("Data loss" in str(e) and
            (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
                It's likely that this is a V2 checkpoint and you need to provide the filename
                *prefix*.  Try removing the '.' and extension.  Try:
                inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))


def get_parameters_from_checkpoint_file(path, param_name_filter=lambda x: True):
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return {
        key: reader.get_tensor(key)
        for key in sorted(var_to_shape_map) if param_name_filter(key)
    }


def translate_resnet_variable_names(names: list):
    simple_replacements = {  # by priority
        'resnet_v2_50/conv1': 'resnet/conv_root',
        'resnet_v2_50/logits': 'conv_logits',
        'resnet_v2_50': 'resnet',
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
    drop_out_words = ['oving', 'Momentum', 'global_step']
    drop_out_words += ['logits', 'bias']
    filters = [
        lambda x: not any(map(lambda d: d in x, drop_out_words)),
        #lambda x: 'bias' not in x or x == 'resnet_v2_50/logits/biases',
    ]

    def filter_param_name(param_name):
        return all(map(lambda f: f(param_name), filters))

    param_name_to_array = get_parameters_from_checkpoint_file(
        checkpoint_file_path, filter_param_name)
    new_names = translate_resnet_variable_names(param_name_to_array.keys())
    name_to_new_name = dict(zip(param_name_to_array.keys(), new_names))
    return {
        name_to_new_name[n]: param_name_to_array[n]
        for n in param_name_to_array.keys()
    }


if __name__ == "__main__":
    from ioutils.filesystem import find_in_ancestor
    path = find_in_ancestor(
        __file__, '/data/pretrained_parameters/resnetv2_50/resnet_v2_50.ckpt')
    param_name_to_array = get_resnet_parameters_from_checkpoint_file(path)

    for n in param_name_to_array.keys():
        print(n, param_name_to_array[n])