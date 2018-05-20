import datetime

import tensorflow as tf

from .models import Model, ModelDef
from .models import BlockStructure, TrainingComponent
from .models import InferenceComponents, TrainingComponents
from .evaluation import ClassificationEvaluator
from . import dirs, parameter_loading


class StandardInferenceComponents:

    @staticmethod
    def resnet(
            ic_kwargs,
            depth,
            cifar_root_block,
            base_width,  # for all models?
            dropout):
        assert not dropout
        print(f'ResNet-{depth}-{base_width}')
        for a in ['input_shape', 'class_count', 'problem_id']:
            assert a in ic_kwargs
        normal = ([3, 3], [1, 1], 'id')
        bottleneck = ([1, 3, 1], [1, 1, 4], 'proj')  # last paragraph in [2]
        group_lengths, ksizes, width_factors, dim_change = {
            18: ([2] * 4, *normal),  # [1] bw 64
            34: ([3, 4, 6, 3], *normal),  # [1] bw 64
            110: ([18] * 3, *normal),  # [1] bw 16
            50: ([3, 4, 6, 3], *bottleneck),  # [1] bw 64
            101: ([3, 4, 23, 3], *bottleneck),  # [1] bw 64
            152: ([3, 8, 36, 3], *bottleneck),  # [1] bw 64
            164: ([18] * 3, *bottleneck),  # [1] bw 16
            200: ([3, 24, 36, 3], *bottleneck),  # [2] bw 64
        }[depth]
        return InferenceComponents.resnet(
            **ic_kwargs,
            cifar_root_block=cifar_root_block,
            base_width=base_width,
            group_lengths=group_lengths,
            block_structure=BlockStructure.resnet(
                ksizes=ksizes,
                dropout_locations=dropout_locations,
                width_factors=width_factors),
            dim_change=dim_change)

    @staticmethod
    def wide_resnet(ic_kwargs,
                    depth,
                    width_factor,
                    cifar_root_block,
                    dropout,
                    dim_change='proj'):
        for a in ['input_shape', 'class_count', 'problem_id']:
            assert a in ic_kwargs
        print(f'WRN-{depth}-{width_factor}')
        zagoruyko_depth = depth
        group_count, ksizes = 3, [3, 3]
        group_depth = (group_count * len(ksizes))
        blocks_per_group = (zagoruyko_depth - 4) // group_depth
        depth = blocks_per_group * group_depth + 4
        assert zagoruyko_depth == depth, \
            f"Invalid depth = {zagoruyko_depth} != {depth} = zagoruyko_depth"
        return InferenceComponents.resnet(
            **ic_kwargs,
            cifar_root_block=cifar_root_block,
            base_width=16,
            width_factor=width_factor,
            group_lengths=[blocks_per_group] * group_count,
            block_structure=BlockStructure.resnet(
                ksizes=ksizes, dropout_locations=[0] if dropout else []),
            dim_change=dim_change)

    @staticmethod
    def densenet(ic_kwargs, depth, base_width, cifar_root_block,
                 dropout):  # 0.2 if data augmentation is not used
        for a in ['input_shape', 'class_count', 'problem_id']:
            assert a in ic_kwargs, a
        print(f'DenseNet-{depth}-{base_width}')
        ksizes = [1, 3]
        depth_to_group_lengths = {
            121: [6, 12, 24, 16],  # base_width = 32
            161: [6, 12, 36, 24],  # base_width = 48
            169: [6, 12, 32, 32],  # base_width = 32
        }
        if depth in depth_to_group_lengths:
            group_lengths = depth_to_group_lengths[depth]
        else:
            group_count = 3
            assert (depth - group_count - 1) % 3 == 0, \
                f"invalid depth: (depth-group_count-1) must be divisible by 3"
            blocks_per_group = (depth - group_count - 1) // \
                               (group_count * len(ksizes))
            group_lengths = [blocks_per_group] * group_count

        return InferenceComponents.densenet(
            **ic_kwargs,
            base_width=base_width,
            group_lengths=group_lengths,
            cifar_root_block=cifar_root_block,
            block_structure=BlockStructure.densenet(ksizes=ksizes),
            dropout_rate=0.2 if dropout else 0)

    @staticmethod
    def ladder_densenet(ic_kwargs,
                        depth,
                        base_width,
                        dropout,
                        cifar_root_block=False):
        print(f'Ladder-DenseNet-{depth}')
        for a in ['input_shape', 'class_count']:
            assert a in ic_kwargs
        group_lengths = {
            121: [6, 12, 24, 16],  # base_width = 32
            161: [6, 12, 36, 24],  # base_width = 48
            169: [6, 12, 32, 32],  # base_width = 32
        }[depth]
        return InferenceComponents.ladder_densenet(
            **ic_kwargs,
            base_width=32,
            cifar_root_block=cifar_root_block,
            group_lengths=group_lengths,
            dropout_rate=0.2 if dropout else 0)


def get_training_component(net_name, ds_train, epoch_count, pretrained=False):
    problem_id, ds_id = ds_train.info['problem_id'], ds_train.info['id']
    if problem_id == 'clf':
        base_learning_rate = 1e-1
        resnet_learning_rate_policy = {
            'boundaries': [
                int(i * epoch_count / 200 + 0.5) for i in [60, 120, 160]
            ],
            'values': [base_learning_rate * 0.2**i for i in range(4)]
        }
        densenet_learning_rate_policy = {
            'boundaries': [int(i * epoch_count / 100 + 0.5) for i in [50, 75]],
            'values': [base_learning_rate * 0.1**i for i in range(3)]
        }
        if ds_id in ['cifar', 'svhn']:
            batch_size = 64 if net_name == 'dn' else 128
        elif ds_id == 'mozgalo':
            batch_size = 32 if net_name == 'dn' else 64
        return TrainingComponent(
            batch_size=batch_size,
            loss=problem_id,
            weight_decay={'dn': 1e-4,
                          'rn': 1e-4,
                          'wrn': 5e-4}[net_name],
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
            learning_rate_policy=densenet_learning_rate_policy
            if net_name == 'dn' else resnet_learning_rate_policy,
            pretrained_lr_factor=1 / 5 if pretrained else 1)
    elif problem_id == 'semseg':
        batch_size = {
            'cityscapes': 4,
            'voc2012': 4,
            'camvid': 8,
            'iccv09': 16,
        }[ds_id]
        weight_decay = {
            'ldn': 1e-4,  # ladder-densenet/voc2012/densenet.py
            'dn': 1e-4,  # ladder-densenet/voc2012/densenet.py
            'rn': 1e-4,  # ladder-densenet/voc2012/resnet.py
            'wrn': 5e-4
        }[net_name]
        return TrainingComponents.ladder_densenet(
            epoch_count=epoch_count,
            base_learning_rate=5e-4,
            batch_size=batch_size,
            weight_decay=weight_decay,
            pretrained_lr_factor=1 / 5 if pretrained else 1)
    else:
        assert False


def get_inference_component(
        net_name,
        ds_train,
        depth: int,
        base_width: int = None,  # rn, dn, ldn
        width_factor: int = None,
        dropout=False):  # wrn
    assert bool(base_width) == (net_name in ['rn', 'dn', 'ldn'])
    assert bool(width_factor) == (net_name == 'wrn')
    sic_args = {
        'ic_kwargs': {
            'input_shape': ds_train[0][0].shape,
            'class_count': ds_train.info['class_count'],
            'problem_id': ds_train.info['problem_id'],
        },
        'depth': depth,
        'cifar_root_block': ds_train.info['id'] in ['cifar', 'svhn'],
        'dropout': dropout,
    }
    if net_name in ['rn', 'dn', 'ldn']:
        sic_args['base_width'] = base_width

    if net_name == 'wrn':
        return StandardInferenceComponents.wide_resnet(
            **sic_args, width_factor=width_factor)
    elif net_name == 'rn':
        return StandardInferenceComponents.resnet(**sic_args)
    elif net_name == 'dn':
        return StandardInferenceComponents.densenet(**sic_args)
    elif net_name == 'ldn':
        # pretrained: 30 epochs cityscapes, 100 epochs voc2012
        return StandardInferenceComponents.ladder_densenet(**sic_args)
    else:
        assert False, f"invalid model name: {net_name}"


def get_model(net_name,
              ds_train,
              depth,
              width,
              pretrained=False,
              epoch_count=None):
    # width: width factor for WRN, base_width for others

    tc = get_training_component(
        net_name=net_name,
        ds_train=ds_train,
        epoch_count=epoch_count,
        pretrained=pretrained)

    ic = get_inference_component(
        net_name=net_name,
        ds_train=ds_train,
        depth=depth,  # all
        base_width=None or net_name != 'wrn' and width,
        width_factor=None or net_name == 'wrn' and width)

    ae = ClassificationEvaluator(ds_train.info['class_count'])

    model = Model(
        modeldef=ModelDef(ic, tc),
        training_log_period=len(ds_train) // tc.batch_size // 5,
        accumulating_evaluator=ae)

    if pretrained:
        print("Loading pretrained parameters...")
        if net_name == 'rn' and depth == 50:
            names_to_params = parameter_loading.get_resnet_parameters_from_checkpoint_file(
                f'{dirs.PRETRAINED}/resnetv2_50/resnet_v2_50.ckpt')
        elif net_name == 'dn' and depth == 121 and width == 32:
            names_to_params = parameter_loading.get_densenet_parameters_from_checkpoint_file(
                f'{dirs.PRETRAINED}/densenet_121/tf-densenet121.ckpt')
        elif net_name == 'ldn' and depth == 121 and width == 32:
            names_to_params = parameter_loading.get_ladder_densenet_parameters_from_checkpoint_file(
                f'{dirs.PRETRAINED}/densenet_121/tf-densenet121.ckpt')
        else:
            assert False, "Pretrained parameters not available."
        model.load_parameters(names_to_params)

    return model


def save_trained_model(model,
                       ds_id,
                       net_name,
                       epoch_count,
                       dropout=None,
                       pretrained=None,
                       saved_nets_dir=dirs.SAVED_NETS):
    if dropout:
        net_name += '-do'
    if pretrained:
        net_name += '-pretrained'
    model.save_state(f'{saved_nets_dir}/{ds_id}/' +
                     f'{net_name}-e{epoch_count}/' +
                     f'{datetime.datetime.now():%Y-%m-%d-%H%M}')


def load_trained_model(model,
                       ds_id,
                       net_name,
                       epoch_count,
                       date_code,
                       dropout=None,
                       pretrained=None,
                       saved_nets_dir=dirs.SAVED_NETS):
    if dropout:
        net_name += '-do'
    if pretrained:
        net_name += '-pretrained'
    model.load_state(f'{saved_nets_dir}/{ds_id}/' +
                     f'{net_name}-e{epoch_count}/' + f'{date_code}/Model')
