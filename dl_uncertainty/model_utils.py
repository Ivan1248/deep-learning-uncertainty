import tensorflow as tf

from .models import Model, ModelDef
from .models import BlockStructure, TrainingComponent, EvaluationMetrics
from .models import InferenceComponents, TrainingComponents
from .evaluation import ClassificationEvaluator


class StandardInferenceComponents:

    @staticmethod
    def resnet(
            ic_kwargs,
            depth,
            cifar_root_block,
            base_width=64,  # for all models?
            dropout_locations=[]):
        print(f'ResNet-{depth}-{base_width}')
        for a in ['input_shape', 'class_count', 'problem_id']:
            assert a in ic_kwargs
        normal = ([3, 3], [1, 1], 'id')
        bottleneck = ([1, 3, 1], [1, 1, 4], 'proj')  # last paragraph in [2]
        group_lengths, ksizes, width_factors, dim_change = {
            18: ([2] * 4, *normal),  # [1]
            34: ([3, 4, 6, 3], *normal),  # [1]
            110: ([3, 4, 6, 3], *normal),  # [2]
            50: ([3, 4, 6, 3], *bottleneck),  # [1]
            101: ([3, 4, 23, 3], *bottleneck),  # [1]
            152: ([3, 8, 36, 3], *bottleneck),  # [1]
            200: ([3, 24, 36, 3], *bottleneck),  # [2]
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
                    dropout_locations=[],
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
                ksizes=ksizes, dropout_locations=dropout_locations),
            dim_change=dim_change)

    @staticmethod
    def densenet(ic_kwargs, depth, base_width, cifar_root_block,
                 dropout_rate=0):  # 0.2 if data augmentation is not used
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
            dropout_rate=dropout_rate,
            block_structure=BlockStructure.densenet(ksizes=ksizes))

    @staticmethod
    def ladder_densenet(ic_kwargs,
                        depth,
                        base_width,
                        dropout_rate=0,
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
            dropout_rate=0)


def get_training_component(net_name,
                           problem_id,
                           epoch_count,
                           ds_id=None,
                           pretrained=False):
    base_learning_rate = {
        'clf': 1e-1,
        'semseg': 1e-4
    }[problem_id]  # semseg 5e-4
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

    if problem_id == 'clf':
        if ds_id in ['cifar', 'svhn']:
            batch_size = 64 if net_name == 'dn' else 128
        elif ds_id == 'mozgalo':
            batch_size = 32 if net_name == 'dn' else 64
        return TrainingComponent(
            batch_size=batch_size,
            weight_decay={'dn': 1e-4,
                          'rn': 1e-4,
                          'wrn': 5e-4}[net_name],
            loss=problem_id,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
            learning_rate_policy=densenet_learning_rate_policy
            if net_name == 'dn' else resnet_learning_rate_policy,
            pre_logits_learning_rate_factor=5e-3 if pretrained else 1)
    elif problem_id == 'semseg':
        batch_size = {
            'cityscapes': 4,
            'voc2012': 4,
            'iccv09': 16,
            'camvid': 16
        }[ds_id]
        weight_decay = {
            'ldn': 1e-4,  # ladder-densenet/voc2012/densenet.py
            'dn': 1e-4,  # ladder-densenet/voc2012/densenet.py
            'rn': 1e-4,  # ladder-densenet/voc2012/resnet.py
            'wrn': 5e-4
        }[net_name]
        if net_name in []:
            return TrainingComponent(
                batch_size=batch_size,
                weight_decay=weight_decay,
                loss=problem_id,
                optimizer=lambda lr: tf.train.AdamOptimizer(lr),
                learning_rate_policy={
                    'dn': densenet_learning_rate_policy,
                    'rn': resnet_learning_rate_policy,
                    'wrn': resnet_learning_rate_policy,
                }[net_name],
                pre_logits_learning_rate_factor=1 / 5 if pretrained else 1)
        elif net_name in ['ldn', 'dn', 'rn', 'wrn']:
            # NOTE/TODO?: 'dn' is here now, not up there
            return TrainingComponents.ladder_densenet(
                epoch_count=epoch_count,
                base_learning_rate=
                5e-4,  # /10 77%, /50 resnet0.840 densenet0.841 /100 79.68
                batch_size=batch_size,
                weight_decay=weight_decay,
                pre_logits_learning_rate_factor=1 / 5 if pretrained else 1)
    else:
        assert False


def get_inference_component(
        net_name,
        problem_id,
        ds_id,
        input_shape,
        class_count=None,
        depth=None,  # all
        base_width=None,  # rn, dn, ldn
        width_factor=None):  # wrn
    ic_args = {
        'input_shape': input_shape,
        'class_count': class_count,
        'problem_id': problem_id,
    }
    sic_args = {
        'depth': depth,
        'cifar_root_block': ds_id in ['cifar', 'svhn'],
    }
    if net_name in ['rn', 'dn', 'wrn']:
        sic_args['base_width'] = base_width

    if net_name == 'wrn':
        return StandardInferenceComponents.wide_resnet(
            ic_args,
            **sic_args,
            width_factor=width_factor,
            dropout_locations=[0])
    elif net_name == 'rn':
        return StandardInferenceComponents.resnet(
            ic_args, **sic_args, dropout_locations=[])
    elif net_name == 'dn':
        return StandardInferenceComponents.densenet(ic_args, **sic_args)
    elif net_name == 'ldn':
        # pretrained: 30 epochs cityscapes, 100 epochs voc2012
        return StandardInferenceComponents.ladder_densenet(ic_args, **sic_args)
    else:
        assert False, f"invalid model name: {net_name}"


def get_model(
        net_name,
        problem_id,
        epoch_count,
        ds_id,
        ds_train,
        depth=None,
        width=None,  # width factor for WRN, base_width for others
        pretrained=False):

    class_count = ds_train.info['class_count']

    tc = get_training_component(
        net_name=net_name,
        problem_id=problem_id,
        epoch_count=epoch_count,
        ds_id=ds_id,
        pretrained=pretrained)

    ic = get_inference_component(
        net_name=net_name,
        problem_id=problem_id,
        ds_id=ds_id,
        input_shape=ds_train[0][0].shape,
        class_count=class_count,
        depth=depth,  # all
        base_width=None or net_name != 'wrn' and width,
        width_factor=None or net_name == 'wrn' and width)

    return Model(
        modeldef=ModelDef(ic, tc),
        training_log_period=len(ds_train) // tc.batch_size // 5,
        accumulating_evaluator=ClassificationEvaluator(class_count))
