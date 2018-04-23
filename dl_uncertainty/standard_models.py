from .models import BlockStructure, ResNet, DenseNet


def resnet(input_shape,
           class_count,
           group_lengths=None,
           base_width=16,
           block_structure=BlockStructure.resnet(
               ksizes=[3, 3], dropout_locations=[0]),
           initial_learning_rate=1e-1,
           epoch_count=200):
    model = ResNet(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=128,
        learning_rate_policy={
            'boundaries': [
                int(i * epoch_count / 200 + 0.5) for i in [60, 120, 160]
            ], 'values': [initial_learning_rate * 0.2**i for i in range(4)]
        },
        block_structure=block_structure,
        group_lengths=group_lengths,
        base_width=base_width,
        weight_decay=5e-4,
        training_log_period=39)
    return model

def densenet(depth,
             input_shape,
             class_count,
             base_width=12,
             dropout_locations=[0],
             initial_learning_rate=1e-1,
             epoch_count=200):
    group_count = 3
    ksizes = [1, 3]
    blocks_per_group = (depth - 5) // (group_count * len(ksizes))
    dm = (depth - group_count - 1) % 3
    assert dm == 0, f"invalid depth ((depth-group_count-1) mod 3 = {dm} must be divisible by 3)"
    print(f"group count: {group_count}, blocks per group: {blocks_per_group}")
    model = DenseNet(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=64,
        learning_rate_policy={
            'boundaries': [int(i * epoch_count / 100 + 0.5) for i in [50, 75]],
            'values': [initial_learning_rate * 0.1**i for i in range(3)]
        },
        block_structure=BlockStructure.densenet(
            ksizes=ksizes, dropout_locations=dropout_locations),
        base_width=base_width,
        group_lengths=[blocks_per_group] * group_count,
        weight_decay=1e-4,
        training_log_period=39 * 2)
    return model