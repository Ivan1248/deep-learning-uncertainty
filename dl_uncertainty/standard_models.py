from .models import BlockStructure, ResNet, DenseNet


def resnet(zagoruyko_depth,
           input_shape,
           class_count,
           widening_factor=1,
           dropout_locations=[0],
           initial_learning_rate=1e-1,
           epoch_count=200):
    group_count = 3
    ksizes = [3, 3]
    blocks_per_group = (zagoruyko_depth - 4) // (group_count * len(ksizes))
    print(f"group count: {group_count}, blocks per group: {blocks_per_group}")
    model = ResNet(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=128,
        learning_rate_policy={
            'boundaries': [
                int(i * epoch_count / 200 + 0.5) for i in [60, 120, 160]
            ], 'values': [initial_learning_rate * 0.2**i for i in range(4)]
        },
        block_structure=BlockStructure.resnet(
            ksizes=ksizes, dropout_locations=dropout_locations),
        group_lengths=[blocks_per_group] * group_count,
        widening_factor=widening_factor,
        weight_decay=5e-4,
        training_log_period=39)
    assert zagoruyko_depth == model.zagoruyko_depth, "invalid depth (zagoruyko_depth={}!={}=model.zagoruyko_depth)".format(
        zagoruyko_depth, model.zagoruyko_depth)
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
    dm = (depth - group_count-1) % 3
    assert dm == 0, f"invalid depth ((depth-group_count-1) mod 3 = {dm} must be divisible by 3)"
    print(f"group count: {group_count}, blocks per group: {blocks_per_group}")
    model = DenseNet(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=128,
        learning_rate_policy={
            'boundaries': [
                int(i * epoch_count / 200 + 0.5) for i in [60, 120, 160]
            ], 'values': [initial_learning_rate * 0.2**i for i in range(4)]
        },
        block_structure=BlockStructure.densenet(
            ksizes=ksizes, dropout_locations=dropout_locations),
        base_width=base_width,
        group_lengths=[blocks_per_group] * group_count,
        weight_decay=1e-4,
        training_log_period=39)
    return model