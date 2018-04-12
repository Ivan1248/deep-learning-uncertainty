from models import ResidualBlockProperties, ResNet


def get_wrn(zagoruyko_depth,
            input_shape,
            class_count,
            widening_factor=1,
            aggregation='sum',
            resnet_ctor=ResNet):
    group_count = 3
    ksizes = [3, 3]
    blocks_per_group = (zagoruyko_depth - 4) // (group_count * len(ksizes))
    print("group count: {}, blocks per group: {}".format(
        group_count, blocks_per_group))
    model = resnet_ctor(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=128,
        learning_rate_policy={
            'boundaries': [60, 120, 160],
            'values': [1e-1 * 0.2**i for i in range(4)]
        },
        block_properties=ResidualBlockProperties(
            ksizes=ksizes,
            dropout_locations=[0],
            dropout_rate=0.3,
            dim_increase='id', 
            aggregation=aggregation),
        group_lengths=[blocks_per_group] * group_count,
        widening_factor=widening_factor,
        weight_decay=5e-4,
        training_log_period=39)
    assert zagoruyko_depth == model.zagoruyko_depth, "invalid depth (zagoruyko_depth={}!={}=model.zagoruyko_depth)".format(
        zagoruyko_depth, model.zagoruyko_depth)
    return model