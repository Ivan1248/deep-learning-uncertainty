from .models import InferenceComponents, BlockStructure


class StandardInferenceComponents:

    @staticmethod
    def resnet(depth,
               base_width,
               dropout_locations=[],
               dim_change='id',
               **ic_kwargs):
        print(f'ResNet-{depth}-{base_width}')
        group_lengths, ksizes, width_factors = {
            34: ([3, 4, 6, 3], [3, 3], [1, 1]),
            50: ([3, 4, 6, 3], [1, 3, 1], [1, 1, 4]),
        }[depth]
        return InferenceComponents.resnet(
            **ic_kwargs,
            base_width=base_width,
            group_lengths=group_lengths,
            block_structure=BlockStructure.resnet(
                ksizes=ksizes,
                dropout_locations=dropout_locations,
                width_factors=width_factors),
            dim_change=dim_change)

    @staticmethod
    def wide_resnet(depth,
                    width_factor,
                    dropout_locations=[],
                    dim_change='proj',
                    **ic_kwargs):
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
            base_width=width_factor * 16,
            group_lengths=[blocks_per_group] * group_count,
            block_structure=BlockStructure.resnet(
                ksizes=ksizes, dropout_locations=dropout_locations),
            dim_change=dim_change)

    @staticmethod
    def densenet(depth, base_width, dropout_locations=[], **ic_kwargs):
        print(f'DenseNet-{depth}-{base_width}')
        group_count, ksizes = 3, [1, 3]
        assert (depth - group_count - 1) % 3 == 0, \
            f"invalid depth: (depth-group_count-1) must be divisible by 3"
        blocks_per_group = (depth - 5) // (group_count * len(ksizes))
        return InferenceComponents.densenet(
            **ic_kwargs,
            base_width=base_width,
            group_lengths=[blocks_per_group] * group_count,
            block_structure=BlockStructure.densenet(
                ksizes=ksizes, dropout_locations=dropout_locations))

    @staticmethod
    def ladder_densenet(depth, base_width, dropout_locations=[], **ic_kwargs):
        print(f'DenseNet-{depth}-{base_width}')
        group_count, ksizes = 3, [1, 3]
        assert (depth - group_count - 1) % 3 == 0, \
            f"invalid depth: (depth-group_count-1) must be divisible by 3"
        blocks_per_group = (depth - 5) // (group_count * len(ksizes))
        ic = InferenceComponents.densenet(
            **ic_kwargs,
            base_width=base_width,
            group_lengths=[blocks_per_group] * group_count,
            block_structure=BlockStructure.densenet(
                ksizes=ksizes, dropout_locations=dropout_locations))