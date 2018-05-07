from .models import InferenceComponents, BlockStructure


class StandardInferenceComponents:

    @staticmethod
    def resnet(depth,
               base_width=64,
               dropout_locations=[],
               dim_change='id',
               **ic_kwargs):
        print(f'ResNet-{depth}-{base_width}')
        normal, bottleneck = ([3, 3], [1, 1]), ([1, 3, 1], [1, 1, 4])
        group_lengths, ksizes, width_factors = {
            18: ([2] * 4, *normal),  # [1]
            34: ([3, 4, 6, 3], *normal),  # [1]
            50: ([3, 4, 6, 3], *bottleneck),  # [1]
            101: ([3, 4, 23, 3], *bottleneck),  # [1]
            110: ([3, 4, 6, 3], *normal),  # [2]
            152: ([3, 8, 36, 3], *bottleneck),  # [1]
            200: ([3, 24, 36, 3], *bottleneck),  # [2]
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
                    cifar_root_block,
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
            cifar_root_block=cifar_root_block,
            base_width=16,
            width_factor=width_factor,
            group_lengths=[blocks_per_group] * group_count,
            block_structure=BlockStructure.resnet(
                ksizes=ksizes, dropout_locations=dropout_locations),
            dim_change=dim_change)

    @staticmethod
    def densenet(depth,
                 base_width,
                 cifar_root_block,
                 dropout_locations=[],
                 **ic_kwargs):
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
            blocks_per_group = (depth - 5) // (group_count * len(ksizes))
            group_lengths = [blocks_per_group] * group_count

        return InferenceComponents.densenet(
            **ic_kwargs,
            cifar_root_block=cifar_root_block,
            base_width=base_width,
            group_lengths=group_lengths,
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