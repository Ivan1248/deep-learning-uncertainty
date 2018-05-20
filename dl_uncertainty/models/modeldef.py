from collections import namedtuple

import tensorflow as tf

from .tf_utils import layers, losses, regularization, evaluation
from .elements import clf_input_to_output_func, semseg_input_to_output_func
from .elements import standard_loss

# InferenceComponent and TrainingComponent

InferenceComponent = namedtuple('InferenceComponent',
                                ['input_shape', 'input_to_output'])


class TrainingComponent:

    def __init__(
            self,
            batch_size,
            loss,
            weight_decay,
            optimizer,  # function lr -> optimizer(lr)
            learning_rate_policy,  # (epoch -> lr(epoch)) or float or dict
            pretrained_lr_factor=1,
            training_post_step=None):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.training_post_step = training_post_step
        self.pretrained_lr_factor = pretrained_lr_factor

        self.loss = loss

        if type(loss) is str:
            self.loss = standard_loss(problem_id=self.loss)

        self.learning_rate_policy = learning_rate_policy
        if type(learning_rate_policy) is dict:
            self.learning_rate_policy = lambda ep: \
                tf.train.piecewise_constant(ep, **learning_rate_policy)
        elif type(learning_rate_policy) in [int, float]:
            self.learning_rate_policy = lambda ep: \
                tf.constant(learning_rate_policy)


class ModelDef:

    class ModelNodes:

        def __init__(self,
                     input,
                     label,
                     output,
                     loss,
                     training_step,
                     learning_rate,
                     additional_outputs=dict(),
                     training_post_step=None):
            self.input = input
            self.label = label
            self.outputs = {**additional_outputs, 'output': output}
            self.loss = loss
            self.training_step = training_step
            self.learning_rate = learning_rate
            self.training_post_step = training_post_step

    def __init__(self, inference_component: InferenceComponent,
                 training_component: TrainingComponent):
        self.inference_component = inference_component
        self.training_component = training_component

    def build_graph(self, epoch, is_training):
        ic = self.inference_component
        tc = self.training_component

        # Input
        input_shape = [None] + list(ic.input_shape)
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')

        # Inference
        output, additional_outputs = ic.input_to_output(
            input,
            is_training=is_training,
            pretrained_lr_factor=tc.pretrained_lr_factor)

        # Label
        label = tf.placeholder(output.dtype, output.shape, name='label')

        # Accessible nodes
        outputs = {**additional_outputs, 'output': output}
        evnodes = {**outputs, 'label': label}

        # Loss and regularization

        loss_fn, loss_args = tc.loss
        loss = loss_fn(* [evnodes[arg] for arg in loss_args])

        if tc.weight_decay > 0:
            w_vars = filter(lambda x: 'weights' in x.name,
                            tf.global_variables())
            loss += tc.weight_decay * regularization.l2_regularization(w_vars)

        # Optimization
        learning_rate = tc.learning_rate_policy(epoch)
        optimizer = tc.optimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        return ModelDef.ModelNodes(
            input=input,
            label=label,
            output=output,
            loss=loss,
            training_step=training_step,
            learning_rate=learning_rate,
            additional_outputs=additional_outputs)


# Special cases


class InferenceComponents:
    # http://ethereon.github.io/netscope/#/editor
    # https://github.com/shicai/DenseNet-Caffe
    # https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt
    # https://github.com/binLearning/caffe_toolkit
    # https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

    @staticmethod
    def generic(
            problem_id,  # clf, regr, semseg, other
            input_shape,
            input_to_features,
            features_to_logits=None,  # clf, semseg
            features_to_output=None,  # regr
            logits_to_probs=None,
            resize_probs=False,  # semseg
            class_count=None):  # clf, semseg
        kwargs = dict()
        if problem_id == 'clf':
            input_to_output_func = clf_input_to_output_func
            kwargs['logits_to_probs'] = logits_to_probs
        if problem_id == 'semseg':
            input_to_output_func = semseg_input_to_output_func
            kwargs['resize_probs'] = resize_probs
            kwargs['input_shape'] = input_shape
            kwargs['logits_to_probs'] = logits_to_probs
        input_to_output = input_to_output_func(
            input_to_features=input_to_features,
            features_to_logits=features_to_logits,
            class_count=class_count,
            **kwargs)
        return InferenceComponent(input_shape, input_to_output)

    @staticmethod
    def resnet(input_shape,
               base_width,
               group_lengths,
               block_structure,
               dim_change,
               cifar_root_block,
               width_factor=1,
               problem_id='clf',
               class_count=None,
               resize_probs=False):
        if problem_id in ['semseg', 'clf']:
            assert class_count is not None
        p = [
            'base_width', 'width_factor', 'group_lengths', 'block_structure',
            'dim_change'
        ]
        params = {k: v for k, v in locals().items() if k in p}

        def input_to_features(x, is_training, pretrained_lr_factor):
            x = layers.resnet_root_block(
                x, base_width=base_width, cifar_type=cifar_root_block)
            return layers.resnet_middle(x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    **layers.default_arg(layers.resnet_middle, 'dropout_params'),
                    'is_training': is_training
                },
                pretrained_lr_factor=pretrained_lr_factor)

        return InferenceComponents.generic(
            input_shape=input_shape,
            input_to_features=input_to_features,
            problem_id=problem_id,
            class_count=class_count,
            resize_probs=resize_probs)

    @staticmethod
    def densenet(input_shape,
                 base_width,
                 group_lengths,
                 block_structure,
                 cifar_root_block,
                 dropout_rate,
                 problem_id='clf',
                 class_count=None):
        if problem_id in ['semseg', 'clf']:
            assert class_count
        p = ['base_width', 'group_lengths', 'block_structure']
        params = {k: v for k, v in locals().items() if k in p}

        def input_to_features(x, is_training, pretrained_lr_factor):
            x = layers.densenet_root_block(
                x, base_width=base_width, cifar_type=cifar_root_block)
            return layers.densenet_middle(
                x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    'rate': dropout_rate,
                    'is_training': is_training
                },
                pretrained_lr_factor=pretrained_lr_factor)

        return InferenceComponents.generic(
            input_shape=input_shape,
            input_to_features=input_to_features,
            problem_id=problem_id,
            class_count=class_count)

    @staticmethod
    def ladder_densenet(input_shape,
                        class_count,
                        base_width=32,
                        group_lengths=[6, 12, 24, 16],
                        block_structure=layers.BlockStructure.densenet(),
                        dropout_rate=0,
                        problem_id='semseg',
                        cifar_root_block=False):
        p = [
            'base_width', 'group_lengths', 'block_structure', 'cifar_root_block'
        ]
        params = {k: v for k, v in locals().items() if k in p}

        def input_to_features(x, is_training, pretrained_lr_factor=1):
            return layers.ladder_densenet(
                x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    'rate': dropout_rate,
                    'is_training': is_training
                },
                pretrained_lr_factor=pretrained_lr_factor)

        def features_to_logits(all_pre_logits, is_training):
            pre_logits, pre_logits_aux = all_pre_logits
            return layers.ladder_densenet_logits(
                pre_logits,
                pre_logits_aux,
                image_shape=input_shape[0:2],
                class_count=class_count,
                bn_params={'is_training': is_training})

        return InferenceComponents.generic(
            input_shape=input_shape,
            input_to_features=input_to_features,
            features_to_logits=features_to_logits,
            problem_id='semseg',
            class_count=class_count)


class TrainingComponents:

    @staticmethod
    def ladder_densenet(
            epoch_count,
            base_learning_rate=5e-4,  # 1e-4 pretrained part
            batch_size=4,
            weight_decay=1e-4,
            pretrained_lr_factor=1):

        def learning_rate_policy(epoch):
            return tf.train.polynomial_decay(
                base_learning_rate,
                global_step=epoch - 1,  # epochs start at 1
                decay_steps=epoch_count,
                end_learning_rate=0,
                power=1.5)

        return TrainingComponent(
            batch_size=batch_size,
            loss='semseg',
            weight_decay=weight_decay,
            optimizer=tf.train.AdamOptimizer,
            learning_rate_policy=learning_rate_policy,
            pretrained_lr_factor=pretrained_lr_factor)