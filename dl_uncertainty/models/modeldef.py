from collections import namedtuple

import tensorflow as tf

from ..tf_utils import layers, losses, regularization, evaluation


def global_pool_affine_logits_func(class_count):

    def f(h, **k):
        h = tf.reduce_mean(h, axis=[1, 2], keep_dims=True)
        h = layers.conv(h, 1, class_count, bias=True, name='conv_logits')
        return tf.reshape(h, [-1, class_count])

    return f


def semseg_affine_logits_func(class_count):

    def f(h, **k):
        return layers.conv(h, 1, class_count, bias=True, name='conv_logits')

    return f


def semseg_affine_resized_logits_func(class_count, spatial_shape):

    def f(h, **k):
        h = layers.conv(h, 1, class_count, bias=True, name='conv_logits')
        return tf.image.resize_bilinear(h, spatial_shape)

    return f


def semseg_affine_resized_probs_func(spatial_shape):

    def f(h):
        h = tf.nn.softmax(h)
        return tf.image.resize_bilinear(h, spatial_shape)

    return f


def clf_input_to_output_func(input_to_features,
                             features_to_logits=None,
                             logits_to_probs=None,
                             class_count=None):
    """
    Creates a function that maps the network input to a single output tensor.
    :param input_to_features: a function that returns 1 or more tensors.
    :param features_to_logits: a function that returns 1 or more tensors. argmax
        of the first of the returned values is taken as the classification 
        output. Default: global pooling followed by a 1x1 convolution.
    :param logits_to_probs: a function that maps 1 logits tensor to 1 probs 
        tensor. Default: tf.nn.softmax.
    :param class_count: needs to be given if features_to_logits is None.
    """
    assert features_to_logits or class_count
    if features_to_logits is None:
        features_to_logits = global_pool_affine_logits_func(class_count)
    logits_to_probs = logits_to_probs or tf.nn.softmax

    def input_to_output(input, pre_logits_learning_rate_factor=1, **kwargs):
        """
        Maps input to a single output tensor.
        :param pre_logits_learning_rate_factor: int or list[int]. A number
            learning rate is to be multiplied by for the pre-trained (pre-logit)
            part of the network. If input_to_features returns a list, it is
            applied only on the first tensor's subgraph.
        :param kwargs: additional arguments to be supplied to input_to_features
        and features_to_logits (bn_params, do_params, ...).
        """
        features = input_to_features(input, **kwargs)
        additional_outputs = {'features': features}

        if type(features) in [list, tuple]:
            features = list(features)
            features[0] = layers.amplify_gradient(
                features[0], pre_logits_learning_rate_factor)
        else:
            features = layers.amplify_gradient(features,
                                               pre_logits_learning_rate_factor)

        all_logits = features_to_logits(features, **kwargs)
        if type(all_logits) not in [list, tuple]:
            all_logits = [all_logits]
        all_probs = list(map(logits_to_probs, all_logits))

        output = tf.argmax(all_logits[0], -1, output_type=tf.int32)
        additional_outputs.update({
            'logits': all_logits[0],
            'probs': all_probs[0],
            'all_logits': all_logits,
            'all_probs': all_probs,
        })
        return output, additional_outputs

    return input_to_output


def semseg_input_to_output_func(input_shape,
                                input_to_features,
                                features_to_logits=None,
                                logits_to_probs=None,
                                class_count=None,
                                resize_probs=False):
    assert features_to_logits or class_count
    if features_to_logits is None:
        if resize_probs:
            features_to_logits = semseg_affine_logits_func(class_count)
            logits_to_probs = logits_to_probs or \
                              semseg_affine_resized_probs_func(input_shape[0:2])
        else:
            features_to_logits = semseg_affine_resized_logits_func(
                class_count, input_shape[0:2])
    return clf_input_to_output_func(input_to_features, features_to_logits,
                                    logits_to_probs)


InferenceComponent = namedtuple('InferenceComponent',
                                ['input_shape', 'input_to_output'])


class TrainingComponent:

    def __init__(self,
                 batch_size,
                 weight_decay,
                 loss='clf',
                 optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
                 learning_rate_policy=lambda epoch: 5e-2,
                 pre_logits_learning_rate_factor=1,
                 training_post_step=None):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.training_post_step = training_post_step
        self.pre_logits_learning_rate_factor = pre_logits_learning_rate_factor
        self.loss = loss
        if type(loss) is str:
            assert self.loss in ['clf', 'semseg', 'regr']
            if self.loss in ['clf', 'semseg']:
                self.loss = (['logits', 'label'], losses.cross_entropy_loss)
            elif self.loss == 'regr':
                self.loss = (['output', 'label'], losses.mean_squared_error)
        self.learning_rate_policy = learning_rate_policy
        if type(learning_rate_policy) is dict:
            self.learning_rate_policy = lambda ep: \
                tf.train.piecewise_constant(ep, **learning_rate_policy)


# name can be list if func returns multiple metrics
EvaluationMetric = namedtuple('EvaluationMetric', ['name', 'arg_names', 'func'])


class ModelNodes:

    def __init__(
            self,
            input,
            label,
            output,
            loss,
            training_step,
            learning_rate,
            evaluation=dict(),
            additional_outputs=dict(),  # probs, logits, ...
            training_post_step=None):
        self.input = input
        self.label = label
        self.outputs = {**additional_outputs, 'output': output}
        self.loss = loss
        self.training_step = training_step
        self.learning_rate = learning_rate
        self.evaluation = evaluation
        self.training_post_step = training_post_step


class ModelDef:

    def __init__(self,
                 inference_component: InferenceComponent,
                 training_component: TrainingComponent,
                 evaluation_metrics=[]):
        self.inference_component = inference_component
        self.training_component = training_component
        self.evaluation_metrics = evaluation_metrics

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
            pre_logits_learning_rate_factor=tc.pre_logits_learning_rate_factor)

        # Label
        label = tf.placeholder(output.dtype, output.shape, name='label')

        # Accessible nodes
        outputs = {**additional_outputs, 'output': output}
        evnodes = {**outputs, 'label': label}

        # Loss and regularization

        loss_args, loss_fn = tc.loss
        loss = loss_fn(* [evnodes[arg] for arg in loss_args])

        if tc.weight_decay > 0:
            w_vars = filter(lambda x: 'weights' in x.name,
                            tf.global_variables())      
            loss += tc.weight_decay * regularization.l2_regularization(w_vars)

        # Optimization
        learning_rate = tc.learning_rate_policy(epoch)
        optimizer = tc.optimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        # Evaluation
        evaluation = dict()
        for name, args, fn in self.evaluation_metrics:
            if type(name) is list:
                evs = fn(* [evnodes[arg] for arg in args])
                evaluation.update(dict(zip(name, evs)))
            else:
                evaluation[name] = fn(* [evnodes[arg] for arg in args])

        return ModelNodes(
            input=input,
            label=label,
            output=output,
            loss=loss,
            training_step=training_step,
            learning_rate=learning_rate,
            evaluation=evaluation,
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
            f_ito = clf_input_to_output_func
            kwargs['logits_to_probs'] = logits_to_probs
        if problem_id == 'semseg':
            f_ito = semseg_input_to_output_func
            kwargs['resize_probs'] = resize_probs
            kwargs['input_shape'] = input_shape
            kwargs['logits_to_probs'] = logits_to_probs
        input_to_output = f_ito(
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

        def input_to_features(x, is_training, **kwargs):
            x = layers.resnet_root_block(
                x, base_width=base_width, cifar_type=cifar_root_block)
            return layers.resnet_middle(x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    **layers.default_arg(layers.resnet_middle, 'dropout_params'),
                    'is_training': is_training
                })

        return InferenceComponents.generic(
            input_shape=input_shape,
            input_to_features=input_to_features,
            problem_id=problem_id,
            class_count=class_count,
            resize_probs=resize_probs)  # TODO: resize_probs=False

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

        def input_to_features(x, is_training, **kwargs):
            x = layers.densenet_root_block(
                x, base_width=base_width, cifar_type=cifar_root_block)
            return layers.densenet_middle(
                x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    'rate': dropout_rate,
                    'is_training': is_training
                })

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

        def input_to_features(x, is_training, **kwargs):
            return layers.ladder_densenet(
                x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    'rate': dropout_rate,
                    'is_training': is_training
                })

        def features_to_logits(all_pre_logits, is_training, **kwargs):
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
            pre_logits_learning_rate_factor=1):

        def learning_rate_policy(epoch):
            return tf.train.polynomial_decay(
                base_learning_rate,
                global_step=epoch - 1,  # epochs start at 1
                decay_steps=epoch_count,
                end_learning_rate=0,
                power=1.5)

        return TrainingComponent(
            batch_size=batch_size,
            weight_decay=weight_decay,
            loss=(['logits', 'label'], losses.cross_entropy_loss),
            optimizer=tf.train.AdamOptimizer,
            learning_rate_policy=learning_rate_policy,
            pre_logits_learning_rate_factor=pre_logits_learning_rate_factor)


class EvaluationMetrics:
    accuracy = EvaluationMetric('accuracy', ['label', 'output'],
                                evaluation.accuracy)

    @staticmethod
    def semantic_segementation(class_count, returns=['mP', 'mR', 'mF1',
                                                     'mIoU']):

        def func(labels, predictions):
            return evaluation.multiclass_scores(labels, predictions,
                                                class_count, returns)

        return EvaluationMetric(returns, ['label', 'output'], func)