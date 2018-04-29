import tensorflow as tf

from ..tf_utils import layers, regularization, losses
from ..tf_utils import layers

from .model import ModelNodes


def global_pool_affine_logits_func(class_count):

    def f(h):
        h = tf.reduce_mean(h, axis=[1, 2], keep_dims=True)
        h = layers.conv(h, 1, class_count, bias=True)
        return tf.reshape(h, [-1, class_count])

    return f


def semseg_affine_resize_logits_func(class_count, spatial_shape):

    def f(h):
        h = layers.conv(h, 1, class_count, bias=True)
        return tf.image.resize_bilinear(h, spatial_shape)

    return f


class InferenceComponent:

    def __init__(
            self,
            input_to_features,
            class_count=None,  # clf
            features_to_logits='auto',  # clf
            features_to_output=None,  # regr
            logits_to_probs=tf.nn.softmax):
        self._input_to_features = input_to_features
        self._features_to_logits = features_to_logits
        self._logits_to_probs = logits_to_probs
        self._features_to_output = features_to_output
        self._class_count = class_count
        self.input_to_output = None

    def initialize(self, problem, input_shape):
        if problem in ['clf', 'semseg']:
            if self._features_to_logits == 'auto':
                if problem == 'semseg':
                    self._features_to_logits = semseg_affine_resize_logits_func(
                        self._class_count, input_shape[1:3])
                else:
                    self._features_to_logits = global_pool_affine_logits_func(
                        self._class_count)
        elif problem == 'regr' and self._features_to_output is None:

            def identity(x, **kwargs):
                return x

            self._features_to_output = identity

        def input_to_output(input, **kwargs):
            features = self._input_to_features(input, **kwargs)
            additional_outputs = {'features': features}
            if problem == 'clf':
                logits = self._features_to_logits(features)
                probs = self._logits_to_probs(logits)
                output = tf.argmax(logits, 1, output_type=tf.int32)
                additional_outputs.update({'logits': logits, 'probs': probs})
            else:
                output = self._features_to_output(features)
            return output, additional_outputs

        self.input_to_output = input_to_output


class TrainingComponent:

    def __init__(self,
                 batch_size,
                 weight_decay,
                 loss='auto',
                 optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
                 learning_rate_policy=lambda epoch: 5e-2,
                 training_post_step=None):
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate_policy = learning_rate_policy
        self.weight_decay = weight_decay
        self.training_post_step = training_post_step

    def initialize(self, problem):
        if self.loss == 'auto':
            self.loss = {
                'clf': losses.cross_entropy_loss,
                'semseg': losses.cross_entropy_loss,
                'regr': losses.mean_squared_error
            }[problem]
        if type(self.learning_rate_policy) is dict:

            def lrp(epoch):
                return tf.train.piecewise_constant(
                    epoch,
                    boundaries=self.learning_rate_policy['boundaries'],
                    values=self.learning_rate_policy['values'])

            self.learning_rate_policy = lrp


class ModelDef():

    def __init__(
            self,
            problem: str,  # clf, regr, semseg, other
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            inference_component: InferenceComponent,
            training_component: TrainingComponent,
            evaluation_metrics=[]):
        assert problem in ['clf', 'regr', 'semseg', 'other']
        self.input_shape = input_shape
        self.problem = problem

        inference_component.initialize(problem, input_shape)
        self.inference_component = inference_component
        training_component.initialize(problem)
        self.training_component = training_component.initialize(problem)
       
        self.evaluation_metrics = evaluation_metrics

    def build_graph(self, epoch, is_training):
        clf = self.problem == 'clf'
        ic = self.inference_component
        tc = self.training_component

        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')

        # Inference
        output, additional_outputs = ic.input_to_output(
            input, is_training=is_training)

        # Loss and regularization
        target = tf.placeholder(output.dtype, output.shape, name='target')
        loss = tc.loss(additional_outputs['logits'] if clf else output, target)

        if tc.weight_decay > 0:
            w_vars = filter(lambda x: 'weights' in x.name,
                            tf.global_variables())
            loss += tc.weight_decay * regularization.l2_regularization(w_vars)

        # Optimization
        learning_rate = tc.learning_rate_policy(epoch)
        optimizer = tc.optimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        # Evaluation
        outputs = {**additional_outputs, 'output': output}
        evaluation = {
            name: fn(outputs[arg] for arg in args)
            for name, args, fn in self.evaluation_metrics
        }

        return ModelNodes(
            input=input,
            target=target,
            output=output,
            loss=loss,
            training_step=training_step,
            evaluation=evaluation,
            additional_outputs=additional_outputs)


class InferenceComponents:

    @staticmethod
    def resnet(base_width,
               group_lengths,
               block_structure,
               dim_change,
               large_input=None,
               problem='clf',
               class_count=None):
        if problem in ['semseg', 'clf']:
            assert class_count is not None
        if problem in ['semseg', 'regr'] and large_input is None:
            large_input = False
        p = [
            'base_width', 'group_lengths', 'block_structure', 'dim_change',
            'large_input'
        ]
        params = {k: v for k, v in locals().items() if k in p}

        def input_to_features(x, is_training, **kwargs):
            return layers.resnet(x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    **layers.default_arg(layers.resnet, 'dropout_params'),
                    'is_training': is_training
                })

        return InferenceComponent(
            input_to_features=input_to_features, class_count=class_count)

    @staticmethod
    def densenet(base_width,
                 group_lengths,
                 block_structure,
                 large_input=None,
                 problem='clf',
                 class_count=None):
        if problem in ['semseg', 'clf']:
            assert class_count is not None
        if problem in ['semseg', 'regr'] and large_input is None:
            large_input = False
        p = ['base_width', 'group_lengths', 'block_structure', 'large_input']
        params = {k: v for k, v in locals().items() if k in p}

        def input_to_features(x, is_training, **kwargs):
            return layers.densenet(x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    **layers.default_arg(layers.resnet, 'dropout_params'),
                    'is_training': is_training
                })

        return InferenceComponent(
            input_to_features=input_to_features, class_count=class_count)


def resnet(input_shape,
           base_width=64,
           group_lengths=[3, 3, 3],
           block_structure=layers.BlockStructure.resnet(),
           dim_change='id',
           large_input=None,
           problem='clf',
           class_count=None,
           batch_size=128,
           weight_decay=5e-4,
           optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
           learning_rate_policy={
               'boundaries': [int(i + 0.5) for i in [60, 120, 160]],
               'values': [1e-1 * 0.2**i for i in range(4)]
           }):
    if problem in ['semseg', 'regr'] and large_input is None:
        large_input = False
    ic = InferenceComponents.resnet(base_width, group_lengths, block_structure,
                                    dim_change, large_input, problem,
                                    class_count)
    tc = TrainingComponent(batch_size, weight_decay, 'auto', optimizer,
                           learning_rate_policy)
    return ModelDef(
        problem=problem,  # clf, regr, semseg, other
        input_shape=input_shape,
        inference_component=ic,
        training_component=tc,
        evaluation_metrics=[])
