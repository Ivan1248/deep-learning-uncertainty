import tensorflow as tf

from ..tf_utils import layers, regularization, losses
from ..tf_utils import layers

from .abstract_model import AbstractModel


def global_pool_logits_func(class_count):

    def f(h):
        h = tf.reduce_mean(h, axis=[1, 2], keep_dims=True)
        h = layers.conv(h, 1, class_count, bias=True)
        return tf.reshape(h, [-1, class_count])

    return f


def semseg_conv_resize_logits_func(class_count, spatial_shape):

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
        self.input_to_features = input_to_features
        self.features_to_logits = features_to_logits
        self.logits_to_probs = logits_to_probs
        self.features_to_output = features_to_output
        self.class_count = class_count

    def initialize(self, problem, input_shape):
        if problem in ['clf', 'semseg']:
            if self.features_to_logits == 'auto':
                if problem == 'semseg':
                    self.features_to_logits = semseg_conv_resize_logits_func(
                        self.class_count, input_shape[1:3])
                else:
                    self.features_to_logits = global_pool_logits_func(
                        self.class_count)
        elif problem == 'regr' and self.features_to_output is None:

            def identity(x, **kwargs):
                return x

            self.features_to_output = identity


class TrainingComponent:

    def __init__(self,
                 weight_decay,
                 loss='auto',
                 optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
                 learning_rate_policy=lambda epoch: 5e-2):
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate_policy = learning_rate_policy
        self.weight_decay = weight_decay

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


class StandardModel(AbstractModel):

    def __init__(
            self,
            problem: str,  # clf, regr, semseg, other
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            batch_size,
            inference_component: InferenceComponent,
            training_component: TrainingComponent,
            evaluation_metrics=[],
            training_log_period=40,
            name='Model'):
        assert problem in ['clf', 'regr', 'semseg', 'other']
        self.input_shape = input_shape
        self.problem = problem

        inference_component.initialize(problem, input_shape)
        self.inference_component = inference_component
        training_component.initialize(problem)
        self.training_component = training_component.initialize(problem)

        self.evaluation_metrics = evaluation_metrics
        super().__init__(
            batch_size=batch_size,
            learning_rate_policy=1,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        clf = self.problem == 'clf'
        ic = self.inference_component
        tc = self.training_component

        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')

        # Hidden layers
        features = ic.input_to_features(input, is_training=is_training)

        # Features to output
        if clf:
            logits = ic.features_to_logits(features)
            probs = ic.logits_to_probs(logits)
            output = tf.argmax(logits, 1, output_type=tf.int32)
        else:
            output = ic.features_to_output(features)

        # Loss and regularization
        target = tf.placeholder(output.dtype, output.shape, name='target')
        loss = tc.loss(logits if clf else output, target)

        if tc.weight_decay > 0:
            w_vars = filter(lambda x: 'weights' in x.name,
                            tf.global_variables())
            loss += tc.weight_decay * regularization.l2_regularization(w_vars)

        # Optimization
        learning_rate = tc.learning_rate_policy(epoch)
        optimizer = tc.optimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        # Additional outputs
        additional_outputs = {'features': features}
        if clf:
            additional_outputs.update({'probs': probs, 'logits': logits})
        outputs = {**additional_outputs, 'output': output}

        # Evaluation
        evaluation = {
            name: fn(outputs[arg] for arg in args)
            for name, args, fn in self.evaluation_metrics
        }

        return AbstractModel.EssentialNodes(
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
           batch_size=128,
           base_width=64,
           group_lengths=[3, 3, 3],
           block_structure=layers.BlockStructure.resnet(),
           dim_change='id',
           large_input=None,
           problem='clf',
           class_count=None,
           weight_decay=5e-4,
           optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
           learning_rate_policy={
               'boundaries': [int(i + 0.5) for i in [60, 120, 160]],
               'values': [1e-1 * 0.2**i for i in range(4)]
           },
           training_log_period=40):
    if problem in ['semseg', 'regr'] and large_input is None:
        large_input = False
    ic = InferenceComponents.resnet(base_width, group_lengths, block_structure,
                                    dim_change, large_input, problem,
                                    class_count)
    tc = TrainingComponent(weight_decay, optimizer, learning_rate_policy)
    return StandardModel(
        problem=problem,  # clf, regr, semseg, other
        input_shape=input_shape,
        batch_size=batch_size,
        inference_component=ic,
        training_component=tc,
        evaluation_metrics=[],
        training_log_period=training_log_period,
        name='ResNet')
