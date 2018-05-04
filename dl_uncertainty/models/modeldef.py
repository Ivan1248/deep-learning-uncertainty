import tensorflow as tf

from ..tf_utils import layers, losses, regularization, evaluation


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
            input_shape,
            input_to_features,
            features_to_logits='auto',  # clf
            features_to_output=None,  # regr
            logits_to_probs=tf.nn.softmax,
            problem='clf',  # clf, regr, semseg, other
            class_count=None):  # clf, semseg
        assert problem in ['clf', 'semseg', 'regr']
        if problem in ['clf', 'semseg']:
            assert class_count is not None

        if problem in ['clf', 'semseg']:
            if features_to_logits == 'auto':
                if problem == 'semseg':
                    features_to_logits = semseg_affine_resize_logits_func(
                        class_count, input_shape[0:2])
                else:
                    features_to_logits = global_pool_affine_logits_func(
                        class_count)
        elif problem == 'regr' and features_to_output is None:

            def identity(x, **kwargs):
                return x

            features_to_output = identity

        def input_to_output(input, **kwargs):
            features = input_to_features(input, **kwargs)
            additional_outputs = {'features': features}
            if problem in ['clf', 'semseg']:
                logits = features_to_logits(features)
                probs = logits_to_probs(logits)
                output = tf.argmax(logits, -1, output_type=tf.int32)
                additional_outputs.update({'logits': logits, 'probs': probs})
            else:
                output = features_to_output(features)
            return output, additional_outputs

        self.problem = problem
        self.input_shape = input_shape
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
            if problem in ['clf', 'semseg']:
                self.loss = (['logits', 'label'], losses.cross_entropy_loss)
            elif problem == 'regr':
                self.loss = (['output', 'label'], losses.mean_squared_error)
            else:
                assert False
        if type(self.learning_rate_policy) is dict:
            lrp = self.learning_rate_policy

            def learning_rate_policy(epoch):
                return tf.train.piecewise_constant(
                    epoch, boundaries=lrp['boundaries'], values=lrp['values'])

            self.learning_rate_policy = learning_rate_policy


class ModelNodes:

    def __init__(
            self,
            input,
            label,
            output,
            loss,
            training_step,
            evaluation=dict(),
            additional_outputs=dict(),  # probs, logits, ...
            training_post_step=None):
        self.input = input
        self.label = label
        self.outputs = {**additional_outputs, 'output': output}
        self.loss = loss
        self.training_step = training_step
        self.evaluation = evaluation
        self.training_post_step = training_post_step


class ModelDef():

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
        tc.initialize(ic.problem)

        # Input
        input_shape = [None] + list(ic.input_shape)
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')

        # Inference
        output, additional_outputs = ic.input_to_output(
            input, is_training=is_training)

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
            evaluation=evaluation,
            additional_outputs=additional_outputs)


# Special cases
class TrainingComponents:

    @staticmethod
    def ladder_densenet(epoch_count,
                        base_learning_rate=5e-4,
                        batch_size=4,
                        weight_decay=1e-4):
        p = ['base_width', 'group_lengths', 'block_structure', 'large_input']
        params = {k: v for k, v in locals().items() if k in p}

        def learning_rate_policy(epoch):
            return tf.train.polynomial_decay(
                base_learning_rate,
                global_step=epoch,
                decay_steps=epoch_count,
                end_learning_rate=0,
                power=1.5)

        return TrainingComponent(
            batch_size=batch_size,
            weight_decay=weight_decay,
            loss='auto',
            optimizer=lambda lr: tf.train.AdamOptimizer(lr),
            learning_rate_policy=learning_rate_policy)


class InferenceComponents:

    @staticmethod
    def resnet(input_shape,
               base_width,
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
            input_shape=input_shape,
            input_to_features=input_to_features,
            problem=problem,
            class_count=class_count)

    @staticmethod
    def densenet(input_shape,
                 base_width,
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
                    **layers.default_arg(layers.densenet, 'dropout_params'),
                    'is_training': is_training
                })

        return InferenceComponent(
            input_shape=input_shape,
            input_to_features=input_to_features,
            problem=problem,
            class_count=class_count)

    @staticmethod
    def ladder_densenet(input_shape,
                        class_count,
                        base_width=32,
                        group_lengths=[6, 12, 24, 16],
                        block_structure=layers.BlockStructure.densenet(
                            dropout_locations=[])):
        p = ['base_width', 'group_lengths', 'block_structure', 'large_input']
        params = {k: v for k, v in locals().items() if k in p}

        def input_to_features(x, is_training, **kwargs):
            return layers.ladder_densenet(x,
                **params,
                bn_params={'is_training': is_training},
                dropout_params={
                    **layers.default_arg(layers.ladder_densenet, 'dropout_params'),
                    'is_training': is_training
                })

        def features_to_logits(all_pre_logits, is_training, **kwargs):
            pre_logits, pre_logits_aux = all_pre_logits
            return layers.ladder_densenet_logits(
                pre_logits,
                pre_logits_aux,
                image_shape=input_shape[1:3],
                class_count=class_count,
                bn_params={'is_training': is_training})

        return InferenceComponent(
            input_shape=input_shape,
            input_to_features=input_to_features,
            features_to_logits=features_to_logits,
            problem='semseg',
            class_count=class_count)


class EvaluationMetrics:
    accuracy = ('accuracy', ['label', 'output'], evaluation.accuracy)

    @staticmethod
    def semantic_segementation(class_count,
                               returns=['m_p', 'm_r', 'm_f1', 'm_iou']):

        def func(labels, predictions):
            return evaluation.evaluate_semantic_segmentation(
                labels, predictions, class_count, returns)

        return (returns, ['label', 'output'], func)
