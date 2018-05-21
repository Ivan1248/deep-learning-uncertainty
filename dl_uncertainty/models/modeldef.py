from collections import namedtuple

import tensorflow as tf

from .tf_utils import layers, losses, regularization, evaluation

# InferenceComponent and TrainingComponent

InferenceComponent = namedtuple('InferenceComponent',
                                ['input_shape', 'input_to_output'])

TrainingComponent = namedtuple('TrainingComponent', [
    'batch_size', 'loss', 'optimizer', 'weight_decay', 'training_post_step',
    'pretrained_lr_factor', 'learning_rate_policy'
])


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
