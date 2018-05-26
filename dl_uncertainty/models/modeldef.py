from collections import namedtuple
from ..utils.collections import UnoverwritableDict

import tensorflow as tf

from .tf_utils import layers, losses, regularization, evaluation

# InferenceComponent and TrainingComponent

InferenceComponent = namedtuple('InferenceComponent', ['input_to_output'])

TrainingComponent = namedtuple('TrainingComponent', [
    'batch_size', 'loss', 'optimizer', 'weight_decay', 'training_post_step',
    'pretrained_lr_factor', 'learning_rate_policy'
])


class ModelDef:

    def __init__(self, inference_component: InferenceComponent,
                 training_component: TrainingComponent):
        self.inference_component = inference_component
        self.training_component = training_component

    def build_graph(self, epoch, is_training, dropout_active):
        ic = self.inference_component
        tc = self.training_component

        # Input
        input = tf.placeholder(tf.float32, shape=[None] * 3 + [3], name='input')
        image_shape = tf.shape(input)[1:3]

        c = UnoverwritableDict({
            'input': input,
            'is_training': is_training,
            'dropout_active': dropout_active,
            'pretrained_lr_factor': tc.pretrained_lr_factor,
            'image_shape': image_shape,
        })

        # Inference
        output, additional_nodes = ic.input_to_output(input, **c)
        c.update(additional_nodes)

        # Label
        c['label'] = tf.placeholder(output.dtype, output.shape, name='label')

        # Loss and regularization
        loss_fn, loss_args = tc.loss
        c['loss_emp'] = loss_fn(* [c[arg] for arg in loss_args])
        loss = c['loss_emp']
        if tc.weight_decay > 0:
            w_vars = filter(lambda x: 'weights' in x.name,
                            tf.global_variables())
            c['loss_reg'] = tc.weight_decay * regularization.l2_regularization(
                w_vars)
            loss += c['loss_reg']
        c['loss'] = loss

        # Optimization
        c['learning_rate'] = tc.learning_rate_policy(epoch)
        c['optimizer'] = tc.optimizer(c['learning_rate'])
        c['training_step'] = c['optimizer'].minimize(c['loss'])

        return c