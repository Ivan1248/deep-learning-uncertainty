from ..tf_utils.layers import BlockStructure

"""
from .abstract_model import AbstractModel
from .dummy import Dummy
from .baseline_a import BaselineA
from .resnet import ResNet
from .densenet import DenseNet

from .resnet_ss import ResNetSS
from .ladder_densenet import LadderDenseNet
"""

from .model import Model
from .modeldef import ModelDef, InferenceComponent, TrainingComponent, ModelDef 
from .modeldef import InferenceComponents, EvaluationMetrics 

"""
from ..tf_utils import layers

def resnet_components(input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            class_count,
            group_lengths=[3, 3, 3],
            block_structure=layers.BlockStructure.resnet(),
            base_width=None,
            weight_decay=5e-4,
            batch_size=128,
            learning_rate_policy=1e-2,
            training_log_period=1):

    return GenericModel(input_shape=input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable input size
            input_to_features= lambda x: resnet(
                x,
                **self.resnet_params,
                bn_params={'is_training': is_training},
                dropout_params={
                    **layers.default_arg(layers.resnet, 'dropout_params'),
                    'is_training': is_training
                }),
            features_to_output=lambda x: x,
            features_to_logits=lambda x: x,
            logits_to_probs=tf.nn.softmax,
            loss='auto',
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
            problem=['classification', 'regression'][0],
            dense_predictions=False,
            evaluation_metrics=[],
            batch_size=128,
            learning_rate_policy=1e-2,
            training_log_period=1,
            weight_decay=1e-4,
            name='Model')

"""