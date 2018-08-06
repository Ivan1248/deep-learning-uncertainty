import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .tf_utils.adversarial_examples import fgsm
from functools import lru_cache

from ..data import DataLoader


class AdversarialExampleGenerator:

    def __init__(self, model, targeted=False):
        self.model = model
        with model._graph.as_default():
            with tf.variable_scope('adv'):
                self.epsilon_node = tf.placeholder(
                    tf.float32, name='epsilon', shape=())
                self.y_node = tf.placeholder(
                    tf.float32, name='y',
                    shape=model.nodes['probs'].shape) if targeted else None
                self.perturb_node = fgsm(
                    x=model.nodes['input'],
                    probs=model.nodes['probs'],
                    y=self.y_node,
                    eps=self.epsilon_node)

    def perturb(self, x, y=None, eps=0.01, single_input=False):
        feed_dict = {
            self.model.nodes['input']: [x] if single_input else x,
            self.model.nodes['is_training']: False,
            self.epsilon_node: eps
        }
        if y is None:
            assert self.y_node is None
        else:
            assert self.y_node is not None
        if self.y_node is not None:
            feed_dict[self.y_node] = np.expand_dims(np.eye(10)[y], 0)
        xp = self.model._sess.run(self.perturb_node, feed_dict)
        if single_input:
            xp = xp[0]
        return xp

    def perturb_most_confident(self, x, ys=[0, 1], eps=0.01,
                               single_input=False):
        assert single_input
        assert self.y_node is not None
        pertxs = []
        probs = []
        for y in ys:
            feed_dict = {
                self.model.nodes['input']: [x] if single_input else x,
                self.model.nodes['is_training']: False,
                self.epsilon_node: eps
            }
            if self.y_node is not None:
                feed_dict[self.y_node] = np.expand_dims(np.eye(10)[y], 0)
            xp, p = self.model._sess.run(
                [self.perturb_node, self.model.nodes['probs']], feed_dict)
            xp = xp[0]
            p = p[0]
            pertxs += [xp]
            probs += [p]
        y = np.argmax(np.max(probs,0))
        #y = np.argmax(np.mean(probs,0))
        return pertxs[y]