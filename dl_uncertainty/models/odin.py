import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .tf_utils.adversarial_examples import fgsm
from functools import lru_cache

from ..data import DataLoader


class Odin:

    def __init__(self, model, max_epsilon=0.012, epsilon_count=13, temp=1000):
        self.model = model
        self.temp = temp
        with model._graph.as_default():
            with tf.variable_scope('odin'):
                self.epsilon_node = tf.placeholder(
                    tf.float32, name='epsilon', shape=())
                self.temp_node = tf.placeholder(
                    tf.float32, name='temp', shape=())
                self.temp_softmax = tf.nn.softmax(
                    model.nodes['logits'] / self.temp_node)
                self.perturb_node = fgsm(
                    x=model.nodes['input'],
                    probs=self.temp_softmax,
                    eps=-self.epsilon_node)
        self.max_epsilon = max_epsilon
        self.epsilon_count = epsilon_count
        self.epsilon = 0

    def _perturb(self, x, eps, single_input=False):
        p = self.model._sess.run(self.perturb_node, {
            self.model.nodes['input']: [x] if single_input else x,
            self.model.nodes['is_training']: False,
            self.temp_node: self.temp,
            self.epsilon_node: eps
        })
        if single_input:
            p = p[0]
        return p

    def _get_score(self, x, eps, scores_name='probs', single_input=False):
        if scores_name == 'probs':
            class_scores = self.model._sess.run(self.temp_softmax, {
                self.model.nodes['input']: [x] if single_input else x,
                self.model.nodes['is_training']: False,
                self.temp_node: self.temp,
                self.epsilon_node: eps
            })
            if single_input:
                class_scores = class_scores[0]
        else:
            class_scores = self.model.predict(
                x, single_input=single_input, outputs=scores_name)
        return np.max(class_scores, axis=-1)

    def _get_pert_score(self, x, eps, scores_name='probs', single_input=False):
        if eps > 0:
            x = self._perturb(x, eps, single_input)
        return self._get_score(x, eps, scores_name, single_input)

    @lru_cache(maxsize=2048)
    def _get_scores(self, ds, scores_name, eps, temp):  # temp for caching
        dl = DataLoader(
            ds, self.model.batch_size, drop_last=False, num_workers=1)
        return np.concatenate(
            [self._get_pert_score(xy[0], eps, scores_name) for xy in dl],
            axis=0)

    def fit_epsilon(self,
                    in_ds,
                    out_ds_val,
                    performance_measure,
                    scores_name='probs'):
        perfs = []
        epsilons = np.linspace(0, self.max_epsilon, self.epsilon_count)
        for eps in tqdm(epsilons):
            in_scores_pert = self._get_scores(in_ds, scores_name, eps,
                                              self.temp)
            out_scores_pert = self._get_scores(out_ds_val, scores_name, eps,
                                               self.temp)
            perf_pert = performance_measure(in_scores_pert, out_scores_pert)
            perfs += [perf_pert]
        perfs = np.array(perfs)
        self.epsilon = epsilons[np.argmax(perfs)]
        return epsilons, perfs

    def get_scores(self, ds, scores_name='probs'):
        return self._get_scores(ds, scores_name, self.epsilon, self.temp)