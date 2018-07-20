import numpy as np
import sklearn
from sklearn import svm, linear_model
import tensorflow as tf
from tqdm import tqdm

from .tf_utils.adversarial_examples import fgsm
from functools import lru_cache

from ..data import DataLoader


class Godin:

    def __init__(self,
                 model,
                 classifier=None,
                 max_epsilon=0.010,
                 epsilon_count=11):
        self.model = model
        self.classifier = None
        with model._graph.as_default():
            with tf.variable_scope('godin'):
                self.epsilon_node = tf.placeholder(
                    tf.float32, name='epsilon', shape=())
                self.perturb_node = fgsm(
                    x=model.nodes['input'],
                    probs=model.nodes['probs'],
                    eps=self.epsilon_node,
                    antiadversarial=True)
        self.max_epsilon = max_epsilon
        self.epsilon_count = epsilon_count
        self.epsilon = 0

    def _perturb(self, x, eps):
        return self.model._sess.run(self.perturb_node, {
            self.model.nodes['input']: x,
            self.model.nodes['is_training']: False,
            self.epsilon_node: eps
        })

    @lru_cache()
    def get_logits(self, ds, eps):
        """ 
        Returns an array with shape [len(ds), C], where C is class/logit count.
        """
        dl = DataLoader(
            ds, self.model.batch_size, drop_last=False, num_workers=1)

        def inner_get_logits(x):
            if eps > 0:
                x = self._perturb(x, eps)
            return self.model.predict(x, outputs='logits')

        return np.concatenate([inner_get_logits(x) for x, y in dl], axis=0)

    #@lru_cache() no caching because ds_in ds_out pair needs to be known
    def _get_scores(self, ds, eps, clf):  # eps and clf for caching
        logits = self.get_logits(ds, eps)
        return clf.predict_scores(logits)

    def get_scores(self, ds, eps=None):
        """ 
        Returns an array with shape [len(ds)] containing scores.
        """
        if eps is None:
            eps = self.epsilon
        return self._get_scores(ds, eps, self.classifier)

    def fit(self, ds_in, ds_out_val, performance_measure, fit_epsilon=True):
        """ 
        Finds the optimal epsilon and trains the clasifier.
        """
        perfs = []
        epsilons = np.linspace(0, self.max_epsilon,
                               self.epsilon_count) if fit_epsilon else [0]
        y = [1] * len(ds_in) + [0] * len(ds_out_val)

        def fit_classifier(eps):
            in_logits = self.get_logits(ds_in, eps)
            out_logits = self.get_logits(ds_out_val, eps)
            self.classifier.fit(
                np.concatenate([in_logits, out_logits], axis=0), y)

        for eps in tqdm(epsilons):
            fit_classifier(eps)
            in_scores = self.get_scores(ds_in, eps)
            out_scores = self.get_scores(ds_out_val, eps)
            perf_pert = performance_measure(in_scores, out_scores)
            perfs += [perf_pert]

        perfs = np.array(perfs)
        self.epsilon = epsilons[np.argmax(perfs)]
        fit_classifier(self.epsilon)
        return epsilons, perfs


class MaxLogitClassifier:

    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def predict_scores(self, x):
        return np.max(x, axis=-1)


class MaxProbClassifier:

    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def predict_scores(self, x):
        expx = np.exp(x)
        return np.max(expx / np.sum(expx), axis=-1)


class ClasswiseNormalizingMaxLogitClassifier:

    def __init__(self):
        self.mean, self.std = None, None

    def fit(self, x, y):
        self.mean, self.std = np.mean(x, 0), np.std(x, 0)

    def predict_scores(self, x):
        return np.max((x - self.mean) / self.std, axis=-1)


class MaxSumLogitClassifier:

    def __init__(self,
                 classifier=None,
                 degree=2,
                 include_max_logit_in_sum=True,
                 classwise_scaling=False):
        self.include_max_logit_in_sum = include_max_logit_in_sum
        self.classwise_scaling = classwise_scaling
        self.mean, self.std = None, None
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.poly = sklearn.preprocessing.PolynomialFeatures(degree)
        if classifier is None:
            #classifier = svm.SVC()
            classifier = linear_model.LogisticRegression()
        self.classifier = classifier

    def _extract_input_features(self, x):
        if self.classwise_scaling:
            x = (x - self.mean) / self.std
        maxs = x.max(-1, keepdims=True)
        sums = x.sum(-1, keepdims=True)
        if not self.include_max_logit_in_sum:
            sums -= maxs
        return np.concatenate([maxs, sums], -1)

    def _extract_final_features_from_input_features(self, x):
        x = self.scaler.transform(x)
        return self.poly.fit_transform(x)

    def _extract_final_features_from_logits(self, x):
        x = self._extract_input_features(x)
        return self._extract_final_features_from_input_features(x)

    def fit(self, x, y):
        if self.classwise_scaling:
            self.mean, self.std = np.mean(x, 0), np.std(x, 0)
        self.scaler.fit(self._extract_input_features(x))
        x = self._extract_final_features_from_logits(x)
        self.classifier.fit(x, y)

    def predict_scores(self, x):
        x = self._extract_final_features_from_logits(x)
        return self.classifier.decision_function(x)
