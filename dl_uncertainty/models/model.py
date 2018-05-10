import abc
import datetime
import os
from functools import reduce

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..data import DataLoader
from ..ioutils import file, console

from .modeldef import ModelDef


class Model(object):

    def __init__(self, modeldef: ModelDef, training_log_period=1, name="Model"):
        self.name = name

        self.batch_size = modeldef.training_component.batch_size

        self.training_log_period = training_log_period
        self.log = []
        self.training_step_event_handler = lambda i: \
            console.read_line(impatient=True, discard_non_last=True)

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        with self._graph.as_default():
            self._epoch = tf.Variable(0, False, dtype=tf.int32, name='epoch')
            self._increment_epoch = tf.assign(
                self._epoch, self._epoch + 1, name='increment_epoch')
            self._is_training = tf.placeholder(tf.bool, name='is_training')

            self.nodes = modeldef.build_graph(self._epoch, self._is_training)

            self._sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver(
                max_to_keep=10, keep_checkpoint_every_n_hours=2)

            param_count = sum(
                reduce((lambda x, y: x * y), var.shape.as_list())
                for var in tf.trainable_variables())
            print(f"Number of parameters: {param_count}")

    def __str__(self):
        return self.name

    def save_state(self, path, save_log=True):
        """
            Saves the trained model as `file_path`.
            If `save_log == True`, `self.log` is saved as `file_path`+'.log'.
        """
        file_path = os.path.join(path, str(self))
        os.makedirs(path, exist_ok=True)
        self._saver.save(self._sess, file_path)
        with open(file_path + ".log", mode='w') as fs:
            fs.write("\n".join(self.log))
            fs.flush()
        print("State saved as '" + file_path + "'.")
        return file_path

    def load_state(self, path):
        self._saver.restore(self._sess, path)
        try:
            self.log = file.read_all_lines(path + ".log")
        except:
            self._log("Log file not found.")
        self._log("State loaded (" + str(self.epoch) + " epochs completed).")

    def load_parameters(self, names_to_values):
        self._log("Loading parameters...")
        with self._graph.as_default():
            name_to_variable = {v.name: v for v in tf.global_variables()}
            for name, value in tqdm(names_to_values.items()):
                var = name_to_variable[name]
                var.load(value, self._sess)
        self._log("Parameters loaded...")

    def predict(self, inputs: list, outputs="output"):
        """
        :param outputs: string list of strings which can be a list like
            ["output", "probs", "logits", "uncertainty"]
        """
        if type(outputs) is str:
            fetches = self.nodes.outputs[outputs]
        if type(outputs) is list:
            fetches = [self.nodes.outputs[o] for o in outputs]
        return self._run(fetches, inputs, None, False)

    def train(self, train_data, epoch_count=1):
        self._train(train_data, epoch_count, self.nodes.evaluation)

    def test(self, dataset, test_name=None):
        return self._test(dataset, self.nodes.evaluation, test_name)

    @property
    def epoch(self):
        return self._sess.run(self._epoch)

    def _train_minibatch(self, inputs, labels, extra_fetches: list = []):
        fetches = [self.nodes.training_step, self.nodes.loss] + \
            list(extra_fetches)
        evals = self._run(fetches, inputs, labels, is_training=True)
        cost, extra = evals[1], evals[2:]
        if self.nodes.training_post_step is not None:
            self._sess.run(self.nodes.training_post_step)
        return cost, extra

    def _test_minibatch(self, inputs, labels, extra_fetches: list = []):
        fetches = [self.nodes.loss] + list(extra_fetches)
        evals = self._run(fetches, inputs, labels, is_training=False)
        cost, extra = evals[0], evals[1:]
        return cost, extra

    def _train(self, data: DataLoader, epoch_count=1, extra_fetches=dict()):

        def handle_step_completed(batch, cost, extra):
            if b % self.training_log_period == 0:
                ev = zip(extra_fetches.keys(), extra)
                self._log(
                    f" {self.epoch:3d}.{b:3d}: {self._eval_str(cost, ev)}")
            return self.training_step_event_handler(b)

        for _ in range(epoch_count):
            self._log(f"Training: epoch {self.epoch:d} " +
                      f"({len(data)} batches of size {self.batch_size}, " +
                      f"lr={self._sess.run(self.nodes.learning_rate):.2e})")
            for b, (inputs, labels) in enumerate(data):
                ef = extra_fetches.values()
                cost, extra = self._train_minibatch(inputs, labels, ef)
                end = handle_step_completed(b, cost, extra) == 'q'
            self._sess.run(self._increment_epoch)
            if end:
                return False

    def _test(self,
              data: DataLoader,
              extra_fetches: dict = dict(),
              test_name=None):
        self._log('Testing%s...' %
                  ("" if test_name is None else " (" + test_name + ")"))
        cost_sum, extra_sum = 0, np.zeros(len(extra_fetches))
        for inputs, labels in data:
            cost, extra = self._test_minibatch(inputs, labels,
                                               extra_fetches.values())
            cost_sum += cost
            extra_sum += np.array(extra)
        cost = cost_sum / len(data)
        extra = extra_sum / len(data)
        ev = list(zip(extra_fetches.keys(), extra))
        self._log(" " + self._eval_str(cost, ev))
        return cost, dict(ev)

    def _run(self, fetches: list, inputs, labels=None, is_training=None):
        feed_dict = {self.nodes.input: inputs}
        if labels is not None:
            feed_dict[self.nodes.label] = np.array(labels)
        if self._is_training is not None:
            feed_dict[self._is_training] = is_training
        return self._sess.run(fetches, feed_dict)

    def _eval_str(self, cost: float, ev: list):
        return f"loss {cost:.4f}, " + ", ".join([f"{k} {v:.4f}" for k, v in ev])

    def _log(self, text: str):
        timestr = datetime.datetime.now().strftime('%H:%M:%S')
        text = f"[{timestr}] {text}"
        self.log.append(text)
        print(text)