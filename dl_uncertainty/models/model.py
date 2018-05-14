import abc
import datetime
import os
from functools import reduce

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..data import DataLoader
from ..ioutils import file, console
from ..evaluation import AccumulatingEvaluator, DummyAccumulatingEvaluator

from .modeldef import ModelDef


class Model(object):

    def __init__(self,
                 modeldef: ModelDef,
                 accumulating_evaluator:AccumulatingEvaluator = \
                     DummyAccumulatingEvaluator(),
                 training_log_period=1,
                 name="Model"):
        self.name = name

        self.batch_size = modeldef.training_component.batch_size

        self.accumulating_evaluator = accumulating_evaluator

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

            self.accumulating_evaluator = self.accumulating_evaluator()
            self.ae_accum_batch = self.accumulating_evaluator.accumulate_batch(
                self.nodes.label, self.nodes.outputs['output'])
            ae_evals = self.accumulating_evaluator.evaluate()
            self.ae_eval_names = list(ae_evals.keys())
            self.ae_evals = list(ae_evals.values())
            self.ae_reset = self.accumulating_evaluator.reset()

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

    def load_parameters(self, name_to_value):
        self._log("Loading parameters...")
        with self._graph.as_default():
            name_to_variable = {v.name: v for v in tf.global_variables()}
            for name, value in tqdm(name_to_value.items()):
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

    @property
    def epoch(self):
        return self._sess.run(self._epoch)

    def train(self, data: DataLoader, epoch_count=1):

        def train_minibatch(inputs, labels, extra_fetches=[]):
            fetches = [
                self.nodes.training_step,
                self.nodes.loss,
            ] + list(extra_fetches)
            _, loss, *extra = self._run(
                fetches, inputs, labels, is_training=True)
            loss += loss
            #evaluator.accumulate(labels, output)
            if self.nodes.training_post_step is not None:
                self._sess.run(self.nodes.training_post_step)
            return loss, extra

        def handle_step_completed(batch, loss):
            ev = zip(self.ae_eval_names, self._sess.run(self.ae_evals))
            self._sess.run(self.ae_reset)
            self._log(f" {self.epoch:3d}.{(b + 1):3d}: " +
                      f"{self._eval_str(loss, ev)}")
        
        self._sess.run(self.ae_reset)
        for _ in range(epoch_count):
            self._log(f"Training: epoch {self.epoch:d} " +
                      f"({len(data)} batches of size {self.batch_size}, " +
                      f"lr={self._sess.run(self.nodes.learning_rate):.2e})")
            for b, (inputs, labels) in enumerate(data):
                loss, *_ = train_minibatch(
                    inputs, labels, extra_fetches=[self.ae_accum_batch])
                if (b + 1) % self.training_log_period == 0:
                    end = handle_step_completed(b, loss) == 'q'
                end = self.training_step_event_handler(b)
            self._sess.run(self._increment_epoch)
            if end:
                return False

    def test(self, data: DataLoader, test_name=None):
        self._log('Testing%s...' %
                  ("" if test_name is None else " (" + test_name + ")"))
        loss_sum = 0
        self._sess.run(self.ae_reset)
        for inputs, labels in data:
            fetches = [
                self.nodes.loss, self.nodes.outputs['output'],
                self.ae_accum_batch
            ]
            loss, output, _ = self._run(
                fetches, inputs, labels, is_training=False)
            loss_sum += loss
        loss = loss_sum / len(data)
        ev = zip(self.ae_eval_names, self._sess.run(self.ae_evals))
        self._log(" " + self._eval_str(loss, ev))
        return loss, ev

    def _test_old(self,
                  data: DataLoader,
                  extra_fetches: dict = dict(),
                  test_name=None):
        self._log('Testing%s...' %
                  ("" if test_name is None else " (" + test_name + ")"))
        cost_sum, extra_sum = 0, np.zeros(len(extra_fetches))
        for inputs, labels in data:
            fetches = [self.nodes.loss] + list(extra_fetches.values())
            evals = self._run(fetches, inputs, labels, is_training=False)
            cost, extra = evals[0], evals[1:]
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

    def _eval_str(self, loss: float, ev):
        if type(ev) is dict:
            ev = ev.items()
        return f"loss={loss:.4f}, " + ", ".join([f"{k}={v:.4f}" for k, v in ev])

    def _log(self, text: str):
        timestr = datetime.datetime.now().strftime('%H:%M:%S')
        text = f"[{timestr}] {text}"
        self.log.append(text)
        print(text)