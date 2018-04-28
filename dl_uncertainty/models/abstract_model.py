import abc
import datetime
import os

import numpy as np
import tensorflow as tf

from ..data import Dataset, MiniBatchReader
from ..ioutils import file


class AbstractModel(object):

    class EssentialNodes:

        def __init__(
                self,
                input,
                target,
                output,
                loss,
                training_step,
                evaluation=dict(),
                additional_outputs=dict(),  # probs, logits, ...
                training_post_step=None):
            self.input = input
            self.target = target
            self.outputs = {**additional_outputs, 'output': output}
            self.loss = loss
            self.training_step = training_step
            self.evaluation = evaluation
            self.training_post_step = training_post_step

    def __init__(
            self,
            batch_size: int,  # mini-batch size
            learning_rate_policy,
            training_log_period=1,
            input_mean=None,
            input_stddev=None,
            name="DCNN"):
        self.name = name

        self.batch_size = batch_size
        self.learning_rate_policy = learning_rate_policy

        self.input_mean, self.input_stddev = input_mean, input_stddev

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        with self._graph.as_default():
            self._epoch = tf.Variable(0, False, dtype=tf.int32, name='epoch')
            self._increment_epoch = tf.assign(
                self._epoch, self._epoch + 1, name='increment_epoch')
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            lr = self.learning_rate_policy
            if type(lr) is dict:
                lr = tf.train.piecewise_constant(
                    self._epoch,
                    boundaries=lr['boundaries'],
                    values=lr['values'])
            self.nodes = self._build_graph(lr, self._epoch, self._is_training)

            self._sess.run(tf.global_variables_initializer())

            self.training_log_period = training_log_period
            self.log = []
            self._saver = tf.train.Saver(
                max_to_keep=10, keep_checkpoint_every_n_hours=2)

            self.training_step_event_handler = lambda step: False

            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print(f"Number of parameters: {total_parameters}")

    def __del__(self):  # I am not sure whether this is good
        self._sess.close()
        #with self._graph.as_default():
        #    tf.reset_default_graph()

    def __str__(self):
        return self.name

    def save_state(self, path, save_log=True):
        """
            Saves the trained model as `file_path`.
            If `save_log == True`, `self.log` is saved as `file_path`+'.log'.
        """
        file_path = os.path.join(path, str(self))
        if not os.path.exists(path):
            os.makedirs(path)
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
            self.log = "Log file not found."
        self._log("State loaded (" + str(self.epoch) + " epochs completed).")

    def predict(self, inputs: list, outputs="output"):
        """
        :param outputs: string list of strings which can at most be a list like
            ["output", "probs", "logits", "uncertainty"]
        """
        if type(outputs) is str:
            fetches = self.nodes.outputs[outputs]
        if type(outputs) is list:
            fetches = [self.nodes.outputs[o] for o in outputs]
        return self._run(fetches, inputs, None, False)

    def train(self,
              train_data: Dataset,
              validation_data: Dataset = None,
              epoch_count: int = 1):
        self._train(train_data, validation_data, epoch_count,
                    self.nodes.evaluation)

    def test(self, dataset, test_name=None):
        return self._test(dataset, self.nodes.evaluation, test_name)

    @property
    def epoch(self):
        return self._sess.run(self._epoch)

    def _train_minibatch(self, inputs, labels, extra_fetches: list = []):
        fetches = [self.nodes.training_step, self.nodes.loss
                  ] + list(extra_fetches)
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

    def _train(self,
               train_data: Dataset,
               validation_data: Dataset = None,
               epoch_count: int = 1,
               extra_fetches: dict = dict()):

        def log_training_start(epoch_count, batch_count, batch_size):
            self._log('Training (epochs: {}; {} batches of size {} per epoch)'
                      .format(epoch_count, batch_count, batch_size))

        def handle_step_completed(batch, cost, extra):
            if b % self.training_log_period == 0 or b == dr.number_of_batches - 1:
                ev = zip(extra_fetches.keys(), extra)
                self._log(
                    f" {self.epoch:3d}.{b:3d}: {self._eval_str(cost, ev)}")
            if self.training_step_event_handler(b):
                nonlocal end
                end = True

        dr = MiniBatchReader(train_data, self.batch_size)
        log_training_start(epoch_count, dr.number_of_batches, self.batch_size)
        end = False
        for _ in range(epoch_count):
            self._log(f"epoch {self.epoch:d}")
            dr.reset(shuffle=True)
            for b in range(dr.number_of_batches):
                inputs, labels = dr.get_next_batch()
                ef = extra_fetches.values()
                cost, extra = self._train_minibatch(inputs, labels, ef)
                handle_step_completed(b, cost, extra)
            self._sess.run(self._increment_epoch)
            if end:
                break

    def _test(self, dataset, extra_fetches: dict = dict(), test_name=None):
        self._log('Testing%s...' %
                  ("" if test_name is None else " (" + test_name + ")"))
        cost_sum, extra_sum = 0, np.zeros(len(extra_fetches))
        dr = MiniBatchReader(dataset, self.batch_size)
        for _ in range(dr.number_of_batches):
            inputs, labels = dr.get_next_batch()
            cost, extra = self._test_minibatch(inputs, labels,
                                               extra_fetches.values())
            cost_sum += cost
            extra_sum += np.array(extra)
        cost = cost_sum / dr.number_of_batches
        extra = extra_sum / dr.number_of_batches
        ev = list(zip(extra_fetches.keys(), extra))
        self._log(" " + self._eval_str(cost, ev))
        return cost, dict(ev)

    @abc.abstractmethod
    def _build_graph(self, learning_rate, epoch, is_training):
        """ 
            Builds the TensorFlow graph for the model.
            Override this. It will be automatically called by the constructor
            (assuming super().__init__(...) is called in the constructor of the
            subclass).
            Returns tuple (input node, target labels node, probs node) (nodes 
            are of type tf.Tensor, the first 2 being placeholders)
         """
        return AbstractModel.EssentialNodes(*([None] * 6))

    def _run(self, fetches: list, inputs, labels=None, is_training=None):
        if self.input_mean is not None:
            inputs = (inputs - self.input_mean) / self.input_stddev
        feed_dict = {self.nodes.input: inputs}
        if labels is not None:
            feed_dict[self.nodes.target] = np.array(labels)
        if self._is_training is not None:
            feed_dict[self._is_training] = is_training
        return self._sess.run(fetches, feed_dict)

    def _eval_str(self, cost: float, ev: list):
        return f"cost {cost:.4f}, " + ", ".join([f"{k} {v:.4f}" for k, v in ev])

    def _log(self, text: str):
        timestr = datetime.datetime.now().strftime('%H:%M:%S')
        text = f"[{timestr}] {text}"
        self.log.append(text)
        print(text)

    #def _print_vars(self):
    #    vars = sorted([v.name for v in tf.global_variables()])
    #    for i, v in enumerate(vars):
    #        print(i, v)
    #    exit()
