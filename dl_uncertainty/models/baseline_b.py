import numpy as np
import tensorflow as tf

from ..data import Dataset, MiniBatchReader

from .abstract_model import AbstractModel


class BaselineB(AbstractModel):
    def __init__(self,
                 input_shape,
                 class_count,
                 batch_size=32,
                 conv_layer_count=4,
                 learning_rate_policy=1e-3,
                 training_log_period=1,
                 weight_decay=1e-4,
                 ortho_penalty=0,
                 use_multiclass_hinge_loss=False,
                 name='ClfBaselineA'):
        self.conv_layer_count = conv_layer_count
        self.weight_decay = weight_decay
        self.ortho_penalty = ortho_penalty
        self.use_multiclass_hinge_loss = use_multiclass_hinge_loss
        self.completed_epoch_count = 0
        super().__init__(
            input_shape=input_shape,
            class_count=class_count,
            batch_size=batch_size,
            learning_rate_policy=learning_rate_policy,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        from tensorflow.contrib import layers
        from tf_utils.layers import conv2d, max_pool, rescale_bilinear, avg_pool, bn_relu
        from tf_utils.losses import multiclass_hinge_loss

        def get_ortho_penalty():
            vars = tf.contrib.framework.get_variables('')
            filt = lambda x: 'conv' in x.name and 'weights' in x.name
            weight_vars = list(filter(filt, vars))
            loss = tf.constant(0.0)
            for v in weight_vars:
                m = tf.reshape(v, (-1, v.shape[3].value))
                d = tf.matmul(
                    m, m, True) - tf.eye(v.shape[3].value) / v.shape[3].value
                loss += tf.reduce_sum(d**2)
            return loss

        input_shape = [None] + list(self.input_shape)
        output_shape = [None, self.class_count]

        # Input image and labels placeholders
        input = tf.placeholder(tf.float32, shape=input_shape, name='input')
        target = tf.placeholder(tf.float32, shape=output_shape, name='target')

        # L2 regularization
        weight_decay = tf.constant(self.weight_decay, dtype=tf.float32)

        # Hidden layers
        h = input
        with tf.contrib.framework.arg_scope(
            [layers.conv2d],
                kernel_size=5,
                data_format='NHWC',
                padding='SAME',
                activation_fn=tf.nn.relu,
                weights_initializer=layers.variance_scaling_initializer(),
                weights_regularizer=layers.l2_regularizer(weight_decay)):
            h = layers.conv2d(h, 16, scope='convrelu1')
            h = layers.max_pool2d(h, 2, 2, scope='pool1')
            h = layers.conv2d(h, 32, scope='convrelu2')
            h = layers.max_pool2d(h, 2, 2, scope='pool2')
        with tf.contrib.framework.arg_scope(
            [layers.fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=layers.variance_scaling_initializer(),
                weights_regularizer=layers.l2_regularizer(weight_decay)):
            h = layers.flatten(h, scope='flatten3')
            h = layers.fully_connected(h, 512, scope='fc3')
        
        self._print_vars()

        # Softmax classification
        logits = layers.fully_connected(
            h, self.class_count, activation_fn=None, scope='logits')
        probs = tf.nn.softmax(logits, name='probs')

        # Loss
        mhl = lambda t, lo: 0.1 * multiclass_hinge_loss(t, lo)
        sce = tf.losses.softmax_cross_entropy
        loss = (mhl if self.use_multiclass_hinge_loss else sce)(target, logits)
        loss = loss + tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        if self.ortho_penalty > 0:
            loss += self.ortho_penalty * get_ortho_penalty()

        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        # Dense predictions and labels
        preds, dense_labels = tf.argmax(probs, 1), tf.argmax(target, 1)

        # Other evaluation measures
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            probs=probs,
            loss=loss,
            training_step=training_step,
            evaluation={
                'accuracy': accuracy
            })


def main(epoch_count=1):
    from data import Dataset, loaders
    from data.preparers import Iccv09Preparer
    import ioutils
    from ioutils import console

    print("Loading and deterministically shuffling data...")
    data_path = os.path.join(
        ioutils.path.find_ancestor(os.path.dirname(__file__), 'projects'),
        'datasets/cifar-10-batches-py')
    ds = loaders.load_cifar10_train(data_path)
    labels = np.array([np.bincount(l.flat).argmax() for l in ds.labels])
    ds = Dataset(ds.images, labels, ds.class_count)
    ds.shuffle(order_determining_number=0.5)

    print("Splitting dataset...")
    ds_train, ds_val = ds.split(0, int(ds.size * 0.8))
    print(ds_train.size, ds_val.size)

    print("Initializing model...")
    model = ClfBaselineB(
        input_shape=ds.image_shape,
        class_count=ds.class_count,
        batch_size=128,
        learning_rate_policy=1e-6,
        #learning_rate_policy={
        #    'boundaries': [3, 5, 7],
        #    'values': [10**-i for i in range(1, 5)]
        #},
        ortho_penalty=0,
        use_multiclass_hinge_loss=False,
        training_log_period=100)

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'd':
            viz.display(ds_val, lambda im: model.predict([im])[0])
        elif text == 'q':
            return True
        return False

    model.training_step_event_handler = handle_step

    #viz = visualization.Visualizer()
    print("Starting training and validation loop...")
    #model.test(ds_val)
    #ds_train = ds_train[:128]
    for i in range(epoch_count):
        model.train(ds_train, epoch_count=1)
        model.test(ds_train, "Training")
        model.test(ds_val, "Validation")
    model.save_state()


if __name__ == '__main__':
    main(epoch_count=200)

# "GTX 970" 43 times faster than "Pentium 2020M @ 2.40GHz × 2"
