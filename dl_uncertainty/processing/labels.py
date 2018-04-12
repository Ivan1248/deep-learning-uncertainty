import numpy as np


def dense_to_one_hot(labels_dense, class_count):
    """ Converts dense labels of any shape to one-hot vectors. """
    flat_oh = (np.arange(class_count) == labels_dense.flatten()[:, None])
    return flat_oh.reshape(list(labels_dense.shape) + [class_count])

def one_hot_to_dense(labels_one_hot):
    """ Converts pixel labels from one-hot vectors to scalars. """
    return np.argmax(labels_one_hot, axis=len(labels_one_hot.shape)-1)
