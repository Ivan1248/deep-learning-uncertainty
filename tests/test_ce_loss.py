import numpy as np
import tensorflow as tf

from _context import dl_uncertainty

from dl_uncertainty.models.tf_utils.losses import cross_entropy_loss, weighted_cross_entropy_loss

logits_np = np.array([[[1, 2], [1, 3], [1, 2.0]]])
labels_np = np.array([[-1, 0, 1]])

logits = tf.constant(logits_np, dtype=tf.float32)
labels = tf.constant(labels_np)

logits_nonign = tf.constant(logits_np[:, 1:, :], dtype=tf.float32)
labels_nonign = tf.constant(labels_np[:, 1:])

labels_oh = tf.one_hot(labels, 2)
labels_nonign_oh = tf.one_hot(labels_nonign, 2)

loss = cross_entropy_loss(logits=logits, labels=labels)
loss_nonign = cross_entropy_loss(logits=logits_nonign, labels=labels_nonign)

sess = tf.Session()
lo, lo_nign = sess.run([loss, loss_nonign])

print(sess.run(labels_oh))
print(sess.run(labels_nonign_oh))
print(lo, lo_nign)
assert lo == lo_nign
