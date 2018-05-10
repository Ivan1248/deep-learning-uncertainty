import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from _context import dl_uncertainty
from dl_uncertainty.tf_utils import layers

one = tf.constant(1.0)
a = tf.Variable(0.0)
b = tf.Variable(0.0)

subgraph_a = lambda x: x * a
subgraph_b = lambda x: x * b

a1 = layers.amplify_gradient_for_subgraph(one, subgraph_a, 2)
b1 = layers.amplify_gradient_for_subgraph(one, subgraph_b, 0.5)

Y = a1 + b1

loss = tf.reduce_mean((Y - 10)**2)

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(1)
train_op = trainer.minimize(loss)

## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.initialize_all_variables())

## 3. učenje
# neka igre počnu!
for i in range(5):
    loss_, a_, b_ = sess.run([train_op, a, b])
    print(i, a_, b_)
    assert round(a_) == 4 * round(b_)
