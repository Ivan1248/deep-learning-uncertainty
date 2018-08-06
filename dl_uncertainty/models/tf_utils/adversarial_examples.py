import numpy as np
import tensorflow as tf


def model_loss(y, model, mean=True):
    """
    Taken from https://github.com/tensorflow/cleverhans.
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        loss = tf.reduce_mean(loss)
    return loss


def fgsm(x,
         probs,
         eps=0.3,
         y=None,
         clip_min=None,
         clip_max=None,
         targeted=False):
    return fgm(
        x=x,
        probs=probs,
        eps=eps,
        y=y,
        clip_min=clip_min,
        clip_max=clip_max,
        targeted=targeted)


def fgm(x,
        probs,
        y=None,
        eps=0.3,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        targeted=False):
    """
    Taken from https://github.com/tensorflow/cleverhans.
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param probs: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        probs_max = tf.reduce_max(probs, 1, keepdims=True)
        y = tf.to_float(tf.equal(probs, probs_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keepdims=True)

    # Compute loss
    loss = model_loss(y, probs, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(xrange(1, len(x.get_shape())))
        normalized_grad = grad / reduce_sum(
            tf.abs(grad), reduction_indices=red_ind, keepdims=True)
    elif ord == 2:
        red_ind = list(xrange(1, len(x.get_shape())))
        square = reduce_sum(
            tf.square(grad), reduction_indices=red_ind, keepdims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x
