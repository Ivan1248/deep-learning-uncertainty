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

def fgm(x,
        probs,
        eps=0.3,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        antiadversarial=False):
    """
    Taken from https://github.com/tensorflow/cleverhans.
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param probs: the model's output tensor (the attack expects the
        probabilities, i.e., the output of the softmax)
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy). 
        Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param antiadversarial: Move in the direction of decreasing the error.
    :return: a tensor for the adversarial example
    """

    # Using model predictions as ground truth to avoid label leaking
    probs_max = tf.reduce_max(probs, 1, keep_dims=True)
    y = tf.to_float(tf.equal(probs, probs_max))
    y = tf.stop_gradient(y)

    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = model_loss(y, probs, mean=False)
    if antiadversarial:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        normalized_grad = grad / tf.reduce_sum(
            tf.abs(grad), np.arange(1, len(x.get_shape())), keep_dims=True)
    elif ord == 2:
        square = tf.reduce_sum(
            tf.square(grad), np.arange(1, len(x.get_shape())), keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        assert False

    scaled_grad = eps * normalized_grad

    adv_x = x + scaled_grad

    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x