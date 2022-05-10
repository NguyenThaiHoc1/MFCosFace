from math import pi

import tensorflow as tf


def SoftmaxLoss():
    """softmax loss"""

    def softmax_loss(y_true, y_pred):
        # y_true: sparse target
        # y_pred: logist
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return tf.reduce_mean(ce)

    return softmax_loss


def ArcfaceLoss(margin, scale, n_classes):
    """arcface loss """

    def arcface_loss(y_true, y_pred):
        # y_true: spare target
        # y_pred: logist

        threshold = tf.math.cos(pi - margin)
        cos_m = tf.math.cos(margin)
        sin_m = tf.math.sin(margin)

        safe_margin = sin_m * margin

        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)

        # Calculate the cosine value of theta + margin.
        cos_t = y_pred
        sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))

        cos_t_margin = tf.where(cos_t > threshold,
                                cos_t * cos_m - sin_t * sin_m,
                                cos_t - safe_margin)

        # The labels here had already been onehot encoded.
        mask = y_true
        cos_t_onehot = cos_t * mask
        cos_t_margin_onehot = cos_t_margin * mask

        # Calculate the final scaled logits.
        logits = (cos_t + cos_t_margin_onehot - cos_t_onehot) * scale

        losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

        return tf.reduce_mean(losses)

    def arcface_loss_v2(y_true, y_pre):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)

        original_target_logit = tf.clip_by_value(tf.math.reduce_sum(y_pre * y_true, axis=1), -1 + (1e-12),
                                                 1 - (1e-12))
        theta = tf.acos(original_target_logit)
        marginal_target_logit = tf.cos(theta + margin)

        one_hot = y_true
        logits = y_pre + tf.multiply(one_hot,
                                     tf.expand_dims(marginal_target_logit - original_target_logit, axis=1))
        logits = logits * scale

        losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

        return tf.reduce_mean(losses)

    return arcface_loss_v2
