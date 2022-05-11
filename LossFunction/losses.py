import math
from math import pi

import tensorflow as tf
import tensorflow.keras.backend as K


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

    def arcface_loss_v3(y_true, y_pre):
        onehot_labels = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)

        theta = tf.acos(K.clip(y_pre, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + margin)
        logits = y_pre * (1 - onehot_labels) + target_logits * onehot_labels
        logits = logits * scale

        losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)

        return tf.reduce_mean(losses)

    def arcface_loss_v4(y_true, y_pre):
        cos_t = y_pre

        cos_m = tf.identity(math.cos(margin), name='cos_m')

        sin_m = tf.identity(math.sin(margin), name='sin_m')

        th = tf.identity(math.cos(math.pi - margin), name='th')

        mm = tf.multiply(sin_m, margin, name='mm')

        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(cos_t * cos_m, sin_t * sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > th, cos_mt, cos_t - mm)

        mask = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes, name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)

        logists = tf.multiply(logists, scale, 'arcface_logist')

        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logists)

        return tf.reduce_mean(ce)

    return arcface_loss_v4


def CosfaceLoss(margin, scale, n_classes):

    def cosface_loss(y_true, y_pred):

        cos_theta = y_pred

        mask = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes, name='one_hot_mask')

        target_logist = cos_theta - margin

        logist = target_logist * mask + (1 - mask) * y_pred

        logist *= scale

        out = tf.nn.softmax(logist)

        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=out)

        return tf.reduce_mean(ce)

    return cosface_loss
