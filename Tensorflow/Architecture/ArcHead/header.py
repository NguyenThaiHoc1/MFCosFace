"""
    https://github.com/YeongHyeon/ArcFace-TF2/blob/ad786aea2d5a41f8b4756f7f26f3ef4d3a1182d6/source/neuralnet.py#L23
    https://github.com/dsskim/arc_face_tensorflow2/blob/3b776ac0faaef10ec752ed5603673d1bbfc1c1a0/models/loss_layers.py#L4
    https://github.com/yinguobing/arcface/blob/58987e24c2f381cf0e7e3cedbae54e71806a8150/train.py

    https://github.com/4uiiurz1/keras-arcface/blob/877ea00531d74c735354a6cee92788110542196e/metrics.py#L94
"""
import tensorflow as tf


class ArcHead(tf.keras.layers.Layer):

    def __init__(self, num_classes, kernel_regularizer, **kwargs):
        super(ArcHead, self).__init__(**kwargs)
        self.units = num_classes
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.units),
                                 dtype=tf.float32,
                                 initializer=tf.keras.initializers.HeNormal(),
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)

        self.built = True

    @tf.function
    def call(self, inputs):
        # Step 1: l2_norm input
        inputs = tf.nn.l2_normalize(inputs, axis=1)

        # Step 2: l2_norm weight
        weight = tf.nn.l2_normalize(self.W, axis=0)

        return tf.matmul(inputs, weight)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units,
                       "kernel_regularizer": self.kernel_regularizer})
        return config


class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs

        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)

        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)

        # dot product
        logits = x @ W

        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(
            tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)

        y = tf.one_hot(tf.cast(y, tf.int32), depth=self.n_classes)  # convert label to onehot
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        # out = tf.nn.softmax(logits)

        return logits

    def compute_output_shape(self, input_shape):
        return None, self.n_classes
