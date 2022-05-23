import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import Input, Concatenate, Dropout, Flatten, Dense
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape

from Settings import config
from Tensorflow.Architecture.ArcHead.header import ArcHead
from Tensorflow.Architecture.utlis import utlis


######################## FUNCTION ########################


def regular(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def convolution2d_block(outputs, kernel, strides=1, padding="same", activation="relu", use_bias=False,
                        name_block=None, name_layer=None):
    sequence_block = Sequential(name=name_block)
    sequence_block.add(Conv2D(filters=outputs, kernel_size=kernel, strides=strides,
                              padding=padding, use_bias=use_bias, name=name_layer))

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = utlis.generate_layer_name('BatchNorm', prefix=name_layer)
        sequence_block.add(BatchNormalization(axis=bn_axis, momentum=0.995, epsilon=0.001,
                                              scale=False, name=bn_name))

    if activation is not None:
        activation_name = utlis.generate_layer_name('Activation', prefix=name_layer)
        sequence_block.add(Activation(activation=activation, name=activation_name))

    return sequence_block


class NormHead(Model):

    def __init__(self, name, num_classes):
        super(NormHead, self).__init__(name=name)

        self.dense = Dense(num_classes, kernel_regularizer=regular(weights_decay=5e-4), name='NormHead_FC_Dense')

    def call(self, inputs, training=False, mask=None):
        out = self.dense(inputs)
        return out


#######################################################

class StemLayer(Model):

    def __init__(self, name):
        super(StemLayer, self).__init__(name=name)

        self.convolution2d_1 = convolution2d_block(32, 3, strides=2, padding='valid',
                                                   name_block='Stem_1',
                                                   name_layer='Convolution2d_1a_3x3')

        self.convolution2d_2 = convolution2d_block(32, 3, strides=1, padding='valid',
                                                   name_block='Stem_2',
                                                   name_layer='Convolution2d_2a_3x3')

        self.convolution2d_3 = convolution2d_block(64, 3, strides=1, padding='same',
                                                   name_block='Stem_3',
                                                   name_layer='Convolution2d_2b_3x3')

        self.max_pooling2d = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')

        self.convolution2d_4 = convolution2d_block(80, 1, strides=1, padding='valid',
                                                   name_block='Stem_4',
                                                   name_layer='Convolution2d_3b_1x1')

        self.convolution2d_5 = convolution2d_block(192, 3, strides=1, padding='valid',
                                                   name_block='Stem_5',
                                                   name_layer='Convolution2d_4a_3x3')

        self.convolution2d_6 = convolution2d_block(256, 3, strides=2, padding='valid',
                                                   name_block='Stem_6',
                                                   name_layer='Convolution2d_4b_3x3')

    def call(self, inputs, training=False, *args, **kwargs):
        out = self.convolution2d_1(inputs)
        out = self.convolution2d_2(out)
        out = self.convolution2d_3(out)
        out = self.max_pooling2d(out)
        out = self.convolution2d_4(out)
        out = self.convolution2d_5(out)
        out = self.convolution2d_6(out)
        return out


######################## CBAM ########################

class ChannelAttention(Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg = GlobalAveragePooling2D()
        self.max = GlobalMaxPooling2D()
        self.conv1 = Conv2D(in_planes // ratio, kernel_size=1, strides=1, padding='same',
                            kernel_regularizer=regular(5e-4),
                            use_bias=False, activation=tf.nn.relu)

        self.conv2 = Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                            kernel_regularizer=regular(5e-4),
                            use_bias=False)

    def call(self, inputs, *args, **kwargs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = Reshape((1, 1, avg.shape[1]))(avg)  # shape (None, 1, 1 feature)
        max = Reshape((1, 1, max.shape[1]))(max)  # shape (None, 1, 1 feature)

        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))

        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out


class SpatialAttention(Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = convolution2d_block(outputs=1, kernel=kernel_size, strides=1,
                                         use_bias=False, activation='sigmoid')

    def call(self, inputs, *args, **kwargs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)
        out = self.conv1(out)

        return out


class AttentionMachines(Model):

    def __init__(self, in_planes, name=None):
        super(AttentionMachines, self).__init__(name)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=16)
        self.sa = SpatialAttention()

    def call(self, inputs, training=False, *args, **kwargs):
        out = self.ca(inputs) * inputs
        out = self.sa(out) * out

        return out


#######################################################


######################## INCEPTION BLOCK ###############################

class InceptionBlockA(Model):

    def __init__(self):
        super(InceptionBlockA, self).__init__()

        # branch 1
        self.branch_1_0 = convolution2d_block(32, 1, use_bias=True,
                                              name_block='sequence_blockA_1_0', name_layer='Conv2d_1x1')

        # branch 2
        self.branch_2_0 = convolution2d_block(32, 1, use_bias=True,
                                              name_block='sequence_blockA_2_0', name_layer='Conv2d_1x1')

        self.branch_2_1 = convolution2d_block(32, 3, use_bias=True,
                                              name_block='sequence_blockA_2_1', name_layer='Conv2d_1x1')

        # branch 3
        self.branch_3_0 = convolution2d_block(32, 1, use_bias=True,
                                              name_block='sequence_blockA_3_0', name_layer='Conv2d_1x1')

        self.branch_3_1 = convolution2d_block(32, 3, use_bias=True,
                                              name_block='sequence_blockA_3_1', name_layer='Conv2d_1x1')

        self.branch_3_2 = convolution2d_block(32, 3, use_bias=True,
                                              name_block='sequence_blockA_3_2', name_layer='Conv2d_1x1')

    def call(self, inputs, training=False, *args, **kwargs):
        out = inputs

        branch_1 = self.branch_1_0(out)

        branch_2 = self.branch_2_0(out)
        branch_2 = self.branch_2_1(branch_2)

        branch_3 = self.branch_3_0(out)
        branch_3 = self.branch_3_1(branch_3)
        branch_3 = self.branch_3_2(branch_3)

        return [branch_1, branch_2, branch_3]


class InceptionBlockB(Model):

    def __init__(self):
        super(InceptionBlockB, self).__init__()
        self.branch_0_0 = convolution2d_block(128, 1, use_bias=True,
                                              name_block='sequence_block17_1', name_layer='Conv2d_1x1')
        self.branch_1_0 = convolution2d_block(128, 1, use_bias=True,
                                              name_block='sequence_block17_2', name_layer='Conv2d_0a_1x1')
        self.branch_1_1 = convolution2d_block(128, [1, 7], use_bias=True,
                                              name_block='sequence_block17_3', name_layer='Conv2d_0b_1x7')
        self.branch_1_2 = convolution2d_block(128, [7, 1], use_bias=True,
                                              name_block='sequence_block17_4', name_layer='Conv2d_0c_7x1')

    def call(self, inputs, training=False, *args, **kwargs):
        out1 = self.branch_0_0(inputs)

        out2 = self.branch_1_0(inputs)
        out2 = self.branch_1_1(out2)
        out2 = self.branch_1_2(out2)

        return [out1, out2]


class InceptionBlockC(Model):

    def __init__(self):
        super(InceptionBlockC, self).__init__()
        self.branch_0_0 = convolution2d_block(192, 1, use_bias=True,
                                              name_block='sequence_block8_1', name_layer='Conv2d_1x1')
        self.branch_1_0 = convolution2d_block(192, 1, use_bias=True,
                                              name_block='sequence_block8_2', name_layer='Conv2d_0a_1x1')
        self.branch_1_1 = convolution2d_block(192, [1, 3], use_bias=True,
                                              name_block='sequence_block8_3', name_layer='Conv2d_0b_1x3')
        self.branch_1_2 = convolution2d_block(192, [3, 1], use_bias=True,
                                              name_block='sequence_block8_4', name_layer='Conv2d_0c_3x1')

    def call(self, inputs, training=False, *args, **kwargs):
        out1 = self.branch_0_0(inputs)

        out2 = self.branch_1_0(inputs)
        out2 = self.branch_1_1(out2)
        out2 = self.branch_1_2(out2)

        return [out1, out2]


class InceptionBlock(Model):

    def __init__(self, type_block, activation, attention_module=True, name=None):
        super(InceptionBlock, self).__init__(name=name)
        self.type_block = type_block
        self.check_activation = activation
        self.attention_module = attention_module
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

        if self.type_block == 'BlockA':
            self.inception_res_block = InceptionBlockA()
            self.number_channel = 256

        elif self.type_block == 'BlockB':
            self.inception_res_block = InceptionBlockB()
            self.number_channel = 896

        elif self.type_block == 'BlockC':
            self.inception_res_block = InceptionBlockC()
            self.number_channel = 1792

        self.mixed = Concatenate(axis=self.channel_axis)

        self.up_1 = convolution2d_block(self.number_channel, 1, activation=None, use_bias=True)

        self.bn = BatchNormalization()

        self.activation = Activation(activation)

        if self.attention_module:
            self.cbam = AttentionMachines(in_planes=self.number_channel, name=f"{name}_CBAM")

    def call(self, inputs, training=False, *args, **kwargs):
        out = self.inception_res_block(inputs, training=training)

        out = self.mixed(out)

        out = self.up_1(out)

        out = self.bn(out, training=training)

        out = self.activation(out, training=training)

        if self.attention_module:
            out = self.cbam(out, training=training)

        return out


#######################################################################


######################## REDUCTION LAYER #############################

class ReductionBlockA(Model):

    def __init__(self, name):
        super(ReductionBlockA, self).__init__(name=name)
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

        # branch 1
        self.reduction_1 = convolution2d_block(384, 3, strides=2, padding='valid',
                                               name_block='seq_reductionA_1', name_layer='Conv2d_3x3')

        # branch 2
        self.reduction_2 = convolution2d_block(192, 1, name_block='seq_reductionA_2', name_layer='Conv2d_0a_1x1')
        self.reduction_3 = convolution2d_block(192, 3, name_block='seq_reductionA_3', name_layer='Conv2d_0b_3x3')

        # branch 3
        self.reduction_4 = convolution2d_block(256, 3, strides=2, padding='valid',
                                               name_block='seq_reductionA_4', name_layer='Conv2d_0a_3x3')

        self.reduction_pooling_1 = MaxPooling2D(3, strides=2, padding='valid')

        # Concatenate
        self.concatenate = Concatenate(axis=channel_axis)

    def call(self, inputs, training=False, *args, **kwargs):
        out1 = self.reduction_1(inputs)

        out2 = self.reduction_2(inputs)
        out2 = self.reduction_3(out2)
        out2 = self.reduction_4(out2)

        out3 = self.reduction_pooling_1(inputs)

        out = self.concatenate([out1, out2, out3])
        return out


class ReductionBlockB(Model):

    def __init__(self, name):
        super(ReductionBlockB, self).__init__(name=name)
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

        # branch 1
        self.reduction_0_0 = convolution2d_block(256, 1,
                                                 name_block='seq_reductionB_1',
                                                 name_layer='Conv2d_0a_1x1')
        self.reduction_0_1 = convolution2d_block(384, 3, strides=2, padding='valid',
                                                 name_block='seq_reductionB_2',
                                                 name_layer='Conv2d_0b_3x3')

        # branch 2
        self.reduction_1_0 = convolution2d_block(256, 1, name_block='seq_reductionB_3',
                                                 name_layer='Conv2d_0a_1x1')
        self.reduction_1_1 = convolution2d_block(256, 3, strides=2, padding='valid',
                                                 name_block='seq_reductionB_4',
                                                 name_layer='Conv2d_0b_3x3')

        # branch 3
        self.reduction_2_0 = convolution2d_block(256, 1, name_block='seq_reductionB_5',
                                                 name_layer='Conv2d_0a_1x1')
        self.reduction_2_1 = convolution2d_block(256, 3, name_block='seq_reductionB_6',
                                                 name_layer='Conv2d_0b_3x3')
        self.reduction_2_2 = convolution2d_block(256, 3, strides=2, padding='valid',
                                                 name_block='seq_reductionB_7',
                                                 name_layer='Conv2d_0c_3x3')

        # pooling 4
        self.reduction_pooling_1 = MaxPooling2D(3, strides=2, padding='valid')

        # Concatenate
        self.concatenate = Concatenate(axis=channel_axis)

    def call(self, inputs, training=False, *args, **kwargs):
        out1 = self.reduction_0_0(inputs)
        out1 = self.reduction_0_1(out1)

        out2 = self.reduction_1_0(inputs)
        out2 = self.reduction_1_1(out2)

        out3 = self.reduction_2_0(inputs)
        out3 = self.reduction_2_1(out3)
        out3 = self.reduction_2_2(out3)

        out4 = self.reduction_pooling_1(inputs)

        out = self.concatenate([out1, out2, out3, out4])
        return out


#######################################################################

class InceptionCBAM(Model):

    def __init__(self, num_classes=None, embedding_size=512, model_type='NormHead', **kwargs):
        super(InceptionCBAM, self).__init__(**kwargs)
        dropout_keep_prob = 0.6
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.model_type = model_type

        """
            Building model
        """
        # 1: Stem layer
        self.stem = StemLayer(name="Stem")

        # 2: CBAM block A
        self.cbam_a = Sequential([
            InceptionBlock(type_block="BlockA",
                           activation="relu",
                           attention_module=True,
                           name=f"BlockA_{idx}")
            for idx in range(1, 5)
        ], name="Att-Inception_A")

        # reduction A
        self.reduction_blockA = ReductionBlockA(name="ReductionA")

        # 2: CBAM block B
        self.cbam_b = Sequential([
            InceptionBlock(type_block="BlockB",
                           activation="relu",
                           attention_module=True,
                           name=f"BlockB_{idx}")
            for idx in range(1, 5)
        ], name="Att-Inception_B")

        # reduction B
        self.reduction_blockB = ReductionBlockB(name="ReductionB")

        # 3: CBAM block C
        self.cbam_c = Sequential([
            InceptionBlock(type_block="BlockC",
                           activation="relu",
                           attention_module=True,
                           name=f"BlockC_{idx}")
            for idx in range(1, 4)
        ], name="Att-Inception_C")

        self.average_pooling = GlobalAveragePooling2D(name='AvgPool')

        self.flatten = Flatten()

        self.dropout = Dropout(1.0 - dropout_keep_prob, name='Dropout')

        # Bottleneck
        self.dense = Dense(embedding_size, use_bias=False, name='Bottleneck')

        # FC ---- for training
        if self.num_classes is not None:
            if self.model_type == 'NormHead':
                self.fc = NormHead(num_classes=self.num_classes, name="Head_FullyConnection")

            elif self.model_type == 'ArcHead':
                self.fc = ArcHead(num_classes=self.num_classes, kernel_regularizer=regular())

    def call(self, inputs, training=False, *args, **kwargs):
        out = self.stem(inputs, training=training)

        out = self.cbam_a(out)

        out = self.reduction_blockA(out)

        out = self.cbam_b(out)

        out = self.reduction_blockB(out)

        out = self.cbam_c(out)

        out = self.average_pooling(out)

        out = self.flatten(out)

        out = self.dropout(out)

        out = self.dense(out)

        if training:
            out = self.fc(out)

        return out


if __name__ == '__main__':
    tf.keras.backend.clear_session()

    model = InceptionCBAM(num_classes=config.NUM_CLASSES,
                          embedding_size=config.EMBEDDING_SIZE,
                          model_type='ArcHead',
                          name="InceptionResNetV1")

    model.build(input_shape=(None, 160, 160, 3))
    model.summary()

    x = Input(shape=(160, 160, 3))
    out = model(x, training=True)
    print(out)
    print("DONE ...")
