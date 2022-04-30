"""
    Base architecture: https://iq.opengenus.org/inception-resnet-v1/
    https://github.com/kobiso/CBAM-keras/blob/796ae9ea31253d87f46ac4908e94ad5d799fbdd5/models/attention_module.py#L5
    https://github.com/zhangkaifang/CBAM-TensorFlow2.0/blob/master/resnet.py
    Author: Nguyen Thai Hoc
"""
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, add, Lambda, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Dense, Input

from Settings import config
from Tensorflow.Architecture.AttentionArchitecture.cbam import channel_attention, spatial_attention
from Tensorflow.Architecture.utlis import utlis


def scaling(x, scale):
    return x * scale


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


class InceptionResnetBlock35(Model):

    def __init__(self):
        super(InceptionResnetBlock35, self).__init__()
        self.branch_0_0 = convolution2d_block(32, 1, name_block='sequence_block35_1', name_layer='Conv2d_1x1')

        self.branch_1_0 = convolution2d_block(32, 1, name_block='sequence_block35_2', name_layer='Conv2d_0a_1x1')
        self.branch_1_1 = convolution2d_block(32, 3, name_block='sequence_block35_3', name_layer='Conv2d_0b_3x3')

        self.branch_2_0 = convolution2d_block(32, 1, name_block='sequence_block35_4', name_layer='Conv2d_0a_1x1')
        self.branch_2_1 = convolution2d_block(32, 3, name_block='sequence_block35_5', name_layer='Conv2d_0b_3x3')
        self.branch_2_2 = convolution2d_block(32, 3, name_block='sequence_block35_6', name_layer='Conv2d_0c_3x3')

    def call(self, inputs, training=False, *args, **kwargs):
        out1 = self.branch_0_0(inputs)

        out2 = self.branch_1_0(inputs)
        out2 = self.branch_1_1(out2)

        out3 = self.branch_2_0(inputs)
        out3 = self.branch_2_1(out3)
        out3 = self.branch_2_2(out3)

        return [out1, out2, out3]


class InceptionResnetBlock17(Model):

    def __init__(self):
        super(InceptionResnetBlock17, self).__init__()
        self.branch_0_0 = convolution2d_block(128, 1, name_block='sequence_block17_1', name_layer='Conv2d_1x1')
        self.branch_1_0 = convolution2d_block(128, 1, name_block='sequence_block17_2', name_layer='Conv2d_0a_1x1')
        self.branch_1_1 = convolution2d_block(128, [1, 7], name_block='sequence_block17_3', name_layer='Conv2d_0b_1x7')
        self.branch_1_2 = convolution2d_block(128, [7, 1], name_block='sequence_block17_4', name_layer='Conv2d_0c_7x1')

    def call(self, inputs, training=False, *args, **kwargs):
        out1 = self.branch_0_0(inputs)

        out2 = self.branch_1_0(inputs)
        out2 = self.branch_1_1(out2)
        out2 = self.branch_1_2(out2)

        return [out1, out2]


class InceptionResnetBlock8(Model):

    def __init__(self):
        super(InceptionResnetBlock8, self).__init__()
        self.branch_0_0 = convolution2d_block(192, 1, name_block='sequence_block8_1', name_layer='Conv2d_1x1')
        self.branch_1_0 = convolution2d_block(192, 1, name_block='sequence_block8_2', name_layer='Conv2d_0a_1x1')
        self.branch_1_1 = convolution2d_block(192, [1, 3], name_block='sequence_block8_3', name_layer='Conv2d_0b_1x3')
        self.branch_1_2 = convolution2d_block(192, [3, 1], name_block='sequence_block8_4', name_layer='Conv2d_0c_3x1')

    def call(self, inputs, training=False, *args, **kwargs):
        out1 = self.branch_0_0(inputs)

        out2 = self.branch_1_0(inputs)
        out2 = self.branch_1_1(out2)
        out2 = self.branch_1_2(out2)

        return [out1, out2]


class InceptionResnetBlock(Model):

    def __init__(self, scale, type_block, activation, attention_module=True, name=None):
        super(InceptionResnetBlock, self).__init__(name=name)
        self.type_block = type_block
        self.scale = scale
        self.check_activation = activation
        self.attention_module = attention_module
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

        if self.type_block == 'Block35':
            self.inception_res_block = InceptionResnetBlock35()
            self.number_channel = 256
        elif self.type_block == 'Block17':
            self.inception_res_block = InceptionResnetBlock17()
            self.number_channel = 896
        elif self.type_block == 'Block8':
            self.inception_res_block = InceptionResnetBlock8()
            self.number_channel = 1792
        else:
            raise ValueError("Type block is wrong ! Please checking")

        self.mixed = Concatenate(axis=self.channel_axis)

        self.up_1 = convolution2d_block(self.number_channel, 1, activation=None, use_bias=True)

        self.up_2 = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           output_shape=self.number_channel,
                           arguments={'scale': self.scale})

        # self.up_2 = Lambda(scaling, output_shape=, arguments={'scale': self.scale})

        self.add = add

        self.bn = BatchNormalization()

        self.activation = Activation(activation)

    def call(self, inputs, training=False, *args, **kwargs):
        out = self.inception_res_block(inputs)
        out = self.mixed(out)

        out = self.up_1(out)

        # CBAM
        if self.attention_module:
            out = channel_attention(out, ratio=8) * out
            out = spatial_attention(out) * out

        out = self.up_2([inputs, out])

        out = self.bn(out)

        if self.check_activation is not None:
            out = self.activation(out)

        return out


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


class NormHead(Model):

    def __init__(self, name, num_classes):
        super(NormHead, self).__init__(name=name)

        self.dense = Dense(num_classes, kernel_regularizer=regular(weights_decay=5e-4), name='NormHead_FC_Dense')

    def call(self, inputs, training=False, mask=None):
        out = self.dense(inputs)
        return out


class InceptionResNetV1(Model):

    def __init__(self, num_classes, embedding_size=512, model_type='NormHead', **kwargs):
        super(InceptionResNetV1, self).__init__(**kwargs)
        dropout_keep_prob = 0.8
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.model_type = model_type

        # Stem layer
        self.stem = StemLayer(name="Stem")

        # 5x Block35 (Inception-ResNet-A block):
        self.block35 = Sequential([
            InceptionResnetBlock(scale=0.17, type_block="Block35", activation="relu", name=f"Block35_{idx}")
            for idx in range(1, 6)
        ], name="Block35")

        # reduction A
        self.reduction_blockA = ReductionBlockA(name="ReductionA")

        # 10x Block17 (Inception-ResNet-B block)
        self.block17 = Sequential([
            InceptionResnetBlock(scale=0.1, type_block="Block17", activation="relu", name=f"Block17_{idx}")
            for idx in range(1, 11)
        ], name="Block17")

        # reduction B
        self.reduction_blockB = ReductionBlockB(name="ReductionB")

        # 5x Block8 (Inception-ResNet-C block)
        self.block8 = Sequential([
            InceptionResnetBlock(scale=0.2, type_block="Block8", activation="relu", name=f"Block8_{idx}")
            for idx in range(1, 6)
        ], name="Block8")

        # Classification block
        self.average_pooling = GlobalAveragePooling2D(name='AvgPool')
        self.dropout = Dropout(1.0 - dropout_keep_prob, name='Dropout')

        # Bottleneck
        self.dense = Dense(embedding_size, use_bias=False, name='Bottleneck')
        self.bn = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False)

        # FC ---- for training
        if self.model_type == 'NormHead':
            self.fc = NormHead(name="Head_FullyConnection", num_classes=self.num_classes)

        elif self.model_type == 'ArcHead':
            raise NotImplementedError

    def call(self, inputs, training=False, *args, **kwargs):
        out = self.stem(inputs)

        out = self.block35(out)

        out = self.reduction_blockA(out)

        out = self.block17(out)

        out = self.reduction_blockB(out)

        out = self.block8(out)

        out = self.average_pooling(out)

        out = self.dropout(out)

        out = self.dense(out)

        out = self.bn(out)

        if training:
            out = self.fc(out)

        return out


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    model = InceptionResNetV1(num_classes=config.NUM_CLASSES,
                              embedding_size=512,
                              name="InceptionResNetV1")
    model.build(input_shape=(None, 299, 299, 3))
    model.summary()

    # demo get layer
    # layer = utlis.get_layer_byname(model,
    #                                name='Block35/Block35_1/inception_resnet_block35/sequence_block35_1/Conv2d_1x1')

    # Test Output of Model
    x = Input(shape=(160, 160, 3))
    out = model(x, training=True)
    print(out)

    print("DONE ...")
