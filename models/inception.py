"""
Inception 1D_CNN Models in Tensorflow-Keras.
References -
Inception_v1 (GoogLeNet): https://arxiv.org/abs/1409.4842 [Going Deeper with Convolutions]
Inception_v2: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v3: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v4: https://arxiv.org/abs/1602.07261 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning]
"""
# Author: Sakib Mahmud
# Email: sakib.mahmud@qu.edu.qa
# License: MIT
# Ref. code: https://github.com/Sakib1263/VGG-1D-2D-Tensorflow-Keras/tree/main

from keras import backend as K
from keras import layers, models, activations, regularizers

def Conv_1D_Block(x, model_width, kernel, strides=1, padding="same"):
    # 1D Convolutional Block with BatchNormalization
    x = layers.Conv1D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = layers.Dense(class_number, activation='softmax')(inputs)
    return out

def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = layers.Dense(feature_number, activation='linear')(inputs)
    return out

def Inceptionv1_Module(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_1D_Block(inputs, filterB1_1, 1, padding='valid')

    branch3x3 = Conv_1D_Block(inputs, filterB2_1, 1, padding='valid')
    branch3x3 = Conv_1D_Block(branch3x3, filterB2_2, 3)

    branch5x5 = Conv_1D_Block(inputs, filterB3_1, 1, padding='valid')
    branch5x5 = Conv_1D_Block(branch5x5, filterB3_2, 5)

    branch_pool = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(inputs)
    branch_pool = Conv_1D_Block(branch_pool, filterB4_1, 1)
    out = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_Block_'+str(i))

    return out

def Inceptionv2_Module(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_1D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3 = Conv_1D_Block(branch3x3, filterB2_2, 3)

    branch3x3dbl = Conv_1D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_2, 3)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_3, 3)

    branch_pool = layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(inputs)
    branch_pool = Conv_1D_Block(branch_pool, filterB4_1, 1)

    out = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_'+str(i))

    return out

def Inception_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_1D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3 = Conv_1D_Block(branch3x3, filterB2_2, 5)

    branch3x3dbl = Conv_1D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_2, 3)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_3, 3)

    branch_pool = layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(inputs)
    branch_pool = Conv_1D_Block(branch_pool, filterB4_1, 1)

    out = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_A'+str(i))

    return out

def Inception_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_1D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3 = Conv_1D_Block(branch3x3, filterB2_2, 7)

    branch3x3dbl = Conv_1D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_2, 7)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_3, 7)

    branch_pool = layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(inputs)
    branch_pool = Conv_1D_Block(branch_pool, filterB4_1, 1)

    out = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_B'+str(i))

    return out

def Inception_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_1D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3 = Conv_1D_Block(branch3x3, filterB2_2, 3)

    branch3x3dbl = Conv_1D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_2, 3)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB3_3, 3)

    branch_pool = layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(inputs)
    branch_pool = Conv_1D_Block(branch_pool, filterB4_1, 1)

    out = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_C'+str(i))

    return out

def Reduction_Block_A(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block A (i)
    branch3x3 = Conv_1D_Block(inputs, filterB1_1, 1)
    branch3x3 = Conv_1D_Block(branch3x3, filterB1_2, 3, strides=2)

    branch3x3dbl = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB2_2, 3)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB2_3, 3, strides=2)

    branch_pool = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(inputs)
    out = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out


def Reduction_Block_B(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block B (i)
    branch3x3 = Conv_1D_Block(inputs, filterB1_1, 1)
    branch3x3 = Conv_1D_Block(branch3x3, filterB1_2, 3, strides=2)

    branch3x3dbl = Conv_1D_Block(inputs, filterB2_1, 1)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB2_2, 7)
    branch3x3dbl = Conv_1D_Block(branch3x3dbl, filterB2_3, 3, strides=2)

    branch_pool = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(inputs)
    out = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out

def Inception_v1(input_dim, num_channel = 1, num_filters = 16, output_nums=4, pooling = 'avg', auxilliary_outputs=False):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    x = Conv_1D_Block(inputs, num_filters, 7, strides=2)
    x = layers.MaxPooling1D(3, strides=2)(x)
    x = Conv_1D_Block(x, num_filters, 1, padding='valid')
    x = Conv_1D_Block(x, num_filters * 3, 3)
    x = layers.MaxPooling1D(3, strides=2)(x)

    x = Inceptionv1_Module(x, 64, 96, 128, 16, 32, 32, 1)  # Inception Block 1
    x = Inceptionv1_Module(x, 128, 128, 192, 32, 96, 64, 2)  # Inception Block 2

    aux_output_0 = []
    if auxilliary_outputs:
        # Auxilliary Output 0
        aux_pool = layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
        aux_conv = Conv_1D_Block(aux_pool, 64, 1)
        if pooling == 'avg':
            aux_conv_0 = layers.GlobalAveragePooling1D()(aux_conv)
        elif pooling == 'max':
            aux_conv_0 = layers.GlobalMaxPool1D()(aux_conv)
        # Final Dense Outputting Layer for the outputs
        aux_conv_0 = layers.Flatten()(aux_conv_0)
        aux_output_0 = layers.Dense(output_nums, activation='softmax')(aux_output_0)

    x = layers.MaxPooling1D(3, strides=2)(x)
    x = Inceptionv1_Module(x, 192, 96, 208, 16, 48, 64, 3)  # Inception Block 3
    x = Inceptionv1_Module(x, 160, 112, 224, 24, 64, 64, 4)  # Inception Block 4
    x = Inceptionv1_Module(x, 128, 128, 256, 24, 64, 64, 5)  # Inception Block 5
    x = Inceptionv1_Module(x, 112, 144, 288, 32, 64, 64, 6)  # Inception Block 6
    x = Inceptionv1_Module(x, 256, 160, 320, 32, 128, 128, 7)  # Inception Block 7

    aux_output_1 = []
    if auxilliary_outputs:
        # Auxilliary Output 1
        aux_pool = layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
        aux_conv = Conv_1D_Block(aux_pool, 64, 1)
        if pooling == 'avg':
            aux_conv_0 = layers.GlobalAveragePooling1D()(aux_conv)
        elif pooling == 'max':
            aux_conv_0 = layers.GlobalMaxPool1D()(aux_conv)
        # Final Dense Outputting Layer for the outputs
        aux_conv_1 = layers.Flatten()(aux_conv_1)
        aux_output_1 = layers.Dense(output_nums, activation='softmax')(aux_output_1)

    x = layers.MaxPooling1D(3, strides=2)(x)
    x = Inceptionv1_Module(x, 256, 160, 320, 32, 128, 128, 8)  # Inception Block 8
    x = Inceptionv1_Module(x, 384, 192, 384, 48, 128, 128, 9)  # Inception Block 9

    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = layers.Flatten()(x)
    outputs = layers.Dense(aux_conv_0, activation='softmax')(x)

    # Create the model 
    model = models.Model(inputs=inputs, outputs=[outputs, aux_output_0, aux_output_1], name='Inception_v1')
    model.summary()

    return model

def Inception_v2(input_dim, num_channel = 1, num_filters = 16, output_nums=4, pooling = 'avg', auxilliary_outputs=False):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    x = layers.SeparableConv1D(num_filters, kernel_size=7, strides=2, depth_multiplier=1, padding='same')(inputs)
    x = layers.MaxPooling1D(3, strides=2)(x)
    x = Conv_1D_Block(x, num_filters * 2, 1, padding='valid')
    x = Conv_1D_Block(x, num_filters * 6, 3, padding='valid')
    x = layers.MaxPooling1D(3, strides=2)(x)

    x = Inceptionv2_Module(x, 64, 64, 64, 64, 96, 96, 32, 1)  # Inception Block 1: 28 x 192
    x = Inceptionv2_Module(x, 64, 64, 96, 64, 96, 96, 64, 2)  # Inception Block 2: 28 x 256

    aux_output_0 = []
    if auxilliary_outputs:
        # Auxilliary Output 0
        aux_pool = layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
        aux_conv = Conv_1D_Block(aux_pool, 64, 1)
        if pooling == 'avg':
            aux_conv_0 = layers.GlobalAveragePooling1D()(aux_conv)
        elif pooling == 'max':
            aux_conv_0 = layers.GlobalMaxPool1D()(aux_conv)
        # Final Dense Outputting Layer for the outputs
        aux_conv_0 = layers.Flatten()(aux_conv_0)
        aux_output_0 = layers.Dense(output_nums, activation='softmax')(aux_output_0)

    x = Reduction_Block_A(x, 128, 160, 64, 96, 96, 1)  # Reduction Block 1: 28 x 320
    x = Inceptionv2_Module(x, 224, 64, 96, 96, 128, 128, 128, 3)  # Inception Block 3: 14 x 576
    x = Inceptionv2_Module(x, 192, 96, 128, 96, 128, 128, 128, 4)  # Inception Block 4: 14 x 576
    x = Inceptionv2_Module(x, 160, 128, 160, 128, 160, 160, 96, 5)  # Inception Block 5: 14 x 576
    x = Inceptionv2_Module(x, 96, 128, 192, 160, 192, 192, 96, 6)  # Inception Block 6: 14 x 576

    aux_output_1 = []
    if auxilliary_outputs:
        # Auxilliary Output 1
        aux_pool = layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
        aux_conv = Conv_1D_Block(aux_pool, 192, 1)
        if pooling == 'avg':
            aux_conv_1 = layers.GlobalAveragePooling1D()(aux_conv)
        elif pooling == 'max':
            aux_conv_1 = layers.GlobalMaxPool1D()(aux_conv)
        # Final Dense Outputting Layer for the outputs
        aux_conv_1 = layers.Flatten()(aux_conv_1)
        aux_output_1 = layers.Dense(output_nums, activation='softmax')(aux_output_1)

    x = Reduction_Block_A(x, 128, 192, 192, 256, 256, 2)  # Reduction Block 2: 14 x 576
    x = Inceptionv2_Module(x, 352, 192, 320, 160, 224, 224, 128, 7)  # Inception Block 7: 7 x 1024
    x = Inceptionv2_Module(x, 352, 192, 320, 192, 224, 224, 128, 8)  # Inception Block 8: 7 x 1024

    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = layers.Flatten()(x)
    outputs = layers.Dense(aux_conv_0, activation='softmax')(x)

    # Create the model 
    model = models.Model(inputs=inputs, outputs=[outputs, aux_output_0, aux_output_1], name='Inception_v1')
    model.summary()

    return model

def Inception_v3(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', auxilliary_outputs=False):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    x = Conv_1D_Block(inputs, num_filters, 3, strides=2, padding='valid')
    x = Conv_1D_Block(x, num_filters, 3, padding='valid')
    x = Conv_1D_Block(x, num_filters * 2, 3)
    x = layers.MaxPooling1D(3, strides=2)(x)

    x = Conv_1D_Block(x, num_filters * 2.5, 1, padding='valid')
    x = Conv_1D_Block(x, num_filters * 6, 3, padding='valid')
    x = layers.MaxPooling1D(3, strides=2)(x)

    # 3x Inception-A Blocks
    x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 32, 1)  # Inception-A Block 1: 35 x 256
    x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 2)  # Inception-A Block 2: 35 x 256
    x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 3)  # Inception-A Block 3: 35 x 256

    aux_output_0 = []
    if auxilliary_outputs:
        # Auxilliary Output 0
        aux_pool = layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
        aux_conv = Conv_1D_Block(aux_pool, 64, 1)
        if pooling == 'avg':
            aux_conv_0 = layers.GlobalAveragePooling1D()(aux_conv)
        elif pooling == 'max':
            aux_conv_0 = layers.GlobalMaxPool1D()(aux_conv)
        # Final Dense Outputting Layer for the outputs
        aux_conv_0 = layers.Flatten()(aux_conv_0)
        aux_output_0 = layers.Dense(output_nums, activation='softmax')(aux_output_0)

    x = Reduction_Block_A(x, 64, 384, 64, 96, 96, 1)  # Reduction Block 1: 17 x 768

    # 4x Inception-B Blocks
    x = Inception_Module_B(x, 192, 128, 192, 128, 128, 192, 192, 1)  # Inception-B Block 1: 17 x 768
    x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 2)  # Inception-B Block 2: 17 x 768
    x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 3)  # Inception-B Block 3: 17 x 768
    x = Inception_Module_B(x, 192, 192, 192, 192, 192, 192, 192, 4)  # Inception-B Block 4: 17 x 768

    aux_output_1 = []
    if auxilliary_outputs:
        # Auxilliary Output 1
        aux_pool = layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
        aux_conv = Conv_1D_Block(aux_pool, 192, 1)
        if pooling == 'avg':
            aux_conv_1 = layers.GlobalAveragePooling1D()(aux_conv)
        elif pooling == 'max':
            aux_conv_1 = layers.GlobalMaxPool1D()(aux_conv)
        # Final Dense Outputting Layer for the outputs
        aux_conv_1 = layers.Flatten()(aux_conv_1)
        aux_output_1 = layers.Dense(output_nums, activation='softmax')(aux_output_1)

    x = Reduction_Block_B(x, 192, 320, 192, 192, 192, 2)  # Reduction Block 2: 8 x 1280

    # 2x Inception-C Blocks: 8 x 2048
    x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 1)  # Inception-C Block 1: 8 x 2048
    x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 2)  # Inception-C Block 2: 8 x 2048

    # Final Dense MLP Layer for the outputs
    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = layers.Flatten()(x)
    outputs = layers.Dense(aux_conv_0, activation='softmax')(x)

    # Create the model 
    model = models.Model(inputs=inputs, outputs=[outputs, aux_output_0, aux_output_1], name='Inception_v1')
    model.summary()

    return model

def Inception_v4(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', auxilliary_outputs=False):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    x = Conv_1D_Block(inputs, 32, 3, strides=2, padding='valid')
    x = Conv_1D_Block(x, 32, 3, padding='valid')
    x = Conv_1D_Block(x, 64, 3)

    branch1 = Conv_1D_Block(x, 96, 3, strides=2, padding='valid')
    branch2 = layers.MaxPooling1D(3, strides=2)(x)
    x = layers.concatenate([branch1, branch2], axis=-1)

    branch1 = Conv_1D_Block(x, 64, 1)
    branch1 = Conv_1D_Block(branch1, 96, 3, padding='valid')
    branch2 = Conv_1D_Block(x, 64, 1)
    branch2 = Conv_1D_Block(branch2, 64, 7)
    branch2 = Conv_1D_Block(branch2, 96, 3, padding='valid')
    x = layers.concatenate([branch1, branch2], axis=-1)

    branch1 = Conv_1D_Block(x, 192, 3, padding='valid')
    branch2 = layers.MaxPooling1D(3, strides=2)(x)
    x = layers.concatenate([branch1, branch2], axis=1)

    # 4x Inception-A Blocks - 35 x 256
    for i in range(4):
        x = Inception_Module_A(x, 96, 64, 96, 64, 96, 96, 96, i)

    aux_output_0 = []
    if auxilliary_outputs:
        # Auxilliary Output 0
        aux_pool = layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
        aux_conv = Conv_1D_Block(aux_pool, 96, 1)
        if pooling == 'avg':
            aux_conv_0 = layers.GlobalAveragePooling1D()(aux_conv)
        elif pooling == 'max':
            aux_conv_0 = layers.GlobalMaxPool1D()(aux_conv)
        # Final Dense Outputting Layer for the outputs
        aux_conv_0 = layers.Flatten()(aux_conv_0)
        aux_output_0 = layers.Dense(output_nums, activation='softmax')(aux_output_0)

    x = Reduction_Block_A(x, 64, 384, 192, 224, 256, 1)  # Reduction Block 1: 17 x 768

    # 7x Inception-B Blocks - 17 x 768
    for i in range(7):
        x = Inception_Module_B(x, 384, 192, 256, 192, 224, 256, 128, i)

    aux_output_1 = []
    if auxilliary_outputs:
        # Auxilliary Output 1
        aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
        aux_conv = Conv_1D_Block(aux_pool, 128, 1)
        if pooling == 'avg':
            aux_conv_1 = layers.GlobalAveragePooling1D()(aux_conv)
        elif pooling == 'max':
            aux_conv_1 = layers.GlobalMaxPool1D()(aux_conv)
        # Final Dense Outputting Layer for the outputs
        aux_conv_1 = layers.Flatten()(aux_conv_1)
        aux_output_1 = layers.Dense(output_nums, activation='softmax')(aux_output_1)

    x = Reduction_Block_B(x, 192, 192, 256, 320, 320, 2)  # Reduction Block 2: 8 x 1280

    # 3x Inception-C Blocks: 8 x 2048
    for i in range(3):
        x = Inception_Module_C(x, 256, 384, 512, 384, 512, 512, 256, i)

    # Final Dense MLP Layer for the outputs
    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = layers.Flatten()(x)
    outputs = layers.Dense(aux_conv_0, activation='softmax')(x)

    # Create the model 
    model = models.Model(inputs=inputs, outputs=[outputs, aux_output_0, aux_output_1], name='Inception_v1')
    model.summary()

    return model