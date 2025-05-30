"""MobileNet 1DCNN in Keras.
MovileNet_v1: https://arxiv.org/abs/1704.04861 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications]
MovileNet_v2: https://arxiv.org/abs/1801.04381 [Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation]
MovileNet_v3: https://arxiv.org/abs/1905.02244 [Searching for MobileNetV3]
"""
# Author: Sakib Mahmud
# Email: sakib.mahmud@qu.edu.qa
# License: MIT
# Ref. code: https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras 

from keras import backend as K
from keras import layers, models, activations, regularizers

def Conv_1D_block(inputs, model_width, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def Conv_1D_block_2(inputs, model_width, kernel, strides, nl):
    # This function defines a 1D convolution operation with BN and activation.
    x = layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    if nl == 'HS':
        x = x * activations.relu(x + 3.0, max_value=6.0) / 6.0
    elif nl == 'RE':
        x = activations.relu(x, max_value=6.0)
    return x

def Conv_1D_DW(inputs, model_width, kernel, strides, alpha):
    # 1D Depthwise Separable Convolutional Block with BatchNormalization
    model_width = int(model_width * alpha)
    x = layers.SeparableConv1D(model_width, kernel, strides=strides, depth_multiplier=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(model_width, 1, strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def bottleneck_block(inputs, filters, kernel, t, alpha, s, r=False):
    tchannel = K.int_shape(inputs)[-1] * t
    cchannel = int(filters * alpha)
    x = Conv_1D_block(inputs, tchannel, 1, 1)
    x = layers.SeparableConv1D(filters, kernel, strides=s, depth_multiplier=1, padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(cchannel, 1, strides=1, padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('linear')(x)
    if r:
        x = layers.concatenate([x, inputs], axis=-1)
    return x

def bottleneck_block_2(inputs, filters, kernel, e, s, squeeze, nl, alpha):
    # This function defines a basic bottleneck structure.
    input_shape = K.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(alpha * filters)

    r = s == 1 and input_shape[2] == filters

    x = Conv_1D_block_2(inputs, tchannel, 1, 1, nl)
    x = layers.SeparableConv1D(filters, kernel, strides=s, depth_multiplier=1, padding='same')(x)
    #x = layers.BatchNormalization()(x)

    if nl == 'HS':
        x = x * activations.relu(x + 3.0, max_value=6.0) / 6.0
    if nl == 'RE':
        x = activations.relu(x, max_value=6.0)

    if squeeze:
        x = _squeeze(x)

    x = layers.Conv1D(cchannel, 1, strides=1, padding='same')(x)
    #x = layers.BatchNormalization()(x)

    if r:
        x = layers.Add()([x, inputs])

    return x


def inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    if strides == 1:
        x = bottleneck_block(inputs, filters, kernel, t, alpha, strides, True)
    else:
        x = bottleneck_block(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = bottleneck_block(x, filters, kernel, t, alpha, 1, True)

    return x

def _squeeze(inputs):
    # This function defines a squeeze structure.
    input_channels = int(inputs.shape[-1])

    x = layers.GlobalAveragePooling1D()(inputs)
    x = layers.Dense(input_channels, activation='relu')(x)
    x = layers.Dense(input_channels, activation='hard_sigmoid')(x)
    x = layers.Reshape((1, input_channels))(x)
    x = layers.Multiply()([inputs, x])

    return x

def MobileNet_v1(input_dim, num_channel = 1, num_filters = 16, output_dim=4, pooling = 'avg', alpha=1.0):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    x = Conv_1D_block(inputs, num_filters * (2 ** 0), 3, 2)
    x = Conv_1D_DW(x, num_filters, 3, 1, alpha)
    x = Conv_1D_DW(x, num_filters * (2 ** 1), 3, 2, alpha)
    x = Conv_1D_DW(x, num_filters, 3, 1, alpha)
    x = Conv_1D_DW(x, num_filters * (2 ** 2), 3, 2, alpha)
    x = Conv_1D_DW(x, num_filters, 3, 1, alpha)
    x = Conv_1D_DW(x, num_filters * (2 ** 3), 3, 2, alpha)
    for i in range(5):
        x = Conv_1D_DW(x, num_filters, 3, 1, alpha)
    x = Conv_1D_DW(x, num_filters * (2 ** 4), 3, 2, alpha)
    x = Conv_1D_DW(x, num_filters * (2 ** 5), 3, 2, alpha)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_dim, activation='softmax')(x)
    # Create the model 
    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def MobileNet_v2(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', alpha=1.0):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    x = Conv_1D_block(inputs, num_filters, 3, 2)

    x = inverted_residual_block(x, 16, 3, t=1, alpha=alpha, strides=1, n=1)
    x = inverted_residual_block(x, 24, 3, t=6, alpha=alpha, strides=2, n=2)
    x = inverted_residual_block(x, 32, 3, t=6, alpha=alpha, strides=2, n=3)
    x = inverted_residual_block(x, 64, 3, t=6, alpha=alpha, strides=2, n=4)
    x = inverted_residual_block(x, 96, 3, t=6, alpha=alpha, strides=1, n=3)
    x = inverted_residual_block(x, 160, 3, t=6, alpha=alpha, strides=2, n=3)
    x = inverted_residual_block(x, 320, 3, t=6, alpha=alpha, strides=1, n=1)
    x = Conv_1D_block(x, 1280, 1, 1)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_nums, activation='softmax')(x)
    # Create the model 
    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model

def MobileNet_v3_Small(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', alpha=1.0):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    x = Conv_1D_block_2(inputs, 16, 3, strides=2, nl='HS')
    x = bottleneck_block_2(x, 16, 3, e=16, s=2, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 24, 3, e=72, s=2, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 24, 3, e=88, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=96, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 48, 5, e=120, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 48, 5, e=144, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=288, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = Conv_1D_block_2(x, 576, 1, strides=1, nl='HS')
    x = x * activations.relu(x + 3.0, max_value=6.0) / 6.0
    x = layers.Conv1D(1280, 1, padding='same')(x)
        
    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_nums, activation='softmax')(x)
    # Create the model 
    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model

def MobileNet_v3_Large(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', alpha=1.0):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    x = Conv_1D_block_2(inputs, 16, 3, strides=2, nl='HS')
    x = bottleneck_block_2(x, 16, 3, e=16, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 24, 3, e=64, s=2, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 24, 3, e=72, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=72, s=2, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=120, s=1, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=120, s=1, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 80, 5, e=240, s=2, squeeze=False, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 80, 3, e=200, s=1, squeeze=False, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 80, 3, e=184, s=1, squeeze=False, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 80, 3, e=184, s=1, squeeze=False, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 112, 3, e=480, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 112, 3, e=672, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 160, 5, e=672, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 160, 5, e=960, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 160, 5, e=960, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = Conv_1D_block_2(x, 960, 1, strides=1, nl='HS')
    x = x * activations.relu(x + 3.0, max_value=6.0) / 6.0
    x = layers.Conv1D(1280, 1, padding='same')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_nums, activation='softmax')(x)
    # Create the model 
    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model