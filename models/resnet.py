"""
ResNet models for Keras.
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf))

"""
# Author: Sakib Mahmud
# Email: sakib.mahmud@qu.edu.qa
# License: MIT
# Ref. code: https://github.com/Sakib1263/TF-1D-2D-ResNetV1-2-SEResNet-ResNeXt-SEResNeXt 

import tensorflow as tf
from keras import layers, models, activations
from keras import backend as K

def conv_1D_block(inputs, model_width, kernel, strides):
    # 1D convolutional block without batch normalization
    x = layers.Conv1D(model_width, kernel, strides, padding='same', kernel_initializer = 'he_normal')(inputs)   
    # x = keras.layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = conv_1D_block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
    return pool

def conv_block(inputs, num_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = conv_1D_block(inputs, num_filters, 3, 2)
    conv = conv_1D_block(conv, num_filters, 3, 1)
    return conv

def residual_block(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = inputs
    #
    conv = conv_1D_block(inputs, num_filters, 3, 1)
    conv = conv_1D_block(conv, num_filters, 3, 1)
    conv = layers.Add()([conv, shortcut])
    out = layers.Activation('relu')(conv)
    return out

def residual_group(inputs, num_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for i in range(n_blocks):
        out = residual_block(out, num_filters)
    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)
    return out

def stem_bottleneck(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs :input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = conv_1D_block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
    return pool

def residual_block_bottleneck(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = conv_1D_block(inputs, num_filters * 4, 1, 1)
    conv = conv_1D_block(inputs, num_filters, 1, 1)
    conv = conv_1D_block(conv, num_filters, 3, 1)
    conv = conv_1D_block(conv, num_filters * 4, 1, 1)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)
    return out

def residual_group_bottleneck(inputs, num_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for i in range(n_blocks):
        out = residual_block_bottleneck(out, num_filters)
    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)
    return out

def learner18(inputs, num_filters):
    # Construct the Learner
    x = residual_group(inputs, num_filters, 2)          # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, 1)           # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, 1)           # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, 1, False)  # Fourth Residual Block Group of 512 filters
    return out

def learner34(inputs, num_filters):
    # Construct the Learner
    x = residual_group(inputs, num_filters, 3)          # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, 3)           # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, 5)           # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters    
    return out

def learner50(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 5)   # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters
    return out

def learner101(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 22)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters
    return out

def learner152(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 7)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 35)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters
    return out

def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = layers.Dense(class_number, activation='softmax')(inputs)
    return out

def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs       : input vector
    # feature_number : number of output features
    out = layers.Dense(feature_number, activation='linear')(inputs)
    return out


def ResNet18(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg'):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_ = stem(inputs, num_filters)  # Stem Convolution Group
    x = learner18(stem_, num_filters)  # Learner
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

def ResNet34(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg'):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_ = stem(inputs, num_filters)  # Stem Convolution Group
    x = learner34(stem_, num_filters)  # Learner
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

def ResNet50(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg'):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_ = stem_bottleneck(inputs, num_filters)  # Stem Convolution Group
    x = learner50(stem_, num_filters)  # Learner
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

def ResNet101(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg'):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_ = stem_bottleneck(inputs, num_filters)  # Stem Convolution Group
    x = learner101(stem_, num_filters)  # Learner
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

def ResNet152(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg'):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_ = stem_bottleneck(inputs, num_filters)  # Stem Convolution Group
    x = learner152(stem_, num_filters)  # Learner
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