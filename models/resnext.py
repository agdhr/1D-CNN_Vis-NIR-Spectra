"""
ResNeXt models for Keras.
# Reference - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf))

"""
# Author: Sakib Mahmud
# Email: sakib.mahmud@qu.edu.qa
# License: MIT
# Ref. code: https://github.com/Sakib1263/TF-1D-2D-ResNetV1-2-SEResNet-ResNeXt-SEResNeXt 

import tensorflow as tf
from keras import layers, models, activations
from keras import backend as K

def Conv_1D_Block(x, model_width, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def stem_bottleneck(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
    return pool

def conv_block(inputs, num_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv_1D_Block(inputs, num_filters, 3, 2)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    return conv

def grouped_convolution_block(inputs, num_filters, kernel_size, strides, cardinality):
    # Adds a grouped convolution block
    group_list = []
    grouped_channels = int(num_filters / cardinality)

    if cardinality == 1:
        # When cardinality is 1, it is just a standard convolution
        x = Conv_1D_Block(inputs, num_filters, 1, strides=strides)
        x = Conv_1D_Block(x, grouped_channels, kernel_size, strides)
        return x

    for c in range(cardinality):
        x = tf.keras.layers.Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels])(inputs)
        x = Conv_1D_Block(x, num_filters, 1, strides=strides)
        x = Conv_1D_Block(x, grouped_channels, kernel_size, strides=strides)
        group_list.append(x)

    group_merge = tf.keras.layers.concatenate(group_list, axis=-1)
    x = tf.keras.layers.BatchNormalization()(group_merge)
    x = tf.keras.layers.Activation('relu')(x)

    return x

def residual_block_bottleneck(inputs, num_filters, cardinality):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = Conv_1D_Block(inputs, num_filters * 2, 1, 1)
    
    x = grouped_convolution_block(inputs, num_filters, 3, 1, cardinality)
    x = Conv_1D_Block(x, num_filters * 2, 1, 1)
    
    conv = tf.keras.layers.Add()([x, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out

def residual_group_bottleneck(inputs, num_filters, n_blocks, cardinality, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for i in range(n_blocks):
        out = residual_block_bottleneck(out, num_filters, cardinality)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)

    return out

def learner18(inputs, num_filters, cardinality):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 2, cardinality)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 1, cardinality)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 1, cardinality)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 1, cardinality, False)  # Fourth Residual Block Group of 512 filters
    return out

def learner34(inputs, num_filters, cardinality):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3, cardinality)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3, cardinality)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 5, cardinality)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, cardinality, False)  # Fourth Residual Block Group of 512 filters
    return out

def learner50(inputs, num_filters, cardinality):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3, cardinality)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3, cardinality)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 5, cardinality)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, cardinality, False)  # Fourth Residual Block Group of 512 filters
    return out

def learner101(inputs, num_filters, cardinality):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3, cardinality)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3, cardinality)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 22, cardinality)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, cardinality, False)  # Fourth Residual Block Group of 512 filters
    return out

def learner152(inputs, num_filters, cardinality):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3, cardinality)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 7, cardinality)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 35, cardinality)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, cardinality, False)  # Fourth Residual Block Group of 512 filters
    return out

def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)
    return out

def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs       : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)
    return out

def ResNeXt18(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', cardinality=8):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_ = stem_bottleneck(inputs, num_filters)  # The Stem Convolution Group
    x = learner18(stem_, num_filters, cardinality)  # The learner
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

def ResNeXt34(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', cardinality=8):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_ = stem_bottleneck(inputs, num_filters)  # The Stem Convolution Group
    x = learner34(stem_, num_filters, cardinality)  # The learner
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

def ResNeXt50(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', cardinality=8):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_b = stem_bottleneck(inputs, num_filters)  # The Stem Convolution Group
    x = learner50(stem_b, num_filters, cardinality)  # The learner
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

def ResNeXt101(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', cardinality=8):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_b = stem_bottleneck(inputs, num_filters)  # The Stem Convolution Group
    x = learner101(stem_b, num_filters, cardinality)  # The learner
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

def ResNeXt152(input_dim, num_channel = 1, num_filters = 64, output_nums=4, pooling = 'avg', cardinality=8):
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Blocks
    stem_b = stem_bottleneck(inputs, num_filters)  # The Stem Convolution Group
    x = learner152(stem_b, num_filters, cardinality)  # The learner
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