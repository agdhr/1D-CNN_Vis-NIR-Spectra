# VGGNet model for 1D data
# This code is a 1D version of the VGGNet architecture, which is originally designed for 2D images.
# The VGGNet architecture is known for its simplicity and depth, using small convolutional filters (3x3) and a deep stack of layers.
# Author: Sakib Mahmud
# Email: sakib.mahmud@qu.edu.qa
# License: MIT
# Ref. code: https://github.com/Sakib1263/VGG-1D-2D-Tensorflow-Keras/tree/main

from tensorflow import keras
from keras.models import Sequential
from keras import layers, models
from keras import backend as K

def conv_1d_block(x, model_width, kernel):
    # 1D convolutional block without batch normalization
    x = layers.Conv1D(model_width, kernel, padding='same', kernel_initializer = 'he_normal')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #x = layers.Dropout(0.25)(x)  # Dropout layer to prevent overfitting
    return x

def VGG11(input_dim, num_channel = 1, num_filters = 16, output_nums=4):
    # Construct the VGG11 model
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Block 1
    conv1 = conv_1d_block(inputs, num_filters * (2 ** 0), kernel=3)
    if conv1.shape[1] <= 2:
        pool1 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv1)
    else:
        pool1 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv1)

    # Block 2
    conv2 = conv_1d_block(pool1, num_filters * (2 ** 1), kernel=3)
    if conv2.shape[1] <= 2:
        pool2 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv2)
    else:
        pool2 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv2)

    # Block 3
    conv3 = conv_1d_block(pool2, num_filters * (2 ** 2), kernel=3)
    conv4 = conv_1d_block(conv3, num_filters * (2 ** 2), kernel=3)
    if conv4.shape[1] <= 2:
        pool3 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv4)   
    else:
        pool3 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv4)
    
    # Block 4
    conv5 = conv_1d_block(pool3, num_filters * (2 ** 3), kernel=3)
    conv6 = conv_1d_block(conv5, num_filters * (2 ** 3), kernel=3)
    if conv6.shape[1] <= 2:
        pool4 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv6)
    else:
        pool4 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv6)
    
    # Block 5
    conv7 = conv_1d_block(pool4, num_filters * (2 ** 3), kernel=3)
    conv8 = conv_1d_block(conv7, num_filters * (2 ** 3), kernel=3)
    if conv8.shape[1] <= 2:
        pool5 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv8)
    else:
        pool5 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv8)
    
    # Fully connected layers
    flatten = layers.Flatten(name='flatten')(pool5)
    # Dense layer
    dense1 = layers.Dense(512, activation='relu')(flatten)
    dense2 = layers.Dense(512, activation='relu')(dense1)
    # Output layer
    out = layers.Dense(output_nums, activation='softmax')(dense2)
    # Create the model 
    model = models.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

def VGG13(input_dim, num_channel = 1, num_filters = 16, output_nums=4):
    # Construct the VGG11 model
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Block 1
    conv1 = conv_1d_block(inputs, num_filters * (2 ** 0), kernel=3)
    conv2 = conv_1d_block(conv1, num_filters * (2 ** 0), kernel=3)
    if conv2.shape[1] <= 2:
        pool1 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv2)
    else:
        pool1 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv2)

    # Block 2
    conv3 = conv_1d_block(pool1, num_filters * (2 ** 1), kernel=3)
    conv4 = conv_1d_block(conv3, num_filters * (2 ** 1), kernel=3)
    if conv4.shape[1] <= 2:
        pool2 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv4)
    else:
        pool2 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv4)

    # Block 3
    conv5 = conv_1d_block(pool2, num_filters * (2 ** 2), kernel=3)
    conv6 = conv_1d_block(conv5, num_filters * (2 ** 2), kernel=3)
    if conv6.shape[1] <= 2:
        pool3 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv6)   
    else:
        pool3 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv6)
    
    # Block 4
    conv7 = conv_1d_block(pool3, num_filters * (2 ** 3), kernel=3)
    conv8 = conv_1d_block(conv7, num_filters * (2 ** 3), kernel=3)
    if conv8.shape[1] <= 2:
        pool4 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv8)
    else:
        pool4 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv8)
    
    # Block 5
    conv9 = conv_1d_block(pool4, num_filters * (2 ** 3), kernel=3)
    conv10 = conv_1d_block(conv9, num_filters * (2 ** 3), kernel=3)
    if conv10.shape[1] <= 2:
        pool5 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv10)
    else:
        pool5 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv10)
    
    # Fully connected layers
    flatten = layers.Flatten(name='flatten')(pool5)
    # Dense layer   
    dense1 = layers.Dense(4096, activation='relu')(flatten)
    dense2 = layers.Dense(4096, activation='relu')(dense1)
    # Output layer
    out = layers.Dense(output_nums, activation='softmax')(dense2)
    # Create the model
    model = models.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

def VGG16(input_dim, num_channel = 1, num_filters = 16, output_dim=4):
    # Construct the VGG11 model
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Block 1
    conv1 = conv_1d_block(inputs, num_filters * (2 ** 0), kernel=3)
    conv2 = conv_1d_block(conv1, num_filters * (2 ** 0), kernel=3)
    if conv2.shape[1] <= 2:
        pool1 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv2)
    else:
        pool1 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv2)

    # Block 2
    conv3 = conv_1d_block(pool1, num_filters * (2 ** 1), kernel=3)
    conv4 = conv_1d_block(conv3, num_filters * (2 ** 1), kernel=3)
    if conv4.shape[1] <= 2:
        pool2 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv4)
    else:
        pool2 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv4)

    # Block 3
    conv5 = conv_1d_block(pool2, num_filters * (2 ** 2), kernel=3)
    conv6 = conv_1d_block(conv5, num_filters * (2 ** 2), kernel=3)
    conv7 = conv_1d_block(conv6, num_filters * (2 ** 2), kernel=3)
    if conv7.shape[1] <= 2:
        pool3 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv7)   
    else:
        pool3 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv7)
    
    # Block 4
    conv8 = conv_1d_block(pool3, num_filters * (2 ** 3), kernel=3)
    conv9 = conv_1d_block(conv8, num_filters * (2 ** 3), kernel=3)
    conv10 = conv_1d_block(conv9, num_filters * (2 ** 3), kernel=3)
    if conv10.shape[1] <= 2:
        pool4 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv10)
    else:
        pool4 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv10)
    
    # Block 5
    conv11 = conv_1d_block(pool4, num_filters * (2 ** 3), kernel=3)
    conv12 = conv_1d_block(conv11, num_filters * (2 ** 3), kernel=3)
    conv13 = conv_1d_block(conv12, num_filters * (2 ** 3), kernel=3)
    if conv13.shape[1] <= 2:
        pool5 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv13)
    else:
        pool5 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv13)

    # Fully connected layers
    flatten = layers.Flatten(name='flatten')(pool5)
    # Dense layer
    dense1 = layers.Dense(256)(flatten)
    dense1 = layers.Activation('relu')(dense1)
    dense2 = layers.Dense(256)(dense1)
    dense2 = layers.Activation('relu')(dense2)
    # Output layer
    out = layers.Dense(output_dim, activation='softmax')(dense2)
    # Create the model
    model = models.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

def VGG16_v2(input_dim, num_channel = 1, num_filters = 16, output_nums=4):
    # Construct the VGG11 model
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Block 1
    conv1 = conv_1d_block(inputs, num_filters * (2 ** 0), kernel=3)
    conv2 = conv_1d_block(conv1, num_filters * (2 ** 0), kernel=3)
    if conv2.shape[1] <= 2:
        pool1 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv2)
    else:
        pool1 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv2)

    # Block 2
    conv3 = conv_1d_block(pool1, num_filters * (2 ** 1), kernel=3)
    conv4 = conv_1d_block(conv3, num_filters * (2 ** 1), kernel=3)
    if conv4.shape[1] <= 2:
        pool2 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv4)
    else:
        pool2 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv4)

    # Block 3
    conv5 = conv_1d_block(pool2, num_filters * (2 ** 2), kernel=3)
    conv6 = conv_1d_block(conv5, num_filters * (2 ** 2), kernel=3)
    conv7 = conv_1d_block(conv6, num_filters * (2 ** 2), kernel=1)
    if conv7.shape[1] <= 2:
        pool3 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv7)   
    else:
        pool3 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv7)
    
    # Block 4
    conv8 = conv_1d_block(pool3, num_filters * (2 ** 3), kernel=3)
    conv9 = conv_1d_block(conv8, num_filters * (2 ** 3), kernel=3)
    conv10 = conv_1d_block(conv9, num_filters * (2 ** 3), kernel=1)
    if conv10.shape[1] <= 2:
        pool4 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv10)
    else:
        pool4 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv10)
    
    # Block 5
    conv11 = conv_1d_block(pool4, num_filters * (2 ** 3), kernel=3)
    conv12 = conv_1d_block(conv11, num_filters * (2 ** 3), kernel=3)
    conv13 = conv_1d_block(conv12, num_filters * (2 ** 3), kernel=1)
    if conv13.shape[1] <= 2:
        pool5 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv13)
    else:
        pool5 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv13)

    # Fully connected layers
    flatten = layers.Flatten(name='flatten')(pool5)
    # Dense layer
    dense1 = layers.Dense(4096, activation='relu')(flatten)
    dense2 = layers.Dense(4096, activation='relu')(dense1)
    # Output layer
    out = layers.Dense(output_nums, activation='softmax')(dense2)
    # Create the model
    model = models.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

def VGG19(input_dim, num_channel = 1, num_filters = 16, output_nums=4):
    # Construct the VGG11 model
    # inputs       : input vector
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    # Block 1
    conv1 = conv_1d_block(inputs, num_filters * (2 ** 0), kernel=3)
    conv2 = conv_1d_block(conv1, num_filters * (2 ** 0), kernel=3)
    if conv2.shape[1] <= 2:
        pool1 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv2)
    else:
        pool1 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv2)

    # Block 2
    conv3 = conv_1d_block(pool1, num_filters * (2 ** 1), kernel=3)
    conv4 = conv_1d_block(conv3, num_filters * (2 ** 1), kernel=3)
    if conv4.shape[1] <= 2:
        pool2 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv4)
    else:
        pool2 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv4)

    # Block 3
    conv5 = conv_1d_block(pool2, num_filters * (2 ** 2), kernel=3)
    conv6 = conv_1d_block(conv5, num_filters * (2 ** 2), kernel=3)
    conv7 = conv_1d_block(conv6, num_filters * (2 ** 2), kernel=3)
    conv8 = conv_1d_block(conv7, num_filters * (2 ** 2), kernel=3)
    if conv8.shape[1] <= 2:
        pool3 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv8)   
    else:
        pool3 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv8)
    
    # Block 4
    conv9 = conv_1d_block(pool3, num_filters * (2 ** 3), kernel=3)
    conv10 = conv_1d_block(conv9, num_filters * (2 ** 3), kernel=3)
    conv11 = conv_1d_block(conv10, num_filters * (2 ** 3), kernel=3)
    conv12 = conv_1d_block(conv11, num_filters * (2 ** 2), kernel=3)
    if conv12.shape[1] <= 2:
        pool4 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv12)
    else:
        pool4 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv12)
    
    # Block 5
    conv13 = conv_1d_block(pool4, num_filters * (2 ** 3), kernel=3)
    conv14 = conv_1d_block(conv13, num_filters * (2 ** 3), kernel=3)
    conv15 = conv_1d_block(conv14, num_filters * (2 ** 3), kernel=3)
    conv16 = conv_1d_block(conv15, num_filters * (2 ** 2), kernel=3)
    if conv16.shape[1] <= 2:
        pool5 = layers.MaxPooling1D(pool_size = 1, strides=2, padding='valid')(conv16)
    else:
        pool5 = layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid')(conv16)

    # Fully connected layers
    flatten = layers.Flatten(name='flatten')(pool5)
    # Dense layer
    dense1 = layers.Dense(4096, activation='relu')(flatten)
    dense2 = layers.Dense(4096, activation='relu')(dense1)
    # Output layer
    out = layers.Dense(output_nums, activation='softmax')(dense2)
    # Create the model
    model = models.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

def VGGNet(input_dim, output_dim): # VGGNet-16
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim, )))
    model.add(keras.layers.GaussianNoise(0.05))
    model.add(keras.layers.Reshape((input_dim, 1)))

    # Layer 1
    model.add(layers.Conv1D(1024, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    
    # Layer 2
    model.add(layers.Conv1D(1024, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    # Layer 3
    model.add(layers.Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    
    # Layer 4
    model.add(layers.Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    # Layer 5
    model.add(layers.Conv1D(256, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    
    # Layer 6
    model.add(layers.Conv1D(256, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    # Layer 7
    model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    # Layer 8
    model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    # Layer 9
    model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    # Layer 10
    model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    # Layer 11
    model.add(layers.Conv1D(96, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    # Layer 12
    model.add(layers.Conv1D(96, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    # Layer 13
    model.add(layers.Conv1D(96, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    # Layer 14
    model.add(layers.Conv1D(96, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    
    model.add(layers.Flatten())
    
    # Layer 15
    model.add(layers.Dense(96))
    model.add(layers.Activation('relu'))
    
    # Layer 16
    model.add(layers.Dense(output_dim))
    model.add(layers.Activation('softmax'))

    model.summary()
    return model

    


    


    
