"""
Name:           DenseNet

Description:    An implementation of DenseNet for 1D inputs in Keras'

Reference:      Densely Connected Convolutional Networks 
                [https://arxiv.org/abs/1608.06993]

Code:           https://github.com/Sakib1263/DenseNet-1D-2D-Tensorflow-Keras/tree/main
"""
from tensorflow import keras
from keras import layers, models
def conv_1d_block(x, model_width, kernel, strides):
    # 1D convolutional block without batch normalization
    x = layers.Conv1D(model_width, kernel, strides, padding='same', kernel_initializer = 'he_normal')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    conv = conv_1d_block(inputs, num_filters, kernel=7, strides=2)
    if conv.shape[1] <= 2:
        pool = layers.MaxPooling1D(pool_size = 1, strides=2, padding='same')(conv)
    else:
        pool = layers.MaxPooling1D(pool_size = 3, strides=2, padding='same')(conv)
    return pool

def conv_block(x, num_filters, bottleneck=True):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    if bottleneck:
        num_filters_bottleneck = num_filters * 4
        x = conv_1d_block(x, num_filters, kernel=1, strides=1)
    out = conv_1d_block(x, num_filters, kernel=3, strides=1)
    return out

def dense_block(x, num_filters, num_layers, bottleneck=True):
    for i in range(num_layers):
        cb = conv_block(x, num_filters, bottleneck=True)
        x = layers.concatenate([x, cb], axis=-1)
    return x

def transition_block(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = layers.Dense(class_number, activation='softmax')(inputs)
    return out

def DenseNet121(input_dim, num_channel = 1, num_filters = 16, output_nums=4, pooling='avg', bottleneck=True):
    inputs = layers.Input(shape=(input_dim, num_channel))    
    stem_block = stem(inputs, num_filters)
    Dense_Block_1 = dense_block(stem_block, num_filters * 2, 6, bottleneck=bottleneck)
    Transition_Block_1 = transition_block(Dense_Block_1, num_filters)
    Dense_Block_2 = dense_block(Transition_Block_1, num_filters * 4, 12, bottleneck=bottleneck)
    Transition_Block_2 = transition_block(Dense_Block_2, num_filters)
    Dense_Block_3 = dense_block(Transition_Block_2, num_filters * 8, 24, bottleneck=bottleneck)
    Transition_Block_3 = transition_block(Dense_Block_3, num_filters)
    Dense_Block_4 = dense_block(Transition_Block_3, num_filters * 16, 16, bottleneck=bottleneck)
    if pooling == 'avg':
        out = layers.GlobalAveragePooling1D()(Dense_Block_4)
    elif pooling == 'max':
        out = layers.GlobalMaxPooling1D()(Dense_Block_4)
    # Final dense outputting layer for the outputs
    out = layers.Flatten(name='flatten')(out)
    outputs = layers.Dense(output_nums, activation='softmax')(out)
    # Instantiate the Model
    model = models.Model(inputs, outputs)
    model.summary()
    return model

def DenseNet161(input_dim, num_channel = 1, num_filters = 16, output_nums=4, pooling='avg', bottleneck=True):
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    stem_block = stem(inputs, num_filters)  # The Stem Convolution Group
    Dense_Block_1 = dense_block(stem_block, num_filters * 2, 6, bottleneck=bottleneck)
    Transition_Block_1 = transition_block(Dense_Block_1, num_filters * 2)
    Dense_Block_2 = dense_block(Transition_Block_1, num_filters * 4, 12, bottleneck=bottleneck)
    Transition_Block_2 = transition_block(Dense_Block_2, num_filters * 4)
    Dense_Block_3 = dense_block(Transition_Block_2, num_filters * 8, 36, bottleneck=bottleneck)
    Transition_Block_3 = transition_block(Dense_Block_3, num_filters * 8)
    Dense_Block_4 = dense_block(Transition_Block_3, num_filters * 16, 24, bottleneck=bottleneck)
    if pooling == 'avg':
        out = layers.GlobalAveragePooling1D()(Dense_Block_4)
    elif pooling == 'max':
        out = layers.GlobalMaxPooling1D()(Dense_Block_4)
    # Final dense outputting layer for the outputs
    out = layers.Flatten(name='flatten')(out)
    outputs = layers.Dense(output_nums, activation='softmax')(out)
    # Instantiate the Model
    model = models.Model(inputs, outputs)
    model.summary()
    return model

def DenseNet169(input_dim, num_channel = 1, num_filters = 16, output_nums=4, pooling='avg', bottleneck=True):
    inputs = layers.Input(shape=(input_dim,num_channel))  # The input tensor
    stem_block = stem(inputs, num_filters)  # The Stem Convolution Group
    Dense_Block_1 = dense_block(stem_block, num_filters * 2, 6, bottleneck=bottleneck)
    Transition_Block_1 = transition_block(Dense_Block_1, num_filters * 2)
    Dense_Block_2 = dense_block(Transition_Block_1, num_filters * 4, 12, bottleneck=bottleneck)
    Transition_Block_2 = transition_block(Dense_Block_2, num_filters * 4)
    Dense_Block_3 = dense_block(Transition_Block_2, num_filters * 8, 32, bottleneck=bottleneck)
    Transition_Block_3 = transition_block(Dense_Block_3, num_filters * 8)
    Dense_Block_4 = dense_block(Transition_Block_3, num_filters * 16, 32, bottleneck=bottleneck)
    if pooling == 'avg':
        out = layers.GlobalAveragePooling1D()(Dense_Block_4)
    elif pooling == 'max':
        out = layers.GlobalMaxPooling1D()(Dense_Block_4)
    # Final dense outputting layer for the outputs
    out = layers.Flatten(name='flatten')(out)
    outputs = layers.Dense(output_nums, activation='softmax')(out)
    # Instantiate the Model
    model = models.Model(inputs, outputs)
    model.summary()
    return model

def DenseNet201(input_dim, num_channel = 1, num_filters = 16, output_nums=4, pooling='avg', bottleneck=True):
    inputs = layers.Input(shape=(input_dim, num_channel))  # The input tensor
    stem_block = stem(inputs, num_filters)  # The Stem Convolution Group
    Dense_Block_1 = dense_block(stem_block, num_filters * 2, 6, bottleneck=bottleneck)
    Transition_Block_1 = transition_block(Dense_Block_1, num_filters)
    Dense_Block_2 = dense_block(Transition_Block_1, num_filters * 4, 12, bottleneck=bottleneck)
    Transition_Block_2 = transition_block(Dense_Block_2, num_filters)
    Dense_Block_3 = dense_block(Transition_Block_2, num_filters * 8, 48, bottleneck=bottleneck)
    Transition_Block_3 = transition_block(Dense_Block_3, num_filters)
    Dense_Block_4 = dense_block(Transition_Block_3, num_filters * 16, 32, bottleneck=bottleneck)
    if pooling == 'avg':
        out = layers.GlobalAveragePooling1D()(Dense_Block_4)
    elif pooling == 'max':
        out = layers.GlobalMaxPooling1D()(Dense_Block_4)
    # Final dense outputting layer for the outputs
    out = layers.Flatten(name='flatten')(out)
    outputs = layers.Dense(output_nums, activation='softmax')(out)
    # Instantiate the Model
    model = models.Model(inputs, outputs)
    model.summary()
    return model

def DenseNet264(input_dim, num_channel = 1, num_filters = 16, output_nums=4, pooling='avg', bottleneck=True):
    inputs = layers.Input(shape=(input_dim,num_channel))  # The input tensor
    stem_block = stem(inputs, num_filters)  # The Stem Convolution Group
    Dense_Block_1 = dense_block(stem_block, num_filters * 2, 6, bottleneck=bottleneck)
    Transition_Block_1 = transition_block(Dense_Block_1, num_filters * 2)
    Dense_Block_2 = dense_block(Transition_Block_1, num_filters * 4, 12, bottleneck=bottleneck)
    Transition_Block_2 = transition_block(Dense_Block_2, num_filters * 4)
    Dense_Block_3 = dense_block(Transition_Block_2, num_filters * 8, 64, bottleneck=bottleneck)
    Transition_Block_3 = transition_block(Dense_Block_3, num_filters * 8)
    Dense_Block_4 = dense_block(Transition_Block_3, num_filters * 16, 48, bottleneck=bottleneck)
    if pooling == 'avg':
        out = layers.GlobalAveragePooling1D()(Dense_Block_4)
    elif pooling == 'max':
        out = layers.GlobalMaxPooling1D()(Dense_Block_4)
    # Final dense outputting layer for the outputs
    out = layers.Flatten(name='flatten')(out)
    outputs = layers.Dense(output_nums, activation='softmax')(out)
    # Instantiate the Model
    model = models.Model(inputs, outputs)
    model.summary()
    return model