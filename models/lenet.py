from tensorflow import keras
from keras.models import Sequential
from keras import layers, models
from keras import backend as K

def LeNet(input_dim, output_dim, num_filters=16):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim, )))
    model.add(keras.layers.GaussianNoise(0.05))
    model.add(keras.layers.Reshape((input_dim, 1)))

    # CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters=num_filters *(2 ** 0), kernel_size=5, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling1D(pool_size=2, strides=4))
    #model.add(layers.Dropout(0.5))
    
    # CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters=num_filters * (2 ** 1), kernel_size=5, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling1D(pool_size=2, strides=2))
    #model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    # FC -> RELU
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.5))
    
    # FC -> RELU
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.5))
    
    # FC -> SOFTMAX
    model.add(layers.Dense(output_dim))
    model.add(layers.Activation('softmax'))

    model.summary()
    return model
