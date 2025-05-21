from tensorflow import keras
from keras.models import Sequential
from keras import layers, models
from keras import backend as K

def LeNet(input_dim, output_dim):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim, )))
    model.add(keras.layers.GaussianNoise(0.05))
    model.add(keras.layers.Reshape((input_dim, 1)))

    # BLOCK 1: CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling1D(pool_size=3, strides=4))
    
    # BLOCK 2: CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling1D(pool_size=3, strides=2))

    # BLOCK 4: FC -> RELU
    model.add(layers.Flatten())
    
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    
    # BLOCK 5: FC -> RELU
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    
    # BLOCK 6: FC -> RELU
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(output_dim))
    model.add(layers.Activation('softmax'))

    model.summary()
    return model
