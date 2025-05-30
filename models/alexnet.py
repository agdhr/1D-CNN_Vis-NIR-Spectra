from tensorflow import keras
from keras.models import Sequential
from keras import layers, models
from keras import backend as K

def AlexNet(input_dim, output_dim, num_filters=16):
    # Initialize the model along with the input shape to be "channel last"
    # and the channels dimension itself
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim, )))
    model.add(layers.GaussianNoise(0.05))
    model.add(layers.Reshape((input_dim, 1)))

    # BLOCK 1: CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters= num_filters * (2 ** 0), kernel_size=11, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=3, strides=2))
    #model.add(layers.Dropout(0.5))

    # BLOCK 2: CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters=num_filters * (2 ** 1), kernel_size=5, strides=2, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=3, strides=2))
    #model.add(layers.Dropout(0.5))
    
    # BLOCK 3: CONV -> RELU -> CONV -> RELU -> CONV -> RELU
    model.add(layers.Conv1D(filters= num_filters * (2 ** 2), kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.Conv1D(filters= num_filters * (2 ** 3), kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.Conv1D(filters=num_filters * (2 ** 4), kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPooling1D(pool_size=3, strides=2))
    #model.add(layers.Dropout(0.5))
    
    # BLOCK 4: FC -> RELU
    model.add(layers.Flatten())
    
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.5))

    # BLOCK 5: FC -> RELU
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.5))

    # BLOCK 6: FC -> RELU
#    model.add(layers.Dense(512))
#    model.add(layers.Activation('relu'))
#    model.add(layers.BatchNormalization())
#    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(output_dim))
    model.add(layers.Activation('softmax'))

    model.summary()
    return model

