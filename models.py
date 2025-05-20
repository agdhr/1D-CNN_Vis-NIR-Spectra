from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras import layers, models
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from scikeras.wrappers import KerasClassifier
from math import floor
from sklearn.metrics import make_scorer, accuracy_score, ConfusionMatrixDisplay, classification_report, auc, confusion_matrix

from bayes_opt.bayesian_optimization import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from keras import callbacks
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option("display.max_columns", None)

def AlexNet(input_dim, output_dim):
    # Initialize the model along with the input shape to be "channel last"
    # and the channels dimension itself
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim, )))
    model.add(keras.layers.GaussianNoise(0.05))
    model.add(keras.layers.Reshape((input_dim, 1)))

    # BLOCK 1: CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters=1024, kernel_size=3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=3, strides=4))
    #model.add(layers.Dropout(0.25))

    # BLOCK 2: CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=3, strides=2))
    #model.add(layers.Dropout(0.25))
    
    # BLOCK 3: CONV -> RELU -> CONV -> RELU -> CONV -> RELU
    model.add(layers.Conv1D(filters=256, kernel_size=2, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.Conv1D(filters=256, kernel_size=2, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPooling1D(pool_size=3, strides=2))
    #model.add(layers.Dropout(0.25))
    
    # BLOCK 4: FC -> RELU
    model.add(layers.Flatten())
    
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.5))

    # BLOCK 5: FC -> RELU
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.5))

    # BLOCK 6: FC -> RELU
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.5))

    model.add(layers.Dense(output_dim))
    model.add(layers.Activation('softmax'))

    model.summary()
    return model

def LeNet(input_dim, output_dim):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim, )))
    model.add(keras.layers.GaussianNoise(0.05))
    model.add(keras.layers.Reshape((input_dim, 1)))

    # BLOCK 1: CONV -> RELU -> POOL 
    model.add(layers.Conv1D(filters=1024, kernel_size=3, strides=4, padding='same'))
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

def VGGNet(input_dim, output_dim):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim, )))
    model.add(keras.layers.GaussianNoise(0.05))
    model.add(keras.layers.Reshape((input_dim, 1)))


    
