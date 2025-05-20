import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===========================================================================
""" LOAD AND EXTRACT DATA """
# ===========================================================================

"""LOAD DATASET"""
def load_data(data):
    dataset = pd.read_csv(data)
    return dataset

"""EXTRACT VARIABLE DATA"""
def variable_data(data):
    # --- Label
    label = data.values[:,0].astype('uint8')
    # --- Spectra data
    spectra = data.values[:,1:].astype('float')
    # --- Wavelengths
    cols = list(data.columns.values.tolist())
    wls = [float(x) for x in cols[1:]]
    return label, spectra, wls

data_path = 'd://z/master/spectra/cnn_spectrum/outputs/cleaned_data.csv'
vis_data = load_data(data_path)

label, spectra, wavelengths = variable_data(vis_data)
print(spectra.shape)

# ===========================================================================
""" DATA SPLIT """
# ===========================================================================

from sklearn.preprocessing import label_binarize
y_encoded = label_binarize(label, classes=[0,1,2,3])

print(y_encoded)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(spectra, y_encoded, test_size=0.2, random_state=42)
print('TOTAL DATA: ', spectra.shape[0])
print('Number of train data: ', x_train.shape[0])
print('Number of test data: ', x_val.shape[0])
print('Number of features/variables: ', x_train.shape[1])

# ===========================================================================
""" DATA AUGMENTATION """
# ===========================================================================

#Expand dataset, Function also available in ChemUtils
def dataaugment(x, betashift, slopeshift, multishift):
    # Calculate shift of baseline
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    #Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5
    #Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1
    x = multi*x + offset
    return x

shift = np.std(x_train)*0.01
print(shift)

# x_train is simply repeated
x_rep = np.repeat(x_train, repeats=3, axis=0)
# Make augmentation
x_aug = dataaugment(x_rep, betashift = 0.1, slopeshift = 0.1, multishift = shift)
# y_train is simply repeated
y_aug = np.repeat(y_train, repeats=3, axis=0) 

print('Number of augmented spectra: ',len(x_aug))
print('Number of augmented label: ', len(y_aug))

plt.plot(wavelengths, x_aug[0:].T)
plt.title('Plot Augmented Spectra', fontweight='bold', fontsize=12, fontname="Segoe UI")
plt.ylabel('Reflectance (%)', fontsize=12, fontname="Segoe UI")
plt.xlabel('Wavelength (nm)', fontsize=12, fontname="Segoe UI")
plt.show()


# ===========================================================================
""" DATA SCALING """
# ===========================================================================

from nirs_utils import standardscaler
x_train = standardscaler(x_aug)
x_val = standardscaler(x_val)
_ = plt.plot(wavelengths, x_train.T)
print(x_train.shape)

# ===========================================================================
""" MODEL TRAINING """
# ===========================================================================

from tensorflow import keras
from models import AlexNet, LeNet
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

input_dim = x_train.shape[1]
output_dim = y_aug.shape[1]

learning_rate = 0.0001
batch_size = 128
epoch = 100
seed = 7
np.random.seed(seed)
    
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
    
for train, val in kfold.split(x_train, y_aug):
    print(x_train[train].shape[0])
    print(x_train[val].shape[0])

    model = LeNet(input_dim, output_dim)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=learning_rate), 
                  metrics=['accuracy'])
    es = EarlyStopping(monitor='accuracy',
                                     min_delta=0,
                                     mode='max', 
                                     verbose=0, 
                                     baseline=None,
                                     patience=5, 
                                     restore_best_weights=True)
        
    history = model.fit(x_train[train], y_aug[train],
                         epochs = epoch,
                         batch_size = 1,
                         validation_data = (x_train[val], y_aug[val]), 
                         #callbacks = [es],
                         verbose=1)
        
scores = model.evaluate(x_train[val], y_aug[val], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))