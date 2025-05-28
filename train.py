import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from nirs_utils import load_csv
from nirs_utils import standardscaler, minmaxscaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

from datetime import datetime
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from models.alexnet import AlexNet
from models.lenet import LeNet
from models.vggnet import VGG16
from models.densenet import DenseNet121
from models.resnet import ResNet18
from models.inception import Inception_v1
from models.mobilenet import MobileNet_v1
# =============================================================================================
""" LOAD AND EXTRACT DATA """
# =============================================================================================
# Load and extract data from CSV file
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

# Set model name

model_name = 'densenet'  # Change this to any of the model names: 'LeNet', 'AlexNet', 'VGG11', 'DenseNet121', 'ResNet18', 'Inception_v1', 'MobileNet_v1'
# Set the relevant paths
output_path = 'd://z/master/spectra/cnn_spectrum/outputs/'+ model_name + '/'
data_path = 'd://z/master/spectra/cnn_spectrum/data/'
# ---------------------------------------------------------------------------------------------------
# load data files in the paths
data_name = 'data_ori'  # Change this to any of the loaded data variables
data = load_csv(data_path + 'cleaned_data.csv') # Change this to any of the loaded data variables
# ---------------------------------------------------------------------------------------------------
# Extract variable data (label, spectra, and wavelengths) from the loaded data
label, spectra, wavelengths = variable_data(data)
print(spectra.shape)
print('TOTAL DATA: ', spectra.shape[0])
print('TOTAL VARIABLES: ', spectra.shape[1])
#---------------------------------------------------------------------------------------------------
# Define the model
model = AlexNet if model_name == 'alexnet' else \
        LeNet if model_name == 'lenet' else \
        VGG16 if model_name == 'vgg' else \
        DenseNet121 if model_name == 'densenet' else \
        ResNet18 if model_name == 'resnet' else \
        Inception_v1 if model_name == 'inception' else \
        MobileNet_v1 if model_name == 'mobilenet' else None
if model is None:
    raise ValueError("Invalid model name. Choose from 'alexnet', 'lenet', 'vgg', 'densenet', 'resnet', 'inception', or 'mobilenet'.")

# Define other parameters
learning_rate = 0.001
batch_size = 16
epoch = 100
seed = 7
np.random.seed(seed)
#----------------------------------------------------------------------------------------------------

# ===================================================================================================
""" DATA SPLIT """
# ===================================================================================================
# Binarize the output
y_encoded = label_binarize(label, classes=[0,1,2,3])

# Split the data into training and validation sets: 80% training and 20% validation
x_train, x_test, y_train, y_test = train_test_split(spectra, y_encoded, test_size=0.2, random_state=42)

# ===================================================================================================
""" DATA AUGMENTATION """
# ===================================================================================================

def dataaugment(x, betashift, slopeshift, multishift):
    # Calculate shift of baseline
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    # Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    # Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5
    # Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1
    x = multi*x + offset
    return x

shift = np.std(x_train)*0.01

# Data Augmentation a single spectrum
X_ = x_train[0:1]
X_ = np.repeat(X_, repeats=5, axis=0)       # Repeating the spectrum 3x
X_aug = dataaugment(X_, betashift = 0.1, slopeshift = 0.1,multishift = shift)
plt.plot(wavelengths, X_aug.T, lw=5, label='augmented spectrum')
plt.plot(wavelengths, X_.T, lw=1, c='red', label = 'original spectrum')
plt.ylabel('Reflectance (%)', fontsize=12, fontname="Segoe UI")
plt.xlabel('Wavelength (nm)', fontsize=12, fontname="Segoe UI")
plt.title('Plot Sample Augmented Spectra', fontweight='bold', fontsize=12, fontname="Segoe UI")
plt.legend(loc='best')
plt.savefig(f'{output_path}Plot_sample_augmented_spectra.png')
plt.show()

# Data Augmentation for all spectra
## x_train is simply repeated
x_rep = np.repeat(x_train, repeats=5, axis=0)
## Make augmentation
x_aug = dataaugment(x_rep, betashift = 0.1, slopeshift = 0.1, multishift = shift)
## y_train is simply repeated
y_aug = np.repeat(y_train, repeats=5, axis=0) 
print('Number of augmented spectra: ',len(x_aug))
print('Number of augmented label: ', len(y_aug))


x_train, x_val, y_train, y_val = train_test_split(x_aug, y_aug, test_size=0.5, random_state=42)

print('Number of train data: ', x_train.shape[0])
print('Number of validation data: ', x_val.shape[0])
print('Number of test data: ', x_test.shape[0])

# ===================================================================================================
""" DATA SCALING """
# ===================================================================================================
# Standardize the data
x_train = minmaxscaler(x_train)
x_test = minmaxscaler(x_test)
x_val = minmaxscaler(x_val)

_ = plt.plot(wavelengths, x_train[0:1].T)
plt.title('Plot Standardized Spectra', fontweight='bold', fontsize=12, fontname="Segoe UI")
plt.ylabel('Reflectance (%)', fontsize=12, fontname="Segoe UI")
plt.xlabel('Wavelength (nm)', fontsize=12, fontname="Segoe UI")
plt.legend(loc='best')
plt.grid(False)
plt.savefig(f'{output_path}Plot_standardized_spectra.png')
plt.show()

# ===================================================================================================
""" MODEL TRAINING """
# ===================================================================================================

# Input and output dimensions
input_dim = x_train.shape[1]
output_dim = y_aug.shape[1]
print('\nInput dimension of the model: ', input_dim)
print('\nOutput dimension of the model: ', output_dim)

model = model(input_dim, output_dim = 4, num_filters=16)


cvscores = []

print('\nK-fold cross validation: ')
print('----------------------------------')

# Gets the current date and time to start compiling model
start1 = datetime.now()
    
# Compile the model
model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=learning_rate), 
                  metrics=['accuracy'])
# Early stopping
es = EarlyStopping(monitor='accuracy',
                       min_delta=0,
                       mode='max', 
                       verbose=0, 
                       baseline=None,
                       patience=5, 
                       restore_best_weights=True)
    
# Fit the model
print('\nFitting model ....')
history = model.fit(x_train, y_train,
                         epochs = epoch,
                         batch_size = batch_size,
                         validation_data = (x_val, y_val),
                         shuffle = False, 
                         #callbacks = [es],
                         verbose=1)
    
# Save the model
model.save(f'{output_path}{model_name}_{data_name}_model.keras')
model.save_weights(f'{output_path}{model_name}_{data_name}.model.weights.h5')

# Gets the current date and time while stopping model compilation.
stop1 = datetime.now()

# Execution time of the model
execution_time = stop1 - start1
print("Model execution time is: ", execution_time)
        
scores = model.evaluate(x_val, y_val, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print('----------------------------------')
# Convert the history dict to a pandas dataframe and save as csv for future plotting
history_df = pd.DataFrame(history.history)

with open(f'{output_path}{model_name}_{data_name}_history_df.csv', mode='w') as f:
    history_df.to_csv(f)

# Plot the training and validation accuracy and loss at each epoch

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='best')
plt.savefig(f'{output_path}{model_name}_{data_name}_loss.png')
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.yscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
#plt.ylim([0,1])
plt.legend(loc='lower right')
plt.savefig(f'{output_path}{model_name}_{data_name}_accuracy.png')
plt.show()

# Save the test data
acc_time = [['mean_acc', np.mean(cvscores)], 
            ['training_time', execution_time]]
acc_time_df = pd.DataFrame(acc_time, columns=['params.', 'values'])
acc_time_df.to_csv(f'{output_path}{model_name}_{data_name}_acc_time.csv', index=False)

# ===================================================================================================
""" MAKE PREDICTION USING TEST DATA """
# ===================================================================================================

# Load the model at a time for testing
#model_path = f'{output_path}{model_name}_model.keras'
#model = keras.models.load_model(model_path, compile=False)

# Predict the test data
y_pred = model.predict(x_test)

# Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

# Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

# Calculate accuracy score
a = accuracy_score(pred,test)
print('The accuracy of test data is:', a*100)

# Calculate confusion matrix
label_name = ['Gayo', 'Kintamani', 'Temanggung', 'Toraja']

# Create confusion matrix
cm_ = confusion_matrix(pred, test)
print(cm_)

# Plot confusion matrix
display = ConfusionMatrixDisplay(cm_,display_labels=label_name)
display.plot()
display.figure_.suptitle("Confusion Matrix of Test Data")
plt.gca().invert_yaxis()
plt.grid(False)
plt.show()

# Save confusion matrix
cm = pd.DataFrame(cm_, index=label_name, columns=label_name)
cm.to_csv(f'{output_path}{model_name}_{data_name}_conmat.csv')

# Plot confusion matrix using seaborn
plt.figure(figsize=(10,7))
sns.heatmap(cm_, annot=True, fmt='d', cmap='Blues', xticklabels=label_name, yticklabels=label_name)
plt.title('Confusion Matrix of Training Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f'{output_path}{model_name}_{data_name}_conmat.png')
plt.show()

# Save the prediction
pred_df = pd.DataFrame(pred, columns=['Predicted'])
pred_df.to_csv(f'{output_path}{model_name}_{data_name}_predictions.csv', index=False)

# Save the test data
test_df = pd.DataFrame(test, columns=['True'])
test_df.to_csv(f'{output_path}{model_name}_{data_name}_test.csv', index=False)

# Show performance metrics
print(classification_report(test, pred, target_names=label_name))

# Save classification report
report = classification_report(test, pred, target_names=label_name, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f'{output_path}{model_name}_{data_name}_classification_report.csv')

# ROC-AUC curve
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(output_dim):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i], )
    roc_auc[i] = auc(fpr[i], tpr[i])

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot()
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
plt.plot(fpr[0], tpr[0], color = 'darkorange', lw = 1, label = 'Gayo (AUC = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color = 'navy', lw = 1, label = 'Kintamani (AUC = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], color = 'deeppink', lw = 1, label = 'Temanggung (AUC = %0.2f)' % roc_auc[2])
plt.plot(fpr[3], tpr[3], color = 'green', lw = 1, label = 'Toraja (AUC = %0.2f)' % roc_auc[3])
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.grid(False)
plt.savefig(f'{output_path}{model_name}_{data_name}_roc_curve.png')
plt.show()

# Save ROC curve data
roc_df = pd.DataFrame({'fpr': fpr[0], 'tpr': tpr[0]})
roc_df.to_csv(f'{output_path}{model_name}_{data_name}_roc_curve.csv', index=False)

# Save ROC curve data for all classes
for i in range(output_dim):
    roc_df = pd.DataFrame({'fpr': fpr[i], 'tpr': tpr[i]})
    roc_df.to_csv(f'{output_path}{model_name}_{data_name}_roc_curve_{i}.csv', index=False)