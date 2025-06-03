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
from keras.optimizers import Adam, SGD, RMSprop, Adagrad

from keras import models, layers
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
model_name = 'lenet'  
# Set label names
label_name = ['Gayo', 'Kintamani', 'Temanggung', 'Toraja']
# Set the relevant paths
output_path = 'd://z/master/spectra/cnn_spectrum/BayesianOpt/'+ model_name + '/'
data_path = 'd://z/master/spectra/cnn_spectrum/data/'
# ----------------------------------------------------------------------------------------------
# load data files in the paths
data = load_csv(data_path + 'cleaned_data.csv') # Change this to any of the loaded data variables
# ----------------------------------------------------------------------------------------------
# Extract variable data (label, spectra, and wavelengths) from the loaded data
label, spectra, wavelengths = variable_data(data)
print('TOTAL DATA: ', spectra.shape[0])
print('TOTAL VARIABLES: ', spectra.shape[1])
#----------------------------------------------------------------------------------------------

# =============================================================================================
""" DATA SPLIT """
# =============================================================================================
# Binarize the output
y_encoded = label_binarize(label, classes=[0,1,2,3])
# Extract output dimension
output_dim = y_encoded.shape[1]
# Split the data into training and test sets: 80% training and 20% test
x_train, x_test, y_train, y_test = train_test_split(spectra, y_encoded, test_size=0.2, random_state=42)

# =============================================================================================
""" DATA AUGMENTATION """
# =============================================================================================
# Function to perform data augmentation
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

# Calculate the shift value based on the standard deviation of the training data
shift = np.std(x_train)*0.01

# Data Augmentation a single spectrum
X_ = x_train[0:1]
X_ = np.repeat(X_, repeats=5, axis=0)       # Repeating the spectrum 5x
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
print('Number of train data (before augmentation): ', x_train.shape[0])

# Split the augmented data into training and validation sets: 50% training and 50% validation
x_train, x_val, y_train, y_val = train_test_split(x_aug, y_aug, test_size=0.5, random_state=42)
print('Number of train data (after augmentation): ', x_train.shape[0])
print('Number of validation data (after augmentation): ', x_val.shape[0])
print('Number of test data: ', x_test.shape[0])

# =============================================================================================
""" DATA SCALING """
# =============================================================================================
# Standardize the data
x_train = minmaxscaler(x_train)
x_test = minmaxscaler(x_test)
x_val = minmaxscaler(x_val)
# Plot the standardized spectra
plt.figure(figsize=(10, 6))
_ = plt.plot(wavelengths, x_train[0:1].T)
plt.title('Plot Standardized Spectra', fontweight='bold', fontsize=12, fontname="Segoe UI")
plt.ylabel('Reflectance (%)', fontsize=12, fontname="Segoe UI")
plt.xlabel('Wavelength (nm)', fontsize=12, fontname="Segoe UI")
plt.legend(loc='best')
plt.grid(False)
plt.savefig(f'{output_path}Plot_standardized_spectra.png')
plt.show()

# =============================================================================================
""" MODEL DEFINITION """    
# =============================================================================================
# Define the model
def lenet_model(input_dim, output_dim,
                num_filters1, num_filters2,
                filter_size1, filter_size2,
                conv_stride1, conv_stride2,
                pool_size1, pool_size2,
                pool_stride1, pool_stride2,
                dense_units1, dense_units2,
                optimizer, 
                learning_rate):
    """ Define the LeNet model for Bayesian Optimization """
    model = keras.Sequential()
    # Input layer with Gaussian noise for regularization
    model.add(layers.Input(shape=(input_dim, )))
    # Reshape the input to be compatible with Conv1D
    model.add(layers.GaussianNoise(0.05))
    # Reshape the input to have a single channel
    model.add(layers.Reshape((input_dim, 1)))

    """ Define the architecture of the LeNet model """
    # CONV -> RELU -> POOL
    model.add(layers.Conv1D(filters=num_filters1, 
                            kernel_size=filter_size1, 
                            strides=conv_stride1, 
                            padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling1D(pool_size=pool_size1, 
                                      strides=pool_stride1))
    # CONV -> RELU -> POOL
    model.add(layers.Conv1D(filters=num_filters2, 
                            kernel_size=filter_size2, 
                            strides=conv_stride2, 
                            padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling1D(pool_size=pool_size2, 
                                      strides=pool_stride2))
    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())
    # FC -> RELU
    model.add(layers.Dense(dense_units1))
    model.add(layers.Activation('relu'))
    # FC -> RELU
    model.add(layers.Dense(dense_units2))
    model.add(layers.Activation('relu'))
    # FC -> SOFTMAX
    model.add(layers.Dense(output_dim))
    model.add(layers.Activation('softmax'))
    
    """ Compile the model """
    # Compile the model with the specified optimizer and learning rate
    model.compile(optimizer=optimizer(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# =============================================================================================
""" BAYESIAN OPTIMIZATION SETUP """
# =============================================================================================
from keras.layers import LeakyReLU
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from keras.callbacks import EarlyStopping

start2 = datetime.now()

def lenet_bo(input_data, output_data,
             num_filters1, num_filters2,
             filter_size1, filter_size2,
             conv_stride1, conv_stride2,
             pool_size1, pool_size2,
             pool_stride1, pool_stride2,
             dense_units1, dense_units2,
             optimizer, learning_rate, 
             batch_size, epochs=50):
    """Setup hyperparameters for Bayesian Optimization"""
    
    # Extract input and output dimensions
    input_dim = input_data.shape[1]
    output_dim = output_data.shape[1]
    # Convert hyperparameters to integers or floats as needed
    num_filters1 = int(num_filters1)
    num_filters2 = int(num_filters2)
    filter_size1 = int(filter_size1)
    filter_size2 = int(filter_size2)
    conv_stride1 = int(conv_stride1)
    conv_stride2 = int(conv_stride2)
    pool_size1 = int(pool_size1)
    pool_size2 = int(pool_size2)
    pool_stride1 = int(pool_stride1)
    pool_stride2 = int(pool_stride2)
    dense_units1 = int(dense_units1)
    dense_units2 = int(dense_units2)
    optimizer = [Adam, SGD, RMSprop, Adagrad][int(optimizer)]
    learning_rate = float(learning_rate)
    batch_size = int(batch_size)

    """ Define the LeNet model for Bayesian Optimization """
    model = lenet_model(input_dim=input_dim,
                        output_dim=output_dim,
                        num_filters1=num_filters1, 
                        num_filters2=num_filters2,
                        filter_size1=filter_size1, 
                        filter_size2=filter_size2,
                        conv_stride1=conv_stride1, 
                        conv_stride2=conv_stride2,
                        pool_size1=pool_size1, 
                        pool_size2=pool_size2,
                        pool_stride1=pool_stride1, 
                        pool_stride2=pool_stride2,
                        dense_units1=dense_units1, 
                        dense_units2=dense_units2,
                        optimizer=optimizer, 
                        learning_rate=learning_rate)
    # Define the early stopping monitor
    early_stopping = EarlyStopping(monitor='accuracy',
                            min_delta=4e-5,
                            patience=5, 
                            mode ='auto',
                            verbose=0,
                            restore_best_weights=True)
    # Create the KerasClassifier wrapper
    nn = KerasClassifier(model, 
                         epochs=epochs, 
                         batch_size=batch_size,
                         verbose=0)
    # Fit the model with cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Define the scoring function for cross-validation. Use accuracy as the scoring metric.
    score_acc = make_scorer(accuracy_score)
    # Perform cross-validation with early stopping 
    scores = cross_val_score(nn, input_data, output_data, 
                             cv=kfold, scoring=score_acc, 
                             params={'callbacks': [early_stopping]})
    # Print the cross-validation scores
    return np.mean(scores)

# Set parameter bounds for Bayesian Optimization
params_lenet = {
    'num_filters1': (1, 256),  # Number of filters in the first convolutional layer
    'num_filters2': (1, 256),  # Number of filters in the second convolutional layer
    'filter_size1': (1, 15),    # Size of the filter in the first convolutional layer
    'filter_size2': (1, 15),    # Size of the filter in the second convolutional layer
    'conv_stride1': (1, 5),     # Stride for the first convolutional layer
    'conv_stride2': (1, 5),     # Stride for the second convolutional layer
    'pool_size1': (1, 5),       # Pool size for the first pooling layer
    'pool_size2': (1, 5),       # Pool size for the second pooling layer
    'pool_stride1': (1, 5),     # Stride for the first pooling layer
    'pool_stride2': (1, 5),     # Stride for the second pooling layer
    'dense_units1': (4, 256),   # Number of units in the first dense layer
    'dense_units2': (4, 256),   # Number of units in the second dense layer
    'optimizer': (0, 3),        # Optimizer choice: Adam(0), SGD(1), RMSprop(2), Adagrad(3)
    'learning_rate': (1e-4, 1e-2),   # Learning rate
    'batch_size': (16, 64),     # Batch size choice: [16, 64]
}
# Run Bayesian Optimization
from bayes_opt.bayesian_optimization import BayesianOptimization
# Wrap lenet_bo so only hyperparameters are passed by the optimizer
lenet_bo_wrapper = lambda **params: lenet_bo(
    x_train, y_train,  # <-- your data
    **params
)
# Initialize Bayesian Optimization  
lenet_bo_optimizer = BayesianOptimization(
    f=lenet_bo_wrapper, 
    pbounds=params_lenet, 
    random_state=42
)
# Maximize the Bayesian Optimization function
lenet_bo_optimizer.maximize(
    init_points=10, # Number of initial random points
    n_iter=20,      # Number of iterations to perform
#    acq='ei',      # Acquisition function: Expected Improvement
#    xi=0.01        # Exploration-exploitation trade-off parameter
)

params_lenet_ = lenet_bo_optimizer.max['params']
# Print the best parameters found by Bayesian Optimization  
print("Best parameters found by Bayesian Optimization:")
for param, value in params_lenet_.items():
    print(f"{param}: {value}")

# Extract the best parameters
optimizerL = [Adam, SGD, RMSprop, Adagrad]  # List of optimizers
# Convert the best parameters to integers where applicable
params_lenet_ = {k: round(v) if isinstance(v, (int, float)) else v for k, v in params_lenet_.items()}

# Convert the best parameters to their actual values
num_filters1 = round(params_lenet_['num_filters1'])
num_filters2 = round(params_lenet_['num_filters2'])
filter_size1 = round(params_lenet_['filter_size1'])
filter_size2 = round(params_lenet_['filter_size2'])
conv_stride1 = round(params_lenet_['conv_stride1'])
conv_stride2 = round(params_lenet_['conv_stride2'])
pool_size1 = round(params_lenet_['pool_size1'])
pool_size2 = round(params_lenet_['pool_size2'])
pool_stride1 = round(params_lenet_['pool_stride1'])
pool_stride2 = round(params_lenet_['pool_stride2'])
dense_units1 = round(params_lenet_['dense_units1'])
dense_units2 = round(params_lenet_['dense_units2'])
learning_rate = params_lenet_['learning_rate']
batch_size = round(params_lenet_['batch_size'])
optimizer = optimizerL[round(params_lenet_['optimizer'])]

# Print the best parameters
print("Best parameters found by Bayesian Optimization:")
print(f"num_filters1: {num_filters1}")
print(f"num_filters2: {num_filters2}")
print(f"filter_size1: {filter_size1}")
print(f"filter_size2: {filter_size2}")
print(f"conv_stride1: {conv_stride1}")
print(f"conv_stride2: {conv_stride2}")
print(f"pool_size1: {pool_size1}")
print(f"pool_size2: {pool_size2}")
print(f"pool_stride1: {pool_stride1}")
print(f"pool_stride2: {pool_stride2}")
print(f"dense_units1: {dense_units1}")
print(f"dense_units2: {dense_units2}")
print(f"optimizer: {optimizer.__name__}")
print(f"learning_rate: {learning_rate}")
print(f"batch_size: {batch_size}")

# Save the best parameters to a CSV file
best_params = pd.DataFrame({
    'num_filters1': [num_filters1],
    'num_filters2': [num_filters2],
    'filter_size1': [filter_size1],
    'filter_size2': [filter_size2],
    'conv_stride1': [conv_stride1],
    'conv_stride2': [conv_stride2],
    'pool_size1': [pool_size1],
    'pool_size2': [pool_size2],
    'pool_stride1': [pool_stride1],
    'pool_stride2': [pool_stride2],
    'dense_units1': [dense_units1],
    'dense_units2': [dense_units2],
    'optimizer': [optimizer.__name__],
    'learning_rate': [learning_rate],
    'batch_size': [batch_size]})
best_params.to_csv(f'{output_path}{model_name}_best_params.csv', index=False)

# Execution time of the Bayesian Optimization
stop2 = datetime.now()
execution_time2 = stop2 - start2
print("Bayesian Optimization execution time is: ", execution_time2)

# =============================================================================================
""" MODEL TRAINING """  
# =============================================================================================
# Define the model with the best parameters found by Bayesian Optimization
lenet_best_model = lenet_model(
    input_dim=x_train.shape[1],
    output_dim=y_train.shape[1],
    num_filters1=num_filters1, 
    num_filters2=num_filters2,
    filter_size1=filter_size1, 
    filter_size2=filter_size2,
    conv_stride1=conv_stride1, 
    conv_stride2=conv_stride2,
    pool_size1=pool_size1, 
    pool_size2=pool_size2,
    pool_stride1=pool_stride1, 
    pool_stride2=pool_stride2,
    dense_units1=dense_units1, 
    dense_units2=dense_units2,
    optimizer=optimizer, 
    learning_rate=learning_rate,
)
# Initialize the list to store cross-validation scores
cvscores = []

start1 = datetime.now()

# Early stopping callback to monitor validation accuracy
es = EarlyStopping(monitor='val_accuracy',
                       min_delta=0,
                       mode='max', 
                       verbose=1, 
                       baseline=None,
                       patience=5, 
                       restore_best_weights=True)
# Train the model with the training data
print('\nFitting model ....')
history = lenet_best_model.fit(
    x_train, 
    y_train, 
    batch_size=batch_size, 
    epochs=100, 
    validation_data=(x_val, y_val),
    #callbacks=[es],  # Early stopping callback
)
# Save the trained model
lenet_best_model.save(f'{output_path}lenet_best_model.keras')
lenet_best_model.save_weights(f'{output_path}lenet_best_model.model.weights.h5')

# Print the model summary
lenet_best_model.summary()

# Execution time of the model
stop1 = datetime.now()
execution_time1 = stop1 - start1
print("Execution time to train model is: ", execution_time1)

# =============================================================================================
""" MODEL EVALUATION """
# =============================================================================================
# Evaluate the model on the test data
print('----------------------------------')
print('Evaluating model ....')
scores = lenet_best_model.evaluate(x_val, y_val, verbose=0)
print("%s: %.2f%%" % (lenet_best_model.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print('----------------------------------')
# Convert the history dict to a pandas dataframe and save as csv for future plotting
history_df = pd.DataFrame(history.history)
with open(f'{output_path}{model_name}_history_df.csv', mode='w') as f:
    history_df.to_csv(f)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='best')
plt.savefig(f'{output_path}{model_name}_loss.png')
plt.show()
# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.yscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.savefig(f'{output_path}{model_name}_accuracy.png')
plt.show()

# =============================================================================================
""" MAKE PREDICTION USING TEST DATA """
# =============================================================================================

print('----------------------------------')
print('Predicting test data ....')

# Predict the test data
y_pred = lenet_best_model.predict(x_test)

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
print('\nThe accuracy of test data is:', a*100)

# Save model performance 
acc_time = [['train_mean_acc', np.mean(cvscores)]
            ['bayes_opt_time', execution_time2], 
            ['training_time', execution_time1], 
            ['test_accuracy', a]]
acc_time_df = pd.DataFrame(acc_time, columns=['params.', 'values'])
acc_time_df.to_csv(f'{output_path}{model_name}_acc_time.csv', index=False)

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
cm.to_csv(f'{output_path}{model_name}_conmat.csv')

# Plot confusion matrix using seaborn
plt.figure(figsize=(7,5))
sns.heatmap(cm_, annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=label_name, 
            yticklabels=label_name, 
            annot_kws={"size": 12})
#plt.title('Confusion Matrix of Training Data')
plt.xlabel('Predicted Label', fontsize=12, fontname="Segoe UI")
plt.ylabel('True Label', fontsize=12, fontname="Segoe UI")
plt.savefig(f'{output_path}{model_name}_conmat.png')
plt.show()

# Save the prediction
pred_df = pd.DataFrame(pred, columns=['Predicted'])
pred_df.to_csv(f'{output_path}{model_name}_predictions.csv', index=False)

# Save the test data
test_df = pd.DataFrame(test, columns=['True'])
test_df.to_csv(f'{output_path}{model_name}_test.csv', index=False)

# Show and save classification report (performance metrics)
report = classification_report(test, pred, target_names=label_name, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f'{output_path}{model_name}_classification_report.csv')
print(classification_report(test, pred, target_names=label_name))

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
plt.savefig(f'{output_path}{model_name}_roc_curve.png')
plt.show()

# Save ROC curve data
roc_df = pd.DataFrame({'fpr': fpr[0], 'tpr': tpr[0]})
roc_df.to_csv(f'{output_path}{model_name}_roc_curve.csv', index=False)

# Save ROC curve data for all classes
for i in range(output_dim):
    roc_df = pd.DataFrame({'fpr': fpr[i], 'tpr': tpr[i]})
    roc_df.to_csv(f'{output_path}{model_name}_roc_curve_{i}.csv', index=False)

# Save the model summary to a text file
with open(f'{output_path}{model_name}_model_summary.txt', 'w') as f:
    lenet_best_model.summary(print_fn=lambda x: f.write(x + '\n'))

# =============================================================================================
""" END OF THE SCRIPT """
# =============================================================================================
print("Script execution completed successfully.")
# =============================================================================================