import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================================
""" LOAD AND VISUALIZE DATASET """
# ===============================================================================

img_path = 'd://z/master/spectra/datasets/'
agavis_path = img_path +'VISNIR_AGA1.csv'
akivis_path = img_path + 'VISNIR_AKI1.csv'
atevis_path = img_path + 'VISNIR_ATE2.csv'
atovis_path = img_path + 'VISNIR_ATO1.csv'
output_path = 'd://z/master/spectra/cnn_spectrum/outputs/'
# Load data
def load_csv(data):
    dataset = pd.read_csv(data)
    return dataset

agavis = load_csv(agavis_path)
akivis = load_csv(akivis_path)
atevis = load_csv(atevis_path)
atovis = load_csv(atovis_path)

# Combine all cvs 
arabica_vis = pd.concat([agavis, akivis, atevis, atovis])

# VARIABLES
def variable_data(data):
    # --- Label
    label = data.values[:,1].astype('uint8')
    # --- Spectra data
    spectra = data.values[:,2:].astype('float')
    # --- Wavelengths
    cols = list(data.columns.values.tolist())
    wls = [float(x) for x in cols[2:]]
    return label, spectra, wls

_, spectra_agavis, wls_agavis = variable_data(agavis)
_, spectra_akivis, wls_akivis = variable_data(akivis)
_, spectra_atevis, wls_atevis = variable_data(atevis)
_, spectra_atovis, wls_atovis = variable_data(atovis)

mean_agavis = np.mean(spectra_agavis, axis=0)
mean_akivis = np.mean(spectra_akivis, axis=0) 
mean_atevis = np.mean(spectra_atevis, axis=0)
mean_atovis = np.mean(spectra_atovis, axis=0) 

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot()
ax.xaxis.label.set_color('black')        #setting up X-axis label color to yellow
ax.yaxis.label.set_color('black')          #setting up Y-axis label color to blue
ax.tick_params(axis='x', colors='black')    #setting up X-axis tick color to red
ax.tick_params(axis='y', colors='black')  #setting up Y-axis tick color to black
ax.spines['left'].set_color('black')        # setting up Y-axis tick color to red
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
plt.plot(wls_agavis, mean_agavis.T, label = 'arabica kintamani')
plt.plot(wls_akivis, mean_akivis.T, label = 'arabica gayo')
plt.plot(wls_atevis, mean_atevis.T, label = 'arabica temanggung')
plt.plot(wls_atovis, mean_atovis.T, label = 'arabica toraja')
plt.xticks(fontsize=12, fontname="Segoe UI")    # np.arange(400, 1000, step=50),
plt.yticks(np.arange(0, 40, step=5), fontsize=12, fontname="Segoe UI")
plt.title('Plot Average Spectra', fontweight='bold', fontsize=12, fontname="Segoe UI")
plt.ylabel('Reflectance (%)', fontsize=12, fontname="Segoe UI")
plt.xlabel('Wavelength (nm)', fontsize=12, fontname="Segoe UI")
plt.legend(loc = 'upper left')
plt.grid(False) # visible=None
plt.savefig(f'{output_path}vis_average-spectra_plot.png')
plt.show()

# Average spectra
label, spectra, wls = variable_data(arabica_vis)

# ===============================================================================
""" OUTLIER'S REMOVAL """
# ===============================================================================
from nirs_utils import pls_data_optimization, pls_prediction1, pls_prediction2

"""Find the number of PLS LV that best simulates dataset 1"""
pls_data_optimization(spectra, label, output_path, plot_components=True)

"""Applying the PLS model with the found number of latent variables to the full datasets and compute error metrics"""
## Visualize the prediction of the PLS model and get the predicted values
print('\n Optimal PLS model applyed to full dataset from instrument 1 \n')    
y1, ypred1, _= pls_prediction2(spectra, 
                               label, 
                               spectra, 
                               label,
                               components=15,
                               plot_components=True)     
   
## instrument 1
pred_error1=np.abs(y1-np.ravel(ypred1))
pred_error1_std=pred_error1.std()
print('\nStandard Deviation of error = {}'.format(pred_error1_std))
ind1=np.ravel(np.where(pred_error1>=4.0*pred_error1_std)).reshape(-1,1)

plt.figure(figsize=(12,4))
plt.plot(pred_error1, c='orange',label='dataset 1')
plt.plot(ind1, pred_error1[pred_error1>=4.0*pred_error1_std],'ro',label='outliers 1')
plt.axhline(4.0*pred_error1_std,c='k',ls='--',lw=0.5)

plt.xlabel('Sample number')
plt.ylabel('Absolute prediction error')
plt.legend()
plt.savefig(f'{output_path}points_categorized_outliers.png')
plt.show()

"""Take a look at the indices of the outlier points in both datasets"""
print('\nDefine datapoints to remove\n')
print(np.ravel(ind1))

"""Recompute PLS for the full clean datasets"""
print('\n PLS model for full clean dataset from instrument 1 \n')
X1_clean=np.delete(spectra,ind1,axis=0)
Y1_clean=np.delete(label,ind1,axis=0)
_=pls_prediction2(X1_clean, 
                  Y1_clean, 
                  X1_clean , 
                  Y1_clean, 
                  components=15, 
                  plot_components=True)  

## check for nans
print('NANs on X1_clean? = ',np.isnan(np.sum(X1_clean)))
print('NANs on Y1_clean? = ',np.isnan(np.sum(Y1_clean)))

print('Instrument 1 clean data size after outlier removed = {}'.format(X1_clean.shape))

# Create dataframe exclude outliers
df = pd.DataFrame(X1_clean, columns=wls)
df.insert(0, "label", Y1_clean)
df.to_csv(f'{output_path}cleaned_data.csv', index=False)

from nirs_utils import SG_derivative
agavis_sgd1 = SG_derivative(mean_agavis, window_size=49, polyorder=2, derivative=1)
akivis_sgd1 = SG_derivative(mean_akivis, window_size=49, polyorder=2, derivative=1)
atevis_sgd1 = SG_derivative(mean_atevis, window_size=49, polyorder=2, derivative=1)
atovis_sgd1 = SG_derivative(mean_atovis, window_size=49, polyorder=2, derivative=1)

from nirs_utils import plot_average_spectra
plot_average_spectra(wls, agavis_sgd1, akivis_sgd1, atevis_sgd1, atovis_sgd1)

""" PLOT SPECTRA AUGMENTATION """
def dataaugment(x, betashift, slopeshift, multishift):
    #Shift of baseline
    #calculate arrays
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

#Data Augment a single spectrum

#First Spectrum
X = X1_clean[0:1]
#Repeating the spectrum 10x
X = np.repeat(X, repeats=3, axis=0)
#Augment (Large pertubations for illustration)
X_aug = dataaugment(X, betashift = 0.1, slopeshift = 0.1,multishift = 0.1)
X_aug.shape  
plt.plot(wls, X_aug.T, label='augmented spectrum')
plt.plot(wls, X.T, lw=5, c='b', label = 'original spectrum')
plt.ylabel('Reflectance (%)', fontsize=12, fontname="Segoe UI")
plt.xlabel('Wavelength (nm)', fontsize=12, fontname="Segoe UI")
plt.legend(loc='upper left')
plt.show()