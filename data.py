import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nirs_utils import plot_spectra, plot_average_spectra

# ======================================================================================================================
""" LOAD AND VISUALIZE DATASET """
# ======================================================================================================================
# file paths
img_path = 'd://z/master/spectra/datasets/'
#agavis_path = img_path +'VISNIR_AGA1.csv'
#akivis_path = img_path + 'VISNIR_AKI1.csv'
#atevis_path = img_path + 'VISNIR_ATE2.csv'
#atovis_path = img_path + 'VISNIR_ATO1.csv'
aganir_path = img_path + 'SWNIR_AGA1.csv'
akinir_path = img_path + 'SWNIR_AKI1.csv'
atenir_path = img_path + 'SWNIR_ATE2.csv'
atonir_path = img_path + 'SWNIR_ATO1.csv'
output_path = 'd://z/master/spectra/cnn_spectrum/data/'

from nirs_utils import load_csv
#agavis = load_csv(agavis_path)
#akivis = load_csv(akivis_path)
#atevis = load_csv(atevis_path)
#atovis = load_csv(atovis_path)
aganir = load_csv(aganir_path)
akinir = load_csv(akinir_path)
atenir = load_csv(atenir_path)
atonir = load_csv(atonir_path)

# Combine all cvs 
arabica_nir = pd.concat([aganir, akinir, atenir, atonir])

# VARIABLES
from nirs_utils import variable_data

# Get variable data: spectra, label, and wavelengths
label, spectra, wls = variable_data(arabica_nir)

# Get variable data of each sample group
_, spectra_aganir, wls = variable_data(aganir)
_, spectra_akinir, wls = variable_data(akinir)
_, spectra_atenir, wls = variable_data(atenir)
_, spectra_atonir, wls = variable_data(atonir)

# Get average spectra from each sample group
mean_aganir = np.mean(spectra_aganir, axis=0)
mean_akinir = np.mean(spectra_akinir, axis=0) 
mean_atenir = np.mean(spectra_atenir, axis=0)
mean_atonir = np.mean(spectra_atonir, axis=0) 

# Plot average spectra
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
plt.plot(wls, mean_aganir.T, label = 'arabica gayo')
plt.plot(wls, mean_akinir.T, label = 'arabica kintamani')
plt.plot(wls, mean_atenir.T, label = 'arabica temanggung')
plt.plot(wls, mean_atonir.T, label = 'arabica toraja')
plt.xticks(fontsize=12, fontname="Segoe UI")
plt.yticks(np.arange(0, 40, step=5), fontsize=12, fontname="Segoe UI")
plt.title('Plot Average Spectra', fontweight='bold', fontsize=12, fontname="Segoe UI")
plt.ylabel('Reflectance (%)', fontsize=12, fontname="Segoe UI")
plt.xlabel('Wavelength (nm)', fontsize=12, fontname="Segoe UI")
plt.legend(loc = 'best')
plt.grid(False) # visible=None
plt.savefig(f'{output_path}Plot_average_spectra.png')
plt.show()

# ======================================================================================================================
""" OUTLIER'S REMOVAL """
# ======================================================================================================================
from nirs_utils import pls_data_optimization, pls_prediction2

# Find the number of PLS LV that best simulates dataset 1
pls_data_optimization(spectra, label, output_path, plot_components=True)

# Applying the PLS model with the found number of latent variables to the full datasets and compute error metrics

## Visualize the prediction of the PLS model and get the predicted values
print('\n Optimal PLS model applyed to full dataset from instrument 1 \n')    
y1, ypred1, _= pls_prediction2(spectra, label, spectra, label, components=15, plot_components=True)
   
## Get prediction error of instrument 1
pred_error1=np.abs(y1-np.ravel(ypred1))
pred_error1_std=pred_error1.std()
print('\nStandard Deviation of error = {}'.format(pred_error1_std))
ind1=np.ravel(np.where(pred_error1>=4.0*pred_error1_std)).reshape(-1,1)

## Visualize datapoints categorized as outliers
plt.figure(figsize=(12,4))
plt.plot(pred_error1, c='orange',label='dataset 1')
plt.plot(ind1, pred_error1[pred_error1>=4.0*pred_error1_std],'ro',label='outliers 1')
plt.axhline(4.0*pred_error1_std,c='k',ls='--',lw=0.5)
plt.xlabel('Sample number')
plt.ylabel('Absolute prediction error')
plt.legend()
plt.savefig(f'{output_path}points_categorized_outliers.png')
plt.show()

## Take a look at the indices of the outlier points in both datasets
print('\nDefine datapoints to remove\n')
print(np.ravel(ind1))

## Recompute PLS for the full clean datasets
print('\n PLS model for full clean dataset from instrument 1 \n')
X1_clean=np.delete(spectra,ind1,axis=0)
Y1_clean=np.delete(label,ind1,axis=0)
_=pls_prediction2(X1_clean, Y1_clean, X1_clean, Y1_clean, components=15, plot_components=True)

## Check for nans
print('NANs on X1_clean? = ',np.isnan(np.sum(X1_clean)))
print('NANs on Y1_clean? = ',np.isnan(np.sum(Y1_clean)))

print('Instrument 1 clean data size after outlier removed = {}'.format(X1_clean.shape))

def save_processed_data(data, label, output_path, name):
    # Create DataFrame
    df = pd.DataFrame(data, columns=wls)
    df.insert(0, "label", label)
    df.to_csv(f'{output_path}{name}_data.csv', index=False)

save_processed_data(X1_clean, Y1_clean, output_path, 'cleaned')
plot_spectra(wls, X1_clean, output_path, title='Original spectra')

# ======================================================================================================================
""" DATA PREPROCESSING """
# ======================================================================================================================

# Multiplicative scatter correction
from nirs_utils import msc, snv, SG_smoothing, SG_derivative, detrend 

# Sav-Gol Smoothing
aranir_sgs = SG_smoothing(spectra, window_size=11, polyorder=1)
save_processed_data(aranir_sgs, label, output_path, 'sgs')
plot_spectra(wls, aranir_sgs, output_path, title='Sav-Gol Smoothing spectra')

# Sav-Gol Derivative
aranir_sgd1 = SG_derivative(spectra, window_size=11, polyorder=2, derivative=1)
save_processed_data(aranir_sgd1, label, output_path, 'sgd1')
plot_spectra(wls, aranir_sgd1, output_path, title='Sav-Gol 1st Derivative spectra')

# Sav-Gol Derivative
aranir_sgd2 = SG_derivative(spectra, window_size=11, polyorder=2, derivative=2)
save_processed_data(aranir_sgd2, label, output_path, 'sgd2')
plot_spectra(wls, aranir_sgd2, output_path, title='Sav-Gol 2nd Derivative spectra')

# Detrend
aranir_dt = detrend(spectra)
save_processed_data(aranir_dt, label, output_path, 'detrend')
plot_spectra(wls, aranir_dt, output_path, title='Detrend spectra')

# Standard normal variate
aranir_snv = snv(spectra)
save_processed_data(aranir_snv, label, output_path, 'snv')
plot_spectra(wls, aranir_snv, output_path, title='Standard Normal Variate spectra')

# Multiplicative scatter correction
aranir_msc = msc(spectra)
if isinstance(aranir_msc, tuple):
    aranir_msc = aranir_msc[0]  # Extract the first element if msc returns a tuple
save_processed_data(aranir_msc, label, output_path, 'msc')
plot_spectra(wls, aranir_msc, output_path, title='Multiplicative scatter correction spectra')