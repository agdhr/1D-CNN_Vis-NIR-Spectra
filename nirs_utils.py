import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import newaxis as nA
from scipy.signal import savgol_filter
from obspy.signal.detrend import polynomial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sys import stdout
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score


"""LOAD DATASET"""
def load_data(data):
    dataset = pd.read_csv(data)
    return dataset

"""EXTRACT VARIABLE DATA"""
def variable_data(data):
    # --- Label
    label = data.values[:,1].astype('uint8')
    # --- Spectra data
    spectra = data.values[:,2:].astype('float')
    # --- Wavelengths
    cols = list(data.columns.values.tolist())
    wls = [float(x) for x in cols[2:]]
    return label, spectra, wls

"""PLOT SPECTRA"""
def plot_spectra(x, y):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot()
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.plot(x, y.T)
    plt.xticks()    # np.arange(400, 1000, step=50),
    plt.ylabel('Reflectance (%)')
    plt.xlabel('Wavelength (nm)')
    plt.grid(False) # visible=None, which='major', axis='both', **kwargs
    plt.show()

def plot_average_spectra(wls, mean1, mean2, mean3, mean4):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot()
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.plot(wls, mean1.T, label = 'arabica gayo')
    plt.plot(wls, mean2.T, label = 'arabica kintamani')
    plt.plot(wls, mean3.T, label = 'arabica temanggung')
    plt.plot(wls, mean4.T, label = 'arabica toraja')
    plt.xticks()    # np.arange(400, 1000, step=50),
    plt.yticks()#np.arange(0, 70, step=10))
    plt.ylabel('Reflectance (%)')
    plt.xlabel('Wavelength (nm)')
    plt.legend(loc = 'best')
    plt.grid(False) # visible=None, which='major', axis='both', **kwargs
    plt.show()

# SIMPLE MOVING AVERAGE
def sma(input_spectra, window_size):
    df = pd.DataFrame(input_spectra)
    moving_averages = df.rolling(window_size, min_periods=1).mean()#.iloc[window_size-1:].values
    return moving_averages

# MSC - MULTIPLICATIVE SCATTER CORRECTION
def msc(input_spectra, reference=None):
    # --- Mean center correction
    for i in range(input_spectra.shape[0]):
        input_spectra[i,:] -= input_spectra[i,:].mean()
    # --- Get the reference spektrum. If no given, estimate it from the mean
    if reference is None:
        # --- Calculate mean
        ref = np.mean(input_spectra, axis=0)
    else:
        ref = reference
    # --- Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_spectra)
    for i in range(input_spectra.shape[0]):
        # --- Run regression
        fit = np.polyfit(ref, input_spectra[i,:], 1, full=True)
        # --- Apply correction
        data_msc[i,:] = (input_spectra[i,:] - fit[0][1]) / fit[0][0]
    return data_msc, ref

# STANDARD NORMAL VARIATE
def snv(input_spectra):
    # --- Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_spectra)
    for i in range(input_spectra.shape[0]):
        # --- Apply correction
        output_data[i,:] = (input_spectra[i,:] - np.mean(input_spectra[i,:])) / np.std(input_spectra[i,:])
    return output_data

# SAVITZKY-GOLAY SMOOTHING
def SG_smoothing(input_data, window_size, polyorder):
    SG_smoothing =savgol_filter(input_data,
                                window_length=window_size,
                                polyorder=polyorder,
                                mode="nearest")
    return SG_smoothing

# SAVITZKY-GOLAY DERIVATIVE/FILTER
def SG_derivative(input_data, window_size, polyorder, derivative):
    SG_filter = savgol_filter(input_data,
                              window_length=window_size,
                              polyorder=polyorder,
                              deriv=derivative,
                              delta=1.0,
                              axis=-1,
                              mode='interp', #'nearest'
                              cval=0.0)
    return SG_filter

## Define a function detrend all spectra
def detrend(input_data):
    data_det = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply detrend
        data_det[i,:] = polynomial(input_data[i,:].copy(), order=2, plot=False)
    return data_det

"""STANDARD SCALER"""
def standardscaler(input_spectra):
    return StandardScaler().fit_transform(input_spectra)

def GlobalStandardScaler(input_data):
    input_data = np.array(input_data)
    x_mean = input_data.mean()
    X = input_data - x_mean
    return X

"""PRINCIPAL COMPONENT ANALYSIS"""
def PrincipalComponentAnalysis(input_spectra, wavelengths, label, label_name, num_comps):
    pca = PCA(n_components=num_comps)
    """Fit the spectral data and extract the explained variance ratio"""
    X = pca.fit(input_spectra)
    """Explained variances equal to n_components largest eigenvalues of the covariance matrix of X."""
    var_expl = X.explained_variance_
    for a in var_expl:
        if(a>0.0):
            print('Sorted Eignevalues: {}'.format(round(a, 3)))
    """Percentage of variance explained by each of the selected components."""
    var_expl_ratio = X.explained_variance_ratio_
    for b in var_expl_ratio:
        if (b>(0.1/100)):
            print('Explained Variance: {}%'.format(round(b*100, 2)))
    """Scree Plot"""
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(6,4))
        fig.set_tight_layout(True)
        ax.plot(var_expl_ratio, '-o', label='Explained Variance %')
        ax.plot(np.cumsum(var_expl_ratio), '-o', label = 'Cumulative Variance %')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_xlabel('Number of Components')
        #ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
        #ax.set_xticks(np.arange(0, 10, step=1))
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_facecolor("white")
    plt.legend()
    plt.show()

    """Get Principle Components (PCs) or PCA scores"""
    # Transform on the scaled features
    Xcomps = pca.fit_transform(input_spectra)
    #comps = SS(Xcomps)

    """PCA Score Plot"""
    PCs_var = var_expl_ratio*100
    unique = list(set(label))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(6,5))
        #ax = fig.add_subplot(111, projection='3d')
        for i, u in enumerate(unique):
            col = np.expand_dims(np.array(colors[i]), axis=0)
            pc1 = [Xcomps[j,0] for j in range(len(Xcomps[:,0])) if label[j] == u]    # PC1
            pc2 = [Xcomps[j,1] for j in range(len(Xcomps[:,1])) if label[j] == u]    # PC2
            #pc3 = [Xcomps[j,2] for j in range(len(Xcomps[:,2])) if label[j] == u]    # PC3
            plt.scatter(pc1, pc2, #
                        c=col, s=20, edgecolors='k', label=str(u))
        plt.xlabel('PC1 ('+str(round(PCs_var[0],2))+'%)')   # ---------------------- < CHANGE INPUT
        plt.ylabel('PC2 ('+str(round(PCs_var[1],2))+'%)')   # ---------------------- < CHANGE INPUT
        plt.legend(label_name, loc = 'lower left')
        #plt.title('2D PCA ' + f'(Total Explained Variance: {total_var:.2f}%)')
        ax.axhline(y=0.0, color='black', linestyle='dashed', alpha = 1.0) # label = 'Horizontal Line ')
        ax.axvline(x=0.0, color='black', linestyle='dashed', alpha = 1.0) # label = 'Vertical Line ')
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        xabs_max = abs(max(ax.get_xlim(), key=abs))
        ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_facecolor("white")
        plt.show()

    """Principal components correlation coefficients"""
    loadings = pca.components_
    num_pcs = pca.n_features_in_

    pc_list = ['PC'+ str(i) for i in list(range(1, num_pcs+1))]
    loadings_df =pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))

    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(6,4))
        plt.plot(wavelengths, loadings_df['PC2'], label='PC2')
        plt.plot(wavelengths, loadings_df['PC1'], label='PC1')
        plt.ylabel('PCA Loadings')
        plt.xlabel('Wavelength (nm)')
        plt.axhline(y=0.0, color='black', linestyle='-')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_facecolor("white")
        plt.grid(visible=None)
        plt.legend(loc = 'best')
        plt.show()

    return Xcomps


## Help functions to help tune the PLS models, find outliers, etc...
## TODO: Group these 3 functions into just one function with multiple choice parameters

## Define the Huber distance (source DeepChemometrics)
def huber(y_true, y_pred, delta=1.0):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return np.mean(delta ** 2 * ((1 + ((y_true - y_pred) / delta) ** 2) ** 0.5 - 1))


## Create a huber loss scorer for cross_val_score()
huber_score = make_scorer(huber)


## Function to help find the best number of components of the PLS based on 10 CV Huber Loss
def pls_data_optimization(X, Y, output_path, plot_components=False):
    """
    This function finds the optimal number of PLS components (up to 40) that best models the data
    based on huber loss and 10 CV
    X - The training data X
    Y - The training data Y
    plot_components - Plot the model's optimization and optimized model
    """
    # Run PLS including a variable number of components, up to 40,  and calculate mean of 10 CV huber loss
    cv_huber = []
    component = np.arange(1, 40)
    for i in component:
        pls = PLSRegression(n_components=i)
        cv_score = cross_val_score(pls, X, Y, cv=KFold(10, shuffle=True), \
                                   n_jobs=-1, scoring=huber_score)
        cv_huber.append(np.mean(cv_score))
        comp = 100 * (i + 1) / 40
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum
    cv_hubermin = np.argmin(cv_huber)
    print("Suggested number of components based in Mean of 10 CV huber loss: ", cv_hubermin + 1)
    print('Minimum found in Mean of 10 CV huber loss: {}'.format(np.min(cv_huber)))
    stdout.write("\n")

    # Define PLS with suggested number of components and fit train data
    pls = PLSRegression(n_components=cv_hubermin + 1)
    pls.fit(X, Y)

    # Get predictions for calibration(train) and validation(test) sets
    Y_pred = pls.predict(X)

    # Calculate and print scores for validation set
    R2_p = r2_score(Y, Y_pred)
    mse_p = mean_squared_error(Y, Y_pred)
    hub_p = huber(Y, Y_pred)
    sep = np.std(Y_pred[0] - Y)
    rmse_p = np.sqrt(mse_p)

    print('\nError metrics for test set:')
    print('R2: %5.3f' % R2_p)
    print('Root Mean Squared Error (RMSE): %5.3f' % rmse_p)
    print('Huber loss (huber): %5.3f' % hub_p)
    print('Standard Error Prediction (SEP): %5.3f' % sep)

    # Plot regression and PLS LV search
    rangey = max(Y) - min(Y)
    rangex = max(Y_pred) - min(Y_pred)

    if plot_components is True:
        plt.figure(figsize=(15, 5))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(component, np.array(cv_huber), '-v', color='blue', mfc='blue')
        ax1.plot(component[cv_hubermin], np.array(cv_huber)[cv_hubermin], 'P', ms=10, mfc='red')
        plt.xlabel('Number of PLS components')
        plt.ylabel('Mean od 10 CV Huber loss')
        plt.title('# PLS components')
        plt.xlim(left=-1)

        z = np.polyfit(np.ravel(Y), np.ravel(Y_pred), 1)
        ax2 = plt.subplot(1, 2, 2, aspect=1)
        ax2.scatter(Y, Y_pred, c='k', s=2)
        ax2.plot(Y, z[1] + z[0] * Y, c='blue', linewidth=2, label='linear fit')
        ax2.plot(Y, Y, color='orange', linewidth=1.5, label='y=x')
        plt.ylabel('Predicted')
        plt.xlabel('Measured')
        plt.title('Prediction from PLS')
        plt.legend(loc=4)

        # Print the scores on the plot
        plt.text(min(Y_pred) + 0.02 * rangex, max(Y) - 0.1 * rangey, 'R$^{2}=$ %5.3f' % R2_p)
        plt.text(min(Y_pred) + 0.02 * rangex, max(Y) - 0.15 * rangey, 'RMSE: %5.3f' % rmse_p)
        plt.text(min(Y_pred) + 0.02 * rangex, max(Y) - 0.2 * rangey, 'Huber loss: %5.3f' % hub_p)
        plt.savefig(f'{output_path}pls_outlier-removal_ncomps.png')
        plt.show()
    return


## Function to help find the best number of components of the PLS based in the MSE or Huber Loss
def pls_prediction1(X_calib, Y_calib, X_valid, Y_valid, output_path, loss='huber', plot_components=False):
    """
    This function finds the optimal number of PLS components (up to 40) that best models the data
    Here we compute some internal metrics based on MSE and Huber loss
    X_calib - The training data X
    Y_calib - The training data Y
    X_valid - The validation or test data X
    Y_valid - The validation or test data Y
    loss - Choose the error metric of the optimization. Options are 'mse' and 'huber'(default)
    plot_components - Plot the model's optimization and optimized model
    """
    # Run PLS including a variable number of components, up to 40,  and calculate MSE and huber loss
    mse = []
    hub = []
    component = np.arange(1, 40)
    for i in component:
        pls = PLSRegression(n_components=i)
        # Fit
        pls.fit(X_calib, Y_calib)
        # Prediction
        Y_pred = pls.predict(X_valid)
        mse_p = mean_squared_error(Y_valid, Y_pred)
        hub_p = huber(Y_valid, Y_pred)
        mse.append(mse_p)
        hub.append(hub_p)

        comp = 100 * (i + 1) / 40
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    hubmin = np.argmin(hub)
    print("Suggested number of components based in MSE: ", msemin + 1)
    print("Suggested number of components based in huber: ", hubmin + 1)
    stdout.write("\n")

    # Define PLS with suggested number of components and fit train data
    if loss == 'mse':
        pls = PLSRegression(n_components=msemin + 1)
        loss_name = 'MSE'
    else:
        pls = PLSRegression(n_components=hubmin + 1)
        loss_name = 'Huber loss'
    pls.fit(X_calib, Y_calib)

    # Get predictions for calibration(train) and validation(test) sets
    Y_calib_pred = pls.predict(X_calib)
    Y_valid_pred = pls.predict(X_valid)

    # Calculate and print scores for validation set
    R2_p = r2_score(Y_valid, Y_valid_pred)
    mse_p = mean_squared_error(Y_valid, Y_valid_pred)
    hub_p = huber(Y_valid, Y_valid_pred)
    sep = np.std(Y_valid_pred[0] - Y_valid)
    rmse_p = np.sqrt(mse_p)
    # rpd = np.std(Y_valid)/sep
    bias = np.mean(Y_pred[:,0]-Y_valid)

    print('\nError metrics for test set:')
    print('R2: %5.3f' % R2_p)
    print('Root Mean Squared Error (RMSE): %5.3f' % rmse_p)
    print('Huber loss (huber): %5.3f' % hub_p)
    print('Standard Error Prediction (SEP): %5.3f' % sep)
    # print('RPD: %5.3f' % rpd)
    # print('Bias: %5.3f' %  bias)

    # Plot regression and figures of merit
    rangey = max(Y_valid) - min(Y_valid)
    rangex = max(Y_valid_pred) - min(Y_valid_pred)

    if plot_components is True:
        plt.figure(figsize=(15, 5))
        ax1 = plt.subplot(1, 2, 1)
        if loss == 'mse':
            ax1.plot(component, np.array(mse), '-v', color='blue', mfc='blue')
            ax1.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
        else:
            ax1.plot(component, np.array(hub), '-v', color='blue', mfc='blue')
            ax1.plot(component[hubmin], np.array(hub)[hubmin], 'P', ms=10, mfc='red')
        plt.xlabel('Number of PLS components')
        plt.ylabel(loss_name)
        plt.title('# PLS components')
        plt.xlim(left=-1)

        z = np.polyfit(np.ravel(Y_valid), np.ravel(Y_valid_pred), 1)
        ax2 = plt.subplot(1, 2, 2, aspect=1)
        ax2.scatter(Y_calib, Y_calib_pred, c='k', s=2)
        ax2.scatter(Y_valid, Y_valid_pred, c='r', s=2)
        ax2.plot(Y_valid, z[1] + z[0] * Y_valid, c='blue', linewidth=2, label='linear fit')
        ax2.plot(Y_valid, Y_valid, color='orange', linewidth=1.5, label='y=x')
        plt.ylabel('Predicted')
        plt.xlabel('Measured')
        plt.title('Prediction from PLS')
        plt.legend(loc=4)

        # Print the scores on the plot
        plt.text(min(Y_valid_pred) + 0.2 * rangex, max(Y_valid) - 0.2 * rangey, 'R$^{2}=$ %5.3f' % R2_p)
        plt.text(min(Y_valid_pred) + 0.2 * rangex, max(Y_valid) - 0.15 * rangey, 'RMSE: %5.3f' % rmse_p)
        plt.text(min(Y_valid_pred) + 0.2 * rangex, max(Y_valid) - 0.2 * rangey, 'Huber loss: %5.3f' % hub_p)
        # plt.text(min(Y_pred)+0.02*rangex, max(Y_valid)-0.25*rangey, 'RPD: %5.3f' % rpd)
        # plt.text(min(Y_pred)+0.02*rangex, max(Y_valid)-0.3*rangey, 'Bias: %5.3f' %  bias)
        plt.show()
    return Y_calib, Y_valid, Y_valid_pred, Y_calib_pred


## Function that computes the PLS model and metrics for a train - test set pair and a given number of LV
def pls_prediction2(X_calib, Y_calib, X_valid, Y_valid, components, plot_components=False):
    """
    Very similar to the two previous functions but without the optimization part
    This function is simply used to compute the PLS model and the error metrics.
    NOTE: For the final error metrics we should use the benchmark() because we need
    take into consideration the unscaled version of the data.
    """
    i = components
    pls = PLSRegression(n_components=i)
    # Fit
    pls.fit(X_calib, Y_calib)
    # Prediction
    Y_pred = pls.predict(X_valid)

    # Calculate and print scores
    score_p = r2_score(Y_valid, Y_pred)
    mse_p = mean_squared_error(Y_valid, Y_pred)
    hub_p = huber(Y_valid, Y_pred)
    rmse_p = np.sqrt(mse_p)
    y_err = Y_valid - Y_pred
    sep = np.std(Y_pred[0] - Y_valid)

    print('R2: %5.3f' % score_p)
    #    print('Mean Squared Error (MSE): %5.3f' % mse_p)
    print('Root Mean Squared Error (RMSE): %5.3f' % rmse_p)
    print('Huber loss (huber): %5.3f' % hub_p)
    print('Standard Error Prediction (SEP): %5.3f' % sep)
    # print('RPD: %5.3f' % rpd)
    # print('Bias: %5.3f' %  bias)

    # Plot regression and figures of merit
    rangey = max(Y_valid) - min(Y_valid)
    rangex = max(Y_pred) - min(Y_pred)

    if plot_components is True:
        plt.figure(figsize=(15, 5))
        z = np.polyfit(np.ravel(Y_valid), np.ravel(Y_pred), 1)
        ax2 = plt.subplot(aspect=1)
        #         ax2.fill_between(Y_cal, y_est-y_err, y np.ravel(z[1]+z[0]*np.arange(150,240))-2.5*sep, facecolor='red', alpha=0.5)
        ax2.scatter(Y_valid, Y_pred, c='r', s=2)
        ax2.plot(Y_valid, z[1] + z[0] * Y_valid, c='blue', linewidth=2, label='linear fit')
        ax2.plot(Y_valid, Y_valid, color='orange', linewidth=1.5, label='y=x')
        plt.ylabel('Predicted')
        plt.xlabel('Measured')
        plt.title('Prediction from PLS')
        plt.legend(loc=4)

        # Print the scores on the plot
        plt.text(min(Y_pred) + 0.2 * rangex, max(Y_valid) - 0.1 * rangey, 'R$^{2}=$ %5.3f' % score_p)
        plt.text(min(Y_pred) + 0.2 * rangex, max(Y_valid) - 0.15 * rangey, 'RMSE: %5.3f' % rmse_p)
        plt.text(min(Y_pred) + 0.2 * rangex, max(Y_valid) - 0.2 * rangey, 'Huber loss: %5.3f' % hub_p)
        #plt.text(min(Y_pred)+0.02*rangex, max(Y_valid)-0.25*rangey, 'RPD: %5.3f' % rpd)
        #plt.text(min(Y_pred)+0.02*rangex, max(Y_valid)-0.3*rangey, 'Bias: %5.3f' %  bias)
        plt.show()

    return Y_calib, Y_pred, pls


## Function to benchmark model error metrics (adapted from DeepChemometrics)
## This function computes error between the predictions made by a "model" and the original data.
def benchmark(x_train, y_train, x_test, y_test, model):
    """
    Think of this as:
    model(x_train, y_train) -> trained_model
    predicted_y = trained_model(x_test)
    error_metric = compute_error_between(y_test, predicted_y)
    """
    ## ORIGINAL
    #     rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
    #     rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
    #     hub = huber(y_train, model.predict(X_train))
    #     hub_test = huber(y_test, model.predict(X_test))
    ######
    ## CONVERT Y values to initial scale before computing error metrics
    y_train_true = yscaler.inverse_transform(y_train)
    y_test_true = yscaler.inverse_transform(y_test)
    y_train_pred = yscaler.inverse_transform(model.predict(x_train)).reshape(y_train_true.shape)
    y_test_pred = yscaler.inverse_transform(model.predict(x_test)).reshape(y_test_true.shape)

    ## Compute error metrics
    rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    R2_train = r2_score(y_train_true, y_train_pred)
    R2_test = r2_score(y_test_true, y_test_pred)
    hub_train = huber(y_train_true, y_train_pred)
    hub_test = huber(y_test_true, y_test_pred)

    ## Print stuff
    print('\n\n*********** Benchmark results *********** \n')
    print("R2  \t(Train/Test) = \t{:.3f}\t/  {:.3f}".format(R2_train, R2_test))
    print("RMSE  \t(Train/Test) = \t{:.3f} \t/  {:.3f}".format(rmse_train, rmse_test))
    print("Huber \t(Train/Test) = \t{:.3f}\t/  {:.3f}".format(hub_train, hub_test))


class EmscScaler(object):
    def __init__(self, order=1):
        self.order = order
        self._mx = None

    def mlr(self, x, y):
        """Multiple linear regression fit of the columns of matrix x
        (dependent variables) to constituent vector y (independent variables)

        order -     order of a smoothing polynomial, which can be included
                    in the set of independent variables. If order is
                    not specified, no background will be included.
        b -         fit coeffs
        f -         fit result (m x 1 column vector)
        r -         residual   (m x 1 column vector)
        """

        if self.order > 0:
            s = scipy.ones((len(y), 1))
            for j in range(self.order):
                s = scipy.concatenate((s, (scipy.arange(0, 1 + (1.0 / (len(y) - 1)), 1.0 / (len(y) - 1)) ** j)[:, nA]),
                                      1)
            X = scipy.concatenate((x, s), 1)
        else:
            X = x

        # calc fit b=fit coefficients
        b = scipy.dot(scipy.dot(scipy.linalg.pinv(scipy.dot(scipy.transpose(X), X)), scipy.transpose(X)), y)
        f = scipy.dot(X, b)
        r = y - f

        return b, f, r

    def inverse_transform(self, X, y=None):
        print("Warning: inverse transform not possible with Emsc")
        return X

    def fit(self, X, y=None):
        """fit to X (get average spectrum), y is a passthrough for pipeline compatibility"""
        self._mx = scipy.mean(X, axis=0)[:, nA]

    def transform(self, X, y=None, copy=None):
        if type(self._mx) == type(None):
            print("EMSC not fit yet. run .fit method on reference spectra")
        else:
            # do fitting
            corr = scipy.zeros(X.shape)
            for i in range(len(X)):
                b, f, r = self.mlr(self._mx, X[i, :][:, nA])
                corr[i, :] = scipy.reshape((r / b[0, 0]) + self._mx, (corr.shape[1],))
            return corr

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

def dataaugment(x, betashift = 0.05, slopeshift = 0.05,multishift = 0.05):
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