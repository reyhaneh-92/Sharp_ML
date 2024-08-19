import pandas as pd
import numpy as np
import matplotlib as plt
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# from scikeras.wrappers import KerasClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from tabulate import tabulate
import tensorflow as tf
# from tensorflow.keras.layers import Dense, Activation, Dropout, Input
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras import layers, Sequential
from tensorflow import keras
from sklearn.metrics import roc_auc_score
# from tensorflow.keras import Model
import pickle
from sklearn.metrics import  f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from keras import backend as K
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


path = '/content/sharp_ml/'


def process_and_normalize(data, surf_type):
    """
    Normalize dataset features using provided mean and standard deviation.

    Args:
        data (pd.DataFrame): DataFrame containing features and targets.
                             Features are in the first 18 columns, targets in the last two.
        std (np.ndarray): Standard deviation of training features.
        mean (np.ndarray): Mean of training features.

    Returns:
        tuple: (np.ndarray, np.ndarray) - Normalized features and target variables.
    """
    file_path = path + f'/stats/stat_{surf_type}_subset.mat'
    stat = sio.loadmat(file_path)
    std = stat[f'std_{surf_type}_det']
    mean = stat[f'mean_{surf_type}_det']
    # Extract features and target variables
    X_data = data.iloc[:, :18].values
    y_data = data.iloc[:, 18:20].values

    # Normalize the features
    X_data = (X_data - mean) / std

    return X_data, y_data


class CategoricalFocalLoss(tf.keras.losses.Loss):

        def __init__(self, alpha, gamma):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def call(self, y_true, y_pred):

            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * K.log(y_pred)
            loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy
            return K.mean(K.sum(loss, axis=-1))
        

def classification_score(y_obs, y_prd):
    """
    Calculate and display performance metrics for rain and snow predictions.

    This function computes True Positive Rate (TPR), False Positive Rate (FPR),
    Area Under Curve (AUC), and F1 Score for two classes: rain and snow.
    It prints these metrics in a tabular format.

    Args:
        y_obs (np.ndarray): Array of observed class labels.
        y_prd (np.ndarray): Array of predicted class labels.
        classes: 0-->clear, 1-->rain, 2-->snow

    params:
        2:snow
        1:rain
        0:clear
    
    Returns:
        None
    """
    TR_snow = np.sum((y_obs == 2) & (y_prd == 2))
    FP_snow = np.sum((y_obs != 2) & (y_prd == 2))
    FN_snow = np.sum((y_obs == 2) & (y_prd != 2))
    TN_snow = np.sum((y_obs != 2) & (y_prd != 2))

    TR_rain = np.sum((y_obs == 1) & (y_prd == 1))
    FP_rain = np.sum((y_obs != 1) & (y_prd == 1))
    FN_rain = np.sum((y_obs == 1) & (y_prd != 1))
    TN_rain = np.sum((y_obs != 1) & (y_prd != 1))

    TPR_rain = TR_rain / (TR_rain + FN_rain)
    TPR_snow = TR_snow / (TR_snow + FN_snow)
    FPR_rain = FP_rain / (FP_rain + TN_rain)
    FPR_snow = FP_snow / (FP_snow + TN_snow)
    
    auc = roc_auc_score(to_categorical(y_obs), to_categorical(y_prd), average=None, multi_class='ovr')

    f1 = f1_score(y_obs, y_prd, average=None)

    # Create a table
    table = [
        ["Metric", "Rain", "Snow"],
        ["TPR", f"{TPR_rain:.3f}", f"{TPR_snow:.3f}"],
        ["FPR", f"{FPR_rain:.3f}", f"{FPR_snow:.3f}"],
        ["AUC", f"{auc[1]:.3f}", f"{auc[2]:.3f}"],
        ["F1 Score", f"{f1[1]:.3f}", f"{f1[2]:.3f}"]
    ]

    # Print the table
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


def regression_score(prd_det, prd_rate, obs, label, phase):
    """
    Compute error metrics for a given label in prediction data.

    This function calculates the bias, mean absolute error (MAE), root mean square error (RMSE), 
    and mean square error (MSE) for predictions that are considered true positives. It evaluates 
    the prediction accuracy against the observed values for a specified class label.

    Args:
        prd_det (np.ndarray): Array of predicted class labels.
        prd_rate (np.ndarray): Array of predicted rates corresponding to each observation.
        obs (np.ndarray): Array containing observed class labels and values.
        label (int): The class label to evaluate error metrics for.
        phase (str): A string identifier for the phase of analysis or processing.

    Returns:
        dict: A dictionary containing calculated error metrics with keys:
              'Bias', 'MAE', 'RMSE', 'MSE'.
    """
    # Identifying true positives
    quant_h = np.quantile(obs[:, 1], 0.99)
    quant_l = np.quantile(obs[:, 1], 0.015)

    idx_TP = (prd_det == label) & (obs[:, 0] == label) & (obs[:, 1] < quant_h) & (obs[:, 1] > quant_l)

    # Calculating errors
    error = prd_rate[idx_TP] - obs[idx_TP, 1]
    bias = np.mean(error)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    mse = np.mean(error**2)

    # Return the calculated error metrics
    print(f"bias is {bias:.3f}")
    print(f"mae is {mae:.3f}")
    print(f"rmse is {rmse:.3f}")
    return {'Bias': bias, 'MAE': mae, 'RMSE': rmse}


def find_knn(X, y, k):
    """
    Find the k-nearest neighbors for each sample in the given dataset.

    This function creates a NearestNeighbors object and fits it to the dataset X.
    It then finds the k-nearest neighbors for each sample in the dataset y.

    Args:
        X (np.ndarray): The dataset to fit the nearest neighbors model, typically the training data.
        y (np.ndarray): The dataset for which to find nearest neighbors, typically the test data.
        k (int): The number of nearest neighbors to find.

    Returns:
        tuple: (distances, indices)
            distances (np.ndarray): An array of distances to the nearest neighbors for each sample in y.
            indices (np.ndarray): An array of indices of the nearest neighbors in X for each sample in y.
    """
    # Create the nearest neighbors object
    nn = NearestNeighbors(n_neighbors=k)
    # Fit the nearest neighbors object to X
    nn.fit(X)
    # Find the k-nearest neighbors for each sample in y
    distances, indices = nn.kneighbors(y)
    return distances, indices


def rate_knn(f_train, f_test, y_train, k_nn):
    """
    Predict local rates using k-nearest neighbors.

    This function uses the k-nearest neighbors approach to predict local rates for the test set
    based on the training set.

    Args:
        f_train (np.ndarray): Feature set for training data.
        f_test (np.ndarray): Feature set for test data.
        y_train (np.ndarray): Target values for the training data.
        k_nn (int): Number of nearest neighbors to consider.

    Returns:
        tuple: (y_pred_loc, indices)
            y_pred_loc (np.ndarray): Predicted local rates for each sample in the test set.
            indices (np.ndarray): Indices of nearest neighbors in the training set for each test sample.
    """
    distances, indices = find_knn(f_train, f_test, k_nn)
    
    y_pred_loc = np.zeros(indices.shape)
    for i in range(len(f_test)):
        for j in range(k_nn):
            y_pred_loc[i, j] = y_train[indices[i, j]]
            
    return y_pred_loc, indices



def ridge_estimation(feature_trn, feature_tst, nn_idx, rate_trn, rate_tst_knn):
    """
    Estimate rates using Ridge regression with nearest neighbor features.

    This function estimates rates for a test set using Ridge regression. It leverages
    the nearest neighbor features from the training set to make predictions.

    Args:
        feature_trn (np.ndarray): Feature matrix for the training set.
        feature_tst (np.ndarray): Feature matrix for the test set.
        nn_idx (np.ndarray): Indices of nearest neighbors for each test sample.
        rate_trn (np.ndarray): Rates or target values for the training set.
        rate_tst_knn (np.ndarray): Predicted rates for the test set using k-nearest neighbors.

    Returns:
        np.ndarray: Predicted rates for each sample in the test set.
    """
    rate_prd = np.zeros(feature_tst.shape[0])
    for i in range(len(rate_tst_knn)):
        clf = Ridge(alpha=0.01)
        clf.fit(np.transpose(feature_trn[nn_idx[i, :], :]), feature_tst[i, :])
        rate_prd[i] = np.sum(rate_trn[nn_idx[i, :]] * clf.coef_)
    return rate_prd




plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
    
})



def plot_detailed_confusion_matrix(y_true, y_pred, ax, title, x_label = False, y_label=False):
    cm = confusion_matrix(y_true, y_pred)
    
    # Add totals for each class
    cm_tot = np.vstack([cm, np.sum(cm, axis=0)])
    cm_tot = np.hstack([cm_tot, np.sum(cm_tot, axis=1).reshape(-1, 1)])
    
    # Calculate the percentage equivalence for each value in the confusion matrix
    total_samples = np.sum(cm)  # Exclude the total row and column
    percent_equivalence = cm / total_samples * 100
    
    # Create a list to hold the annotations
    annotations = []
    
    # Iterate through the confusion matrix to create the annotations
    for i in range(cm_tot.shape[0]):
        row_annotations = []
        for j in range(cm_tot.shape[1]):
            value = cm_tot[i, j]
            if i < cm.shape[0] and j < cm.shape[1]:
                percent_value = percent_equivalence[i, j]
                row_annotations.append(f"{value}\n({percent_value:.2f}%)")
            elif i == cm.shape[0] and j < cm.shape[1]:  # Total row
                actual_total = cm_tot[i, j]
                correct = cm[j, j]
                correct_percent = correct / actual_total * 100 if actual_total != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}% )\n({false_percent:.2f}%)")
            elif i < cm.shape[0] and j == cm.shape[1]:  # Total column
                predicted_total = cm_tot[i, j]
                correct = cm[i, i]
                correct_percent = correct / predicted_total * 100 if predicted_total != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}%)\n({false_percent:.2f}% )")
            else:  # Bottom-right total cell
                correct_total = np.trace(cm)
                correct_percent = correct_total / value * 100 if value != 0 else 0
                false_percent = 100 - correct_percent
                row_annotations.append(f"{value}\n({correct_percent:.2f}%)\n({false_percent:.2f}%)")
        annotations.append(row_annotations)
    
    # Convert annotations to a numpy array
    annotations = np.array(annotations)
    
    # Create a mask for different colors
    mask = np.full(cm_tot.shape, '', dtype=object)
    # Diagonal
    for i in range(cm.shape[0]):
        mask[i, i] = 'diag'
    # Last row and last column
    mask[-1, :] = 'total'
    mask[:, -1] = 'total'
    # Remaining cells
    mask[mask == ''] = 'off_diag'
    
    # Create a custom colormap with the desired colors
    colors = {'diag': 'darkseagreen', 'total': 'whitesmoke', 'off_diag': 'wheat'}
    cmap = sns.color_palette([colors[key] for key in ['diag', 'total', 'off_diag']])
    
    # Convert mask to numerical values for colormap
    mask_num = np.zeros_like(mask, dtype=float)
    mask_num[mask == 'diag'] = 0
    mask_num[mask == 'total'] = 1
    mask_num[mask == 'off_diag'] = 2
    
    # Plot the heatmap
    sns.heatmap(mask_num, annot=False, cmap=plt.cm.colors.ListedColormap(cmap),
                xticklabels=['clear', 'rain', 'snow', 'Total'],
                yticklabels=['clear', 'rain', 'snow', 'Total'],
                linewidths=1, linecolor='black', cbar=False, ax=ax)  # Disable colorbar
    
    # Add annotations manually with colored text
    for i in range(cm_tot.shape[0]):
        for j in range(cm_tot.shape[1]):
            text = annotations[i, j]
            if i == cm.shape[0] or j == cm.shape[1]:  # Total rows and columns
                parts = text.split('\n')
                ax.text(j + 0.5, i + 0.45, parts[1],
                        ha='center', va='center', fontsize=12, fontweight='bold', rotation=0, color='green')
                ax.text(j + 0.5, i + 0.65, parts[2],
                        ha='center', va='center', fontsize=12, fontweight='bold', rotation=0, color='salmon')
            else:
                ax.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center', fontsize=12, fontweight='bold', rotation=45)
    if x_label:
        ax.set_xlabel('Predicted labels')
    if y_label:
        ax.set_ylabel('True labels')
    ax.set_title(title)


def plot_density_scatter(ax, y_tst, rate_prd, label_prd, label_class, threshold=0.1):
    """
    Plot a density scatter plot for predicted vs. observed rates with error metrics.

    This function generates a scatter plot where each point's color represents its density,
    highlighting regions with high concentrations of points. It also computes and displays
    error metrics for a specified class label.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis object where the plot will be drawn.
        y_tst (np.ndarray): Array containing true labels and additional observed data.
        rate_prd (np.ndarray): Array of predicted rates.
        label_prd (np.ndarray): Array of predicted class labels.
        label_class (int): The class label for which the plot and metrics are generated.
                            0: clear    1:rain   2: snow
        threshold (float, optional): The minimum value for filtering the observed and predicted rates.
                                     Defaults to 0.1.

    Returns:
        None
    """

    idx_label = (
        (y_tst[:, 0] == label_class) &
        (label_prd == label_class) &
        (y_tst[:, 1] > threshold) &
        (rate_prd > threshold)
    )
    
    # Determine phase for error metrics
    if label_class == 1:
        phase = 'rain'
    elif label_class == 2:
        phase = "snow"
    elif label_class == 0:
        phase = 'clear'
    else:
        print('Phase not defined')
        
    metrics = regression_score(label_prd, rate_prd, y_tst, label_class, phase)

    # Extract x and y data
    x = y_tst[idx_label, 1]
    y = rate_prd[idx_label]
    ulim, llim = np.quantile(x, 0.975), 0

    # Compute the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Create the scatter plot
    scatter = ax.scatter(x, y, c=z, cmap='jet', alpha=0.5)

    # Plot y=x line
    ax.plot([x.min(), x.max()], [x.min(), x.max()], color='black')

    # Set dynamic limits and aspect ratio
    ax.set_xlim(llim, ulim)
    ax.set_ylim(llim, ulim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("obs [mm/hr]")
    ax.set_ylabel("prd [mm/hr]")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('Density')

    # Add error metrics to the plot on the top left with a gray box
    metrics_text = '\n'.join([f"{key}: {value:.3f}" for key, value in metrics.items()])
    ax.text(
        0.02, 0.98, metrics_text,
        fontsize=16, verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        bbox=dict(facecolor='gray', alpha=0.3, edgecolor='black')
    )
    
    
def CDFmatch_orbit(CDF_ref, CDF_prd, biased):
    cdf_x = interp1d(CDF_prd[:,0],CDF_prd[:,1],fill_value="extrapolate")(biased)
    x_cdf= interp1d(CDF_ref[:,1],CDF_ref[:,0], fill_value="extrapolate")(cdf_x)
    RTV_DB=x_cdf
    return RTV_DB


#%% Orbital retrievals 

import tensorflow as tf
from tensorflow import keras
import os
import scipy.io as sio
import numpy as np
import glob
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
import pandas as pd
import pickle



# Function to load models
def load_models():
    # path_models = '/panfs/jay/groups/0/ebtehaj/rahim035/sajad_s/project-4/revision/models/'
    path_models = os.path.join(path, 'models/')

    
    models = {
        'ocean': {
            'dtc': pickle.load(open(path_models + 'ocean_det_xgb.pickle.dat', "rb")),
            'rain': keras.models.load_model(path_models + 'ocean_est_FL_rain_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False),
            'snow': keras.models.load_model(path_models + 'ocean_est_FL_snow_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False)
        },
        'land': {
            'dtc': pickle.load(open(path_models + 'land_det_xgb.pickle.dat', "rb")),
            'rain': keras.models.load_model(path_models + 'land_est_FL_rain_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False),
            'snow': keras.models.load_model(path_models + 'land_est_FL_snow_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False)
        },
        'coast': {
            'dtc': pickle.load(open(path_models + 'coast_det_xgb.pickle.dat', "rb")),
            'rain': keras.models.load_model(path_models + 'coast_est_FL_rain_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False),
            'snow': keras.models.load_model(path_models + 'coast_est_FL_snow_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False)
        },
        'snow': {
            'dtc': pickle.load(open(path_models + 'snowcover_det_xgb.pickle.dat', "rb")),
            'rain': keras.models.load_model(path_models + 'snowcover_est_FL_rain_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False),
            'snow': keras.models.load_model(path_models + 'snowcover_est_FL_snow_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False)
        },
        'ice': {
            'dtc': pickle.load(open(path_models + 'seaice_det_xgb.pickle.dat', "rb")),
            'rain': keras.models.load_model(path_models + 'seaice_est_FL_rain_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False),
            'snow': keras.models.load_model(path_models + 'seaice_est_FL_snow_10class.h5', custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(alpha=0.25, gamma=2)}, compile=False)
        }
    }

    # Extract specific layers as models
    models['ocean']['rain'] = tf.keras.Model(models['ocean']['rain'].input, models['ocean']['rain'].get_layer('fc_3').output)
    models['ocean']['snow'] = tf.keras.Model(models['ocean']['snow'].input, models['ocean']['snow'].get_layer('fc_3').output)

    models['land']['rain'] = tf.keras.Model(models['land']['rain'].input, models['land']['rain'].get_layer('fc_3').output)
    models['land']['snow'] = tf.keras.Model(models['land']['snow'].input, models['land']['snow'].get_layer('fc_3').output)

    models['coast']['rain'] = tf.keras.Model(models['coast']['rain'].input, models['coast']['rain'].get_layer('fc_3').output)
    models['coast']['snow'] = tf.keras.Model(models['coast']['snow'].input, models['coast']['snow'].get_layer('fc_3').output)

    models['snow']['rain'] = tf.keras.Model(models['snow']['rain'].input, models['snow']['rain'].get_layer('fc_3').output)
    models['snow']['snow'] = tf.keras.Model(models['snow']['snow'].input, models['snow']['snow'].get_layer('fc_3').output)

    models['ice']['rain'] = tf.keras.Model(models['ice']['rain'].input, models['ice']['rain'].get_layer('fc_3').output)
    models['ice']['snow'] = tf.keras.Model(models['ice']['snow'].input, models['ice']['snow'].get_layer('fc_3').output)

    return models

# Function to load feature data
def load_feature_data():
    # path_features = '/panfs/jay/groups/0/ebtehaj/rahim035/sajad_s/project-4/revision/features/'
    path_features = os.path.join(path,'features/')
    features_data = {
        'ocean': {
            'X_trn_rain_features_ocean': sio.loadmat(path_features + 'feature_dic_ocean.mat')['X_trn_rain_features_ocean'],
            'y_trn_rain_ocean': sio.loadmat(path_features + 'feature_dic_ocean.mat')['y_trn_rain_ocean'],
            'X_trn_snow_features_ocean': sio.loadmat(path_features + 'feature_dic_ocean.mat')['X_trn_snow_features_ocean'],
            'y_trn_snow_ocean': sio.loadmat(path_features + 'feature_dic_ocean.mat')['y_trn_snow_ocean']
        },
        'land': {
            'X_trn_rain_features_land': sio.loadmat(path_features + 'feature_dic_land.mat')['X_trn_rain_features_land'],
            'y_trn_rain_land': sio.loadmat(path_features + 'feature_dic_land.mat')['y_trn_rain_land'],
            'X_trn_snow_features_land': sio.loadmat(path_features + 'feature_dic_land.mat')['X_trn_snow_features_land'],
            'y_trn_snow_land': sio.loadmat(path_features + 'feature_dic_land.mat')['y_trn_snow_land']
        },
        'coast': {
            'X_trn_rain_features_coast': sio.loadmat(path_features + 'feature_dic_coast.mat')['X_trn_rain_features_coast'],
            'y_trn_rain_coast': sio.loadmat(path_features + 'feature_dic_coast.mat')['y_trn_rain_coast'],
            'X_trn_snow_features_coast': sio.loadmat(path_features + 'feature_dic_coast.mat')['X_trn_snow_features_coast'],
            'y_trn_snow_coast': sio.loadmat(path_features + 'feature_dic_coast.mat')['y_trn_snow_coast']
        },
        'snow': {
            'X_trn_rain_features_snow': sio.loadmat(path_features + 'feature_dic_snowcover.mat')['X_trn_rain_features_snowcover'],
            'y_trn_rain_snow': sio.loadmat(path_features + 'feature_dic_snowcover.mat')['y_trn_rain_snowcover'],
            'X_trn_snow_features_snow': sio.loadmat(path_features + 'feature_dic_snowcover.mat')['X_trn_snow_features_snowcover'],
            'y_trn_snow_snow': sio.loadmat(path_features + 'feature_dic_snowcover.mat')['y_trn_snow_snowcover']
        },
        'ice': {
            'X_trn_rain_features_ice': sio.loadmat(path_features + 'feature_dic_seaice.mat')['X_trn_rain_features_seaice'],
            'y_trn_rain_ice': sio.loadmat(path_features + 'feature_dic_seaice.mat')['y_trn_rain_seaice'],
            'X_trn_snow_features_ice': sio.loadmat(path_features + 'feature_dic_seaice.mat')['X_trn_snow_features_seaice'],
            'y_trn_snow_ice': sio.loadmat(path_features + 'feature_dic_seaice.mat')['y_trn_snow_seaice']
        }
    }
    return features_data

# Function to load stats
def load_stats():
    # path_stat = '/panfs/jay/groups/0/ebtehaj/rahim035/sajad_s/project-4/revision/stats/'
    path_stat = os.path.join(path, 'stats/')
    stats = {
        'ocean': sio.loadmat(path_stat + 'stat_ocean_detection.mat'),
        'land': sio.loadmat(path_stat + 'stat_land_detection.mat'),
        'coast': sio.loadmat(path_stat + 'stat_coast_detection.mat'),
        'snow': sio.loadmat(path_stat + 'stat_snowcover_detection.mat'),
        'ice': sio.loadmat(path_stat + 'stat_seaice_detection.mat')
    }
    return stats

# # Function to load CDFs
# def load_cdfs():
#     CDFs = sio.loadmat("/panfs/jay/groups/0/ebtehaj/rahim035/sajad_s/project-4/revision/orbital/trained_CDF.mat")

#     CDF_data = {
#         'rain': {
#             'ocean': {'prd': CDFs['CDF_prd_rain_ocean'], 'ref': CDFs['CDF_ref_rain_ocean']},
#             'land': {'prd': CDFs['CDF_prd_rain_land'], 'ref': CDFs['CDF_ref_rain_land']},
#             'coast': {'prd': CDFs['CDF_prd_rain_coast'], 'ref': CDFs['CDF_ref_rain_coast']},
#             'snow': {'prd': CDFs['CDF_prd_rain_snow'], 'ref': CDFs['CDF_ref_rain_snow']},
#             'ice': {'prd': CDFs['CDF_prd_rain_ice'], 'ref': CDFs['CDF_ref_rain_ice']}
#         },
#         'snow': {
#             'ocean': {'prd': CDFs['CDF_prd_snow_ocean'], 'ref': CDFs['CDF_ref_snow_ocean']},
#             'land': {'prd': CDFs['CDF_prd_snow_land'], 'ref': CDFs['CDF_ref_snow_land']},
#             'coast': {'prd': CDFs['CDF_prd_snow_coast'], 'ref': CDFs['CDF_ref_snow_coast']},
#             'snow': {'prd': CDFs['CDF_prd_snow_snow'], 'ref': CDFs['CDF_ref_snow_snow']},
#             'ice': {'prd': CDFs['CDF_prd_snow_ice'], 'ref': CDFs['CDF_ref_snow_ice']}
#         }
#     }
#     return CDF_data

# Function to preprocess input data for a specific orbit file
def preprocess_input_data(file_path, stats):
    var_names = ['10v', '10h', '18v', '18h', '23v', '36v', '36h', '89v', '89h', '166v', '166h', '183-3', '183-7', 'tclw', 'tciw', 't2m', 'tcwv', 'cape']
    f_orbit = sio.loadmat(file_path)

    # Prepare detection data
    X_detection = np.transpose(f_orbit['X_detection'][:, :])
    X_detection_df = pd.DataFrame(X_detection, columns=var_names)
    X_detection_df = X_detection_df[['10v', '10h', '18v', '18h', '23v', '36v', '36h', '89v', '89h', '166v', '166h', '183-3', '183-7', 'tciw', 'tclw', 'tcwv', 't2m', 'cape']]

    # Extract mean and std for normalization
    mean_detection = {
        'ocean': stats['ocean']['mean_ocean_det'][0, :18],
        'land': stats['land']['mean_land_det'][0, :18],
        'coast': stats['coast']['mean_coast_det'][0, :18],
        'snow': stats['snow']['mean_snowcover_det'][0, :18],
        'ice': stats['ice']['mean_seaice_det'][0, :18]
    }
    std_detection = {
        'ocean': stats['ocean']['std_ocean_det'][0, :18],
        'land': stats['land']['std_land_det'][0, :18],
        'coast': stats['coast']['std_coast_det'][0, :18],
        'snow': stats['snow']['std_snowcover_det'][0, :18],
        'ice': stats['ice']['std_seaice_det'][0, :18]
    }

    # Normalized data for different regions
    X_normalized = {region: (X_detection_df.values - mean_detection[region]) / std_detection[region] for region in mean_detection}

    return X_normalized

# Function to predict features
def predict_features(models, X_normalized):
    predictions = {
        'ocean': {
            'det': models['ocean']['dtc'].predict(X_normalized['ocean']),
            'rain': models['ocean']['rain'].predict(X_normalized['ocean']),
            'snow': models['ocean']['snow'].predict(X_normalized['ocean'])
        },
        'land': {
            'det': models['land']['dtc'].predict(X_normalized['land']),
            'rain': models['land']['rain'].predict(X_normalized['land']),
            'snow': models['land']['snow'].predict(X_normalized['land'])
        },
        'coast': {
            'det': models['coast']['dtc'].predict(X_normalized['coast']),
            'rain': models['coast']['rain'].predict(X_normalized['coast']),
            'snow': models['coast']['snow'].predict(X_normalized['coast'])
        },
        'snow': {
            'det': models['snow']['dtc'].predict(X_normalized['snow']),
            'rain': models['snow']['rain'].predict(X_normalized['snow']),
            'snow': models['snow']['snow'].predict(X_normalized['snow'])
        },
        'ice': {
            'det': models['ice']['dtc'].predict(X_normalized['ice']),
            'rain': models['ice']['rain'].predict(X_normalized['ice']),
            'snow': models['ice']['snow'].predict(X_normalized['ice'])
        }
    }
    return predictions

# Function to localize retrievals
def localize_retrievals(predictions, features_data, k_nn_rain, k_nn_snow):
    localized_rates = {}
    for region in predictions:
        localized_rates[region] = {
            'rain': rate_knn(features_data[region]['X_trn_rain_features_' + region], predictions[region]['rain'], features_data[region]['y_trn_rain_' + region][:, 1], k_nn_rain),
            'snow': rate_knn(features_data[region]['X_trn_snow_features_' + region], predictions[region]['snow'], features_data[region]['y_trn_snow_' + region][:, 1], k_nn_snow),
            'det': predictions[region]['det']
        }
    return localized_rates



# Function to reconstruct orbit data
# Function to reconstruct orbit data
def reconstruct_orbit(localized_rates, orbit_num):
    orb = path + '/orbital/Orbit_' + orbit_num + '.mat'
    GPROF = sio.loadmat(orb)

    SurfType = GPROF['A2_GPROF']['surfaceType'][0][0]
    Lat = GPROF["A2_GPROF"]["Lat"][0][0]
    Lon = GPROF["A2_GPROF"]["Lon"][0][0]
    [n1, n2] = Lat.shape

    X_precip_label = np.zeros((n1, n2))
    X_snow = np.zeros((n1, n2))
    X_rain = np.zeros((n1, n2))
    X_rain_knn = np.zeros((n1, n2, 20))
    X_snow_knn = np.zeros((n1, n2, 20))

    for z in range(n1 * n2):
        [idx_i, idx_j] = np.unravel_index(z, (n1, n2), 'F')
        if (SurfType[idx_i, idx_j] == 0 or SurfType[idx_i, idx_j] == 16):
            X_precip_label[idx_i, idx_j] = localized_rates['ocean']['det'][z]
            X_rain[idx_i, idx_j] = np.mean(localized_rates['ocean']['rain'][0][z])
            X_snow[idx_i, idx_j] = np.mean(localized_rates['ocean']['snow'][0][z])
        elif SurfType[idx_i, idx_j] == 100 or SurfType[idx_i, idx_j] == 15:
            X_precip_label[idx_i, idx_j] = localized_rates['land']['det'][z]
            X_rain[idx_i, idx_j] = np.mean(localized_rates['land']['rain'][0][z])
            X_snow[idx_i, idx_j] = np.mean(localized_rates['land']['snow'][0][z])
        elif SurfType[idx_i, idx_j] == 200:
            X_precip_label[idx_i, idx_j] = localized_rates['coast']['det'][z]
            X_rain[idx_i, idx_j] = np.mean(localized_rates['coast']['rain'][0][z])
            X_snow[idx_i, idx_j] = np.mean(localized_rates['coast']['snow'][0][z])
        elif SurfType[idx_i, idx_j] == 300:
            X_precip_label[idx_i, idx_j] = localized_rates['snow']['det'][z]
            X_rain[idx_i, idx_j] = np.mean(localized_rates['snow']['rain'][0][z])
            X_snow[idx_i, idx_j] = np.mean(localized_rates['snow']['snow'][0][z])
        elif SurfType[idx_i, idx_j] == 400:
            X_precip_label[idx_i, idx_j] = localized_rates['ice']['det'][z]
            X_rain[idx_i, idx_j] = np.mean(localized_rates['ice']['rain'][0][z])
            X_snow[idx_i, idx_j] = np.mean(localized_rates['ice']['snow'][0][z])
        elif SurfType[idx_i, idx_j] != 0 and SurfType[idx_i, idx_j] != 16 and SurfType[idx_i, idx_j] != 15 and SurfType[idx_i, idx_j] != 100 and SurfType[idx_i, idx_j] != 200 and SurfType[idx_i, idx_j] != 300 and SurfType[idx_i, idx_j] != 400 and not np.isnan(SurfType[idx_i, idx_j]):
            X_precip_label[idx_i, idx_j] = localized_rates['land']['det'][z]
            X_rain[idx_i, idx_j] = np.mean(localized_rates['land']['rain'][0][z])
            X_snow[idx_i, idx_j] = np.mean(localized_rates['land']['snow'][0][z])

    return X_rain, X_snow, X_precip_label, Lat, Lon



def load_colormaps(mat_file_path):
    # Load the colormaps from the .mat file
    Cmap = sio.loadmat(mat_file_path)
    
    # Extract colormaps from the loaded data
    cmap_rain = Cmap['Cmap_rain']
    cmap_snow = Cmap['Cmap_snow']
    
    # Create ListedColormaps
    cmap_snow = mpl.colors.ListedColormap(cmap_snow, name='myColorMap_snow', N=cmap_snow.shape[0])
    cmap_rain = mpl.colors.ListedColormap(cmap_rain, name='myColorMap_rain', N=cmap_rain.shape[0])
    
    # Define a custom color list for rain
    colorlist_rain = ["darkorange", "gold", "lawngreen", "lightseagreen"]
    
    # Create a LinearSegmentedColormap and reverse it
    cmap_rain = LinearSegmentedColormap.from_list('testCmap', colors=colorlist_rain, N=256)
    cmap_rain = cmap_rain.reversed()
    
    return cmap_rain, cmap_snow

