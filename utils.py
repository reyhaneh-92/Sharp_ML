from scipy.io import loadmat
import pandas as pd
import numpy as np
import h5py
import matplotlib as plt
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, Sequential
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, datasets
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Model
import pickle
from sklearn.metrics import make_scorer, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from keras import backend as K
import seaborn as sns

path = '/panfs/jay/groups/0/ebtehaj/rahim035/sajad_s/project-4/revision'


def process_and_normalize(data):
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
    file_path = path + '/stats/stat_ocean_subset.mat'
    stat = sio.loadmat(file_path)
    std = stat['std_ocean_det']
    mean = stat['mean_ocean_det']
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
        

def score(y_obs, y_prd):
    """
    Calculate and display performance metrics for rain and snow predictions.

    This function computes True Positive Rate (TPR), False Positive Rate (FPR),
    Area Under Curve (AUC), and F1 Score for two classes: rain and snow.
    It prints these metrics in a tabular format.

    Args:
        y_obs (np.ndarray): Array of observed class labels.
        y_prd (np.ndarray): Array of predicted class labels.

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


def compute_error_metrics(prd_det, prd_rate, obs, label, phase):
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
    quant = np.quantile(obs[:, 1], 1)
    idx_TP = (prd_det == label) & (obs[:, 0] == label) & (obs[:, 1] < quant)

    # Calculating errors
    error = prd_rate[idx_TP] - obs[idx_TP, 1]
    bias = np.mean(error)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    mse = np.mean(error**2)

    # Creating the table (Commented out)
    table_score = [
        ['Metric', 'Value'],
        ['Bias', bias],
        ['MAE', mae],
        ['RMSE', rmse],
        ['MSE', mse]
    ]

    # Return the calculated error metrics
    return {'Bias': bias, 'MAE': mae, 'RMSE': rmse, 'MSE': mse}


from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
import numpy as np

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


def loc_rate(f_train, f_test, y_train, k_nn):
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
        
    metrics = compute_error_metrics(label_prd, rate_prd, y_tst, label_class, phase)

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
