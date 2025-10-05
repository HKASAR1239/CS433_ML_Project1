# Pre-processing of the data before feeding it to the model

import os
import numpy as np
import matplotlib.pyplot as plt
import implementations as im

#load data from a csv file
def load_data(file_name):
    """Load data from a csv file.
    The csv file should be in the data/dataset/ folder relative to this file.

    input:
    file_name: str, name of the csv file
    output:
    data: numpy array of shape (n_samples, n_features)
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))+'/data/dataset/'
    data = np.array(np.genfromtxt(dir_path+file_name, delimiter=','))
    return data


def assess_missing_features(data): 
    """Assess the percentage of missing data for each feature in the dataset.

    input:
    data: numpy array of shape (n_samples, n_features)

    output:
    missing_data: list of length n_features, where each element is the percentage of missing data for that feature
    """
    missing_data = []
    for column in data.T:
        missing_percentage = np.sum(np.isnan(column)) / len(column)
        missing_data.append(missing_percentage)
    return missing_data

def remove_missing_features(data, threshold=0.8):
    """Remove features with a percentage of missing data above a certain threshold.

    input:
    data: numpy array of shape (n_samples, n_features)
    threshold: float, percentage of missing data above which a feature is removed

    output:
    data: numpy array of shape (n_samples, n_features_removed)
    """
    missing_data = assess_missing_features(data)
    features_to_keep = [i for i, missing in enumerate(missing_data) if missing <= threshold]
    features_to_remove = [i for i, missing in enumerate(missing_data) if missing > threshold]
    return data[:, features_to_keep], features_to_remove


def assess_missing_data_points(data, threshold=0.8):
    """Assess the percentage of missing features for each sample in the dataset.

    input:
    data: numpy array of shape (n_samples, n_features)

    output:
    missing_data: list of length n_samples, where each element is the percentage of missing features for that data point
    """

    missing_data = []
    for row in data:
        missing_percentage = np.sum(np.isnan(row)) / len(row)
        missing_data.append(missing_percentage)
    return missing_data

def remove_missing_data_points(data, threshold=0.5):
    """Remove data points with a percentage of missing features above a certain threshold.

    input:
    data: numpy array of shape (n_samples, n_features)
    threshold: float, percentage of missing features above which a data point is removed

    output:
    data: numpy array of shape (n_samples_removed, n_features)
    """
    missing_data = assess_missing_data_points(data)
    above_threshold = [i for i, missing in enumerate(missing_data) if missing > threshold]
    points_to_keep = [i for i, missing in enumerate(missing_data) if missing <= threshold]
    points_to_remove = [i for i, missing in enumerate(missing_data) if missing > threshold]
    
    return data[points_to_keep, :], points_to_remove


def fill_missing_data_mode(data):
    """Fill missing data with the mode of the feature.

    input:
    data: numpy array of shape (n_samples, n_features)

    output:
    data: numpy array of shape (n_samples, n_features)
    """

    for i in range(data.shape[1]):
        column = data[:, i]
        data[:, i] = fill_column_mode(column)
    return data

def fill_column_mode(column):
    """Fill missing data in a single column with the mode of that column.

    input:
    column: numpy array of shape (n_samples,) with some np.nan values
    output:
    column: numpy array of shape (n_samples,) with np.nan values filled with the mode of the column
    """ 
    if np.all(np.isnan(column)):
        return column
    values, counts = np.unique(column[~np.isnan(column)], return_counts=True)
    mode = values[np.argmax(counts)]
    column[np.isnan(column)] = mode
    return column


def fill_missing_data_0(data):
    """Fill missing data with 0.

    input:
    data: numpy array of shape (n_samples, n_features)

    output:
    data: numpy array of shape (n_samples, n_features)
    """
    data[np.isnan(data)] = 0
    return data


def get_variance(data):
    """Get the variance of each feature in the dataset.

    input:
    data: numpy array of shape (n_samples, n_features)

    output:
    variance: list of length n_features, where each element is the variance of that feature
    """
    variance = []
    for column in data.T:
        variance.append(np.nanvar(column))
    return variance

def IQR(data,factor=1.5):
    """Remove outliers by applying IQR method : computes the 25th and 75th percentile, then form bounds

        Input : numpy array of shape (n_samples,n_features)

        Output : numpy array filtered without the outliers
    """
    # Compute percentiles and IQR 
    Q1 = np.percentile(data,0.25,axis=0)
    Q3 = np.percentile(data,0.75,axis=0)
    IQR = Q3-Q1

    # Determine the bounds
    lowerBound = Q1 - factor * IQR
    upperBound = Q3 + factor * IQR

    # Filtering 
    mask = np.all((data <= upperBound) & (data >= lowerBound),axis=1)
    return data[mask]



def plot_missing_data(missing_data, title):
    print(np.mean(missing_data))  # Print average percentage of missing data across all features
    print(np.median(missing_data))  # Print median percentage of missing data across all features
    plt.hist(missing_data, bins=20)
    plt.title(title)
    plt.xlabel("Percentage of missing data")
    plt.ylabel("Occurences")
    plt.show()

def normalize_feature(column):
    """ Normalize one olumn of the data (one feature) to have mean 0 and variance 1.
    input:
    column: numpy array of shape (n_samples,)
    output:
    column: numpy array of shape (n_samples,) normalized
    """
    mean = np.mean(column)
    std = np.std(column)
    if std == 0:
        return column - mean
    return (column - mean) / std



def fill_data(data, remove_features = [], remove_points = [], threshold = True, threshold_features=0.9, threshold_points=0.6, normalize=True,remove_outliers=True):
    """ Pre-process the data before feeding it to the model.
    Remove data points or features with percentage of missing data above a certain threshold if threshold is True (for training data).
    Remove specified features and data points if remove_features and remove_points are not empty (for test data).
    Fill features that have only one type of value (except np.nan) with 0.
    Fill remaining missing data with the mode of the feature.
    Optionally normalize the data to have mean 0 and variance 1.
    
    input:
    data: numpy array of shape (n_samples, n_features)
    remove_features: list of int, indices of features to remove
    remove_points: list of int, indices of data points to remove
    threshold_features: float, percentage of missing data above which a feature is removed
    threshold_points: float, percentage of missing features above which a data point is removed
    normalize: bool, whether to normalize the data to have mean 0 and variance 1

    output:
    data: numpy array of shape (n_samples_removed, n_features_removed), final pre-processed data
    removed_features: list of int, indices of features that were removed
    removed_points: list of int, indices of data points that were removed
    """
    #Remove specified features and points
    data = np.delete(data, remove_features, axis=1)
    data = np.delete(data, remove_points, axis=0)

    if threshold: # remove features and points based on threshold
        data, removed_features = remove_missing_features(data, threshold=threshold_features)
        data, removed_points = remove_missing_data_points(data, threshold=threshold_points)
    else: # do not remove features and points based on threshold
        removed_features = []
        removed_points = []


    for i in range(data.shape[1]):

        column = data[:, i]

        if np.all(np.isnan(column)): # if all values are np.nan, skip
            continue

        if np.all(~np.isnan(column)): # if there is no missing data, skip
            if normalize:
                column = normalize_feature(column)
            data[:, i] = column
            continue

        unique_values = np.unique(column[~np.isnan(column)]) # unique values excluding np.nan
        if len(unique_values) == 1: # check if there is only one unique value (except np.nan)
            column = fill_missing_data_0(column)

        else:
            column = fill_column_mode(column) 
        

        if normalize:
            column = normalize_feature(column)
        
        data[:, i] = column
    if remove_outliers:
        data = IQR(data)

    return data, removed_features, removed_points

# ------------------------------------------------- FUNCTIONS FOR DATA PROCESSING ---------------------------------------------

def build_k_indices(y: np.ndarray, k_fold: int, seed: int):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    n_row = y.shape[0]
    interval = int(n_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(n_row)
    k_indices = [indices[k * interval : (k+1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def CrossValidation1Fold(y: np.ndarray, x: np.ndarray, k_fold: int, seed: int,model, **model_kwargs):
    """
    Perform a folding for cross validation with a specified model
    """
    k_indices = build_k_indices[y,x,k_fold,seed]
    xTest = x[k_indices]
    yTest = y[k_indices]
    trainIdx = k_indices[np.arange(len(k_indices)) != k].ravel()
    xTrain,yTrain = x[trainIdx], y[trainIdx]

    w,mse = model(yTrain,xTrain,**model_kwargs)

    yPred = xTest @ w
    errorTest = im._mse_loss(yTest,xTest,w)
    errorTrain = im._mse_loss(yTrain,xTrain,w)

    return errorTest,errorTrain

def CrossValidation(y: np.ndarray, x: np.ndarray, k_fold: int, seed: int,model, **model_kwargs):
    """
    Perform a cross validation with a specified model
    """
    errorsTest = []
    errorsTrain = []
    for k in range(k_fold):
        errorTest,errorTrain = CrossValidation1Fold(y,x,k_fold,seed,model,**model_kwargs)
        errorsTest.append(errorTest)
        errorsTrain.append(errorTrain)

    return errorsTest,errorsTrain



if __name__ == "__main__":
    
    x_train = load_data('x_train.csv')
    x_test = load_data('x_test.csv')
    y_train = load_data('y_train.csv')

    filled_x_train, removed_features, removed_points = fill_data(x_train, remove_features = [], remove_points = [], threshold = True, threshold_features=0.9, threshold_points=0.6, normalize=True)
    np.savetxt(os.path.dirname(os.path.realpath(__file__))+'/data/processed/filled_x_train_f09-p06-n.csv', filled_x_train, delimiter=',')

    # remove the same features and points from x_test as were removed from x_train
    filled_x_test, _, _ = fill_data(x_test, remove_features = removed_features, remove_points = removed_points, threshold = False, normalize=True)
    np.savetxt(os.path.dirname(os.path.realpath(__file__))+'/data/processed/filled_x_test_f09-p06-n.csv', filled_x_test, delimiter=',')

    # remove the same points from y_train as were removed from x_train
    y_train, _ = remove_missing_data_points(y_train, threshold=0.6)
    np.savetxt(os.path.dirname(os.path.realpath(__file__))+'/data/processed/filled_y_train_f09-p06-n.csv', y_train, delimiter=',')

    # Print the shape of the processed data to verify that the number of features and points match
    print(filled_x_train.shape)
    print(filled_x_test.shape)
    print(y_train.shape)


