# Pre-processing of the data before feeding it to the model

import os
import numpy as np
import matplotlib.pyplot as plt

#load data from a csv file
def load_data(file_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))+'/data/dataset/'
    data = np.array(np.genfromtxt(dir_path+file_name, delimiter=','))
    return data


def assess_missing_data(data): 
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



def plot_missing_data(missing_data, title):
    print(np.mean(missing_data))  # Print average percentage of missing data across all features
    print(np.median(missing_data))  # Print median percentage of missing data across all features
    plt.hist(missing_data, bins=20)
    plt.title(title)
    plt.xlabel("Percentage of missing data")
    plt.ylabel("Occurences")
    plt.show()



def remove_missing_features(data, threshold=0.8):
    """Remove features with a percentage of missing data above a certain threshold.

    input:
    data: numpy array of shape (n_samples, n_features)
    threshold: float, percentage of missing data above which a feature is removed

    output:
    data: numpy array of shape (n_samples, n_features_removed)
    """
    missing_data = assess_missing_data(data)
    features_to_keep = [i for i, missing in enumerate(missing_data) if missing <= threshold]
    features_to_remove = [i for i, missing in enumerate(missing_data) if missing > threshold]
    return data[:, features_to_keep], features_to_remove


def assess_missing_points(data, threshold=0.8):
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
    missing_data = assess_missing_points(data)
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
        if np.all(np.isnan(column)):
            continue  # Skip if all values are NaN
        values, counts = np.unique(column[~np.isnan(column)], return_counts=True)
        mode = values[np.argmax(counts)]
        column[np.isnan(column)] = mode
        data[:, i] = column
    return data

    


def fill_missing_data_0(data):
    """Fill missing data with 0.

    input:
    data: numpy array of shape (n_samples, n_features)

    output:
    data: numpy array of shape (n_samples, n_features)
    """
    data[np.isnan(data)] = 0
    return data



if __name__ == "__main__":
    
    x_train = load_data('x_train.csv')
    plot_missing_data(assess_missing_data(x_train), title="Percentage of missing data per feature")
    plot_missing_data(assess_missing_points(x_train), title="Percentage of missing features per data point")

    for i in [0.8, 0.9, 0.95, 0.99]:
        x_train_reduced, removed_features = remove_missing_features(x_train, threshold=i)
        print(f"Threshold: {i}, Removed features: {len(removed_features)}")

    for i in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        x_train_reduced, removed_points = remove_missing_data_points(x_train, threshold=i)
        print(f"Threshold: {i}, Removed data points: {len(removed_points)}")
    

        