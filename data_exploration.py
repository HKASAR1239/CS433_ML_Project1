# Pre-processing of the data before feeding it to the model

import os
import numpy as np
import matplotlib.pyplot as plt
import implementations as im
import helpers as hl

# ----------------------------------- LOADING DATA --------------------------------------------------------------


def load_data(file_name):
    """Load data from a csv file.
    The csv file should be in the data/dataset/ folder relative to this file.

    input:
    file_name: str, name of the csv file
    output:
    data: numpy array of shape (n_samples, n_features)
    """
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data/dataset/"
    data = np.array(np.genfromtxt(dir_path + file_name, delimiter=","))
    return data


# ----------------------------------- DATA EXPLORATION ----------------------------------------------------------
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
    features_to_keep = [
        i for i, missing in enumerate(missing_data) if missing <= threshold
    ]
    features_to_remove = [
        i for i, missing in enumerate(missing_data) if missing > threshold
    ]
    return data[:, features_to_keep], features_to_remove


def assess_missing_data_points(data):
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
    above_threshold = [
        i for i, missing in enumerate(missing_data) if missing > threshold
    ]
    points_to_keep = [
        i for i, missing in enumerate(missing_data) if missing <= threshold
    ]
    points_to_remove = [
        i for i, missing in enumerate(missing_data) if missing > threshold
    ]

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


# ----------------------------------- HELPERS FOR DATA PROCESSING -----------------------------------------------


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


def IQR(data, factor=1.5):
    """
    Remove outliers using IQR method. First handles obvious placeholder values,
    then applies statistical outlier detection.

    Input: numpy array of shape (n_samples, n_features)
    Output: numpy array with outliers replaced by NaN
    """
    data = data.copy()

    # Step 1: Replace obvious missing data placeholders FIRST
    placeholder_values = [-999999, 999999, 99999]
    for placeholder in placeholder_values:
        data[data == placeholder] = np.nan

    # Step 2: Replace extreme values (likely data errors)
    data[np.abs(data) > 1e5] = np.nan

    initial_nans = np.isnan(data).sum()

    # Step 3: Apply IQR on remaining valid data
    Q1 = np.nanpercentile(data, 25, axis=0)
    Q3 = np.nanpercentile(data, 75, axis=0)
    IQR_val = Q3 - Q1

    lowerBound = Q1 - factor * IQR_val
    upperBound = Q3 + factor * IQR_val

    # Mark statistical outliers as NaN
    for i in range(data.shape[1]):
        column = data[:, i]
        # Only check non-NaN values
        valid_mask = ~np.isnan(column)
        outlier_mask = (
            (column < lowerBound[i]) | (column > upperBound[i])
        ) & valid_mask
        data[outlier_mask, i] = np.nan

    final_nans = np.isnan(data).sum()
    print(f"IQR processing:")
    print(f"  - Placeholders replaced: {initial_nans} values")
    print(f"  - Statistical outliers marked: {final_nans - initial_nans} values")
    print(f"  - Total NaN: {final_nans} ({final_nans/data.size*100:.2f}%)")

    return data


def plot_missing_data(missing_data, title):
    print(
        np.mean(missing_data)
    )  # Print average percentage of missing data across all features
    print(
        np.median(missing_data)
    )  # Print median percentage of missing data across all features
    plt.hist(missing_data, bins=20)
    plt.title(title)
    plt.xlabel("Percentage of missing data")
    plt.ylabel("Occurences")
    plt.show()


def normalize_feature(column):
    """Normalize one olumn of the data (one feature) to have mean 0 and variance 1.
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


def identify_categorical_features(data, max_unique_ratio=0.05):
    """
    Identify which features are categorical vs continuous.

    A feature is considered categorical if:
    1. It has a small number of unique values (< 5% of samples)
    2. All values are integers
    3. Has <= 20 unique values

    Input:
        data: np.ndarray of shape (n_samples, n_features)
        max_unique_ratio: float, max ratio of unique values to total samples

    Output:
        categorical_mask: boolean array of shape (n_features,)
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    categorical_mask = np.zeros(n_features, dtype=bool)

    for i in range(n_features):
        feature = data[:, i]
        unique_values = np.unique(feature)
        n_unique = len(unique_values)

        # Check if all values are integers (or very close to integers)
        is_integer = np.allclose(feature, np.round(feature))

        # Criteria for categorical:
        # 1. Small number of unique values relative to sample size
        # 2. Integer values
        # 3. Absolute cap on unique values
        is_few_unique = n_unique < max(20, n_samples * max_unique_ratio)
        is_very_few = n_unique <= 20

        if (is_integer and is_few_unique) or is_very_few:
            categorical_mask[i] = True
            # print(f"  Feature {i}: Categorical ({n_unique} unique values)")

    return categorical_mask


# ----------------------------------- HELPERS FOR CROSS VALIDATION ----------------------------------------------


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
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def build_k_indices_knn(y: np.ndarray, k_fold: int, percent_data: float, seed: int):
    """build k indices for training a kNN model with k-fold.
    Args:
        y:      shape=(N,)
        x:      shape=(N,D)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed
    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    # Select a percentage of the data for kNN training
    n_row = y.shape[0]
    n_selected = int(n_row * percent_data)
    np.random.seed(seed)
    selected_indices = np.random.permutation(n_row)[:n_selected]
    interval = int(n_selected / k_fold)
    indices = selected_indices
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# ----------------------------------- FUNCTIONS FOR OUTLIER TREATMENT -------------------------------------------


def clean_placeholders_only(data):
    """
    Replace ONLY placeholder values that are clearly errors.
    Do NOT touch real extreme values that may be informative.

    Input: numpy array of shape (n_samples, n_features)
    Output: numpy array with placeholders replaced by NaN
    """
    data = data.copy()

    print("\n=== CLEANING PLACEHOLDERS ===")
    initial_valid = np.sum(~np.isnan(data))

    # Step 1: Obvious placeholder values
    placeholder_values = [-999, -999999, 999999, -9999, 99999, 9999, 99000, 99900]
    for placeholder in placeholder_values:
        mask = np.isclose(data, placeholder, atol=1e-6)
        count = np.sum(mask)
        if count > 0:
            print(f"  Replacing {count} occurrences of {placeholder}")
            data[mask] = np.nan

    # Step 2: Physically impossible values (very extreme)
    # Only if REALLY absurd (>1 million, etc.)
    extreme_mask = np.abs(data) > 1e6
    extreme_count = np.sum(extreme_mask)
    if extreme_count > 0:
        print(f"  Replacing {extreme_count} extreme values (>1e6)")
        data[extreme_mask] = np.nan

    final_valid = np.sum(~np.isnan(data))
    print(f"  Total replaced: {initial_valid - final_valid} values")
    print(
        f"  Remaining NaN: {np.sum(np.isnan(data))} ({np.sum(np.isnan(data))/data.size*100:.2f}%)"
    )

    return data


def smart_outlier_removal(data, factor=3.0, per_feature_threshold=0.01):
    """
    Intelligent outlier removal for medical data.

    Strategy:
    1. Replace only placeholders
    2. Very permissive IQR (factor=3.0 instead of 1.5)
    3. Limit to 1% of max outliers per feature
    4. Only mark truly extreme values

    Input:
        data: numpy array (n_samples, n_features)
        factor: IQR multiplier (3.0 = very permissive)
        per_feature_threshold: max % of outliers to mark per feature

    Output:
        data: numpy array with outliers replaced by NaN
    """
    data = data.copy()

    print("\n=== SMART OUTLIER REMOVAL ===")
    print(f"IQR factor: {factor} (permissive)")
    print(f"Max outliers per feature: {per_feature_threshold*100}%")

    # Step 1: Clean placeholders first
    data = clean_placeholders_only(data)

    initial_nans = np.isnan(data).sum()

    # Step 2: Permissive IQR per feature
    total_outliers_marked = 0

    for i in range(data.shape[1]):
        column = data[:, i].copy()
        valid_mask = ~np.isnan(column)

        if np.sum(valid_mask) < 10:  # Skip features with too little data
            continue

        valid_data = column[valid_mask]

        # Calculate IQR
        Q1 = np.percentile(valid_data, 25)
        Q3 = np.percentile(valid_data, 75)
        IQR_val = Q3 - Q1

        if IQR_val == 0:  # Constant feature
            continue

        # Very permissive bounds
        lower_bound = Q1 - factor * IQR_val
        upper_bound = Q3 + factor * IQR_val

        # Identify potential outliers
        outlier_mask = ((column < lower_bound) | (column > upper_bound)) & valid_mask
        n_outliers = np.sum(outlier_mask)

        # Limit: only mark if <1% of the feature
        max_outliers = int(per_feature_threshold * np.sum(valid_mask))

        if n_outliers > 0 and n_outliers <= max_outliers:
            data[outlier_mask, i] = np.nan
            total_outliers_marked += n_outliers
        elif n_outliers > max_outliers:
            # If too many outliers detected, it is probably a skewed distribution
            # → Do nothing for this feature
            pass

    final_nans = np.isnan(data).sum()
    print(f"  Statistical outliers marked: {final_nans - initial_nans} values")
    print(f"  Total NaN: {final_nans} ({final_nans/data.size*100:.2f}%)")

    return data


def no_outlier_removal(data):
    """
    Deletes only obvious placeholder values, does not perform any statistical outlier detection.

    Input: numpy array (n_samples, n_features)
    Output: numpy array with only placeholders replaced
    """
    return clean_placeholders_only(data)


def std_outlier_removal(data, std_threshold=5.0, per_feature_threshold=0.01):
    """
    Outlier removal for medical data based on the standard deviation and mean of the data.
    Data points that are more than a certain standard deviations threshold away from the mean are considered outliers and replaced with 0.

    Strategy:
    1. Replace only placeholders
    2. Standard deviation method
    3. Limit to 1% of max outliers per feature
    4. Only mark truly extreme values

    Input:
        data: numpy array (n_samples, n_features)
        std_threshold: number of standard deviations from the mean to consider as outlier
        per_feature_threshold: max % of outliers to mark per feature

    Output:
        data: numpy array with outliers replaced by 0
    """

    data = data.copy()

    print("\n=== STD OUTLIER REMOVAL ===")
    print(f"STD threshold: {std_threshold}")
    print(f"Max outliers per feature: {per_feature_threshold*100}%")

    # Step 1: Clean placeholders first
    data = clean_placeholders_only(data)

    initial_nans = np.isnan(data).sum()
    initial_0 = np.sum(data == 0)

    # Step 2: Std deviation method per feature
    total_outliers_marked = 0

    for i in range(data.shape[1]):
        column = data[:, i].copy()
        valid_mask = ~np.isnan(column)

        if np.sum(valid_mask) < 10:  # Skip features with too little data
            continue

        valid_data = column[valid_mask]

        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)

        if std_val == 0:  # Constant feature
            continue

        lower_bound = mean_val - std_threshold * std_val
        upper_bound = mean_val + std_threshold * std_val

        # Identify potential outliers
        outlier_mask = ((column < lower_bound) | (column > upper_bound)) & valid_mask
        n_outliers = np.sum(outlier_mask)

        # Limit: only mark if <1% of the feature
        max_outliers = int(per_feature_threshold * np.sum(valid_mask))

        if n_outliers > 0 and n_outliers <= max_outliers:
            data[outlier_mask, i] = np.nan
            total_outliers_marked += n_outliers
        elif n_outliers > max_outliers:
            # If too many outliers detected, it is probably a skewed distribution
            # → Do nothing for this feature
            pass

    final_nans = np.isnan(data).sum()
    final_0 = np.sum(data == 0)
    print(f"  Statistical outliers marked: {final_0 - initial_0} values")
    print(f"  Total NaN: {final_nans} ({final_nans/data.size*100:.2f}%)")

    return data


# ----------------------------------- DATA PROCESSING -----------------------------------------------------------
def fill_data(x_train_raw, x_test_raw, y_train_raw,
                       threshold_features=0.8, threshold_points=0.6,
                       normalize=True, outlier_strategy='none', fill_method='median'):
    """
    Robust preprocessing pipeline for training and test data.

    Steps:
    1. Remove extremely sparse features and rows
    2. Handle outliers (placeholders, smart, or aggressive)
    3. Fill remaining NaNs with medians
    4. Properly handle categorical vs continuous features
    5. Encode categorical features safely
    6. Normalize continuous features
    7. Remove zero-variance features

    Returns:
        x_train, y_train, x_test
    """
    import numpy as np

    x_train = x_train_raw.copy()
    x_test = x_test_raw.copy()
    y_train = y_train_raw.copy()

    # -------------------------
    # Step 1: Handle outliers with custom outlier strategy
    # -------------------------
    if outlier_strategy == "none":
        x_train = no_outlier_removal(x_train)
        x_test = no_outlier_removal(x_test)
    elif outlier_strategy == "smart":
        x_train = smart_outlier_removal(x_train, factor=3.0)
        x_test = smart_outlier_removal(x_test, factor=3.0)
    elif outlier_strategy == "aggressive":
        x_train = smart_outlier_removal(x_train, factor=1.5)
        x_test = smart_outlier_removal(x_test, factor=1.5)
    elif outlier_strategy == "std":
        x_train = std_outlier_removal(x_train, std_threshold=3.0)
        x_test = std_outlier_removal(x_test, std_threshold=3.0)

    # -------------------------
    # Step 2: Remove sparse features and rows
    # -------------------------
    feature_mask = np.isnan(x_train).mean(axis=0) < threshold_features
    x_train = x_train[:, feature_mask]
    x_test = x_test[:, feature_mask]

    row_mask = np.isnan(x_train).mean(axis=1) < threshold_points
    x_train = x_train[row_mask]
    y_train = y_train[row_mask]

    # -------------------------
    # Step 3: Fill remaining NaNs with median or mode
    # -------------------------
    if fill_method == "median":
        for i in range(x_train.shape[1]):
            median_val = np.nanmedian(x_train[:, i])
            x_train[:, i] = np.nan_to_num(x_train[:, i], nan=median_val)
            x_test[:, i] = np.nan_to_num(x_test[:, i], nan=median_val)

    elif fill_method == "mode":
        for i in range(x_train.shape[1]):
            x_train[:, i] = fill_column_mode(x_train[:, i])
            x_test[:, i] = fill_column_mode(x_test[:, i])

    # -------------------------
    # Step 4: Identify categorical vs continuous
    # -------------------------
    n_features = x_train.shape[1]
    is_categorical = np.zeros(n_features, dtype=bool)
    for i in range(n_features):
        n_unique = len(np.unique(x_train[:, i]))
        is_int = np.allclose(x_train[:, i], x_train[:, i].astype(int))
        if n_unique <= 10 or (is_int and n_unique <= 50):
            is_categorical[i] = True

    continuous_mask = ~is_categorical
    print(f"Categorical features: {is_categorical.sum()}")
    print(f"Continuous features: {continuous_mask.sum()}")

    # -------------------------
    # Step 5: Normalize continuous features
    # -------------------------
    if normalize and continuous_mask.sum() > 0:
        print(f"\n=== NORMALIZING CONTINUOUS FEATURES ===")
        median = np.median(x_train[:, continuous_mask], axis=0)
        q75 = np.percentile(x_train[:, continuous_mask], 75, axis=0)
        q25 = np.percentile(x_train[:, continuous_mask], 25, axis=0)
        iqr = q75 - q25
        zero_iqr_mask = iqr == 0
        iqr[zero_iqr_mask] = 1.0

        x_train[:, continuous_mask] = (x_train[:, continuous_mask] - median) / iqr
        x_test[:, continuous_mask] = (x_test[:, continuous_mask] - median) / iqr

        x_train[:, continuous_mask] = np.clip(x_train[:, continuous_mask], -10, 10)
        x_test[:, continuous_mask] = np.clip(x_test[:, continuous_mask], -10, 10)
        print(f"Normalized {continuous_mask.sum()} continuous features")

    # -------------------------
    # Step 6: Encode categorical features using one-hot encoding.
    # -------------------------
    print(f"\n=== ENCODING CATEGORICAL FEATURES ===")
    ONE_HOT_THRESHOLD = 20
    x_train_encoded = []
    x_test_encoded = []

    for i in range(n_features):
        if is_categorical[i]:
            unique_vals, counts = np.unique(x_train[:, i], return_counts=True)
            n_unique = len(unique_vals)

            if n_unique <= ONE_HOT_THRESHOLD:
                value_map = {val: idx for idx, val in enumerate(unique_vals)}
                n_classes = len(unique_vals)
                train_idx = np.vectorize(value_map.get)(x_train[:, i])
                test_idx = np.vectorize(lambda v: value_map.get(v, -1))(x_test[:, i])
                train_onehot = np.eye(n_classes)[train_idx.astype(int)]
                test_onehot = np.eye(n_classes)[
                    np.clip(test_idx, 0, n_classes - 1).astype(int)
                ]
                x_train_encoded.append(train_onehot)
                x_test_encoded.append(test_onehot)
            else:
                freq_map = {
                    val: c / len(x_train[:, i]) for val, c in zip(unique_vals, counts)
                }
                train_freq = np.vectorize(freq_map.get)(x_train[:, i])
                test_freq = np.vectorize(lambda v: freq_map.get(v, 0.0))(x_test[:, i])
                x_train_encoded.append(train_freq[:, None])
                x_test_encoded.append(test_freq[:, None])
        else:
            x_train_encoded.append(x_train[:, i][:, None])
            x_test_encoded.append(x_test[:, i][:, None])

    # Concatenate all features
    x_train = np.hstack(x_train_encoded)
    x_test = np.hstack(x_test_encoded)
    print(
        f"Final shapes after encoding: x_train={x_train.shape}, x_test={x_test.shape}"
    )

    # -------------------------
    # Step 7: Remove zero-variance features
    # -------------------------
    feature_vars = np.var(x_train, axis=0)
    valid_features = feature_vars > 1e-10
    n_removed = np.sum(~valid_features)
    if n_removed > 0:
        print(f"Removing {n_removed} zero-variance features")
        x_train = x_train[:, valid_features]
        x_test = x_test[:, valid_features]

    # -------------------------
    # Final diagnostics
    # -------------------------
    print(f"\n=== FINAL DATA SUMMARY ===")
    print(f"Training shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(
        f"Overall - min: {x_train.min():.3f}, max: {x_train.max():.3f}, mean: {x_train.mean():.3f}"
    )
    print(f"Max absolute value: {np.abs(x_train).max():.3f}")
    print(f"NaNs in train: {np.isnan(x_train).sum()}, test: {np.isnan(x_test).sum()}")

    return x_train, y_train, x_test

# ---------------------------------- TEST FUNCTION FOR OUTLIER STRATEGIES COMPARISON -----------------------------

def compare_outlier_strategies(x_train_raw, y_train_raw):
    """
    Compare all three strategies with a simple model, here least squares.
    """
    import implementations as impl

    results = {}
    i = 0
    for strategy in ["none", "smart", "aggressive", "std"]:
        for fill_method in ["median", "mode"]:
            print("\n" + "=" * 70)
            print(f"TESTING STRATEGY: {strategy.upper()}, {fill_method.upper()}")
            print("=" * 70)

            # Preprocess
            x_train, y_train, x_test = fill_data(
                x_train_raw.copy(),
                x_train_raw.copy(),  # Dummy test set, it is the reason why we see twice the same printing
                y_train_raw.copy(),
                threshold_features=0.8,
                threshold_points=0.6,
                normalize=True,
                outlier_strategy=strategy,
                fill_method=fill_method,
            )

            # Add bias
            x_train_bias = np.c_[np.ones(x_train.shape[0]), x_train]

            # Train simple model (least squares)
            w, loss = impl.least_squares(y_train, x_train_bias)

            # Evaluate
            y_pred_cont = x_train_bias @ w

            # Find best threshold
            best_f1 = 0
            for t in np.linspace(y_pred_cont.min(), y_pred_cont.max(), 50):
                y_pred = np.where(y_pred_cont >= t, 1, -1)
                tp = np.sum((y_pred == 1) & (y_train == 1))
                fp = np.sum((y_pred == 1) & (y_train == -1))
                fn = np.sum((y_pred == -1) & (y_train == 1))

                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1

            results[i] = {
                "strategy": strategy,
                "fill_method": fill_method,
                "f1": best_f1,
                "n_samples": x_train.shape[0],
                "n_features": x_train.shape[1],
            }
            i += 1
            print(f"\n✓ Strategy '{strategy}': F1 = {best_f1:.4f}")
            print(f"  Samples: {x_train.shape[0]}, Features: {x_train.shape[1]}")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    for strategy_id, res in results.items():
        print(
            f"{res['strategy'].upper():10s}, {res['fill_method'].upper():6s}: F1={res['f1']:.4f} | Samples={res['n_samples']:5d} | Features={res['n_features']:3d}"
        )
    # for strategy, res in results.items():
    #    print(f"{strategy:12s}: F1={res['f1']:.4f}  |  Samples={res['n_samples']:5d}  |  Features={res['n_features']:3d}")

    best = max(results.items(), key=lambda x: x[1]["f1"])
    print(f" WINNER: {best[0]} (F1={best[1]['f1']:.4f})")

    return results


"""
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/data/dataset/'
    
# Load raw data
print("Loading raw data...")
x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = hl.load_csv_data(dir_path)
print(f"Raw train shape: {x_train_raw.shape}, test shape: {x_test_raw.shape}, labels shape: {y_train_raw.shape}")

# Only keep 10% of data for quick testing 
x_train_sample = x_train_raw[::10]
y_train_sample = y_train_raw[::10]

# Compare outlier strategies
compare_outlier_strategies(x_train_raw, y_train_raw)
"""
