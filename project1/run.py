import numpy as np
import os
import helpers as hl
import implementations as impl
import data_exploration as de
import matplotlib.pyplot as plt


# Parameters
THRESHOLD_FEATURES = 0.8
THRESHOLD_POINTS = 0.6
NORMALIZE = True
REMOVE_OUTLIERS = False
MAX_ITERS = 5000
GAMMA = 0.1
LAMBDA = 0.001

# ----------------------------------- DATA PROCESSING -----------------------------------------------------------

def prepare_data(
    threshold_features=0.8,
    threshold_points=0.6,
    normalize=True,
    remove_outliers=False,
    aberrant_threshold=10.0,
):
    """
        Load the raw training and test data, preprocess it, and return cleaned datasets
        ready for machine learning models. Handles missing values, feature removal,
        normalization, and optional outlier removal.

    INPUTS :
        - threshold_features (float): Maximum allowed fraction of missing values per feature.
        - threshold_points (float): Maximum allowed fraction of missing values per data point (row).
        - normalize (bool): If True, normalize feature values.
        - remove_outliers (bool): If True, removes outlier data points from the training set.
    OUTPUTS:
        - x_train (numpy.ndarray): Preprocessed training features, ready for model input.
        - y_train (numpy.ndarray): Corresponding labels for x_train, aligned after removal of invalid data points.
        - x_test (numpy.ndarray): Preprocessed test features, aligned with the training features (same columns).
    """
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data/dataset/"

    # Load Data
    print("Loading data...")
    x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = hl.load_csv_data(
        dir_path
    )
    print(f"Raw training data shape: {x_train_raw.shape}")
    print(f"Raw test data shape: {x_test_raw.shape}")
    print(f"Raw labels shape: {y_train_raw.shape}")

    # Preprocessing Data
    print("\nPreprocessing training data...")
    x_train, removed_features, removed_points = de.fill_data(
        x_train_raw,
        remove_features=[],
        remove_points=[],
        threshold=True,
        threshold_features=threshold_features,
        threshold_points=threshold_points,
        normalize=normalize,
        remove_outliers=remove_outliers,
    )

    # Prints some info
    print(
        f"Removed {len(removed_features)} features with >{THRESHOLD_FEATURES*100}% missing data"
    )
    print(
        f"Removed {len(removed_points)} data points with >{THRESHOLD_POINTS*100}% missing data"
    )
    print(f"Processed training data shape: {x_train.shape}")

    # Remove corresponding labels for removed datapoints
    y_train = np.delete(y_train_raw, removed_points, axis=0)
    print(f"Processed labels shape: {y_train.shape}")

    # Repeat operation for x_test
    print("\nPreprocessing test data...")
    x_test, _, _ = de.fill_data(
        x_test_raw,
        remove_features=removed_features,
        remove_points=[],
        threshold=False,
        threshold_features=threshold_features,
        threshold_points=threshold_points,
        normalize=normalize,
        remove_outliers=remove_outliers,
    )
    print(f"Processed test data shape: {x_test.shape}")

    if normalize:
        print("\nApplying consistent normalization...")

        # Identify categorical vs continuous features
        categorical_mask = de.identify_categorical_features(x_train)
        continuous_mask = ~categorical_mask

        print(f"Found {categorical_mask.sum()} categorical features")
        print(f"Found {continuous_mask.sum()} continuous features")

        # Normalize only continuous features
        if continuous_mask.sum() > 0:
            train_mean = x_train[:, continuous_mask].mean(axis=0)
            train_std = x_train[:, continuous_mask].std(axis=0)

            # Avoid division by zero
            train_std[train_std == 0] = 1.0

            # Apply SAME transformation to both
            x_train[:, continuous_mask] = (
                x_train[:, continuous_mask] - train_mean
            ) / train_std
            x_test[:, continuous_mask] = (
                x_test[:, continuous_mask] - train_mean
            ) / train_std

    # Set the aberrant values to zero after normalization (aberant values: more than 10 std from mean)
    aberrant_values = []
    for column in range(x_train.shape[1]):
        col_mean = x_train[:, column].mean()
        col_std = x_train[:, column].std()
        aberrant_mask_train = (
            np.abs(x_train[:, column] - col_mean) > aberrant_threshold * col_std
        )
        aberrant_mask_test = (
            np.abs(x_test[:, column] - col_mean) > aberrant_threshold * col_std
        )
        x_train[aberrant_mask_train, column] = 0.0
        x_test[aberrant_mask_test, column] = 0.0
        aberrant_values.append(np.sum(aberrant_mask_train))

    # print("\nAberrant values per feature (set to 0):")
    # for idx, count in enumerate(aberrant_values):
    #    if count > 0:
    #        print(f" Feature {idx}: {count} aberrant values")

    return x_train, y_train, x_test, train_ids, test_ids


def prepare_data2(
    threshold_features=0.8,
    threshold_points=0.6,
    normalize=True,
    outlier_strategy="smart",
    fill_method="median",
):
    """
    Load raw training and test data, preprocess it, and return cleaned datasets
    ready for ML models. Handles missing values, feature removal, outlier removal,
    and normalization.

    INPUTS:
        - threshold_features (float): Max fraction of missing values allowed per feature.
        - threshold_points (float): Max fraction of missing values allowed per data point (row).
        - normalize (bool): If True, normalize continuous features.
        - outlier_strategy (str): 'none', 'smart', or 'aggressive' outlier handling.

    OUTPUTS:
        - x_train (np.ndarray): Preprocessed training features.
        - y_train (np.ndarray): Aligned labels for x_train.
        - x_test (np.ndarray): Preprocessed test features aligned with x_train.
        - train_ids (np.ndarray): Original training IDs.
        - test_ids (np.ndarray): Original test IDs.
    """
    import os
    import numpy as np

    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data/dataset/"

    # Load raw data
    print("Loading raw data...")
    x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = hl.load_csv_data(
        dir_path
    )
    print(
        f"Raw train shape: {x_train_raw.shape}, test shape: {x_test_raw.shape}, labels shape: {y_train_raw.shape}"
    )

    # Add this at the start of prepare_data2
    print(f"RAW data - Train min: {x_train_raw.min()}, max: {x_train_raw.max()}")
    print(f"RAW data - Unique values sample: {np.unique(x_train_raw[:, 0])[:10]}")
    print(f"RAW labels: {np.unique(y_train_raw)}")
    # Apply robust preprocessing pipeline
    print("\nApplying robust preprocessing pipeline...")
    x_train, y_train, x_test = de.fill_data_robust3(
        x_train_raw,
        x_test_raw,
        y_train_raw,
        threshold_features=threshold_features,
        threshold_points=threshold_points,
        normalize=normalize,
        outlier_strategy=outlier_strategy,
        fill_method=fill_method,
    )

    print(
        f"\nFinal training data shape: {x_train.shape}, test data shape: {x_test.shape}"
    )
    print(f"Final labels shape: {y_train.shape}")
    print(
        f"Remaining NaNs in train: {np.isnan(x_train).sum()}, test: {np.isnan(x_test).sum()}"
    )

    return x_train, y_train, x_test, train_ids, test_ids


# ----------------------------------- HELPERS FOR EVALUATION -----------------------------------------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)


def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + fn + 1e-8)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)


def compute_f1_score(y_train_binary, x_train_bias, weights, threshold=0.2):
    """
    Compute F1 score.

    Input : - y_train_binary (np.ndarray) : target of shape (n_datapoints,)
            - x_train_bias (nd.ndarray) : dataset of shape (n_datapoints,n_features)
            - threshold (float) : threshold for binary classification
    Output :
            - f1 (float) : F1 score
    """
    # Calculate metrics
    z_train = x_train_bias @ weights
    y_train_pred_prob = impl._sigmoid(z_train)
    y_train_pred_binary = (y_train_pred_prob >= threshold).astype(
        int
    )  # Same threshold as test

    # F1 Score
    tp = np.sum((y_train_pred_binary == 1) & (y_train_binary == 1))
    fp = np.sum((y_train_pred_binary == 1) & (y_train_binary == 0))
    fn = np.sum((y_train_pred_binary == 0) & (y_train_binary == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1, tp, fp, fn


def compute_f1_score_KNN(y_true, y_pred):
    """
    Compute F1 score for KNN predictions.

    Inputs:
      - y_true: np.ndarray of shape (n_samples,) with labels in {-1, 1}
      - y_pred: np.ndarray of shape (n_samples,) with predicted labels in {-1, 1}

    Returns:
      - f1: float — F1 score
    """
    # Compute F1 score
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1, tp, fp, fn


# ----------------------------------- HELPERS FOR PLOTING RESULTS ---------------------------------------------------
def plot_and_save(x, y, xlabel, ylabel, color, filename, model_name, best_param):
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plots")
    full_path = os.path.join(dir_path, filename)
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, label=ylabel, linewidth=2, color=color)
    plt.axvline(best_param, color="r", linestyle="--", label=f"Best {xlabel} : {best_param:.3f}")
    plt.xlabel(xlabel,fontsize = 19)
    plt.ylabel(ylabel,fontsize=19)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16)  
    plt.title(f"{model_name}: {ylabel} vs {xlabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = dir_path + filename
    plt.savefig(f"{full_path}")
    print(f"Saved plot: {filename}")
    plt.show()


# ----------------------------------- TRAINING FUNCTIONS  ------------------------------------------------------------
def train_mean_square_error_gd(
    y_train,
    x_train,
    x_test,
    test_ids,
    gammas=np.logspace(-8, -1, 10),
    max_iters=20000,
    k_fold=4,
    seed=42,
    save_plots=False,
):
    """
    Train a MSE with gradient descent model with hyperparameter tuning
    using k-fold cross-validation, then generate predictions for the test set.

    This function:
      1. Prepares the data (adds bias, converts labels to {0, 1})
      2. Performs cross-validation over a grid of (lambda, gamma)
      3. Selects the combination yielding the best mean F1 score
      4. Retrains the final model on the full training set with best parameters
      5. Generates and saves predictions for the test set

    Input  : - y_train : np.ndarray : Training labels, shape (N,) with values in {-1, 1}.
             - x_train : np.ndarray : Training features, shape (N, D).
             - x_test : np.ndarray : Test features, shape (M, D) (no labels).
             - test_ids : np.ndarray : Indices of the the test features.
             - gammas : list of float, optional : Learning rates to test during cross-validation.
             - max_iters : int, optional :Maximum number of iterations for gradient descent.
             - k_fold : int, optional : Number of folds for cross-validation.
             - seed : int, optional : Random seed for reproducibility.
    """
    threshold = 0.2
    # Prepare data for logistic regression
    print("\nPreparing data for MSE GD...")
    # Add bias term (column of ones) to features
    x_train_bias = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test_bias = np.c_[np.ones(x_test.shape[0]), x_test]
    print(f"Training data with bias: {x_train_bias.shape}")
    print(f"Test data with bias: {x_test_bias.shape}")

    # Initialize weights
    initial_w = np.zeros(x_train_bias.shape[1])

    # Compute k indices for k-folding
    k_indices = de.build_k_indices(y_train, k_fold, seed)

    # Cross validation over lambdas
    mean_losses, mean_accs, mean_f1s = [], [], []

    # ---- Cross-validation over gammas ----
    for gamma in gammas:
        print(f"\nRunning cross-validation for gamma = {gamma}")
        fold_losses, fold_accs, fold_f1s = [], [], []

        for k in range(k_fold):
            val_idx = k_indices[k]
            train_idx = np.delete(np.arange(y_train.shape[0]), val_idx)
            X_tr, X_val = x_train_bias[train_idx], x_train_bias[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Train model
            w, loss = impl.mean_squared_error_gd(
                y_tr,
                X_tr,
                initial_w=initial_w,
                max_iters=max_iters,
                gamma=gamma,
            )

            # Predict validation set
            y_val_pred_cont = X_val @ w
            y_val_pred = np.where(y_val_pred_cont >= threshold, 1, -1)

            # Compute metrics
            acc = accuracy(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred)
            fold_losses.append(loss)
            fold_accs.append(acc)
            fold_f1s.append(f1)

        # Average across folds
        mean_losses.append(np.mean(fold_losses))
        mean_accs.append(np.mean(fold_accs))
        mean_f1s.append(np.mean(fold_f1s))

    # ---- Select best gamma ----
    best_idx = np.argmax(mean_f1s)
    best_gamma = gammas[best_idx]
    print(f"\nBest gamma by F1 score: {best_gamma:.4f}")
    print(
        f"Mean F1: {mean_f1s[best_idx]:.4f}, Mean Acc: {mean_accs[best_idx]:.4f}, Mean Loss: {mean_losses[best_idx]:.4f}"
    )

    # ---- Retrain with best gamma ----
    final_w, final_loss = impl.mean_squared_error_gd(
        y_train,
        x_train_bias,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=best_gamma,
    )

    # ---- Training and test predictions ----
    print("\nGenerating predictions with best gamma...")
    y_train_pred = x_train_bias @ final_w
    y_train_bin = np.where(y_train_pred >= threshold, 1, -1)
    train_acc = accuracy(y_train, y_train_bin)
    train_f1 = f1_score(y_train, y_train_bin)
    print(
        f"Final training accuracy: {train_acc:.3f}, F1: {train_f1:.4f}, loss: {final_loss:.3f}"
    )

    # Test predictions
    y_test_pred = np.where(x_test_bias @ final_w >= threshold, 1, -1)
    output_path = "submission.csv"
    hl.create_csv_submission(test_ids, y_test_pred, output_path)
    print(f"Submission saved at: {output_path}")

    # ---- Plot results ----
    if save_plots:
        plot_and_save(
            gammas,
            mean_f1s,
            "Gamma",
            "F1",
            "blue",
            "MSE_GD_F1_VS_gamma.png",
            "MSE Gradient Descent",
            best_gamma,
        )
        plot_and_save(
            gammas,
            mean_accs,
            "Gamma",
            "Accuracy",
            "orange",
            "MSE_GD_acc_VS_gamma.png",
            "MSE Gradient Descent",
            best_gamma,
        )


def train_mean_square_error_sgd(
    y_train,
    x_train,
    x_test,
    test_ids,
    gammas=np.logspace(-8, -1, 10),
    max_iters=20000,
    k_fold=4,
    seed=42,
    save_plots=False,
):
    """
    Train a MSE with gradient descent model with hyperparameter tuning
    using k-fold cross-validation, then generate predictions for the test set.

    This function:
      1. Prepares the data (adds bias, converts labels to {0, 1})
      2. Performs cross-validation over a grid of (lambda, gamma)
      3. Selects the combination yielding the best mean F1 score
      4. Retrains the final model on the full training set with best parameters
      5. Generates and saves predictions for the test set

    Input  : - y_train : np.ndarray : Training labels, shape (N,) with values in {-1, 1}.
             - x_train : np.ndarray : Training features, shape (N, D).
             - x_test : np.ndarray : Test features, shape (M, D) (no labels).
             - test_ids : np.ndarray : Indices of the the test features.
             - gammas : list of float, optional : Learning rates to test during cross-validation.
             - max_iters : int, optional :Maximum number of iterations for gradient descent.
             - k_fold : int, optional : Number of folds for cross-validation.
             - seed : int, optional : Random seed for reproducibility.
             - save_plots : Boolean, optional : Plot metrics w.r.t gamma hyperparameter and save.
    """
    threshold = 0.0
    # Prepare data for logistic regression
    print("\nPreparing data for MSE SGD...")
    # Add bias term (column of ones) to features
    x_train_bias = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test_bias = np.c_[np.ones(x_test.shape[0]), x_test]
    print(f"Training data with bias: {x_train_bias.shape}")
    print(f"Test data with bias: {x_test_bias.shape}")

    # Initialize weights
    initial_w = np.zeros(x_train_bias.shape[1])

    # Compute k indices for k-folding
    k_indices = de.build_k_indices(y_train, k_fold, seed)

    mean_losses, mean_accs, mean_f1s = [], [], []

    # ---- Cross-validation over gammas ----
    for gamma in gammas:
        print(f"\nRunning cross-validation for gamma = {gamma}")
        fold_losses, fold_accs, fold_f1s = [], [], []

        for k in range(k_fold):
            val_idx = k_indices[k]
            train_idx = np.delete(np.arange(y_train.shape[0]), val_idx)
            X_tr, X_val = x_train_bias[train_idx], x_train_bias[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Train model
            w, loss = impl.mean_squared_error_sgd(
                y_tr,
                X_tr,
                initial_w=initial_w,
                max_iters=max_iters,
                gamma=gamma,
            )

            # Predict validation set
            y_val_pred_cont = X_val @ w
            y_val_pred = np.where(y_val_pred_cont >= threshold, 1, -1)

            # Compute metrics
            acc = accuracy(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred)
            fold_losses.append(loss)
            fold_accs.append(acc)
            fold_f1s.append(f1)

        # Average across folds
        mean_losses.append(np.mean(fold_losses))
        mean_accs.append(np.mean(fold_accs))
        mean_f1s.append(np.mean(fold_f1s))

    # ---- Select best gamma ----
    best_idx = np.argmax(mean_f1s)
    best_gamma = gammas[best_idx]
    print(f"\nBest gamma by F1 score: {best_gamma}")
    print(
        f"Mean F1: {mean_f1s[best_idx]}, Mean Acc: {mean_accs[best_idx]:.4f}, Mean Loss: {mean_losses[best_idx]:.4f}"
    )

    # ---- Retrain with best gamma ----
    final_w, final_loss = impl.mean_squared_error_sgd(
        y_train,
        x_train_bias,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=best_gamma,
    )

    # ---- Training and test predictions ----
    print("\nGenerating predictions with best gamma...")
    y_train_pred = x_train_bias @ final_w
    y_train_bin = np.where(y_train_pred >= threshold, 1, -1)
    train_acc = accuracy(y_train, y_train_bin)
    train_f1 = f1_score(y_train, y_train_bin)
    print(
        f"Final training accuracy: {train_acc:.3f}, F1: {train_f1:.4f}, loss: {final_loss:.3f}"
    )

    # Test predictions
    y_test_pred = np.where(x_test_bias @ final_w >= threshold, 1, -1)
    output_path = "submission.csv"
    hl.create_csv_submission(test_ids, y_test_pred, output_path)
    print(f"Submission saved at: {output_path}")

    # ---- Plot results ----
    if save_plots:
        plot_and_save(
            gammas,
            mean_f1s,
            "Gamma",
            "F1",
            "blue",
            "MSE_SGD_F1_VS_gamma.png",
            "MSE SGD",
            best_gamma,
        )
        plot_and_save(
            gammas,
            mean_accs,
            "Gamma",
            "Accuracy",
            "orange",
            "MSE_SGD_acc_VS_gamma.png",
            "MSE SGD",
            best_gamma,
        )


def train_least_squares(y_train, x_train, x_test, test_ids, save_plots=False):
    """
    Train a MSE with gradient descent model with hyperparameter tuning then generate predictions for the test set.

    This function:
      1. Prepares the data (adds bias, converts labels to {0, 1})
      2. Selects the threshold yielding the best F1 score
      4. Retrains the final model on the full training set with best parameters
      5. Generates and saves predictions for the test set

    Input  : - y_train : np.ndarray : Training labels, shape (N,) with values in {-1, 1}.
             - x_train : np.ndarray : Training features, shape (N, D).
             - x_test : np.ndarray : Test features, shape (M, D) (no labels).
             - test_ids : np.ndarray : Indices of the the test features.
             - save_plots : Boolean, optional : Plot metrics in function of threshold and save them.
    """
    # ===== ADD DIAGNOSTICS =====
    print("\n=== DATA DIAGNOSTICS ===")
    print("TRAINING : ")
    print(f"X_train shape: {x_train.shape}")
    print(
        f"X_train stats: min={np.nanmin(x_train):.3f}, max={np.nanmax(x_train):.3f}, mean={np.nanmean(x_train):.3f}"
    )
    print(f"Contains NaN: {np.isnan(x_train).any()} (count: {np.isnan(x_train).sum()})")
    print(f"Contains Inf: {np.isinf(x_train).any()}")
    print("TEST : ")
    print(f"X_train shape: {x_test.shape}")
    print(
        f"X_train stats: min={np.nanmin(x_test):.3f}, max={np.nanmax(x_test):.3f}, mean={np.nanmean(x_test):.3f}"
    )
    print(f"Contains NaN: {np.isnan(x_test).any()} (count: {np.isnan(x_test).sum()})")
    print(f"Contains Inf: {np.isinf(x_test).any()}")

    # Add bias term
    x_train_bias = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test_bias = np.c_[np.ones(x_test.shape[0]), x_test]

    # Threshold range for tuning
    w, loss = impl.least_squares(y_train, x_train_bias)
    y_train_pred = x_train_bias @ w
    thresholds = np.linspace(-1, 1, 50)
    thresholds = np.linspace(y_train_pred.min(), y_train_pred.max(), 50)
    f1s = []
    accs = []
    for t in thresholds:
        y_train_pred_binary = np.where(y_train_pred >= t, 1, -1)
        f1 = f1_score(y_train, y_train_pred_binary)
        acc = accuracy(y_train, y_train_pred_binary)
        f1s.append(f1)
        accs.append(acc)
    best_f1 = np.max(f1s)
    best_threshold = thresholds[np.argmax(f1s)]
    best_y_train_pred_binary = np.where(y_train_pred >= best_threshold, 1, -1)
    # --- Plot results ---
    if save_plots:
        plot_and_save(
            thresholds,
            f1s,
            "Threshold",
            "F1",
            "blue",
            "least_squares_F1_VS_threshold.png",
            "Least squares",
            best_threshold,
        )
        plot_and_save(
            thresholds,
            accs,
            "Threshold",
            "Accuracy",
            "orange",
            "least_squares_Accuracy_VS_threshold.png",
            "Least squares",
            best_threshold,
        )
    # plot_and_save(thresholds, mean_losses, "Loss", "green", "least_squares_Loss_VS_threshold.png","Least_squares")

    print(
        f"Threshold {best_threshold:.2f}: pred mean={y_train_pred.mean():.3f}, "
        f"std={y_train_pred.std():.3f}, min={y_train_pred.min():.3f}, max={y_train_pred.max():.3f}"
    )
    print(
        f"  Predicted as 1: {(best_y_train_pred_binary == 1).sum()}/{len(best_y_train_pred_binary)}"
    )

    train_acc = accuracy(y_train, best_y_train_pred_binary)
    train_f1 = f1_score(y_train, best_y_train_pred_binary)

    print(f"\nFinal model on full data:")
    print(f"Train accuracy = {train_acc:.3f}, Train F1 = {train_f1:.3f}")

    # --- Generate test predictions ---
    y_test_pred_cont = x_test_bias @ w
    y_test_pred = np.where(y_test_pred_cont >= best_threshold, 1, -1)

    print("Train preds:", y_train_pred.min(), y_train_pred.max(), y_train_pred.mean())
    print(
        "Test preds:",
        y_test_pred_cont.min(),
        y_test_pred_cont.max(),
        y_test_pred_cont.mean(),
    )

    # Save submission
    output_path = "submission.csv"
    hl.create_csv_submission(test_ids, y_test_pred, output_path)
    print(f"Submission file saved to: {output_path}")


def train_reg_logistic_regression(
    y_train,
    x_train,
    x_test,
    test_ids,
    gammas=np.logspace(-5, -2, 5),
    lambdas=np.logspace(-4, -2, 4),
    threshold=0.2,
    max_iters=2000,
    k_fold=4,
    seed=42,
    save_plots=False,
):
    """
    Train a regularized logistic regression model with hyperparameter tuning
    using k-fold cross-validation, then generate predictions for the test set.

    This function:
      1. Prepares the data (adds bias, converts labels to {0, 1})
      2. Performs cross-validation over a grid of (lambda, gamma)
      3. Selects the combination yielding the best mean F1 score
      4. Retrains the final model on the full training set with best parameters
      5. Generates and saves predictions for the test set

    Input  : - y_train : np.ndarray : Training labels, shape (N,) with values in {-1, 1}.
             - x_train : np.ndarray : Training features, shape (N, D).
             - x_test : np.ndarray : Test features, shape (M, D) (no labels).
             - test_ids : np.ndarray : Indices of the the test features.
             - gammas : list of float, optional : Learning rates to test during cross-validation.
            - lambdas : list or np.ndarray of float, optional : Regularization strengths to test during cross-validation.
             - max_iters : int, optional :Maximum number of iterations for gradient descent.
             - k_fold : int, optional : Number of folds for cross-validation.
             - seed : int, optional : Random seed for reproducibility.
             - save_plots : Boolean, optional : Plot metrics w.r.t gamma hyperparameter and save.
    """
    # threshold = 0.05
    # Prepare data for logistic regression
    print("\nPreparing data for logistic regression...")
    # Convert labels from {-1, 1} to {0, 1} for logistic regression
    y_train_binary = (y_train + 1) / 2
    y_train_binary = y_train_binary.astype(int)
    print(f"Label distribution: {np.bincount(y_train_binary)}")
    n_0 = np.sum(y_train_binary == 0)
    n_1 = np.sum(y_train_binary == 1)
    print(f"Class 0: {np.sum(y_train_binary == 0)} samples")
    print(f"Class 1: {np.sum(y_train_binary == 1)} samples")
    # Add bias term (column of ones) to features
    x_train_bias = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test_bias = np.c_[np.ones(x_test.shape[0]), x_test]
    print(f"Training data with bias: {x_train_bias.shape}")
    print(f"Test data with bias: {x_test_bias.shape}")

    # Initialize weights
    initial_w = np.zeros(x_train_bias.shape[1])
    initial_w[0] = np.log(n_1 / n_0)

    # Compute k indices for k-folding
    k_indices = de.build_k_indices(y_train, k_fold, seed)

    plot_data = []
    # CV over lambdas and gammas
    best_f1 = 0.0
    best_gamma, best_lambda = None, None
    for lambda_ in lambdas:
        print(f"=== Lambda = {lambda_} === ")
        for gamma in gammas:
            print(f" Gamma = {gamma}")
            f1_scores = []
            for k in range(k_fold):
                val_idx = k_indices[k]
                train_idx = np.delete(np.arange(y_train_binary.shape[0]), val_idx)
                X_tr, X_val = x_train_bias[train_idx], x_train_bias[val_idx]
                y_tr, y_val = y_train_binary[train_idx], y_train_binary[val_idx]

                # Train
                w, loss = impl.reg_logistic_regression(
                    y_tr,
                    X_tr,
                    lambda_=lambda_,
                    initial_w=initial_w,
                    max_iters=max_iters,
                    gamma=gamma,
                )
                x = X_val @ w
                print(x[0:10])
                f1, _, _, _ = compute_f1_score(y_val, X_val, w, threshold)
                print(f1)
                print(loss)
                f1_scores.append(f1)

            mean_f1 = np.mean(f1_scores)
            plot_data.append((lambda_, gamma, mean_f1))

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_lambda, best_gamma = lambda_, gamma

    print(f"Best lambda/gamma: {best_lambda}/{best_gamma}, F1: {best_f1:.4f}")
    # Retrain on full data with best params
    final_w, _ = impl.reg_logistic_regression(
        y_train_binary,
        x_train_bias,
        lambda_=best_lambda,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=best_gamma,
    )

    # Training set predictions
    y_train_prob = impl._sigmoid(x_train_bias @ final_w)
    y_train_pred = (y_train_prob >= threshold).astype(int)
    x = x_train_bias @ final_w
    print(
        f"F1 score : {compute_f1_score(y_train_binary,x_train_bias,final_w,threshold)}"
    )

    print(f"\nTraining predictions:")
    print(
        f"  Probabilities: min={y_train_prob.min():.3f}, max={y_train_prob.max():.3f}, mean={y_train_prob.mean():.3f}"
    )
    print(
        f"  Predicted class 0: {np.sum(y_train_pred == 0)} ({np.sum(y_train_pred == 0)/len(y_train_pred)*100:.1f}%)"
    )
    print(
        f"  Predicted class 1: {np.sum(y_train_pred == 1)} ({np.sum(y_train_pred == 1)/len(y_train_pred)*100:.1f}%)"
    )
    print(
        f"  Actual class 0: {np.sum(y_train_binary == 0)} ({np.sum(y_train_binary == 0)/len(y_train_binary)*100:.1f}%)"
    )
    print(
        f"  Actual class 1: {np.sum(y_train_binary == 1)} ({np.sum(y_train_binary == 1)/len(y_train_binary)*100:.1f}%)"
    )
    # Predict on test set
    y_test_prob = impl._sigmoid(x_test_bias @ final_w)
    y_test_binary = (y_test_prob >= threshold).astype(int)
    y_test = 2 * y_test_binary - 1

    path = os.path.dirname(os.path.realpath(__file__))
    output_path = path + "/submission_v8.csv"
    hl.create_csv_submission(test_ids, y_test, output_path)
    print("Submission file saved.")
    # ---- Plot F1 scores only ----
    if save_plots:
        # Convert plot_data to arrays
        lambdas_arr = np.array([t[0] for t in plot_data])
        gammas_arr = np.array([t[1] for t in plot_data])
        f1_arr = np.array([t[2] for t in plot_data])

        # Example: F1 vs gamma for fixed best lambda
        mask = lambdas_arr == best_lambda
        plt.figure()
        plt.plot(gammas_arr[mask], f1_arr[mask], marker="o")
        plt.axvline(best_gamma, color="r", linestyle="--", label="Best gamma")
        plt.xlabel("Gamma")
        plt.ylabel("F1 score")
        plt.title("Regularized Logistic Regression: F1 vs Gamma")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path + "/f1.png")
        print("Saved plot: reg_logistic_F1_VS_gamma.png")
        plt.close()

        # Plot the prediction distribution on training and test sets with transparent histograms
        plt.figure()
        plt.hist(
            y_train_prob,
            bins=30,
            alpha=0.5,
            label="Train Predictions",
            color="blue",
            density=True,
        )
        plt.hist(
            y_test_prob,
            bins=30,
            alpha=0.5,
            label="Test Predictions",
            color="orange",
            density=True,
        )
        plt.axvline(threshold, color="r", linestyle="--", label="Threshold")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title("Prediction Distribution on Train and Test Sets")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path + "/Pred_distr.png")
        print("Saved plot: reg_logistic_prediction_distribution.png")
        plt.close()


def train_logistic_regression(
    y_train,
    x_train,
    x_test,
    test_ids,
    gammas=[0.1],
    max_iters=2000,
    k_fold=4,
    seed=42,
    save_plots=False,
):
    """
    Train a logistic regression model with hyperparameter tuning
    using k-fold cross-validation, then generate predictions for the test set.

    This function:
      1. Prepares the data (adds bias, converts labels to {0, 1})
      2. Performs cross-validation over a grid of (lambda, gamma)
      3. Selects the combination yielding the best mean F1 score
      4. Retrains the final model on the full training set with best parameters
      5. Generates and saves predictions for the test set

    Input  : - y_train : np.ndarray : Training labels, shape (N,) with values in {-1, 1}.
             - x_train : np.ndarray : Training features, shape (N, D).
             - x_test : np.ndarray : Test features, shape (M, D) (no labels).
             - test_ids : np.ndarray : Indices of the the test features.
             - gammas : list of float, optional : Learning rates to test during cross-validation.
             - max_iters : int, optional :Maximum number of iterations for gradient descent.
             - k_fold : int, optional : Number of folds for cross-validation.
             - seed : int, optional : Random seed for reproducibility.
    """
    threshold = 0.2
    # Prepare data for logistic regression
    print("\nPreparing data for logistic regression...")
    # Convert labels from {-1, 1} to {0, 1} for logistic regression
    y_train_binary = (y_train + 1) / 2
    y_train_binary = y_train_binary.astype(int)
    print(f"Label distribution: {np.bincount(y_train_binary)}")
    print(f"Class 0: {np.sum(y_train_binary == 0)} samples")
    print(f"Class 1: {np.sum(y_train_binary == 1)} samples")
    # Add bias term (column of ones) to features
    x_train_bias = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test_bias = np.c_[np.ones(x_test.shape[0]), x_test]
    print(f"Training data with bias: {x_train_bias.shape}")
    print(f"Test data with bias: {x_test_bias.shape}")

    # Initialize weights
    initial_w = np.zeros(x_train_bias.shape[1])

    # Compute k indices for k-folding
    k_indices = de.build_k_indices(y_train, k_fold, seed)

    # Cross validation over lambdas and gammas
    # Store F1 scores for plotting
    f1_per_gamma = []

    best_f1 = 0.0
    best_gamma = None
    for gamma in gammas:
        fold_f1s = []
        for k in range(k_fold):
            val_idx = k_indices[k]
            train_idx = np.delete(np.arange(y_train_binary.shape[0]), val_idx)
            X_tr, X_val = x_train_bias[train_idx], x_train_bias[val_idx]
            y_tr, y_val = y_train_binary[train_idx], y_train_binary[val_idx]

            # Train model
            w, _ = impl.logistic_regression(
                y_tr, X_tr, initial_w=initial_w, max_iters=max_iters, gamma=gamma
            )

            f1 = compute_f1_score(y_val, X_val, w, threshold)
            fold_f1s.append(f1)

        mean_f1 = np.mean(fold_f1s)
        f1_per_gamma.append(mean_f1)

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_gamma = gamma

    print(f"Best gamma: {best_gamma}, F1: {best_f1:.4f}")

    # Retrain on full training data
    final_w, _ = impl.logistic_regression(
        y_train_binary,
        x_train_bias,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=best_gamma,
    )

    # Predict on test set
    y_test_prob = impl._sigmoid(x_test_bias @ final_w)
    y_test_binary = (y_test_prob >= threshold).astype(int)
    y_test = 2 * y_test_binary - 1
    hl.create_csv_submission(test_ids, y_test, "submission.csv")
    print("Submission file saved.")

    # ---- Plot F1 vs gamma ----
    if save_plots:
        plot_and_save(
            gammas,
            f1_per_gamma,
            "Gamma",
            "F1",
            "blue",
            "Log_Reg_F1_VS_gamma.png",
            "Log_Reg",
            best_gamma,
        )


def train_ridge_regression(
    y_train,
    x_train,
    x_test,
    test_ids,
    lambdas=np.logspace(-5, -1, 4),
    k_fold=4,
    seed=42,
    save_plots=False,
):
    """
    Train a ridge regression model with hyperparameter tuning
    using k-fold cross-validation, then generate predictions for the test set.

    This function:
      1. Prepares the data (adds bias, converts labels to {0, 1})
      2. Performs cross-validation over a grid of (lambda, gamma)
      3. Selects the combination yielding the best mean F1 score
      4. Retrains the final model on the full training set with best parameters
      5. Generates and saves predictions for the test set

    Input  : - y_train : np.ndarray : Training labels, shape (N,) with values in {-1, 1}.
             - x_train : np.ndarray : Training features, shape (N, D).
             - x_test : np.ndarray : Test features, shape (M, D) (no labels).
             - test_ids : np.ndarray : Indices of the the test features.
            - lambdas : list or np.ndarray of float, optional : Regularization strengths to test during cross-validation.
            - k_fold : int, optional : Number of folds for cross-validation.
            - seed : int, optional : Random seed for reproducibility.
    """
    threshold = 0.2
    # Prepare data for logistic regression
    print("\nPreparing data for logistic regression...")
    # Add bias term (column of ones) to features
    x_train_bias = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test_bias = np.c_[np.ones(x_test.shape[0]), x_test]
    print(f"Training data with bias: {x_train_bias.shape}")
    print(f"Test data with bias: {x_test_bias.shape}")
    # Compute k indices for k-folding
    k_indices = de.build_k_indices(y_train, k_fold, seed)

    mean_losses, mean_f1s = [], []

    best_loss = float("inf")
    best_lambda = None

    # Cross-validation over lambdas
    for lambda_ in lambdas:
        fold_losses, fold_f1s = [], []

        for k in range(k_fold):
            val_idx = k_indices[k]
            train_idx = np.delete(np.arange(y_train.shape[0]), val_idx)
            X_tr, X_val = x_train_bias[train_idx], x_train_bias[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Train ridge regression
            w, loss = impl.ridge_regression(y_tr, X_tr, lambda_=lambda_)
            fold_losses.append(loss)
            # Convert predictions to ±1
            y_val_pred = np.where(X_val @ w >= threshold, 1, -1)
            f1 = f1_score(y_val, y_val_pred)
            fold_f1s.append(f1)

        mean_loss = np.mean(fold_losses)
        mean_f1 = np.mean(fold_f1s)

        mean_losses.append(mean_loss)
        mean_f1s.append(mean_f1)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_lambda = lambda_

    print(
        f"Best lambda: {best_lambda}, mean validation loss: {best_loss:.4f}, mean F1: {mean_f1s[lambdas.tolist().index(best_lambda)]:.4f}"
    )

    # Retrain on full training data
    final_w, final_loss = impl.ridge_regression(
        y_train, x_train_bias, lambda_=best_lambda
    )

    # Predictions
    y_test_pred = np.where(x_test_bias @ final_w >= threshold, 1, -1)
    hl.create_csv_submission(test_ids, y_test_pred, "submission.csv")
    print("Submission file saved.")

    # ---- Plot loss vs lambda ----
    if save_plots:
        plot_and_save(
            lambdas,
            mean_f1s,
            "Lambda",
            "F1",
            "blue",
            "Ridge_Reg_F1_VS_lambda.png",
            "Ridge_Reg",
            best_lambda,
        )
        plot_and_save(
            lambdas,
            mean_losses,
            "Lambda",
            "Loss",
            "orange",
            "Ridge_Reg_Loss_VS_lambda.png",
            "Ridge_Reg",
            best_lambda,
        )

    # Final prediction by majority vote
    y_pred = np.where(y_pred >= 0, 1, -1)
    return y_pred


def knn_predict(x_train, y_train, x_test, k=3, factor=1):
    """
    K-Nearest Neighbors classifier.

    Input  : - x_train : np.ndarray : Training features, shape (N, D).
             - y_train : np.ndarray : Training labels, shape (N,)
             - x_test : np.ndarray : Test features, shape (M, D).
             - k : int, optional : Number of neighbors to consider.
    Output : - y_pred : np.ndarray : Predicted labels for test set, shape (M,)."""

    print("\nPreparing data for KNN...")

    # Code the KNN algorithm
    y_pred = []
    print("Number of test samples :", x_test.shape[0])
    for i in range(x_test.shape[0]):
        # Compute distances from test point to all training points
        distances = np.linalg.norm(x_train - x_test[i], axis=1)
        # Get indices of k nearest neighbors
        knn_indices = np.argsort(distances)[:k]
        # Get the labels of the k nearest neighbors
        knn_labels = y_train[knn_indices]
        # Select class, positive count counts more (x2)
        count_pos = factor * np.sum(knn_labels == 1)
        count_neg = np.sum(knn_labels == -1)
        label = np.sign(count_pos - count_neg)
        y_pred.append(label)
    return np.array(y_pred)


def train_knn(
    y_train,
    x_train,
    x_test,
    test_ids,
    ks=[3, 5, 10, 15, 20, 30, 40],
    factors=[1, 2, 5, 7, 10],
    k_fold=4,
    seed=42,
    create_submission_file=False,
):
    """
    Train a KNN model with hyperparameter tuning
    using k-fold cross-validation, then generate predictions for the test set.

    This function:
      1. Performs cross-validation over a grid of (k, factor)
      2. Selects the combination yielding the best mean F1 score
      3. Retrains the final model on the full training set with best parameters
      4. Generates and saves predictions for the test set

    Input:
        - y_train : np.ndarray : Training labels, shape (N,) with values in {-1, 1}.
        - x_train : np.ndarray : Training features, shape (N, D).
        - x_test : np.ndarray : Test features, shape (M, D) (no labels).
        - test_ids : np.ndarray : Indices of the the test features.
        - ks : list of int, optional : Number of neighbors to test during cross-validation.
        - factors : list of int, optional : Weighting factors for positive class during cross-validation.
        - k_fold : int, optional : Number of folds for cross-validation.
        - seed : int, optional : Random seed for reproducibility.
        - save_plots : Boolean, optional : Plot metrics w.r.t hyperparameters and save.

    Output  : - y_pred : np.ndarray : Predicted labels for test set, shape (M,).
    """
    # CV over k and factor
    best_f1 = 0.0
    best_k, best_factor = None, None
    plot_f1_scores = []
    parameters = []
    for k in ks:
        for factor in factors:
            parameters.append((k, factor))
            print(f"=== K = {k}, factor = {factor} === ")
            f1_scores = []
            # Compute k indices for k-folding
            k_indices = de.build_k_indices_knn(y_train, k_fold, 0.002, seed)
            for fold in range(k_fold):
                val_idx = k_indices[fold]
                train_idx = np.delete(np.arange(y_train.shape[0]), val_idx)
                X_tr, X_val = x_train[train_idx], x_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Predict using KNN
                y_val_pred = knn_predict(X_tr, y_tr, X_val, k=k, factor=factor)

                f1 = f1_score(y_val, y_val_pred)
                f1_scores.append(f1)

            mean_f1 = np.mean(f1_scores)
            print(f"Mean F1 score: {mean_f1:.4f}")

            plot_f1_scores.append(mean_f1)

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_k, best_factor = k, factor
    print(f"Best k/factor: {best_k}/{best_factor}, F1: {best_f1:.4f}")

    if create_submission_file:
        # Retrain on full data with best params
        y_test_pred = knn_predict(x_train, y_train, x_test, best_k, best_factor)

        # Create and save submission file
        output_path = os.path.dirname(os.path.realpath(__file__))
        output_path += "/submission_knn_1.csv"
        hl.create_csv_submission(test_ids, y_test_pred, output_path)
        print("Submission file saved at:", output_path)

    return plot_f1_scores, parameters


# ----------------------------------- TRAINING & EVALUATION ---------------------------------------------------------


# x_train,y_train,x_test,train_ids, test_ids = prepare_data(threshold_features = 0.5,threshold_points = 0.5, normalize = True, remove_outliers = False, aberrant_threshold=10)
# train_reg_logistic_regression(y_train,x_train,x_test,test_ids,max_iters=2000,lambdas=[1e-6],gammas=[0.1], threshold=0.2, k_fold=4)                            #F1 : 0.422

# x_train,y_train,x_test,train_ids, test_ids = prepare_data(threshold_features = 0.5,threshold_points = 0.5, normalize = True, remove_outliers = False, aberrant_threshold=5)

x_train, y_train, x_test, train_ids, test_ids = prepare_data2(
    threshold_features=0.8,
    threshold_points=0.5,
    normalize=True,
    outlier_strategy="smart",
    fill_method="mode",
)
#train_least_squares(y_train, x_train, x_test, test_ids, save_plots=True)
# train_reg_logistic_regression(y_train,x_train,x_test,test_ids,max_iters=5000,lambdas=[1e-6],gammas=[0.1], threshold=0.2, k_fold=4, save_plots=True)
# threshold: 0.5     #smart, median: F1: 0.4296                       #aggressive, median: F1 : 0.4293           #smart, mode: F1: 0.4298                #none, mode: F1: 0.4229
# threshold: 0.8     #smart, median: F1: 0.4314                       #smart, median: F1 : 0.4309                #std (threshold=5), mode: F1: 0.4304
# std (threshold=3), mode: F1: 0.4305
# -- best so far: smart, mode, threshold 0.8


# x_train,y_train,x_test,train_ids, test_ids = prepare_data(threshold_features = 0.5,threshold_points = 0.5, normalize = True, remove_outliers = False, aberrant_threshold=1000)
# train_reg_logistic_regression(y_train,x_train,x_test,test_ids,max_iters=2000,lambdas=[1e-6],gammas=[0.1], threshold=0.2, k_fold=4)                            #F1 : 0.413

train_mean_square_error_gd(y_train,x_train,x_test,test_ids,gammas=np.logspace(-8,-3,10),max_iters=1000,save_plots=True)
#train_mean_square_error_sgd(y_train,x_train,x_test,test_ids,save_plots=True)