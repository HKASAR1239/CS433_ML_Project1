import numpy as np
import os
import helpers as hl
import implementations as impl
import data_exploration as de
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
THRESHOLD_FEATURES = 0.8  
THRESHOLD_POINTS = 0.6   
NORMALIZE = True          
REMOVE_OUTLIERS = False  
MAX_ITERS = 5000          
GAMMA = 0.1  
LAMBDA = 0.001
# best gamma / best lambda : 0.1 0.0001
# Training F1 score: 0.413

def prepare_data(threshold_features = 0.8,threshold_points = 0.6, normalize = True, remove_outliers = False):
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
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/data/dataset/'
    
    # Load Data
    print("Loading data...")
    x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = hl.load_csv_data(dir_path)
    print(f"Raw training data shape: {x_train_raw.shape}")
    print(f"Raw test data shape: {x_test_raw.shape}")
    print(f"Raw labels shape: {y_train_raw.shape}")

    #Preprocessing Data
    print("\nPreprocessing training data...")
    x_train, removed_features, removed_points = de.fill_data(
    x_train_raw,
    remove_features=[],
    remove_points=[],
    threshold=True,
    threshold_features=threshold_features,
    threshold_points=threshold_points,
    normalize=normalize,
    remove_outliers=remove_outliers)

    # Prints some info
    print(f"Removed {len(removed_features)} features with >{THRESHOLD_FEATURES*100}% missing data")
    print(f"Removed {len(removed_points)} data points with >{THRESHOLD_POINTS*100}% missing data")
    print(f"Processed training data shape: {x_train.shape}")

    # Remove corresponding labels for removed datapoints
    y_train = np.delete(y_train_raw, removed_points, axis=0)
    print(f"Processed labels shape: {y_train.shape}")

    #Repeat operation for x_test
    print("\nPreprocessing test data...")
    x_test, _, _ = de.fill_data(
    x_test_raw,
    remove_features=removed_features,
    remove_points=[], 
    threshold=False,   
    threshold_features=threshold_features,
    threshold_points=threshold_points,
    normalize=normalize,
    remove_outliers=False) 
    print(f"Processed test data shape: {x_test.shape}")

    return x_train,y_train,x_test,train_ids, test_ids

def compute_f1_score(y_train_binary,x_train_bias,weights,threshold = 0.2):
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
    y_train_pred_binary = (y_train_pred_prob >= threshold).astype(int)  # Same threshold as test

    # F1 Score
    tp = np.sum((y_train_pred_binary == 1) & (y_train_binary == 1))
    fp = np.sum((y_train_pred_binary == 1) & (y_train_binary == 0))
    fn = np.sum((y_train_pred_binary == 0) & (y_train_binary == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def train_reg_logistic_regression(
        y_train,
        x_train,
        x_test,
        test_ids,
        gammas = [0.1],
        lambdas = np.logspace(-5, -1, 4),
        max_iters=2000,                                  
        k_fold=4,
        seed=42):
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
    k_indices = de.build_k_indices(y_train,k_fold,seed)

    # Cross validation over lambdas and gammas
    best_f1_score = 0.0
    best_gamma = None
    best_lambda = None

    for lambda_ in lambdas:
        print("Running for lambda = ",lambda_)
        for gamma in gammas:
            print("Running for gamma = ",gamma)
            f1_scores = []
            for k in range(k_fold):
                # Split into training/validation
                val_idx = k_indices[k]
                train_idx = np.delete(np.arange(y_train_binary.shape[0]), val_idx)
                X_tr, X_val = x_train_bias[train_idx], x_train_bias[val_idx]
                y_tr, y_val = y_train_binary[train_idx], y_train_binary[val_idx]

                # Train model on this fold
                w, loss = impl.reg_logistic_regression(
                    y_tr,
                    X_tr,
                    lambda_=lambda_,
                    initial_w=initial_w,
                    max_iters=max_iters,
                    gamma=gamma
                )
                # Predictions on validation sets
                f1 = compute_f1_score(y_val,X_val,w,threshold)
                f1_scores.append(f1)
            # Check quality of parameters
            f1_score = np.mean(f1_scores)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_gamma = gamma
                best_lambda = lambda_
    print("best gamma / best lambda :",best_gamma,best_lambda)
    # Compute weights and loss using optimal hyperparamaters to generate submission file. 
    print(f"Training F1 score: {best_f1_score:.3f}")
    final_w,final_loss = impl.reg_logistic_regression(
        y_train_binary,
        x_train_bias,
        lambda_ = best_lambda,
        initial_w = initial_w,
        max_iters = max_iters,
        gamma=best_gamma)
    
    # Predictions
    print("\nGenerating predictions...")
    # Predict probabilities
    z_test = x_test_bias @ final_w
    y_pred_prob = impl._sigmoid(z_test)
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_prob >= 0.2).astype(int)
    # Convert back to {-1, 1} format for submission
    y_pred = 2 * y_pred_binary - 1

    # Submission file
    print("\nCreating submission file...")
    output_path = "submission.csv"
    hl.create_csv_submission(test_ids, y_pred, output_path)
    print(f"Submission file at : {output_path}")
    
def train_logistic_regression(
        y_train,
        x_train,
        x_test,
        test_ids,
        gammas = [0.1],
        max_iters=2000,                                  
        k_fold=4,
        seed=42):
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
    k_indices = de.build_k_indices(y_train,k_fold,seed)

    # Cross validation over lambdas and gammas
    best_f1_score = 0.0
    best_gamma = None

    for gamma in gammas:
        print("Running for gamma = ",gamma)
        f1_scores = []
        for k in range(k_fold):
            # Split into training/validation
            val_idx = k_indices[k]
            train_idx = np.delete(np.arange(y_train_binary.shape[0]), val_idx)
            X_tr, X_val = x_train_bias[train_idx], x_train_bias[val_idx]
            y_tr, y_val = y_train_binary[train_idx], y_train_binary[val_idx]

            # Train model on this fold
            w, loss = impl.logistic_regression(
                y_tr,
                X_tr,
                initial_w=initial_w,
                max_iters=max_iters,
                gamma=gamma
            )
            # Predictions on validation sets
            f1 = compute_f1_score(y_val,X_val,w,threshold)
            f1_scores.append(f1)
        # Check quality of parameters
        f1_score = np.mean(f1_scores)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_gamma = gamma
    print("best gamma  :",best_gamma)
    # Compute weights and loss using optimal hyperparamaters to generate submission file. 
    print(f"Training F1 score: {best_f1_score:.3f}")
    final_w,final_loss = impl.logistic_regression(
        y_train_binary,
        x_train_bias,
        initial_w = initial_w,
        max_iters = max_iters,
        gamma=best_gamma)
    
    # Predictions
    print("\nGenerating predictions...")
    # Predict probabilities
    z_test = x_test_bias @ final_w
    y_pred_prob = impl._sigmoid(z_test)
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_prob >= 0.2).astype(int)
    # Convert back to {-1, 1} format for submission
    y_pred = 2 * y_pred_binary - 1

    # Submission file
    print("\nCreating submission file...")
    output_path = "submission.csv"
    hl.create_csv_submission(test_ids, y_pred, output_path)
    print(f"Submission file at : {output_path}")

def train_ridge_regression(
        y_train,
        x_train,
        x_test,
        test_ids,
        lambdas = np.logspace(-5, -1, 4),                                
        k_fold=4,
        seed=42):
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
    k_indices = de.build_k_indices(y_train,k_fold,seed)

    # Cross validation over lambdas and gammas
    best_f1_score = 0.0
    best_lambda = None

    for lambda_ in lambdas:
        print("Running for lambda = ",lambda_)
        f1_scores = []
        for k in range(k_fold):
            # Split into training/validation
            val_idx = k_indices[k]
            train_idx = np.delete(np.arange(y_train_binary.shape[0]), val_idx)
            X_tr, X_val = x_train_bias[train_idx], x_train_bias[val_idx]
            y_tr, y_val = y_train_binary[train_idx], y_train_binary[val_idx]

            # Train model on this fold
            w, loss = impl.ridge_regression(
                y_tr,
                X_tr,
                lambda_=lambda_,
            )
            # Predictions on validation sets
            f1 = compute_f1_score(y_val,X_val,w,threshold)
            f1_scores.append(f1)
        # Check quality of parameters
        f1_score = np.mean(f1_scores)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_lambda = lambda_
    print(" best lambda :",best_lambda)
    # Compute weights and loss using optimal hyperparamaters to generate submission file. 
    print(f"Training F1 score: {best_f1_score:.3f}")
    final_w,final_loss = impl.ridge_regression(
        y_train_binary,
        x_train_bias,
        lambda_ = best_lambda)
    
    # Predictions
    print("\nGenerating predictions...")
    # Predict probabilities
    z_test = x_test_bias @ final_w
    y_pred_prob = impl._sigmoid(z_test)
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_prob >= 0.2).astype(int)
    # Convert back to {-1, 1} format for submission
    y_pred = 2 * y_pred_binary - 1

    # Submission file
    print("\nCreating submission file...")
    output_path = "submission.csv"
    hl.create_csv_submission(test_ids, y_pred, output_path)
    print(f"Submission file at : {output_path}")

x_train,y_train,x_test,train_ids, test_ids = prepare_data(threshold_features = 0.8,threshold_points = 0.6, normalize = True, remove_outliers = False)

train_reg_logistic_regression(y_train,x_train,x_test,test_ids)
"""
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/data/dataset/'

# Load Data
print("Loading data...")
x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = hl.load_csv_data(dir_path)
# Print shape for debug
print(f"Raw training data shape: {x_train_raw.shape}")
print(f"Raw test data shape: {x_test_raw.shape}")
print(f"Raw labels shape: {y_train_raw.shape}")

# Preprocessing training data using de methods
print("\nPreprocessing training data...")
x_train, removed_features, removed_points = de.fill_data(
    x_train_raw,
    remove_features=[],
    remove_points=[],
    threshold=True,
    threshold_features=THRESHOLD_FEATURES,
    threshold_points=THRESHOLD_POINTS,
    normalize=NORMALIZE,
    remove_outliers=REMOVE_OUTLIERS
)
# Prints some info
print(f"Removed {len(removed_features)} features with >{THRESHOLD_FEATURES*100}% missing data")
print(f"Removed {len(removed_points)} data points with >{THRESHOLD_POINTS*100}% missing data")
print(f"Processed training data shape: {x_train.shape}")
# Remove corresponding labels for removed data points
y_train = np.delete(y_train_raw, removed_points, axis=0)
print(f"Processed labels shape: {y_train.shape}")


# Same for test data
print("\nPreprocessing test data...")
x_test, _, _ = de.fill_data(
    x_test_raw,
    remove_features=removed_features,
    remove_points=[], 
    threshold=False,   
    threshold_features=THRESHOLD_FEATURES,
    threshold_points=THRESHOLD_POINTS,
    normalize=NORMALIZE,
    remove_outliers=False 
)
print(f"Processed test data shape: {x_test.shape}")





# Prepare data for logreg
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


# Training
print("\nTraining logistic regression model...")
# Initialize weights
initial_w = np.zeros(x_train_bias.shape[1])
# Train regularized logistic regression
w, loss = impl.reg_logistic_regression(
    y_train_binary,
    x_train_bias,
    lambda_=LAMBDA,
    initial_w=initial_w,
    max_iters=MAX_ITERS,
    gamma=GAMMA
)
print(f"Training completed. Final loss: {loss:.6f}")


# Predictions
print("\nGenerating predictions...")
# Predict probabilities
z_test = x_test_bias @ w
y_pred_prob = impl._sigmoid(z_test)
# Convert probabilities to binary predictions
y_pred_binary = (y_pred_prob >= 0.2).astype(int)
# Convert back to {-1, 1} format for submission
y_pred = 2 * y_pred_binary - 1
print(f"Prediction distribution:")
print(f"Class -1: {np.sum(y_pred == -1)} samples")
print(f"Class 1: {np.sum(y_pred == 1)} samples")


# Submission file
print("\nCreating submission file...")
output_path = "submission.csv"
hl.create_csv_submission(test_ids, y_pred, output_path)
print(f"Submission file at : {output_path}")


# Calculate metrics
print("\nComputing training metrics...")
z_train = x_train_bias @ w
y_train_pred_prob = impl._sigmoid(z_train)
y_train_pred_binary = (y_train_pred_prob >= 0.2).astype(int)  # Same threshold as test

# Accuracy
train_accuracy = np.mean(y_train_pred_binary == y_train_binary)

# F1 Score
tp = np.sum((y_train_pred_binary == 1) & (y_train_binary == 1))
fp = np.sum((y_train_pred_binary == 1) & (y_train_binary == 0))
fn = np.sum((y_train_pred_binary == 0) & (y_train_binary == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Training F1 score: {f1:.3f}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
"""