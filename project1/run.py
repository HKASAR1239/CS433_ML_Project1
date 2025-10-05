import numpy as np
import os
import helpers as hl
import implementations as impl
import data_exploration as de

# Parameters
THRESHOLD_FEATURES = 0.8  
THRESHOLD_POINTS = 0.6   
NORMALIZE = True          
REMOVE_OUTLIERS = False  
MAX_ITERS = 5000          
GAMMA = 0.1  
LAMBDA = 0.001


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