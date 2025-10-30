# ML Project 1 - Heart Disease Prediction

Machine Learning course project (EPFL CS-433, Fall 2025) - Binary classification of cardiovascular disease risk using the BRFSS dataset.

**Team:** Gabriel Taieb, Aurel Bizeau & Alexia Möller

-----

## Table of Contents

  - [Quick Setup](#quick-setup)
  - [Project Overview](#project-overview)
  - [Project Structure](#project-structure)
  - [Methods Implemented](#methods-implemented)
  - [Results Summary](#results-summary)
  - [Usage](#usage)
  - [Reproducibility](#reproducibility)

-----

## Quick Setup

```bash
python3 -m venv venv_project1
source venv_project1/bin/activate
pip install -r requirements.txt
```

-----

## Project Overview

This project implements binary classification to predict cardiovascular disease risk from the Behavioral Risk Factor Surveillance System (BRFSS) dataset containing health-related data from **more than 400,000 individuals**.

**Goal:** Predict whether a person is at risk of developing coronary heart disease (MICHD) based on lifestyle and clinical features.

**Approach:**

  * Comprehensive data preprocessing pipeline (Best: 0.8 feature threshold, "smart" IQR outlier removal, and mode imputation).
  * Implementation of 6 ML methods from scratch (no external ML libraries).
  * Extensive hyperparameter tuning with cross-validation.
  * Comparative analysis of multiple algorithms.

-----

## Project Structure

```
.
├── run.py                          # Main script: trains model & generates predictions
├── implementations.py              # 6 required ML methods (GD, SGD, LS, Ridge, Logistic)
├── data_exploration.py             # Data preprocessing & cross-validation utilities
├── helpers.py                      # Utility functions (given)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── project1_description.pdf        # Project assignment description
├── submission.csv                  # Final predictions for submission
│
├── data/                           # Data directory
│   └── dataset/                    # Training and test data (CSV files)
│       ├── x_train.csv             # Training features
│       ├── y_train.csv             # Training labels (-1, 1)
│       ├── x_test.csv              # Test features
│       └── sample_submission.csv   # Example submission format (given)
│
├── plots/                          # Visualization outputs from experiments
│   ├── Missing_data.png            # Missing data analysis
│   ├── Missing_features.png        # Feature missingness visualization
│   ├── KNN_F1_Scores.png           # KNN hyperparameter tuning results
│   ├── least_squares_*.png         # Least squares threshold tuning
│   ├── Log_Reg_*.png               # Logistic regression experiments
│   ├── MSE_GD_*.png                # Gradient descent experiments
│   ├── MSE_SGD_*.png               # Stochastic gradient descent experiments
│   └── ... (other experimental plots)
│
├── grading_tests/                  # Given grading tests
│   ├── test_project1_public.py     
│   ├── conftest.py                 
│   ├── environment.yml             
│   └── INSTRUCTIONS.md
├── submission_files/               # Intermediate submission files (not tracked, generated when running)      
└── ML_Project_1_AUGAXIA/           # Report materials (LaTeX source, figures)
    ├── references.bib              # Bibliography for report
    ├── Log_Reg_F1_VS_gamma.png     # Figures for report
    └── untitled folder/
        └── latex-template.tex      # LaTeX report source

```

-----

## Methods Implemented

All methods implemented in `implementations.py`:

| Method | Function | Description |
|--------|----------|-------------|
| **Gradient Descent (GD)** | `mean_squared_error_gd` | Linear regression with full-batch gradient descent |
| **Stochastic Gradient Descent (SGD)** | `mean_squared_error_sgd` | Linear regression with batch size = 1 |
| **Least Squares** | `least_squares` | Closed-form solution via normal equations |
| **Ridge Regression** | `ridge_regression` | L2 regularized least squares |
| **Logistic Regression** | `logistic_regression` | Binary classification with gradient descent |
| **Regularized Logistic Regression** | `reg_logistic_regression` | Logistic regression with L2 penalty |

### Additional Methods Explored

  - **K-Nearest Neighbors (KNN)**: Custom implementation with weighted voting

-----

## Results Summary

### Performance Comparison (Final Validation F1 Scores)

This table reflects the final model performance based on 4-fold cross-validation or AICrowd validation scores, as reported in the project paper.

| Method | Best Hyperparameters | Validation F1 Score | Train F1 Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | $\gamma=0.08$ | **\~0.438** | 0.4259 |
| **Regularized Logistic Regression** | $\lambda=10^{-6}$, $\gamma=0.1$ | **\~0.437** | 0.4291 |
| **KNN** | $k=30$, factor=9 | \~0.318 | 0.3258 |
| **Least Squares (LSE)** | N/A | \~0.16 | 0.4220 |
| **MSE SGD** | $\gamma=0.0046$ | \~0.137 | 0.1426 |
| **Ridge Regression** | $\lambda=10^{-5}$ | \~0.033 | 0.0275 |
| **MSE GD** | $\gamma=0.01$ | \~0.0095 | 0.0101 |

### Key Findings

1.  **Model-task alignment is crucial**: Probabilistic classifiers (Logistic Regression) vastly outperformed MSE-based linear models, which failed to handle the severe class imbalance.
2.  **Preprocessing is critical**: The best strategy (smart IQR outlier removal, mode imputation) achieved an F1 score of 0.437, a significant improvement over baseline preprocessing (0.421 F1).
3.  **Best Model**: **Regularized Logistic Regression** (and standard Logistic Regression) provided the best and most stable results for this classification task.
4.  **Overfitting**: The optimal regularization $\lambda$ was minimal ($10^{-6}$), suggesting the model was not severely overfitting with proper preprocessing.

-----

## Usage

### Generate Predictions

Run the main script to train the model and generate submission file:

```bash
python run.py
```

This will:

1.  Load training and test data from `data/dataset/`
2.  Apply preprocessing pipeline
3.  Train the model (default: Regularized Logistic Regression)
4.  Generate `submission.csv` for AIcrowd submission

### Key Configuration Parameters

Edit these parameters in `run.py` (lines 13-19). The defaults below are set to the best-performing configuration from the report.

```python
THRESHOLD_FEATURES = 0.8    # Remove features with >80% missing data
THRESHOLD_POINTS = 0.6      # Remove samples with >60% missing features
NORMALIZE = True            # Apply z-score normalization
REMOVE_OUTLIERS = True      # Enable outlier removal (using 'smart' strategy)
MAX_ITERS = 5000           # Iterations for iterative methods
GAMMA = 0.1                # Optimal learning rate for Reg. Logistic
LAMBDA = 1e-6              # Optimal regularization strength for Reg. Logistic
```

### Switching Between Methods

In `run.py` (line 1255), uncomment your desired method. The default should be set to the best-performing model (Regularized Logistic Regression).

```python
# Best Model: Regularized Logistic Regression
# This model achieved the highest stable validation F1 score (0.437).
train_reg_logistic_regression(
    y_train, x_train, x_test, test_ids,
    max_iters=5000, lambdas=[1e-6], gammas=[0.1], 
    threshold=0.2, k_fold=4, save_plots=True
)

# --- Other Methods ---

# Least Squares (Poor performance: 0.16 F1)
# train_least_squares(y_train, x_train, x_test, test_ids, save_plots=True)

# Or use KNN (Moderate performance: 0.318 F1)
# train_knn(y_train, x_train, x_test, test_ids, 
#           ks=[30], factors=[9], k_fold=4)
```

### Hyperparameter Tuning

Each training function supports grid search over hyperparameters:

**Example: Tune learning rate and regularization**

```python
train_reg_logistic_regression(
    y_train, x_train, x_test, test_ids,
    max_iters=5000,
    lambdas=[1e-7, 1e-6, 1e-5, 1e-4],  # Test multiple λ values
    gammas=[0.05, 0.1, 0.15, 0.2],      # Test multiple γ values
    threshold=0.2,
    k_fold=4,
    save_plots=True
)
```

### Cross-Validation

All training functions use k-fold cross-validation (default k=4):

  - Splits data into k folds
  - Trains on k-1 folds, validates on remaining fold
  - Reports mean validation F1 score
  - Selects best hyperparameters
  - Retrains on full training set with best parameters

-----

## Reproducibility

### System Requirements

  - Python 3.8+
  - NumPy
  - Matplotlib (for visualization only)
  - Seaborn (for visualization only)

### Random Seed

Set `seed=42` in cross-validation functions for reproducible splits.

### Expected Performance

With default settings (Regularized Logistic Regression):

  - **Training F1 Score**: \~0.429
  - **Cross-Validation F1**: \~0.437

### Data Requirements

Place these files in `data/dataset/`:

  - `x_train.csv`: Training features
  - `y_train.csv`: Training labels (-1, 1)
  - `x_test.csv`: Test features

### Output Files

  - `submission.csv`: Predictions in AIcrowd format (Id, Prediction)
  - `plots/*.png`: Visualization outputs (if `save_plots=True`)

-----

## Implementation Notes

### Design Decisions

1.  **No External ML Libraries**: All algorithms implemented from scratch using only NumPy
2.  **Modular Architecture**: Separate files for implementations, preprocessing, and execution
3.  **Extensive Logging**: Progress tracking and performance metrics throughout
4.  **Flexible Pipeline**: Easy to swap preprocessing strategies and models

### Performance Optimizations

  - Vectorized NumPy operations
  - Efficient normal equation solving with `np.linalg.solve`
  - Smart outlier detection avoiding unnecessary computation
  - Categorical feature detection for selective normalization

### Challenges Overcome

1.  **Extreme Class Imbalance**: Addressed with F1 score optimization and threshold tuning
2.  **High Missing Data Rate**: Multi-stage imputation strategy
3.  **Computational Constraints**: Efficient implementations for large dataset
4.  **Overfitting Risk**: Regularization and cross-validation

**Project Submission Date:** October 31st, 2025