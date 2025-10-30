# ML Project 1 - Heart Disease Prediction

Machine Learning course project (EPFL CS-433, Fall 2025) - Binary classification of cardiovascular disease risk using the BRFSS dataset.

**Team:** Gabriel Taieb, Aurel Bizeau & Alexia Möller

---

## Table of Contents
- [Quick Setup](#quick-setup)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Methods Implemented](#methods-implemented)
- [Results Summary](#results-summary)
- [Usage](#usage)
- [Reproducibility](#reproducibility)

---

## Quick Setup

```bash
python3 -m venv venv_project1
source venv_project1/bin/activate
pip install -r requirements.txt
```

---

## Project Overview

This project implements binary classification to predict cardiovascular disease risk from the Behavioral Risk Factor Surveillance System (BRFSS) dataset containing health-related data from 300,000+ individuals.

**Goal:** Predict whether a person is at risk of developing coronary heart disease (MICHD) based on lifestyle and clinical features.

**Approach:** 
- Comprehensive data preprocessing pipeline
- Implementation of 6 ML methods from scratch (no external ML libraries)
- Extensive hyperparameter tuning with cross-validation
- Comparative analysis of multiple algorithms

---

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
---

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
---

## Results Summary

### Performance Comparison (Cross-Validation F1 Scores)

| Method | Best Hyperparameters | Mean F1 Score | Notes |
|--------|---------------------|---------------|-------|
| **Regularized Logistic Regression** | λ=1e-6, γ=0.1, threshold=0.2 | **~0.43** | Best performing method |
| **Least Squares** | threshold=-0.5 | ~0.42 | Good baseline |
| **Ridge Regression** | λ=0.0001, γ=0.1 | ~0.41 | Regularization helped |
| **Logistic Regression** | γ=0.01 | ~0.31 | Needed regularization |
| **MSE SGD** | γ=0.0002 | ~0.22 | High variance |
| **MSE GD** | γ=0.01 | ~0.30 | Slower convergence |
| **KNN** | k=30, factor=9 | ~0.35-0.42 | Computationally expensive |

### Experiment Visualizations

All plots are stored in the `plots/` directory:

1. **Regularized Logistic Regression** (`Log_Reg_F1_VS_gamma.png`)
   - F1 score plateaus at γ ≈ 0.1
   - Optimal performance: F1 ≈ 0.89 (note: this may be on training set)

2. **Least Squares Threshold Tuning**
   - `least_squares_Accuracy_VS_threshold.png`: Accuracy peaks at threshold ≈ -0.5
   - `least_squares_F1_VS_threshold.png`: F1 score maximized at threshold ≈ -0.5

3. **MSE with Gradient Descent**
   - `MSE_GD_F1_VS_gamma.png`: F1 increases monotonically with γ
   - `MSE_GD_acc_VS_gamma.png`: Accuracy follows similar trend
   - Best γ ≈ 0.01 (marked with red line)

4. **MSE with SGD**
   - `MSE_SGD_F1_VS_gamma.png`: Sharp peak at γ ≈ 0.0002
   - `MSE_SGD_acc_VS_gamma.png`: Accuracy drops sharply outside optimal range
   - SGD highly sensitive to learning rate

5. **KNN Hyperparameter Grid Search**
   - `KNN_F1_Scores.png`: 2D heatmap of F1 scores vs. (k, weighting factor)
   - `KNN_F1_Scores_CV1.png`: Single best configuration, k=30, factor=9
   - Higher weighting factors for positive class improved F1

6. **Prediction Distribution** (`Pred_distr.png`)
   - Shows distribution of predicted probabilities on train/test
   - Threshold at 0.2 separates classes

### Key Findings

1. **Regularized Logistic Regression** performed best with proper tuning
2. **Feature thresholding** at 0.8 and **mode imputation** were optimal
3. **Cross-validation** (k=4 folds) prevented overfitting
4. **Threshold tuning** crucial for binary classification (optimal ≈ 0.2-0.5)
5. **Learning rate** critically important for gradient-based methods
6. **KNN** competitive but computationally prohibitive for large datasets

---

## Usage

### Generate Predictions

Run the main script to train the model and generate submission file:

```bash
python run.py
```

This will:
1. Load training and test data from `data/dataset/`
2. Apply preprocessing pipeline
3. Train the model (currently: least squares or regularized logistic regression)
4. Generate `submission.csv` for AIcrowd submission

### Key Configuration Parameters

Edit these parameters in `run.py` (lines 13-19):

```python
THRESHOLD_FEATURES = 0.8    # Remove features with >80% missing data
THRESHOLD_POINTS = 0.6      # Remove samples with >60% missing features
NORMALIZE = True            # Apply z-score normalization
REMOVE_OUTLIERS = False     # Enable/disable outlier removal
MAX_ITERS = 5000           # Iterations for iterative methods
GAMMA = 0.1                # Learning rate
LAMBDA = 0.001             # Regularization strength
```

### Switching Between Methods

In `run.py` (line 1255), uncomment your desired method:

```python
# Least Squares (current default)
train_least_squares(y_train, x_train, x_test, test_ids, save_plots=True)

# Or use Regularized Logistic Regression
# train_reg_logistic_regression(
#     y_train, x_train, x_test, test_ids,
#     max_iters=5000, lambdas=[1e-6], gammas=[0.1], 
#     threshold=0.2, k_fold=4, save_plots=True
# )

# Or use KNN
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

---

## Reproducibility

### System Requirements
- Python 3.8+
- NumPy
- Matplotlib (for visualization only)
- Seaborn (for visualization only)

### Random Seed
Set `seed=42` in cross-validation functions for reproducible splits.

### Expected Performance
With default settings:
- **Training F1 Score**: ~0.41-0.43
- **Cross-Validation F1**: ~0.42-0.43

### Data Requirements
Place these files in `data/dataset/`:
- `x_train.csv`: Training features
- `y_train.csv`: Training labels (-1, 1)
- `x_test.csv`: Test features

### Output Files
- `submission.csv`: Predictions in AIcrowd format (Id, Prediction)
- `plots/*.png`: Visualization outputs (if `save_plots=True`)

---

## Implementation Notes

### Design Decisions

1. **No External ML Libraries**: All algorithms implemented from scratch using only NumPy
2. **Modular Architecture**: Separate files for implementations, preprocessing, and execution
3. **Extensive Logging**: Progress tracking and performance metrics throughout
4. **Flexible Pipeline**: Easy to swap preprocessing strategies and models

### Performance Optimizations

- Vectorized NumPy operations
- Efficient normal equation solving with `np.linalg.solve`
- Smart outlier detection avoiding unnecessary computation
- Categorical feature detection for selective normalization

### Challenges Overcome

1. **Extreme Class Imbalance**: Addressed with F1 score optimization and threshold tuning
2. **High Missing Data Rate**: Multi-stage imputation strategy
3. **Computational Constraints**: Efficient implementations for large dataset
4. **Overfitting Risk**: Regularization and cross-validation

**Project Submission Date:** October 31st, 2025
