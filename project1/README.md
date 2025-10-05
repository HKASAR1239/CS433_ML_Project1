# ML Project 1 - Heart Disease Prediction

Machine Learning course project (EPFL CS-433, Fall 2025) - Binary classification of cardiovascular disease risk.

## Quick Setup

```bash
python3 -m venv project1
source project1/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
.
├── run.py                    # Main script: trains model & generates predictions
├── implementations.py        # 6 required ML methods (GD, SGD, LS, Ridge, Logistic)
├── data_exploration.py       # Data preprocessing & cross-validation
├── helpers.py               # Utility functions (load data, create submission)
├── data/dataset/            # Training and test data (CSV)
└── submission.csv           # Output predictions for AIcrowd
```

## Usage

**Generate predictions:**
```bash
python run.py
```

This will:
- Load and preprocess data (handle missing values, normalize, remove outliers)
- Train (with one of the 6 methods - regularized logistic regression for the moment)
- Create `submission.csv` for AIcrowd submission

**Key parameters** (edit in `run.py`):
- `THRESHOLD_FEATURES = 0.8` - Remove features with >THRESHOLD_FEATURES missing data
- `LAMBDA = 0.001` - Regularization strength
- `GAMMA = 0.1` - Learning rate
- Prediction threshold (change line 100)

## Team

Gabriel Taieb, Aurel Bizeau & Alexia Möller
