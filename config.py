"""
Configuration file for Credit Card Fraud Detection project.
Contains parameters for data preprocessing, model training, and evaluation.
"""

# Data paths
DATA_PATH = "data/creditcard.csv"
MODEL_SAVE_PATH = "models/rf_model.pkl"
SCALER_SAVE_PATH = "models/scaler.pkl"
THRESHOLD_SAVE_PATH = "models/optimal_threshold.pkl"

# Preprocessing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAMPLING_METHOD = "smoteenn"  # Options: "smote", "smoteenn", "adasyn", "undersample", "none"
PCA_THRESHOLD = 50  # Apply PCA if features exceed this number
PCA_MAX_COMPONENTS = 30

# Feature engineering parameters
OUTLIER_THRESHOLD = 3  # For IQR-based outlier detection
HIGH_CORRELATION_THRESHOLD = 0.95  # For dropping highly correlated features

# Random Forest parameters
RF_PARAMS = {
    "n_estimators": 600,
    "class_weight": "balanced",
    "random_state": 42,
    "bootstrap": True,
    "criterion": "gini",
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "ccp_alpha": 0.0,
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    "min_weight_fraction_leaf": 0.0,
    "oob_score": False,
    "n_jobs": -1,
    "verbose": 0,
    "warm_start": False
}

# Directories
DIRS = {
    "models": "models",
    "results": "results",
    "data": "data"
}