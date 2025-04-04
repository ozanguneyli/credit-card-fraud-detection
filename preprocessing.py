"""
Preprocessing module for Credit Card Fraud Detection.

This module contains functions for:
- Loading and cleaning data
- Feature engineering
- Handling class imbalance
- Data preprocessing and transformation
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from scipy import stats
import logging

from config import TEST_SIZE, RANDOM_STATE, PCA_THRESHOLD, PCA_MAX_COMPONENTS, OUTLIER_THRESHOLD, HIGH_CORRELATION_THRESHOLD

# Configure logger
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing the dataset
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Check for required columns
        required_cols = ['Time', 'Amount', 'Class']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' not found in dataset")
        
        return df
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Dataset file at {filepath} is empty")
        raise


def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform advanced feature engineering for credit card fraud detection.
    
    Features created include:
    - Temporal features (Hour, Day, Part of Day) from 'Time'
    - Log-transformed version of 'Amount' to reduce skewness
    - Multiple anomaly scores for 'Amount'
    - Interaction features between time and amount
    - Magnitude and direction features from V components
    - Outlier detection in the V features
    
    Args:
        df (pd.DataFrame): Input dataframe with raw features
        
    Returns:
        pd.DataFrame: Enhanced dataframe with engineered features
    """
    logger.info("Starting advanced feature engineering")
    df_copy = df.copy()
    feature_count_before = df_copy.shape[1]
    
    # Process temporal features if 'Time' exists
    if 'Time' in df_copy.columns:
        logger.info("Engineering temporal features")
        # Convert 'Time' to hours (0-23)
        df_copy['Hour'] = (df_copy['Time'] / 3600) % 24
        
        # Day feature (starting from Day 1)
        df_copy['Day'] = (df_copy['Time'] // 86400) + 1
        
        # Part of Day (categorical time periods)
        df_copy['PartOfDay'] = pd.cut(
            df_copy['Hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        ).astype(str)
        
        # One-hot encode 'PartOfDay'
        part_of_day_dummies = pd.get_dummies(df_copy['PartOfDay'], prefix='TimeOfDay')
        df_copy = pd.concat([df_copy, part_of_day_dummies], axis=1)
        df_copy.drop('PartOfDay', axis=1, inplace=True)
        
        # Weekend indicator (assuming days start from 1 and follow calendar)
        df_copy['IsWeekend'] = (df_copy['Day'] % 7).isin([0, 6]).astype(int)
        
        # Periodic transformation of hours (to capture cyclical nature)
        df_copy['Hour_sin'] = np.sin(2 * np.pi * df_copy['Hour'] / 24)
        df_copy['Hour_cos'] = np.cos(2 * np.pi * df_copy['Hour'] / 24)
    

    # Process amount features if 'Amount' exists
    if 'Amount' in df_copy.columns:
        logger.info("Engineering amount-related features")
        # Log-transform 'Amount' to reduce skewness (log1p to handle zeros)
        df_copy['LogAmount'] = np.log1p(df_copy['Amount'])
        
        # Amount bucketing (categorizing transactions by size)
        df_copy['AmountBucket'] = pd.qcut(
            df_copy['Amount'], 
            q=5, 
            labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'],
            duplicates='drop'
        ).astype(str)
        # One-hot encode amount buckets
        amount_dummies = pd.get_dummies(df_copy['AmountBucket'], prefix='Amount')
        df_copy = pd.concat([df_copy, amount_dummies], axis=1)
        df_copy.drop('AmountBucket', axis=1, inplace=True)
        
        # Compute multiple anomaly scores for 'Amount'
        df_copy['Amount_Zscore'] = stats.zscore(df_copy['Amount'], nan_policy='omit')
        
        # Manual MAD calculation (median absolute deviation)
        median_amount = df_copy['Amount'].median()
        mad = np.median(np.abs(df_copy['Amount'] - median_amount))
        df_copy['Amount_MAD'] = np.abs(df_copy['Amount'] - median_amount) / (mad + 1e-8)
        
        # Mark suspicious amounts (typical online fraud thresholds)
        df_copy['SuspiciousAmount'] = ((df_copy['Amount'] > 1000) | (df_copy['Amount'] < 1)).astype(int)

    # Create interaction features
    if 'Hour' in df_copy.columns and 'LogAmount' in df_copy.columns:
        logger.info("Creating interaction features")
        df_copy['Hour_LogAmount'] = df_copy['Hour'] * df_copy['LogAmount']
    
    if 'IsWeekend' in df_copy.columns and 'LogAmount' in df_copy.columns:
        df_copy['Weekend_LogAmount'] = df_copy['IsWeekend'] * df_copy['LogAmount']

    # Process V features (assuming these are PCA components from original data)
    v_features = [col for col in df_copy.columns if col.startswith('V')]
    if len(v_features) > 0:
        logger.info(f"Processing {len(v_features)} V features")
        # Calculate magnitude of the V components (Euclidean norm)
        df_copy['V_Magnitude'] = np.sqrt(np.sum(df_copy[v_features] ** 2, axis=1))
        
        # Create ratios between certain V components (first 5 as example)
        first_vs = v_features[:5]
        for i, v1 in enumerate(first_vs):
            for j, v2 in enumerate(first_vs):
                if i < j:  # avoid duplicate ratios and division by zero issues
                    ratio_name = f"{v1}_to_{v2}"
                    # Handle division by zero
                    df_copy[ratio_name] = df_copy[v1] / (df_copy[v2] + 1e-8)
        
        # Detect extreme outliers in V features that might indicate fraud
        for v in v_features:
            outlier_col = f"{v}_Outlier"
            q1 = df_copy[v].quantile(0.01)
            q3 = df_copy[v].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - (OUTLIER_THRESHOLD * iqr)
            upper_bound = q3 + (OUTLIER_THRESHOLD * iqr)
            df_copy[outlier_col] = ((df_copy[v] < lower_bound) | (df_copy[v] > upper_bound)).astype(int)
        
        # Count the number of outlier features per transaction
        outlier_cols = [col for col in df_copy.columns if col.endswith('_Outlier')]
        df_copy['Outlier_Count'] = df_copy[outlier_cols].sum(axis=1)
    
    feature_count_after = df_copy.shape[1]
    logger.info(f"Feature engineering complete. Added {feature_count_after - feature_count_before} new features")
    
    return df_copy


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle extreme outliers in the dataset using IQR-based capping.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    logger.info("Handling outliers in numerical features")
    df_copy = df.copy()
    
    # Find numerical columns (excluding 'Time', 'Class', and one-hot encoded columns)
    numerical_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if not (
        col == 'Time' or 
        col == 'Class' or 
        col.startswith('TimeOfDay_') or 
        col.startswith('Amount_')
    )]
    
    # Apply capping to extreme outliers
    for col in numerical_cols:
        q1 = df_copy[col].quantile(0.01)
        q3 = df_copy[col].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - (OUTLIER_THRESHOLD * iqr)
        upper_bound = q3 + (OUTLIER_THRESHOLD * iqr)
        
        # Count outliers before capping
        outliers_count = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
        
        # Cap outliers
        df_copy[col] = np.where(df_copy[col] < lower_bound, lower_bound, df_copy[col])
        df_copy[col] = np.where(df_copy[col] > upper_bound, upper_bound, df_copy[col])
        
        if outliers_count > 0:
            logger.info(f"Capped {outliers_count} outliers in column '{col}'")
    
    return df_copy


def drop_correlated_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drop highly correlated features to reduce multicollinearity.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        
    Returns:
        pd.DataFrame: Dataframe with highly correlated features removed
    """
    logger.info("Checking for highly correlated features")
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    high_corr_features = [column for column in upper.columns if any(upper[column] > HIGH_CORRELATION_THRESHOLD)]
    
    if high_corr_features:
        logger.info(f"Dropping {len(high_corr_features)} highly correlated features: {high_corr_features}")
        return X.drop(columns=high_corr_features)
    else:
        logger.info("No highly correlated features found")
        return X


def apply_sampling_strategy(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    sampling: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the specified sampling strategy to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling: Sampling strategy name ("smote", "smoteenn", "adasyn", "undersample", "none")
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled X_train and y_train
    """
    logger.info(f"Applying {sampling} sampling strategy")
    
    # Get class distribution before sampling
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    logger.info(f"Before sampling - Negative class: {neg_count}, Positive class: {pos_count}, Ratio: {neg_count/pos_count:.2f}:1")
    
    # Apply selected sampling strategy
    if sampling == "smote":
        sampler = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif sampling == "smoteenn":
        sampler = SMOTEENN(random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif sampling == "adasyn":
        sampler = ADASYN(random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif sampling == "undersample":
        # Custom undersampling implementation
        neg_indices = np.where(y_train == 0)[0]
        pos_indices = np.where(y_train == 1)[0]
        
        # Define balance ratio (not fully balanced)
        ratio = min(5, neg_count / max(pos_count, 1))
        target_neg_count = int(pos_count * ratio)
        
        # Randomly select samples from the majority class
        selected_neg_indices = np.random.choice(neg_indices, target_neg_count, replace=False)
        selected_indices = np.concatenate([selected_neg_indices, pos_indices])
        
        X_resampled = X_train[selected_indices]
        y_resampled = y_train[selected_indices]
    else:  # "none"
        X_resampled, y_resampled = X_train, y_train
    
    # Get class distribution after sampling
    neg_count_after = np.sum(y_resampled == 0)
    pos_count_after = np.sum(y_resampled == 1)
    logger.info(f"After sampling - Negative class: {neg_count_after}, Positive class: {pos_count_after}, Ratio: {neg_count_after/pos_count_after:.2f}:1")
    
    return X_resampled, y_resampled


def preprocess_data(
    df: pd.DataFrame, 
    test_size: float = TEST_SIZE, 
    sampling: str = "smoteenn"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    Preprocess the dataset with enhanced methods for fraud detection.
    
    Process includes:
    - Advanced feature engineering
    - Outlier handling
    - Feature selection
    - Train-test split
    - Sampling for class imbalance
    - Feature scaling
    
    Args:
        df: Input dataframe
        test_size: Proportion of data to use for testing
        sampling: Sampling strategy to handle class imbalance
        
    Returns:
        Tuple containing:
        - X_train_scaled: Scaled training features
        - X_test_scaled: Scaled test features  
        - y_train: Training labels
        - y_test: Test labels
        - scaler: Fitted scaler object for future transformations
    """
    logger.info("Starting data preprocessing")
    
    # Apply advanced feature engineering
    df = advanced_feature_engineering(df)

    # Handle outliers
    df = handle_outliers(df)
    
    # Drop the original 'Time' column
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
        logger.info("Dropped 'Time' column after creating temporal features")
    
    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Drop highly correlated features
    X = drop_correlated_features(X)
    
    # Apply dimensionality reduction if the dataset has many features
    if X.shape[1] > PCA_THRESHOLD:
        logger.info(f"Dataset has {X.shape[1]} features, applying PCA")
        n_components = min(PCA_MAX_COMPONENTS, X.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_reduced = pca.fit_transform(X)
        
        # Create a DataFrame with PCA components
        X = pd.DataFrame(
            X_reduced, 
            index=X.index, 
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        logger.info(f"Reduced dimensions from {X.shape[1]} to {n_components} components")
        logger.info(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Split into training and test sets
    logger.info(f"Splitting data with test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Convert to numpy arrays if they're pandas objects
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    # Apply sampling strategy
    if sampling != "none":
        X_train, y_train = apply_sampling_strategy(X_train, y_train, sampling)
    
    # Scale features
    logger.info("Scaling features with RobustScaler")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Preprocessing complete")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler