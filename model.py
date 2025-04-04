"""
Model training and evaluation module for Credit Card Fraud Detection.

This module handles:
- Random Forest model training
- Optimal threshold selection using precision-recall curves
- Model evaluation with comprehensive metrics
- Model persistence
"""

import joblib
import numpy as np
import logging
from typing import Dict, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    precision_recall_curve, roc_auc_score, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier

from config import RF_PARAMS, MODEL_SAVE_PATH

# Configure logger
logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Tuple[RandomForestClassifier, float]:
    """
    Train a Random Forest model and determine the optimal classification threshold.
    
    The function uses precision-recall curves to find the threshold that maximizes
    the F1 score, providing a better balance between precision and recall for 
    imbalanced fraud detection datasets.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (for evaluation)
        y_test: Test labels (for evaluation)
        
    Returns:
        Tuple containing:
        - The trained Random Forest model
        - The optimal threshold for classification
    """
    logger.info("Training Random Forest classifier with parameters:")
    for key, value in RF_PARAMS.items():
        logger.info(f"  {key}: {value}")
    
    # Create and train the model
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    logger.info("Random Forest model training complete")

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:]  # Top 10 features
        logger.info("Top feature importance indices: %s", top_indices)
        logger.info("Top feature importance values: %s", importances[top_indices])

    # Determine optimal threshold on the training set using precision-recall curve
    y_train_probs = model.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_probs)
    
    # Calculate F1 scores for each threshold
    # Add epsilon to avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Find threshold with maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    # Adjust index for thresholds array which might be 1 element shorter
    threshold_idx = min(optimal_idx, len(thresholds) - 1)
    optimal_threshold = thresholds[threshold_idx]
    
    logger.info(f"Optimal threshold (train set): {optimal_threshold:.4f}")
    logger.info(f"Optimal F1 score (train set): {f1_scores[optimal_idx]:.4f}")

    # Evaluate on the test set using the optimal threshold
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_probs >= optimal_threshold).astype(int)

    # Calculate key metrics
    accuracy_val = accuracy_score(y_test, y_pred)
    precision_val = precision_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)
    f1_val = f1_score(y_test, y_pred)
    auc_train = roc_auc_score(y_train, y_train_probs)
    auc_test = roc_auc_score(y_test, y_test_probs)
    avg_precision = average_precision_score(y_test, y_test_probs)

    # Log evaluation metrics
    logger.info("\nTraining Evaluation Metrics:")
    logger.info(f"Train AUC: {auc_train:.4f}")
    logger.info(f"Test AUC: {auc_test:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"Accuracy: {accuracy_val:.4f}")
    logger.info(f"Precision: {precision_val:.4f}")
    logger.info(f"Recall: {recall_val:.4f}")
    logger.info(f"F1 Score: {f1_val:.4f}")
    
    return model, optimal_threshold


def evaluate_model(
    model: RandomForestClassifier, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    threshold: float
) -> Dict[str, float]:
    """
    Evaluate the Random Forest model using the given threshold.
    
    Calculates and returns comprehensive evaluation metrics focusing on
    fraud detection performance measures.
    
    Args:
        model: Trained RandomForest model
        X_test: Test features
        y_test: Test labels 
        threshold: Classification threshold (optimized from training)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating model with threshold: {threshold:.4f}")
    
    # Generate probability predictions
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    
    # Apply threshold to get binary predictions
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Calculate metrics
    accuracy_val = accuracy_score(y_test, y_pred)
    precision_val = precision_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)
    f1_val = f1_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_pred_probs)
    avg_precision = average_precision_score(y_test, y_pred_probs)
    
    # Get confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # Calculate additional fraud-specific metrics
    specificity = tn / (tn + fp)  # True negative rate
    false_positive_rate = fp / (fp + tn)  # FPR
    
    # Financial metrics (assuming average costs)
    # These are placeholder values - should be adjusted based on business context
    avg_fraud_amount = 1000  # Average amount lost in a fraudulent transaction
    investigation_cost = 50  # Cost to investigate a flagged transaction
    
    # Estimated savings from catching frauds minus cost of investigations
    savings = tp * avg_fraud_amount - (tp + fp) * investigation_cost
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy_val,
        'precision': precision_val,
        'recall': recall_val,
        'f1_score': f1_val,
        'auc': auc_val,
        'avg_precision': avg_precision,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'estimated_savings': savings
    }

    # Log metrics
    logger.info("\nFinal Evaluation Metrics:")
    logger.info(f"Accuracy: {accuracy_val:.4f}")
    logger.info(f"Precision: {precision_val:.4f}")
    logger.info(f"Recall: {recall_val:.4f}")
    logger.info(f"F1 Score: {f1_val:.4f}")
    logger.info(f"AUC: {auc_val:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    
    # Log confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info(f"[[{tn}, {fp}],")
    logger.info(f" [{fn}, {tp}]]")
    logger.info(f"True Positives: {tp} | False Negatives: {fn}")
    logger.info(f"False Positives: {fp} | True Negatives: {tn}")
    
    # Log classification report for more detailed metrics per class
    report = classification_report(y_test, y_pred)
    logger.info(f"\nClassification Report:\n{report}")
    
    return metrics


def save_model(model: RandomForestClassifier, filename: str = MODEL_SAVE_PATH) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: The trained model to save
        filename: The path where the model will be saved
        
    Returns:
        None
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Model successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(filename: str = MODEL_SAVE_PATH) -> RandomForestClassifier:
    """
    Load a trained model from disk.
    
    Args:
        filename: Path to the saved model file
        
    Returns:
        The loaded model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    try:
        model = joblib.load(filename)
        logger.info(f"Model successfully loaded from {filename}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {filename}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise