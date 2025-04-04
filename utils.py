"""
Utility functions for the Credit Card Fraud Detection project.

This module provides helper functions for:
- Data visualization
- Performance reporting
- Model interpretation
- Utility functions for prediction
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.inspection import permutation_importance

from config import SCALER_SAVE_PATH, THRESHOLD_SAVE_PATH

# Configure logger
logger = logging.getLogger(__name__)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Plot the ROC curve for the model predictions.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        save_path: Optional path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    threshold: Optional[float] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the Precision-Recall curve for the model predictions.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        threshold: Optional threshold to highlight on the curve
        save_path: Optional path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (area = {pr_auc:.4f})')
    
    # If threshold is provided, find the point on the curve
    if threshold is not None:
        # Find the closest threshold value
        idx = np.argmin(np.abs(thresholds - threshold))
        plt.plot(recall[idx], precision[idx], 'ro', 
                 label=f'Threshold: {threshold:.4f}\nPrecision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()


def plot_feature_importance(
    model: Any, 
    feature_names: List[str], 
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the feature importance of the model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Optional path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def get_permutation_importance(
    model: Any, 
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: List[str],
    top_n: int = 20,
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Calculate permutation importance for features.
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        feature_names: List of feature names
        top_n: Number of top features to return
        n_repeats: Number of times to permute a feature
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    logger.info("Calculating permutation importance...")
    
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )
    
    feature_importance = {
        feature_names[i]: importance 
        for i, importance in enumerate(perm_importance.importances_mean)
    }
    
    # Sort by importance
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )
    
    return sorted_importance


def predict_fraud_probability(
    transaction_data: Union[pd.DataFrame, np.ndarray],
    model_path: str,
    scaler_path: str = SCALER_SAVE_PATH,
    threshold_path: str = THRESHOLD_SAVE_PATH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict fraud probability for new transactions.
    
    Args:
        transaction_data: New transaction data (should have same features as training data)
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        threshold_path: Path to the saved optimal threshold
        
    Returns:
        Tuple containing:
        - Fraud probabilities for each transaction
        - Binary fraud predictions (0 or 1)
    """
    try:
        # Load model, scaler and threshold
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        threshold = joblib.load(threshold_path)
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Loaded scaler from {scaler_path}")
        logger.info(f"Loaded threshold ({threshold:.4f}) from {threshold_path}")
        
        # Scale the data
        if isinstance(transaction_data, pd.DataFrame):
            scaled_data = scaler.transform(transaction_data)
        else:
            scaled_data = scaler.transform(transaction_data)
        
        # Get probabilities
        fraud_probs = model.predict_proba(scaled_data)[:, 1]
        
        # Apply threshold
        fraud_predictions = (fraud_probs >= threshold).astype(int)
        
        return fraud_probs, fraud_predictions
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error predicting fraud: {str(e)}")
        raise


def generate_performance_report(metrics: Dict[str, float], save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive performance report from metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Optional path to save the report
        
    Returns:
        Report as a string
    """
    report = []
    report.append("=" * 60)
    report.append("CREDIT CARD FRAUD DETECTION - PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Classification metrics
    report.append("CLASSIFICATION METRICS")
    report.append("-" * 60)
    if 'accuracy' in metrics:
        report.append(f"Accuracy:      {metrics['accuracy']:.4f}")
    if 'precision' in metrics:
        report.append(f"Precision:     {metrics['precision']:.4f}")
    if 'recall' in metrics:
        report.append(f"Recall:        {metrics['recall']:.4f}")
    if 'f1_score' in metrics:
        report.append(f"F1 Score:      {metrics['f1_score']:.4f}")
    if 'auc' in metrics:
        report.append(f"AUC:           {metrics['auc']:.4f}")
    if 'avg_precision' in metrics:
        report.append(f"Avg Precision: {metrics['avg_precision']:.4f}")
    report.append("")
    
    # Fraud detection specific metrics
    report.append("FRAUD DETECTION METRICS")
    report.append("-" * 60)
    if 'specificity' in metrics:
        report.append(f"Specificity:         {metrics['specificity']:.4f}")
    if 'false_positive_rate' in metrics:
        report.append(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    if 'estimated_savings' in metrics:
        report.append(f"Estimated Savings:   ${metrics['estimated_savings']:.2f}")
    report.append("")
    
    # Format as string
    report_str = "\n".join(report)
    
    # Save if path provided
    if save_path:
        try:
            with open(save_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Performance report saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
    
    return report_str