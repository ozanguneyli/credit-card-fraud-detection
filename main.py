"""
Main execution script for Credit Card Fraud Detection.

This script orchestrates the entire fraud detection workflow:
1. Loads and preprocesses the credit card dataset
2. Trains a Random Forest model with optimized threshold
3. Evaluates the model's performance
4. Saves the trained model, scaler, and optimal threshold
"""

import os
import joblib
import logging
from datetime import datetime

from preprocessing import load_data, preprocess_data
from model import train_random_forest, evaluate_model
from config import DATA_PATH, DIRS, MODEL_SAVE_PATH, SCALER_SAVE_PATH, THRESHOLD_SAVE_PATH, SAMPLING_METHOD


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    for dir_name in DIRS.values():
        os.makedirs(dir_name, exist_ok=True)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    logger.info("Directory setup complete.")


def save_artifacts(model, scaler, threshold) -> None:
    """Save the trained model, scaler, and optimal threshold."""
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(threshold, THRESHOLD_SAVE_PATH)
    
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")
    logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")
    logger.info(f"Optimal threshold saved to {THRESHOLD_SAVE_PATH}")


def main() -> tuple:
    """
    Main function that executes the entire fraud detection workflow.
    
    Returns:
        tuple: Contains the trained model, optimal threshold, and evaluation metrics
    """
    # Setup project directories
    setup_directories()

    # Load and preprocess data
    logger.info("Loading data from %s", DATA_PATH)
    df = load_data(DATA_PATH)
    
    logger.info("Preprocessing data with %s sampling method", SAMPLING_METHOD)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df, 
        sampling=SAMPLING_METHOD
    )
    
    # Train the Random Forest model and determine the optimal threshold
    logger.info("Training Random Forest model...")
    rf_model, optimal_threshold = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Evaluate the model using the optimal threshold
    logger.info(f"Evaluating model with optimal threshold: {optimal_threshold:.4f}")
    eval_metrics = evaluate_model(rf_model, X_test, y_test, threshold=optimal_threshold)
    
    # Print summary
    logger.info("=== Model Performance Summary ===")
    if eval_metrics is not None:
        for metric, value in eval_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    else:
        logger.warning("No metrics returned from evaluate_model function.")
    
    # Save model artifacts
    save_artifacts(rf_model, scaler, optimal_threshold)
    
    logger.info("Processing complete. Results and models saved.")
    
    return rf_model, optimal_threshold, eval_metrics


if __name__ == "__main__":
    main()