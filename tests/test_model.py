"""
Unit tests for the model module.
"""

import unittest
import numpy as np
import os
import tempfile
from sklearn.ensemble import RandomForestClassifier

# Import functions to test
from model import (
    train_random_forest,
    evaluate_model,
    save_model,
    load_model
)


class TestModel(unittest.TestCase):
    """Test cases for the model module."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        # Create features with some predictive power
        X = np.random.normal(0, 1, (n_samples, 10))
        
        # Create imbalanced target (5% fraud)
        y = np.zeros(n_samples)
        fraud_indices = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
        y[fraud_indices] = 1
        
        # Make the first feature more predictive of fraud
        X[y == 1, 0] += 2
        
        # Split into train and test
        train_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
        test_idx = np.array(list(set(range(n_samples)) - set(train_idx)))
        
        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]
        
        # Create a temporary directory for test model files
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        # Remove any test files
        import shutil
        shutil.rmtree(self.test_dir)

    def test_train_random_forest(self):
        """Test training the Random Forest model."""
        # Train the model
        model, threshold = train_random_forest(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        # Check if model and threshold are created correctly
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertIsInstance(threshold, float)
        self.assertTrue(0 <= threshold <= 1)
        
        # Basic check that the model has learned something
        train_probs = model.predict_proba(self.X_train)[:, 1]
        self.assertGreater(np.mean(train_probs[self.y_train == 1]), 
                         np.mean(train_probs[self.y_train == 0]))
    
    def test_evaluate_model(self):
        """Test model evaluation function."""
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate with default threshold
        metrics = evaluate_model(model, self.X_test, self.y_test, threshold=0.5)
        
        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check metric values are valid
        for metric_name, value in metrics.items():
            if metric_name != 'estimated_savings':  # This can be negative
                self.assertTrue(0 <= value <= 1, f"{metric_name} should be between 0 and 1")
    
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Define test path
        test_path = os.path.join(self.test_dir, "test_model.pkl")
        
        # Save model
        save_model(model, test_path)
        self.assertTrue(os.path.exists(test_path))
        
        # Load model
        loaded_model = load_model(test_path)
        
        # Check it's the right type
        self.assertIsInstance(loaded_model, RandomForestClassifier)
        
        # Check predictions match
        original_preds = model.predict(self.X_test)
        loaded_preds = loaded_model.predict(self.X_test)
        self.assertTrue(np.array_equal(original_preds, loaded_preds))
    
    def test_load_model_not_found(self):
        """Test loading a non-existent model file."""
        non_existent_path = os.path.join(self.test_dir, "non_existent_model.pkl")
        with self.assertRaises(FileNotFoundError):
            load_model(non_existent_path)


if __name__ == '__main__':
    unittest.main()