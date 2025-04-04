"""
Unit tests for the preprocessing module.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Import functions to test
from preprocessing import (
    load_data, 
    advanced_feature_engineering, 
    handle_outliers, 
    preprocess_data
)


class TestPreprocessing(unittest.TestCase):
    """Test cases for the preprocessing module."""

    def setUp(self):
        """Set up test data."""
        # Create a mock dataframe
        np.random.seed(42)
        self.mock_df = pd.DataFrame({
            'Time': np.random.randint(0, 172800, 100),  # 2 days in seconds
            'Amount': np.random.exponential(scale=100, size=100),
            'V1': np.random.normal(0, 1, 100),
            'V2': np.random.normal(0, 1, 100),
            'V3': np.random.normal(0, 1, 100),
            'Class': np.random.choice([0, 1], 100, p=[0.95, 0.05])  # 5% fraud
        })
        
        # Add some outliers
        self.mock_df.loc[0, 'Amount'] = 10000  # Large amount
        self.mock_df.loc[1, 'V1'] = 10  # Outlier

    def test_load_data(self):
        """Test load_data function with mock file."""
        # Save mock dataframe to a test CSV file
        temp_file = "test_data.csv"
        self.mock_df.to_csv(temp_file, index=False)
        
        # Test loading the file
        df = load_data(temp_file)
        
        # Check if the dataframe is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, self.mock_df.shape)
        
        # Clean up
        import os
        os.remove(temp_file)
    
    def test_load_data_file_not_found(self):
        """Test load_data function with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")
    
    def test_advanced_feature_engineering(self):
        """Test advanced_feature_engineering function."""
        # Apply feature engineering
        df_engineered = advanced_feature_engineering(self.mock_df)
        
        # Check new features were created
        self.assertIn('Hour', df_engineered.columns)
        self.assertIn('Day', df_engineered.columns)
        self.assertIn('LogAmount', df_engineered.columns)
        
        # Check for temporal features
        self.assertTrue(all(0 <= df_engineered['Hour']) and all(df_engineered['Hour'] < 24))
        
        # Check for amount features
        self.assertTrue(all(df_engineered['LogAmount'] >= 0))
        
        # Check for V features processing if sufficient V columns exist
        v_features = [col for col in df_engineered.columns if col.startswith('V')]
        if len(v_features) >= 5:
            self.assertIn('V_Magnitude', df_engineered.columns)
    
    def test_handle_outliers(self):
        """Test handle_outliers function."""
        # Get original outlier values
        original_amount_max = self.mock_df['Amount'].max()
        original_v1_max = self.mock_df['V1'].max()
        
        # Handle outliers
        df_no_outliers = handle_outliers(self.mock_df)
        
        # Check if outliers were handled
        self.assertLess(df_no_outliers['Amount'].max(), original_amount_max)
        self.assertLess(df_no_outliers['V1'].max(), original_v1_max)
    
    def test_preprocess_data(self):
        """Test the full preprocess_data function."""
        # Apply preprocessing
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            self.mock_df, test_size=0.2, sampling="none"
        )
        
        # Check output types
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertIsInstance(scaler, RobustScaler)
        
        # Check shapes
        self.assertEqual(len(y_train) + len(y_test), len(self.mock_df))
        self.assertEqual(X_train.shape[0], len(y_train))
        self.assertEqual(X_test.shape[0], len(y_test))
    
    def test_sampling_strategies(self):
        """Test different sampling strategies."""
        # Test with SMOTE
        X_train_smote, _, y_train_smote, _, _ = preprocess_data(
            self.mock_df, test_size=0.2, sampling="smote"
        )
        
        # Check if class balance improved
        class_counts = np.bincount(y_train_smote.astype(int))
        self.assertGreater(class_counts[1], 0)  # Should have positive samples
        
        # Test with undersampling
        X_train_under, _, y_train_under, _, _ = preprocess_data(
            self.mock_df, test_size=0.2, sampling="undersample"
        )
        
        # Undersampling should reduce majority class
        self.assertLess(len(y_train_under), len(self.mock_df) * 0.8)


if __name__ == '__main__':
    unittest.main()