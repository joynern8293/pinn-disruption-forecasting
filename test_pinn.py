# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import torch

# Import the classes and functions from your main script
from pinn_forecasting_with_enhanced_logging import (
    StockDataHandler,
    PINN,
    get_user_input_files,
    get_training_hyperparameters,
    get_initial_parameter_guesses
)

class TestPINNComponents(unittest.TestCase):

    def setUp(self):
        """Set up a dummy CSV file for testing."""
        self.test_csv_path = 'test_data.csv'
        data = {
            '<DATE>': pd.to_datetime(pd.date_range(start='2020-01-01', periods=200)),
            '<CLOSE>': [100 + i for i in range(200)]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        """Remove the dummy CSV file after tests."""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_data_handler_split(self):
        """Test the data loading and splitting logic."""
        handler = StockDataHandler(self.test_csv_path, 21, 41, 2, 41, 10.0)
        processed_df = handler.load_and_process_data()
        original_len = len(processed_df)
        self.assertGreater(original_len, 100, "Dataframe should have significant length after processing")

        (t_train, p_train), (t_test, p_test), (df_train, df_test) = handler.get_split_tensors(0.2)

        self.assertEqual(len(t_train) + len(t_test), original_len)
        self.assertAlmostEqual(len(t_test) / original_len, 0.2, places=1)
        self.assertIsInstance(t_train, torch.Tensor)
        self.assertTrue(t_train.requires_grad)

    @patch('builtins.input', side_effect=['test_data.csv', '0.25'])
    def test_get_user_input_files(self, mock_input):
        """Test the file and split ratio input function."""
        file_path, ratio = get_user_input_files()
        self.assertEqual(file_path, 'test_data.csv')
        self.assertEqual(ratio, 0.25)

    @patch('builtins.input', side_effect=['10000', '0.002', '0.05', '0.9'])
    def test_get_training_hyperparameters(self, mock_input):
        """Test the hyperparameter input function."""
        epochs, lr, l_pde, gamma = get_training_hyperparameters()
        self.assertEqual(epochs, 10000)
        self.assertEqual(lr, 0.002)
        self.assertEqual(l_pde, 0.05)
        self.assertEqual(gamma, 0.9)

    @patch('builtins.input', side_effect=['yes'])
    def test_get_initial_parameter_guesses_auto(self, mock_input):
        """Test the automatic parameter guessing."""
        p_train_np = torch.tensor([[100.0], [105.0], [110.0]])
        P0, k, t1, t2, s = get_initial_parameter_guesses(p_train_np)
        self.assertAlmostEqual(P0, 100.0)
        self.assertIsNotNone(k)
        self.assertAlmostEqual(t1, 3 * 0.33)
        self.assertAlmostEqual(t2, 3 * 0.66)
        self.assertEqual(s, 0.1)

if __name__ == '__main__':
    # You might need to adjust how you run this depending on your environment
    # This allows it to run correctly inside some IDEs/notebooks
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
