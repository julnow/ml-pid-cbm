import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from hipe4ml.tree_handler import TreeHandler
from prepare_model import PrepareModel


class TestPrepareModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_model_without_opt = PrepareModel("config.json", False)
        cls.train_model_with_opt = PrepareModel("config.json", True)
        cls.proton_entry = {
            "Complex_q": 1.0,
            "Complex_p": 1.0,
            "Complex_pid": 2212.0,
            "Complex_mass2": 0.8,
        }
        cls.kaon_entry = {
            "Complex_q": 1.0,
            "Complex_p": 1.2,
            "Complex_pid": 321.0,
            "Complex_mass2": 0.4,
        }
        cls.pion_entry = {
            "Complex_q": 1.0,
            "Complex_p": 1.2,
            "Complex_pid": 211.0,
            "Complex_mass2": 0.2,
        }
        cls.json_data = """{"features_for_train":["Complex_mass2", "Complex_p"],
            "hyper_params": {"values": {"n_estimators": 596,"max_depth": 5,"learning_rate": 0.07161792803939408},
            "ranges": {"n_estimators": [400, 1000],"max_depth": [2, 6],"learning_rate": [0.01, 0.1]}}}"""
        cls.proton_tree_handler = TreeHandler()
        cls.proton_tree_handler.set_data_frame(pd.DataFrame([cls.proton_entry] * 10))
        cls.kaon_tree_handler = TreeHandler()
        cls.kaon_tree_handler.set_data_frame(pd.DataFrame([cls.kaon_entry] * 10))
        cls.pion_tree_handler = TreeHandler()
        cls.pion_tree_handler.set_data_frame(pd.DataFrame([cls.pion_entry] * 10))

    def test_load_features_for_train(self):
        target_features = ["Complex_mass2", "Complex_p"]
        with patch("builtins.open", mock_open(read_data=self.json_data)):
            read_features = self.train_model_without_opt.load_features_for_train()
            self.assertEqual(target_features, read_features)

    def test_load_hyper_params_vals(self):
        target_vals = (596, 5, 0.07161792803939408)
        with patch("builtins.open", mock_open(read_data=self.json_data)):
            read_vals = self.train_model_without_opt.load_hyper_params_vals()
            self.assertEqual(target_vals, read_vals)

    def test_load_hyper_params_ranges(self):
        target_ranges = {
            "n_estimators": (400, 1000),
            "max_depth": (2, 6),
            "learning_rate": (0.01, 0.1),
        }
        with patch("builtins.open", mock_open(read_data=self.json_data)):
            read_ranges = self.train_model_without_opt.load_hyper_params_ranges()
            self.assertEqual(read_ranges, target_ranges)

    def test_prepare_train_test_data(self):

        with patch("builtins.open", mock_open(read_data=self.json_data)):
            self.train_model_without_opt.prepare_train_test_data(
                self.proton_tree_handler,
                self.kaon_tree_handler,
                self.pion_tree_handler,
                0.5,
            )
            self.train_model_with_opt.prepare_train_test_data(
                self.proton_tree_handler,
                self.kaon_tree_handler,
                self.pion_tree_handler,
                0.5,
            )

    def test_prepare_model_handler(self):

        with patch("builtins.open", mock_open(read_data=self.json_data)):
            train_test_data = self.train_model_without_opt.prepare_train_test_data(
                self.proton_tree_handler,
                self.kaon_tree_handler,
                self.pion_tree_handler,
                0.1,
            )
            self.train_model_with_opt.prepare_model_handler(  # to modify and suppress output
                train_test_data=train_test_data
            )
            self.assertRaises(
                TypeError, lambda: self.train_model_with_opt.prepare_model_handler()
            )
            self.train_model_without_opt.prepare_model_handler()

    if __name__ == "__main__":
        unittest.main()

