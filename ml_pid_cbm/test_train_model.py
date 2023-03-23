import unittest
from unittest.mock import patch, mock_open
from train_model import TrainModel


class testTrainModel(unittest.TestCase):
    def setUp(self):
        self.train_model = TrainModel("config.json", True)

    def test_load_features_for_train(self):
        target_features = ["Complex_mass2", "Complex_p", "Complex_pT", "Complex_eta"]
        json_data = """{"features_for_train":["Complex_mass2", "Complex_p", "Complex_pT", "Complex_eta"]}"""
        with patch("builtins.open", mock_open(read_data=json_data)):
            read_features = self.train_model.load_features_for_train()
            self.assertEqual(target_features, read_features)

    def test_load_hyper_params_vals(self):
        target_vals = (596, 5, 0.07161792803939408)
        json_data = """{"hyper_params": {"values": {"n_estimators": 596,"max_depth": 5,"learning_rate": 0.07161792803939408},
            "ranges": {"n_estimators": [400, 1000],"max_depth": [2, 6],"learning_rate": [0.01, 0.1]}}}"""
        with patch("builtins.open", mock_open(read_data=json_data)):
            read_vals = self.train_model.load_hyper_params_vals()
            self.assertEqual(target_vals, read_vals)

    def test_load_hyper_params_ranges(self):
        target_ranges = {
            "n_estimators": (400, 1000),
            "max_depth": (2, 6),
            "learning_rate": (0.01, 0.1),
        }
        json_data = """{"hyper_params": {"values": {"n_estimators": 596,"max_depth": 5,"learning_rate": 0.07161792803939408},
            "ranges": {"n_estimators": [400, 1000],"max_depth": [2, 6],"learning_rate": [0.01, 0.1]}}}"""
        with patch("builtins.open", mock_open(read_data=json_data)):
            read_ranges = self.train_model.load_hyper_params_ranges()
            self.assertEqual(read_ranges, target_ranges)
