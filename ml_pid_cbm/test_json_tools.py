import unittest
from unittest.mock import mock_open, patch

import json_tools


class TestJsonTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.json_data = """{"file_names":{
        "training": "training_filename.test",
        "test": "test_filename.test"
        },"var_names":{"momentum": "Complex_p","charge": "Complex_q"},
        "cuts":{"Complex_mass2": {"lower": -1.0,"upper": 2.0},"Complex_pT": {"lower": 0.0,"upper": 2.0}},
        "features_for_train":["Complex_mass2", "Complex_p"],
        "hyper_params": {"values": {"n_estimators": 596,"max_depth": 5,"learning_rate": 0.07161792803939408},
        "ranges": {"n_estimators": [400, 1000],"max_depth": [2, 6],"learning_rate": [0.01, 0.1]}}}"""

    def test_create_cut_string(self):
        # ideally formatted
        expected_string = "0.1 <= test_cut < 13.0"
        self.assertEqual(
            json_tools.create_cut_string(0.1, 13.0, "test_cut"), expected_string
        )
        # not ideally formatted
        self.assertEqual(
            json_tools.create_cut_string(0.141, 13, "test_cut"), expected_string
        )

    def test_load_var_name(self):
        # mocking json file for testing
        with patch("builtins.open", mock_open(read_data=self.json_data)):
            self.assertEqual(
                json_tools.load_var_name("test.json", "momentum"), "Complex_p"
            )

    def test_load_quality_cuts(self):
        # mocking json file for testing
        with patch("builtins.open", mock_open(read_data=self.json_data)):
            quality_cuts = json_tools.load_quality_cuts("test.json")
            expected_cuts = ["-1.0 <= Complex_mass2 < 2.0", "0.0 <= Complex_pT < 2.0"]
            self.assertEqual(quality_cuts, expected_cuts)

    def test_load_features_for_train(self):
        target_features = ["Complex_mass2", "Complex_p"]
        with patch("builtins.open", mock_open(read_data=self.json_data)):
            read_features = json_tools.load_features_for_train("test.json")
            self.assertEqual(target_features, read_features)

    def test_load_hyper_params_vals(self):
        target_vals = (596, 5, 0.07161792803939408)
        with patch("builtins.open", mock_open(read_data=self.json_data)):
            read_vals = json_tools.load_hyper_params_vals("test.json")
            self.assertEqual(target_vals, read_vals)

    def test_load_file_name(self):
            target_name = "training_filename.test"
            with patch("builtins.open", mock_open(read_data=self.json_data)):
                read_name = json_tools.load_file_name("test.json", training_or_test="training")
                self.assertEqual(target_name, read_name)
    if __name__ == "__main__":
        unittest.main()
