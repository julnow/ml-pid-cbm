import unittest
from unittest.mock import patch, mock_open
from hipe4ml.tree_handler import TreeHandler
import pandas as pd
from load_data import LoadData


class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.loader_pos = LoadData(
            data_file_name="data.tree",
            json_file_name="config.json",
            lower_p_cut=0.0,
            upper_p_cut=3.0,
            anti_particles=False,
        )
        self.loader_anti = LoadData(
            data_file_name="data.tree",
            json_file_name="config.json",
            lower_p_cut=0.0,
            upper_p_cut=3.0,
            anti_particles=True,
        )
        self.loader_empty = LoadData(
            data_file_name=None,
            json_file_name=None,
            lower_p_cut=0.0,
            upper_p_cut=3.0,
            anti_particles=True,
        )

    def test_create_cut_string(self):
        # ideally formatted
        expected_string = "0.1 < test_cut < 13.0"
        self.assertEqual(
            LoadData.create_cut_string(0.1, 13.0, "test_cut"), expected_string
        )
        # not ideally formatted
        self.assertEqual(
            LoadData.create_cut_string(0.141, 13, "test_cut"), expected_string
        )

    def test_load_var_name(self):
        json_data = """{"var_names":{"momentum": "Complex_p","charge": "Complex_q"}}"""
        # mocking json file for testing
        with patch("builtins.open", mock_open(read_data=json_data)):
            self.assertEqual(self.loader_pos.load_var_name("test.json", "momentum"), "Complex_p")

    def test_load_quality_cuts(self):
        json_data = """{"cuts":{"Complex_mass2": {"lower": -1.0,"upper": 2.0},"Complex_pT": {"lower": 0.0,"upper": 2.0}}}"""
        # mocking json file for testing
        with patch("builtins.open", mock_open(read_data=json_data)):
            quality_cuts = self.loader_pos.load_quality_cuts("test.json")
            expected_cuts = ["-1.0 < Complex_mass2 < 2.0", "0.0 < Complex_pT < 2.0"]
            self.assertEqual(quality_cuts, expected_cuts)

    def test_clean_tree(self):
        # manually created entries to test data cleaning
        positive_entry = {"Complex_q": 1, "Complex_p": 2, "Complex_eta": 3}
        incorrect_eta_entry = {"Complex_q": 1, "Complex_p": 3, "Complex_eta": 9}
        negative_entry = {"Complex_q": -1, "Complex_p": 1, "Complex_eta": 2}
        incorrect_p_entry = {"Complex_q": -1, "Complex_p": 30, "Complex_eta": 2}
        incorrect_type_entry = {"Complex_q": 1, "Complex_p": 30, "Complex_eta": None}
        complete_data = [
            positive_entry,
            incorrect_eta_entry,
            negative_entry,
            incorrect_p_entry,
            incorrect_type_entry,
        ]
        # mock json file for testing
        json_data = """{"var_names":{"momentum": "Complex_p","charge": "Complex_q"},
        "cuts":{"Complex_p": {"lower": 0,"upper": 3.0},"Complex_eta": {"lower": 0.0,"upper": 6.0}}}"""
        tree_handler = TreeHandler()
        tree_handler.set_data_frame(pd.DataFrame(complete_data))
        with patch("builtins.open", mock_open(read_data=json_data)):
            # positive particles
            positive_tree_handler = self.loader_pos.clean_tree(tree_handler)
            pd.testing.assert_frame_equal(
                positive_tree_handler.get_data_frame().reset_index(drop=True),
                pd.DataFrame([positive_entry]).reset_index(drop=True),
                check_dtype=False,
            )
            # negative (anti) particles
            negative_tree_handler = self.loader_anti.clean_tree(tree_handler)
            pd.testing.assert_frame_equal(
                negative_tree_handler.get_data_frame().reset_index(drop=True),
                pd.DataFrame([negative_entry]).reset_index(drop=True),
                check_dtype=False,
            )

    if __name__ == "__main__":
        unittest.main()
