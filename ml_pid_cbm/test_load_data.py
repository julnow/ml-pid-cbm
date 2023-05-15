import unittest
from unittest.mock import mock_open, patch

import pandas as pd
from hipe4ml.tree_handler import TreeHandler

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
        self.plain_tree = "test_plain_tree.root"

    def test_clean_tree(self):
        # manually created entries to test data cleanin
        # mock json file for testing
        json_data = """{"var_names":{"momentum": "Complex_p","charge": "Complex_q"},
        "cuts":{"Complex_pT": {"lower": 0,"upper": 2.0},"Complex_eta": {"lower": 0.0,"upper": 6.0}}}"""
        with patch("builtins.open", mock_open(read_data=json_data)):
            # only positive particles
            test_str_pos = self.loader_pos.clean_tree()
            expected_str_pos = "(0.0 <= Complex_pT < 2.0) and (0.0 <= Complex_eta < 6.0) and (0.0 <= Complex_p < 3.0) and (Complex_q > 0)"
            test_str_neg = self.loader_anti.clean_tree()
            expected_str_neg = "(0.0 <= Complex_pT < 2.0) and (0.0 <= Complex_eta < 6.0) and (0.0 <= Complex_p < 3.0) and (Complex_q < 0)"
            self.assertEqual(test_str_pos, expected_str_pos)
            self.assertEqual(test_str_neg, expected_str_neg)

    def test_get_particles_type(self):
        proton_entry = {
            "Complex_q": 1,
            "Complex_p": 1,
            "Complex_pid": 2212,
            "Complex_mass2": 0.8,
        }
        proton_entry_specific_p = {
            "Complex_q": 1,
            "Complex_p": 2,
            "Complex_pid": 2212,
            "Complex_mass2": 0.79,
        }
        proton_entry_outside_1sigma = {
            "Complex_q": 1,
            "Complex_p": 1,
            "Complex_pid": 2212,
            "Complex_mass2": 3.8,
        }
        pion_entry = {
            "Complex_q": 1,
            "Complex_p": 1.2,
            "Complex_pid": 211,
            "Complex_mass2": 0.2,
        }
        complete_data = [
            proton_entry,
            proton_entry,
            proton_entry,
            proton_entry,
            proton_entry,
            proton_entry_specific_p,
            proton_entry_outside_1sigma,
            pion_entry,
        ]
        # mock json file for testing
        json_data = """{"var_names":{"momentum": "Complex_p","charge": "Complex_q", "mass2": "Complex_mass2", "pid": "Complex_pid"}}"""
        tree_handler = TreeHandler()
        tree_handler.set_data_frame(pd.DataFrame(complete_data))
        with patch("builtins.open", mock_open(read_data=json_data)):
            protons = self.loader_pos.get_particles_type(
                tree_handler, 2212, 1
            ).get_data_frame()
            # should find proton inside 1sigma region
            self.assertEqual(
                protons[
                    protons["Complex_mass2"] == proton_entry_specific_p["Complex_mass2"]
                ]["Complex_p"].iloc[0],
                proton_entry_specific_p["Complex_p"],
            )
            # should filter out proton outside 1sigma region
            pd.testing.assert_frame_equal(
                protons[
                    protons["Complex_mass2"]
                    == proton_entry_outside_1sigma["Complex_mass2"]
                ],
                protons.drop(protons.index),
            )
            # should filter out pions
            pd.testing.assert_frame_equal(
                protons[protons["Complex_mass2"] == pion_entry["Complex_mass2"]],
                protons.drop(protons.index),
            )

    def test_get_protons_kaons_pions(self):
        proton_entry = {
            "Complex_q": 1.0,
            "Complex_p": 1.0,
            "Complex_pid": 2212.0,
            "Complex_mass2": 0.8,
        }
        kaon_entry = {
            "Complex_q": 1.0,
            "Complex_p": 1.2,
            "Complex_pid": 321.0,
            "Complex_mass2": 0.4,
        }
        pion_entry = {
            "Complex_q": 1.0,
            "Complex_p": 1.2,
            "Complex_pid": 211.0,
            "Complex_mass2": 0.2,
        }
        anti_proton_entry = {
            "Complex_q": -1.0,
            "Complex_p": 1.0,
            "Complex_pid": -2212.0,
            "Complex_mass2": 0.8,
        }
        anti_kaon_entry = {
            "Complex_q": -1.0,
            "Complex_p": 1.2,
            "Complex_pid": -321.0,
            "Complex_mass2": 0.4,
        }
        anti_pion_entry = {
            "Complex_q": -1.0,
            "Complex_p": 1.2,
            "Complex_pid": -211.0,
            "Complex_mass2": 0.2,
        }
        complete_data = [
            proton_entry,
            kaon_entry,
            pion_entry,
            anti_proton_entry,
            anti_kaon_entry,
            anti_pion_entry,
        ]
        # mock json file for testing
        json_data = """{"var_names":{"momentum": "Complex_p", "charge": "Complex_q", "mass2": "Complex_mass2", "pid": "Complex_pid"}}"""
        tree_handler = TreeHandler()
        tree_handler.set_data_frame(pd.DataFrame(complete_data))
        with patch("builtins.open", mock_open(read_data=json_data)):
            protons, kaons, pions = self.loader_pos.get_protons_kaons_pions(
                tree_handler
            )
            # check if each particle type was loaded correctly
            pd.testing.assert_frame_equal(
                protons.get_data_frame().reset_index(drop=True),
                pd.DataFrame([proton_entry]).reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                kaons.get_data_frame().reset_index(drop=True),
                pd.DataFrame([kaon_entry]).reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                pions.get_data_frame().reset_index(drop=True),
                pd.DataFrame([pion_entry]).reset_index(drop=True),
            )
            # repeat for antiparticles
            (
                anti_protons,
                anti_kaons,
                anti_pions,
            ) = self.loader_anti.get_protons_kaons_pions(tree_handler)
            # check if each particle type was loaded correctly
            pd.testing.assert_frame_equal(
                anti_protons.get_data_frame().reset_index(drop=True),
                pd.DataFrame([anti_proton_entry]).reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                anti_kaons.get_data_frame().reset_index(drop=True),
                pd.DataFrame([anti_kaon_entry]).reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                anti_pions.get_data_frame().reset_index(drop=True),
                pd.DataFrame([anti_pion_entry]).reset_index(drop=True),
            )

    def test_load_tree(self):
        with patch.object(LoadData, "clean_tree", return_value=''):
            self.loader_pos.load_tree(self.plain_tree)
