import unittest
from unittest.mock import mock_open, patch
import pandas as pd

from validate_model import ValidateModel
from particles_id import ParticlesId as Pid


class TestValidateModel(unittest.TestCase):
    def setUp(self):
        first_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": 2212,
            "Complex_mass2": 1.1,
            "model_output_0": 0.8,
            "model_output_1": 0.1,
            "model_output_2": 0.1,
        }
        second_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": 2212,
            "Complex_mass2": 0.5,
            "model_output_0": 0.6,
            "model_output_1": 0.2,
            "model_output_2": 0.2,
        }
        proton_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": Pid.PROTON.value,
            "Complex_mass2": 0.8,
            "model_output_0": 0.8,
            "model_output_1": 0.1,
            "model_output_2": 0.1,
        }
        kaon_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": Pid.POS_KAON.value,
            "Complex_mass2": 0.4,
            "model_output_0": 0.4,
            "model_output_1": 0.1,
            "model_output_2": 0.1,
        }
        pion_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": Pid.POS_PION.value,
            "Complex_mass2": 0.2,
            "model_output_0": 0.1,
            "model_output_1": 0.1,
            "model_output_2": 0.1,
        }
        complete_data = [
            first_entry,
            second_entry,
            proton_entry,
            kaon_entry,
            pion_entry,
        ]
        json_data = """{"var_names": {"momentum": "Complex_p","charge": "Complex_q","mass2": "Complex_mass2","pid": "Complex_pid"}}"""
        with patch("builtins.open", mock_open(read_data=json_data)):
            self.validate = ValidateModel(
                2, 4, False, "test.json", pd.DataFrame(complete_data)
            )

    def test_get_n_classes(self):
        self.assertEqual(self.validate.get_n_classes(), 4)

    def test_xgb_preds(self):
        self.validate.xgb_preds(0.7, 0.7, 0.7)
        df = self.validate.particles_df
        self.assertEqual(df[df["Complex_mass2"] == 1.1]["xgb_preds"].item(), 0)
        self.assertEqual(df[df["Complex_mass2"] == 0.5]["xgb_preds"].item(), 3)

    def test_remap_names(self):
        self.validate.remap_names()
        df = self.validate.particles_df
        self.assertEqual(df[df["Complex_mass2"] == 0.2]["Complex_pid"].item(), 2)
        self.assertEqual(df[df["Complex_mass2"] == 0.4]["Complex_pid"].item(), 1)
        self.assertEqual(df[df["Complex_mass2"] == 0.8]["Complex_pid"].item(), 0)

    def test_save_df(self):
        self.validate.save_df()

    def test_parse_model_name(self):
        model_name_positive = "model_0.0_6.0_positive"
        lower_p, upper_p, anti = ValidateModel.parse_model_name(model_name_positive)
        self.assertEqual([lower_p, upper_p, anti], [0.0, 6.0, False])
        model_name_anti = "model_3.0_6.0_anti"
        lower_p, upper_p, anti = ValidateModel.parse_model_name(model_name_anti)
        self.assertEqual([lower_p, upper_p, anti], [3.0, 6.0, True])
        model_name_incorrect = "model_anti_1_4"
        self.assertRaises(
            ValueError, lambda: ValidateModel.parse_model_name(model_name_incorrect)
        )
