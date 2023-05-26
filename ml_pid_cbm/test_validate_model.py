import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd

from .tools.particles_id import ParticlesId as Pid
from .validate_model import ValidateModel


class TestValidateModel(unittest.TestCase):
    def setUp(self):
        first_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": 2212,
            "Complex_mass2": 1.1,
            "Complex_pT": 0.5,
            "Complex_rapidity": 3.0,
            "model_output_0": 0.8,
            "model_output_1": 0.1,
            "model_output_2": 0.1,
        }
        second_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": 2212,
            "Complex_mass2": 0.5,
            "Complex_pT": 0.5,
            "Complex_rapidity": 3.0,
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
            "Complex_pT": 0.5,
            "Complex_rapidity": 3.0,
            "model_output_0": 0.1,
            "model_output_1": 0.8,
            "model_output_2": 0.1,
        }
        pion_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": Pid.POS_PION.value,
            "Complex_mass2": 0.2,
            "Complex_pT": 0.5,
            "Complex_rapidity": 3.0,
            "model_output_0": 0.1,
            "model_output_1": 0.1,
            "model_output_2": 0.8,
        }
        bckgr_entry = {
            "Complex_q": 1.0,
            "Complex_p": 2.0,
            "Complex_pid": 1010,
            "Complex_mass2": 0.1,
            "Complex_pT": 0.5,
            "Complex_rapidity": 3.0,
            "model_output_0": 0.3,
            "model_output_1": 0.4,
            "model_output_2": 0.3,
        }
        complete_data = [
            first_entry,
            second_entry,
            proton_entry,
            {**proton_entry, "Complex_mass2": 0.9},
            kaon_entry,
            {**kaon_entry, "Complex_mass2": 0.6},
            pion_entry,
            {**pion_entry, "Complex_mass2": 0.3},
            bckgr_entry,
            {**bckgr_entry, "Complex_mass2": 0.0},
        ]
        self.json_data = """{"var_names": {"momentum": "Complex_p","charge": "Complex_q","mass2": "Complex_mass2","pid": "Complex_pid"},
                    "vars_to_draw": ["Complex_mass2", "Complex_p"]}"""
        test_config_path = f"{Path(__file__).resolve().parent}/test_config.json"
        with patch("builtins.open", mock_open(read_data=self.json_data)):
            self.validate = ValidateModel(
                2, 4, False, test_config_path, pd.DataFrame(complete_data)
            )
            self.validate_false = ValidateModel(
                2, 4, True, test_config_path, pd.DataFrame(complete_data)
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
        self.validate_false.remap_names()

    def test_save_df(self):
        self.validate.save_df()

    def test_generate_plots(self):
        self.validate.xgb_preds(0.7, 0.7, 0.7)
        # with patch("builtins.open", mock_open(read_data=self.json_data)):
        # should be mock json but
        # https://github.com/julnow/ml-pid-cbm/actions/runs/5004465285/jobs/9029522575
        self.validate.generate_plots()

    def test_evaluate_probas(self):
        self.validate.evaluate_probas(0.1, 0.9, 5, 50)

    def test_confusion_matrix_and_stats(self):
        self.validate.xgb_preds(0.7, 0.7, 0.7)
        self.validate.confusion_matrix_and_stats()

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

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).resolve().parent
        cls.img_dir = f"{cls.test_dir}/testimg"
        if not os.path.exists(cls.img_dir):
            os.makedirs(cls.img_dir)
        os.chdir(cls.img_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.test_dir)
        shutil.rmtree(cls.img_dir)
