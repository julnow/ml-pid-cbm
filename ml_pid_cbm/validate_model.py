import os
import re
import io
import argparse
from typing import Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from hipe4ml.model_handler import ModelHandler
from load_data import LoadData
from particles_id import ParticlesId as Pid
import plotting_tools


class ValidateModel:
    """
    Class for testing the ml model
    """

    def __init__(
        self,
        lower_p_cut: float,
        upper_p_cut: float,
        anti_particles: bool,
        json_file_name: str,
        particles_df: pd.DataFrame,
    ):
        self.lower_p_cut = lower_p_cut
        self.upper_p_cut = upper_p_cut
        self.anti_particles = anti_particles
        self.json_file_name = json_file_name
        self.particles_df = particles_df

    def xgb_preds(self, proba_proton: float, proba_pion: float):
        """Gets particle type as selected by xgboost model if above probability threshold.

        Args:
            proba_proton (float): Probablity threshold to classify particle as proton.
            proba_pion (float): Probablity threshold to classify particle as pion.
        """
        df = self.particles_df
        df["xgb_preds"] = (
            df[["model_output_0", "model_output_1"]]
            .idxmax(axis=1)
            .map(lambda x: x.lstrip("model_output_"))
            .astype(int)
        )
        # setting to bckgr if smaller than probability threshold
        proton = (df["xgb_preds"] == 0) & (df["model_output_0"] > proba_proton)
        pion = (df["xgb_preds"] == 1) & (df["model_output_1"] > proba_pion)
        df.loc[~(proton | pion ), "xgb_preds"] = 3

        self.particles_df = df

    def remap_names(self):
        """
        Remaps Pid of particles to output format from XGBoost Model.
        Protons: 0; Kaons: 1; Pions, Electrons, Muons: 2; Other: 3

        """
        pid_variable_name = LoadData.load_var_name(self.json_file_name, "pid")
        df = self.particles_df
        if self.anti_particles:
            df[pid_variable_name] = (
                df[pid_variable_name]
                .map(
                    defaultdict(
                        lambda: 2.0,
                        {
                            Pid.ANTI_PROTON.value: 0.0,
                            Pid.NEG_PION.value: 1.0,
                            Pid.ELECTRON.value: 1.0,
                            Pid.NEG_MUON.value: 1.0,
                        },
                    ),
                    na_action="ignore",
                )
                .astype(float)
            )
        else:
            df[pid_variable_name] = (
                df[pid_variable_name]
                .map(
                    defaultdict(
                        lambda: 2.0,
                        {
                            Pid.PROTON.value: 0.0,
                            Pid.POS_PION.value: 1.0,
                            Pid.POSITRON.value: 1.0,
                            Pid.POS_MUON.value: 1.0,
                        },
                    ),
                    na_action="ignore",
                )
                .astype(float)
            )
        self.particles_df = df

    def save_df(self):
        """
        Saves dataframe with validated data into pickle format.
        """
        self.particles_df.to_pickle("validated_data.pickle")

    def sigma_selection(self, pid: float, nsigma: float = 5, info: bool = False):
        """Sigma selection for dataframe to remove systmatically (not by the ML model) mismatched particles.

        Args:
            pid (float): Pid of particle for this selection
            nsigma (float, optional): _description_. Defaults to 5.
            info (bool, optional): _description_. Defaults to False.
        """
        df = self.particles_df
        pid_variable_name = LoadData.load_var_name(json_file_name, "pid")
        # for selected pid
        mass2_variable_name = LoadData.load_var_name(json_file_name, "mass2")
        mean = df[df[pid_variable_name] == pid][mass2_variable_name].mean()
        std = df[df[pid_variable_name] == pid][mass2_variable_name].std()
        outside_sigma = (df[pid_variable_name] == pid) & (
            (df[mass2_variable_name] < (mean - nsigma * std))
            | (df[mass2_variable_name] > (mean + nsigma * std))
        )
        df_sigma_selected = df[~outside_sigma]
        if info:
            df_len = len(df)
            df1_len = len(df_sigma_selected)
            print(
                "we get rid of "
                + str(round((df_len - df1_len) / df_len * 100, 2))
                + " % of pid = "
                + str(pid)
                + " particle entries"
            )
        self.particles_df = df_sigma_selected

    def efficiency_stats(
        self,
        cm: np.ndarray,
        pid: float,
        pid_variable_name: str,
        txt_tile: io.TextIOWrapper,
        dataframe: pd.DataFrame = None,
    ):
        """
        Prints efficiency stats from confusion matrix into efficiency_stats.txt file and stdout.
        Efficiency is calculated as correctly identified X / all true simulated X
        Purity is calulated as correctly identified X / all identified X

        Args:
            cm (np.ndarray): Confusion matrix  generetated by sklearn.metrics.confusion_matrix.
            pid (float): Pid of particles to print efficiency stats.
            pid_variable_name (str): Variable name of pid in input tree.
            df (pd.DataFrame): Dataframe with all variables. Defaults to None.
        """
        df = dataframe or self.particles_df
        all_simulated_signal = len(df.loc[df[pid_variable_name] == pid])
        true_signal = cm[pid][pid]
        false_signal = 0
        for i, row in enumerate(cm):
            if i != pid:
                false_signal += row[pid] + cm[pid][i]
        reconstructed_signals = true_signal + false_signal
        efficiency = (
            true_signal / all_simulated_signal * 100
        )  # efficency calculated as true signal/all signal
        purity = (
            true_signal / reconstructed_signals * 100
        )  # purity calculated as true signal/reconstructed_signals
        stats = f"""
        For particle ID = {pid}: 
        Efficiency: {efficiency:.2f}%
        Purity: {purity:.2f}%
        """
        print(stats)
        txt_tile.writelines(stats)

    @staticmethod
    def parse_model_name(
        name: str,
        pattern: str = r"model_([\d.]+)_([\d.]+)_(anti)|model_([\d.]+)_([\d.]+)_([a-zA-Z]+)",
    ) -> Tuple[float, float, bool]:
        """Parser model name to get info about lower momentum cut, upper momentum cut, and if model is trained for anti_particles.

        Args:
            name (str): Name of the model.
            pattern (_type_, optional): Pattern of model name.
             Defaults to r"model_([\d.]+)_([\d.]+)_(anti)|model_([\d.]+)_([\d.]+)_([a-zA-Z]+)".

        Raises:
            ValueError: Raises error if model name incorrect.

        Returns:
            Tuple[float, float, bool]: Tuple containing lower_p_cut, upper_p_cut, is_anti
        """
        match = re.match(pattern, name)
        if match:
            if match.group(3):
                lower_p_cut = float(match.group(1))
                upper_p_cut = float(match.group(2))
                is_anti = True
            else:
                lower_p_cut = float(match.group(4))
                upper_p_cut = float(match.group(5))
                is_anti = False
        else:
            raise ValueError("Incorrect model name, regex not found.")
        return (lower_p_cut, upper_p_cut, is_anti)


if __name__ == "__main__":
    # parser for main class
    parser = argparse.ArgumentParser(
        prog="ML_PID_CBM ValidateModel",
        description="Program for validating PID ML models",
    )
    parser.add_argument(
        "--config",
        "-c",
        nargs=1,
        required=True,
        type=str,
        help="Filename of path of config json file.",
    )
    parser.add_argument(
        "--modelname",
        "-m",
        nargs=1,
        required=True,
        type=str,
        help="Name of folder containing trained ml model.",
    )
    parser.add_argument(
        "--probabilitycuts",
        "-p",
        nargs=2,
        required=True,
        type=float,
        help="Probability cut value for respectively protons, and pions. E.g., 0.9 0.8",
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=1,
        help="Max number of workers for ThreadPoolExecutor which reads Root tree with data.",
    )
    args = parser.parse_args()
    # config  arguments to be loaded from args
    json_file_name = args.config[0]
    model_name = args.modelname[0]
    proba_proton, proba_pion = (
        args.probabilitycuts[0],
        args.probabilitycuts[1],
    )
    n_workers = args.nworkers
    lower_p, upper_p, is_anti = ValidateModel.parse_model_name(model_name)
    # loading test data
    data_file_name = LoadData.load_file_name(json_file_name, "test")

    loader = LoadData(data_file_name, json_file_name, lower_p, upper_p, is_anti)
    # sigma selection
    # loading model handler and applying on dataset
    print(
        f"\nLoading data from {data_file_name}\nApplying model handler from {model_name}"
    )
    os.chdir(f"{model_name}")
    model_hdl = ModelHandler()
    model_hdl.load_model_handler(model_name)
    test_particles = loader.load_tree(model_handler=model_hdl, max_workers=n_workers)
    # validate model object
    validate = ValidateModel(
        lower_p, upper_p, is_anti, json_file_name, test_particles.get_data_frame()
    )
    # apply probabilty cuts
    print(
        f"\nApplying probability cuts.\nFor protons: {proba_proton}\nFor pions: {proba_pion}"
    )
    validate.xgb_preds(proba_proton, proba_pion)
    # remap Pid to match output XGBoost format
    validate.remap_names()
    # sigma selection for each particle type
    for pid in range(0, 2):
        validate.sigma_selection(pid, 4)
    # graphs
    # confusion matrix
    pid_variable_name = LoadData.load_var_name(json_file_name, "pid")
    cnf_matrix = confusion_matrix(
        validate.particles_df[pid_variable_name], validate.particles_df["xgb_preds"]
    )
    plotting_tools.plot_confusion_matrix(cnf_matrix)
    plotting_tools.plot_confusion_matrix(cnf_matrix, normalize=True)
    # confusion matrix statistic
    txt_file = open("efficiency_stats.txt", "w+")
    for pid in range(0, 3):
        validate.efficiency_stats(cnf_matrix, pid, pid_variable_name, txt_file)
    txt_file.close()
    # tof plots
    print("Generating plots.")
    # simulated:
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df[pid_variable_name] == 0],
        json_file_name,
        "protons (all simulated)",
    )
    
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df[pid_variable_name] == 1],
        json_file_name,
        "pions, muons, electrons (all simulated)",
    )
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df[pid_variable_name] == 2],
        json_file_name,
        "bckgr (all simulated)",
    )
    # xgb selected
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df["xgb_preds"] == 0],
        json_file_name,
        "protons (XGB-selected)",
    )
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df["xgb_preds"] == 1],
        json_file_name,
        "pions, muons, electrons (XGB-selected)",
    )
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df["xgb_preds"] == 2],
        json_file_name,
        "bckgr (XGB-selected)"
    )
    # mass2 plots
    mass2_variable_name = LoadData.load_var_name(json_file_name, "mass2")
    plotting_tools.plot_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 0][
            mass2_variable_name
        ],
        validate.particles_df[validate.particles_df[pid_variable_name] == 0][
            mass2_variable_name
        ],
        "Protons",
        (-0.1, 1.5),
    )
    plotting_tools.plot_all_particles_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 0],
        mass2_variable_name,
        pid_variable_name,
        "Protons",
        (-0.1, 1.5),
    )
    plotting_tools.plot_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 1][
            mass2_variable_name
        ],
        validate.particles_df[validate.particles_df[pid_variable_name] == 1][
            mass2_variable_name
        ],
        "Pions (& electrons, muons)",
        (-0.15, 0.15),
    )
    plotting_tools.plot_all_particles_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 1],
        mass2_variable_name,
        pid_variable_name,
        "Pions (& electrons, muons)",
        (-0.15, 0.15),
    )
    plotting_tools.plot_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 2][
            mass2_variable_name
        ],
        validate.particles_df[validate.particles_df[pid_variable_name] == 2][
            mass2_variable_name
        ],
        "Background",
        (-0.15, 0.15),
    )
    plotting_tools.plot_all_particles_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 2],
        mass2_variable_name,
        pid_variable_name,
        "Background",
        (-0.15, 0.15),
    )
    # pt-rapidity plots
    for i in range(2):
        plotting_tools.plot_eff_pT_rap(validate.particles_df, i)
        plotting_tools.plot_pt_rapidity(validate.particles_df, i)
    # save df
    validate.save_df()
