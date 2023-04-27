import argparse
import io
import os
import re
import sys
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from hipe4ml.model_handler import ModelHandler
from sklearn.metrics import confusion_matrix

import plotting_tools
from load_data import LoadData
from particles_id import ParticlesId as Pid
from prepare_model import PrepareModel


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

    def xgb_preds(self, proba_proton: float, proba_kaon: float, proba_pion: float):
        """Gets particle type as selected by xgboost model if above probability threshold.

        Args:
            proba_proton (float): Probablity threshold to classify particle as proton.
            proba_kaon (float): Probablity threshold to classify particle as kaon.
            proba_pion (float): Probablity threshold to classify particle as pion.
        """
        df = self.particles_df
        df["xgb_preds"] = (
            df[["model_output_0", "model_output_1", "model_output_2"]]
            .idxmax(axis=1)
            .map(lambda x: x.lstrip("model_output_"))
            .astype(int)
        )
        # setting to bckgr if smaller than probability threshold
        proton = (df["xgb_preds"] == 0) & (df["model_output_0"] > proba_proton)
        pion = (df["xgb_preds"] == 1) & (df["model_output_1"] > proba_kaon)
        kaon = (df["xgb_preds"] == 2) & (df["model_output_2"] > proba_pion)
        df.loc[~(proton | pion | kaon), "xgb_preds"] = 3

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
                        lambda: 3.0,
                        {
                            Pid.ANTI_PROTON.value: 0.0,
                            Pid.NEG_KAON.value: 1.0,
                            Pid.NEG_PION.value: 2.0,
                            Pid.ELECTRON.value: 2.0,
                            Pid.NEG_MUON.value: 2.0,
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
                        lambda: 3.0,
                        {
                            Pid.PROTON.value: 0.0,
                            Pid.POS_KAON.value: 1.0,
                            Pid.POS_PION.value: 2.0,
                            Pid.POSITRON.value: 2.0,
                            Pid.POS_MUON.value: 2.0,
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
        txt_tile: io.TextIOWrapper = None,
        dataframe: pd.DataFrame = None,
        print_output: bool = True,
    ) -> Tuple[float, float]:
        """
        Prints efficiency stats from confusion matrix into efficiency_stats.txt file and stdout.
        Efficiency is calculated as correctly identified X / all true simulated X
        Purity is calulated as correctly identified X / all identified X

        Args:
            cm (np.ndarray): Confusion matrix  generetated by sklearn.metrics.confusion_matrix.
            pid (float): Pid of particles to print efficiency stats.
            pid_variable_name (str): Variable name of pid in input tree.
            df (pd.DataFrame): Dataframe with all variables. Defaults to None.

        Returns:
            Tuple[float, float]: Tuple with efficiency and purity
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
        if print_output:
            print(stats)
        if txt_tile is not None:
            txt_tile.writelines(stats)
        return (efficiency, purity)

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

    def evaluate_probas(
        self,
        start: float,
        stop: float,
        n_steps: int,
        pid_variable_name: float,
        save_fig: bool = True,
    ):
        print(
            f"Checking efficiency and purity for {int(n_steps)} probablity cuts between {start}, and {stop}..."
        )
        probas = np.linspace(start, stop, n_steps)
        efficienciess_protons, efficiencies_kaons, efficiencies_pions = [], [], []
        efficiencies = [efficienciess_protons, efficiencies_kaons, efficiencies_pions]
        purities_protons, purities_kaons, purities_pions = [], [], []
        purities = [purities_protons, purities_kaons, purities_pions]
        for proba in probas:
            validate.xgb_preds(proba, proba, proba)
            # confusion matrix
            cnf_matrix = confusion_matrix(
                self.particles_df[pid_variable_name],
                self.particles_df["xgb_preds"],
            )
            for pid in range(0, 3):
                efficiency, purity = validate.efficiency_stats(
                    cnf_matrix, pid, pid_variable_name, print_output=False
                )
                efficiencies[pid].append(efficiency)
                purities[pid].append(purity)
        plotting_tools.plot_efficiency_purity(probas, efficiencies, purities, save_fig)
        if save_fig:
            print("Plots ready!")


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Arguments parser for the main method.

    Args:
        args (List[str]): Arguments from the command line, should be sys.argv[1:].

    Returns:
        argparse.Namespace: argparse.Namespace containg args
    """
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
    proba_group = parser.add_mutually_exclusive_group(required=True)
    proba_group.add_argument(
        "--probabilitycuts",
        "-p",
        nargs=3,
        type=float,
        help="Probability cut value for respectively protons, kaons, and pions. E.g., 0.9 0.95 0.9",
    )
    proba_group.add_argument(
        "--evaluateproba",
        "-e",
        nargs=3,
        type=float,
        help="Minimal probability cut, maximal, and number of steps to investigate.",
    )
    parser.add_argument(
        "--nworkers",
        "-n",
        type=int,
        default=1,
        help="Max number of workers for ThreadPoolExecutor which reads Root tree with data.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode allows selection of probability cuts after evaluating them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # parser for main class
    args = parse_args(sys.argv[1:])
    # config  arguments to be loaded from args
    json_file_name = args.config[0]
    model_name = args.modelname[0]
    if args.probabilitycuts is not None:
        proba_proton, proba_kaon, proba_pion = (
            args.probabilitycuts[0],
            args.probabilitycuts[1],
            args.probabilitycuts[2],
        )
    else:
        proba_proton, proba_kaon, proba_pion = -1.0, -1.0, -1.0
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
    # temporary solution for dealing with v_tof = l/t
    #################################################
    test_particles = loader.load_tree(max_workers=n_workers)
    df_with_v_tof = test_particles.get_data_frame()
    df_with_v_tof["Complex_v_tof"] = df_with_v_tof.eval("Complex_l / Complex_t")
    test_particles.set_data_frame(df_with_v_tof)
    test_particles.apply_model_handler(model_hdl)
    ####################################################
    # test_particles = loader.load_tree(model_handler=model_hdl, max_workers=n_workers)
    # validate model object
    validate = ValidateModel(
        lower_p, upper_p, is_anti, json_file_name, test_particles.get_data_frame()
    )
    # remap Pid to match output XGBoost format
    validate.remap_names()
    pid_variable_name = LoadData.load_var_name(json_file_name, "pid")

    # # sigma selection for each particle type
    # for pid in range(0, 3):
    #     validate.sigma_selection(pid, 4)
    if args.evaluateproba is not None:
        validate.evaluate_probas(
            args.evaluateproba[0],
            args.evaluateproba[1],
            int(args.evaluateproba[2]),
            pid_variable_name,
            not args.interactive,
        )
        if args.interactive:
            while proba_proton < 0 or proba_proton > 1:
                proba_proton = float(
                    input(
                        "Enter the probability threshold for proton (between 0 and 1): "
                    )
                )

            while proba_kaon < 0 or proba_kaon > 1:
                proba_kaon = float(
                    input(
                        "Enter the probability threshold for kaon (between 0 and 1): "
                    )
                )

            while proba_pion < 0 or proba_pion > 1:
                proba_pion = float(
                    input(
                        "Enter the probability threshold for pion (between 0 and 1): "
                    )
                )
        else:
            sys.exit(0)
    # if probabilites are set
    # apply probabilty cuts
    print(
        f"\nApplying probability cuts.\nFor protons: {proba_proton}\nFor kaons: {proba_kaon}\nFor pions: {proba_pion}"
    )
    validate.xgb_preds(proba_proton, proba_kaon, proba_pion)
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
        "kaons (all simulated)",
    )
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df[pid_variable_name] == 2],
        json_file_name,
        "pions, muons, electrons (all simulated)",
    )
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df[pid_variable_name] == 3],
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
        "kaons (XGB-selected)",
    )
    plotting_tools.tof_plot(
        validate.particles_df[validate.particles_df["xgb_preds"] == 2],
        json_file_name,
        "pions, muons, electrons (XGB-selected)",
    )
    # plotting_tools.tof_plot(
    #     validate.particles_df[validate.particles_df["xgb_preds"] == 3],
    #     json_file_name,
    #     "bckgr (XGB-selected)"
    # )
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
        "Kaons",
        (-0.1, 0.4),
    )
    plotting_tools.plot_all_particles_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 1],
        mass2_variable_name,
        pid_variable_name,
        "Kaons",
        (-0.1, 0.4),
    )
    plotting_tools.plot_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 2][
            mass2_variable_name
        ],
        validate.particles_df[validate.particles_df[pid_variable_name] == 2][
            mass2_variable_name
        ],
        "Pions (& electrons, muons)",
        (-0.15, 0.15),
    )
    plotting_tools.plot_all_particles_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 2],
        mass2_variable_name,
        pid_variable_name,
        "Pions (& electrons, muons)",
        (-0.15, 0.15),
    )
    plotting_tools.plot_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 3][
            mass2_variable_name
        ],
        validate.particles_df[validate.particles_df[pid_variable_name] == 3][
            mass2_variable_name
        ],
        "Background",
        (-0.15, 0.15),
    )
    plotting_tools.plot_all_particles_mass2(
        validate.particles_df[validate.particles_df["xgb_preds"] == 3],
        mass2_variable_name,
        pid_variable_name,
        "Background",
        (-0.15, 0.15),
    )
    # pt-rapidity plots and for each rapidity
    features_for_train = PrepareModel.load_features_for_train(json_file_name)
    for pid in range(3):
        plotting_tools.plot_before_after_variables(
            validate.particles_df, pid, pid_variable_name, features_for_train
        )
        plotting_tools.plot_eff_pT_rap(validate.particles_df, pid)
        plotting_tools.plot_pt_rapidity(validate.particles_df, pid)
    # save validated dataset
    validate.save_df()
