import argparse
import os
import sys
from collections import defaultdict
from shutil import copy2
from typing import List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from ml_pid_cbm.tools import json_tools, plotting_tools
from ml_pid_cbm.tools.load_data import LoadData
from ml_pid_cbm.tools.particles_id import ParticlesId as Pid
from ml_pid_cbm.validate_model import ValidateModel


class ValidateGauss(ValidateModel):
    """
    Class for testing the ml model
    """

    def evaluate_probas(
        self,
        start: float = 0.35,
        stop: float = 1,
        n_steps: int = 40,
        purity_cut: float = 0.0,
        save_fig: bool = True,
    ) -> Tuple[float, float, float]:
        """Method for evaluating probability (BDT) cut effect on efficency and purity.

        Args:
            start (float, optional): Lower range of probablity cuts. Defaults to 0.3.
            stop (float, optional): Upper range of probablity cuts. Defaults to 0.98.
            n_steps (int, optional): Number of probability cuts to try. Defaults to 30.
            pid_variable_name (str, optional): Name of the variable containing true Pid. Defaults to "Complex_pid".
            purity_cut (float, optional): Minimal purity for automatic cuts selection. Defaults to 0..
            save_fig (bool, optional): Saves figures (BDT cut vs efficiency and purity) to file if True. Defaults to True.

        Returns:
            Tuple[float, float, float]: Probability cut for each variable.
        """
        print(
            f"Checking efficiency and purity for {int(n_steps)} probablity cuts between {start}, and {stop}..."
        )
        probas = np.linspace(start, stop, n_steps)
        efficienciess_protons, efficiencies_kaons, efficiencies_pions = [], [], []
        efficiencies = [efficienciess_protons, efficiencies_kaons, efficiencies_pions]
        purities_protons, purities_kaons, purities_pions = [], [], []
        purities = [purities_protons, purities_kaons, purities_pions]
        best_cuts = [0.0, 0.0, 0.0]
        max_efficiencies = [0.0, 0.0, 0.0]
        max_purities = [0.0, 0.0, 0.0]

        for proba in probas:
            self.gauss_preds(proba, proba, proba)
            # confusion matrix
            cnf_matrix = confusion_matrix(
                self.particles_df[self.pid_variable_name],
                self.particles_df["Complex_gauss_preds"],
            )
            for pid in range(self.get_n_classes() - 1):
                efficiency, purity = self.efficiency_stats(
                    cnf_matrix, pid, print_output=False
                )
                efficiencies[pid].append(efficiency)
                purities[pid].append(purity)
                if purity_cut > 0.0:
                    # Minimal purity for automatic threshold selection.
                    # Will choose the highest efficiency for purity above this value.
                    if purity >= purity_cut:
                        if efficiency > max_efficiencies[pid]:
                            best_cuts[pid] = proba
                            max_efficiencies[pid] = efficiency
                            max_purities[pid] = purity
                    # If max purity is below this value, will choose the highest purity available.
                    else:
                        if purity > max_purities[pid]:
                            best_cuts[pid] = proba
                            max_efficiencies[pid] = efficiency
                            max_purities[pid] = purity

        plotting_tools.plot_efficiency_purity(probas, efficiencies, purities, save_fig)
        if save_fig:
            print("Plots ready!")
        if purity_cut > 0:
            print(f"Selected probaility cuts: {best_cuts}")
            return (best_cuts[0], best_cuts[1], best_cuts[2])
        else:
            return (-1.0, -1.0, -1.0)

    def gauss_preds(self, proba_proton: float, proba_kaon: float, proba_pion: float):
        """Gets particle type as selected by xgboost model if above probability threshold.

        Args:
            proba_proton (float): Probablity threshold to classify particle as proton.
            proba_kaon (float): Probablity threshold to classify particle as kaon.
            proba_pion (float): Probablity threshold to classify particle as pion.
        """
        df = self.particles_df
        df["Complex_gauss_preds"] = df["Complex_gauss_pid"]

        # setting to bckgr if smaller than probability threshold
        proton = (df["Complex_gauss_pid"] == 0) & (df["Complex_prob_p"] > proba_proton)
        pion = (df["Complex_gauss_pid"] == 1) & (df["Complex_prob_K"] > proba_kaon)
        kaon = (df["Complex_gauss_pid"] == 2) & (df["Complex_prob_pi"] > proba_pion)
        df.loc[~(proton | pion | kaon), "Complex_gauss_preds"] = 3

        self.particles_df = df

    def confusion_matrix_and_stats(
        self, efficiency_filename: str = "efficiency_stats.txt"
    ):
        """
        Generates confusion matrix and efficiency/purity stats.
        """
        cnf_matrix = confusion_matrix(
            self.particles_df[self.pid_variable_name],
            self.particles_df["Complex_gauss_preds"],
        )
        plotting_tools.plot_confusion_matrix(cnf_matrix)
        plotting_tools.plot_confusion_matrix(cnf_matrix, normalize=True)
        txt_file = open(efficiency_filename, "w+")
        for pid in range(self.get_n_classes() - 1):
            self.efficiency_stats(cnf_matrix, pid, txt_file)
        txt_file.close()

    def remap_gauss_names(self):
        """
        Remaps Pid of particles to output format from XGBoost Model.
        Protons: 0; Kaons: 1; Pions, Electrons, Muons: 2; Other: 3

        """
        df = self.particles_df
        if self.anti_particles:
            df["Complex_gauss_pid"] = (
                df["Complex_gauss_pid"]
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
            df["Complex_gauss_pid"] = (
                df["Complex_gauss_pid"]
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

    def _tof_plots(self):
        """
        Generates tof plots.
        """
        for pid, particle_name in enumerate(self.classes_names):
            # simulated:
            try:
                plotting_tools.tof_plot(
                    self.particles_df[self.particles_df[self.pid_variable_name] == pid],
                    self.json_file_name,
                    f"{particle_name} (all simulated)",
                )
            except ValueError:
                print(f"No simulated {particle_name}s")
            # xgb selected
            try:
                plotting_tools.tof_plot(
                    self.particles_df[self.particles_df["Complex_gauss_preds"] == pid],
                    self.json_file_name,
                    f"{particle_name} (Gauss-selected)",
                )
            except ValueError:
                print(f"No Gauss-selected {particle_name}s")

    def _mass2_plots(self):
        """
        Generates mass2 plots.
        """
        protons_range = (-0.2, 1.8)
        kaons_range = (-0.2, 0.6)
        pions_range = (-0.3, 0.3)
        ranges = [protons_range, kaons_range, pions_range, pions_range]
        for pid, particle_name in enumerate(self.classes_names):
            plotting_tools.plot_mass2(
                self.particles_df[self.particles_df["Complex_gauss_preds"] == pid][
                    self.mass2_variable_name
                ],
                self.particles_df[self.particles_df[self.pid_variable_name] == pid][
                    self.mass2_variable_name
                ],
                particle_name,
                ranges[pid],
            )
            plotting_tools.plot_all_particles_mass2(
                self.particles_df[self.particles_df["Complex_gauss_preds"] == pid],
                self.mass2_variable_name,
                self.pid_variable_name,
                particle_name,
                ranges[pid],
            )

    def _vars_distributions_plots(self):
        """
        Generates distributions of variables and pT-rapidity graphs.
        """
        vars_to_draw = json_tools.load_vars_to_draw(self.json_file_name)
        for pid, particle_name in enumerate(self.classes_names):
            plotting_tools.var_distributions_plot(
                vars_to_draw,
                [
                    self.particles_df[
                        (self.particles_df[self.pid_variable_name] == pid)
                    ],
                    self.particles_df[
                        (
                            (self.particles_df[self.pid_variable_name] == pid)
                            & (self.particles_df["Complex_gauss_preds"] == pid)
                        )
                    ],
                    self.particles_df[
                        (
                            (self.particles_df[self.pid_variable_name] != pid)
                            & (self.particles_df["Complex_gauss_preds"] == pid)
                        )
                    ],
                ],
                [
                    f"true MC {particle_name}",
                    f"true selected {particle_name}",
                    f"false selected {particle_name}",
                ],
                filename=f"vars_dist_{particle_name}",
            )


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Arguments parser for the main method.

    Args:
        args (List[str]): Arguments from the command line, should be sys.argv[1:].

    Returns:
        argparse.Namespace: argparse.Namespace containg args
    """
    parser = argparse.ArgumentParser(
        prog="ML_PID_CBM ValidatGauss",
        description="Program for validating Gaussian PID model",
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
        "--nworkers",
        "-n",
        type=int,
        default=1,
        help="Max number of workers for ThreadPoolExecutor which reads Root tree with data.",
    )
    parser.add_argument(
        "--momentum",
        "-p",
        nargs=2,
        required=True,
        type=float,
        help="Lower and upper momentum limit, e.g., 1 3",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    # parser for main class
    args = parse_args(sys.argv[1:])
    # config  arguments to be loaded from args
    json_file_name = args.config[0]

    n_workers = args.nworkers
    lower_p, upper_p, is_anti = args.momentum[0], args.momentum[1], False
    # loading test data
    data_file_name = json_tools.load_file_name(json_file_name, "test")

    loader = LoadData(data_file_name, json_file_name, lower_p, upper_p, is_anti)
    # sigma selection
    # loading model handler and applying on dataset
    print(f"\nLoading data from {data_file_name}\n in ranges p = {lower_p}, {upper_p}")
    json_file_path = os.path.join(os.getcwd(), json_file_name)
    folder_name = f"gauss_{lower_p}_{upper_p}"
    if not os.path.exists(f"{folder_name}"):
        os.mkdir(f"{folder_name}")
    os.chdir(f"{folder_name}")
    copy2(json_file_path, os.getcwd())
    test_particles = loader.load_tree(max_workers=n_workers)
    # validate model object
    validate = ValidateGauss(
        lower_p, upper_p, is_anti, json_file_name, test_particles.get_data_frame()
    )
    # remap Pid to match output XGBoost format
    validate.remap_names()
    validate.remap_gauss_names()
    pid_variable_name = json_tools.load_var_name(json_file_name, "pid")
    proba_proton, proba_kaon, proba_pion = validate.evaluate_probas(purity_cut=90)
    # graphs
    validate.gauss_preds(proba_proton, proba_kaon, proba_pion)
    validate.confusion_matrix_and_stats()
    print("Generating plots...")
    validate.generate_plots()
    # save validated dataset
    validate.save_df()
