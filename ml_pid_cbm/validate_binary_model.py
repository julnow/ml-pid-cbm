import argparse
import os
import sys
from collections import defaultdict
from typing import List, Tuple

import json_tools
import numpy as np
import pandas as pd
import plotting_tools
from hipe4ml.model_handler import ModelHandler
from load_data import LoadData
from particles_id import ParticlesId as Pid
from sklearn.metrics import confusion_matrix
from validate_model import ValidateModel


class ValidateBinaryModel(ValidateModel):
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
        super().__init__(
            lower_p_cut, upper_p_cut, anti_particles, json_file_name, particles_df
        )
        self.classes_names = ["kaons", "bckgr"]

    def get_n_classes(self):
        return len(self.classes_names)

    def xgb_preds(self, proba: float):
        """Gets particle type as selected by xgboost model if above probability threshold.

        Args:
            proba(float): Probablity threshold to classify particle as signal.
        """
        df = self.particles_df
        df["xgb_preds"] = 0
        # setting to bckgr if smaller than probability threshold
        particle = df["model_output"] < proba
        df.loc[~(particle), "xgb_preds"] = 1

        self.particles_df = df

    def remap_names(self):
        """
        Remaps Pid of particles to output format from XGBoost Model.
        Kaons: 0; Other: 1

        """
        df = self.particles_df
        if self.anti_particles:
            df[self.pid_variable_name] = (
                df[self.pid_variable_name]
                .map(
                    defaultdict(
                        lambda: 1.0,
                        {
                            Pid.NEG_KAON.value: 0.0,
                        },
                    ),
                    na_action="ignore",
                )
                .astype(float)
            )
        else:
            df[self.pid_variable_name] = (
                df[self.pid_variable_name]
                .map(
                    defaultdict(
                        lambda: 1.0,
                        {
                            Pid.POS_KAON.value: 0.0,
                        },
                    ),
                    na_action="ignore",
                )
                .astype(float)
            )
        self.particles_df = df

    def evaluate_probas(
        self,
        start: float = 0.3,
        stop: float = 0.98,
        n_steps: int = 30,
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
        efficiencies = []
        purities = []
        best_cut = 0.0
        max_efficiency = 0.0
        max_purity = 0.0

        for proba in probas:
            self.xgb_preds(proba)
            # confusion matrix
            cnf_matrix = confusion_matrix(
                self.particles_df[self.pid_variable_name],
                self.particles_df["xgb_preds"],
            )
            pid = 0
            efficiency, purity = self.efficiency_stats(
                cnf_matrix, pid, print_output=False
            )
            efficiencies.append(efficiency)
            purities.append(purity)
            if purity_cut > 0.0:
                # Minimal purity for automatic threshold selection.
                # Will choose the highest efficiency for purity above this value.
                if purity >= purity_cut:
                    if efficiency > max_efficiency:
                        best_cut = proba
                        max_efficiency = efficiency
                        max_purity = purity
                # If max purity is below this value, will choose the highest purity available.
                else:
                    if purity > max_purity:
                        best_cut = proba
                        max_efficiency = efficiency
                        max_purity = purity

        plotting_tools.plot_efficiency_purity(
            probas, [efficiencies], [purities], save_fig, particle_names=["kaons"]
        )
        if save_fig:
            print("Plots ready!")
        if purity_cut > 0:
            print(f"Selected probaility cuts: {best_cut}")
            return best_cut
        else:
            return -1.0

    def confusion_matrix_and_stats(
        self, efficiency_filename: str = "efficiency_stats.txt", save_fig: bool = True
    ):
        """
        Generates confusion matrix and efficiency/purity stats.
        """
        cnf_matrix = confusion_matrix(
            self.particles_df[self.pid_variable_name], self.particles_df["xgb_preds"]
        )
        plotting_tools.plot_confusion_matrix(
            cnf_matrix, classes=self.classes_names, save_fig=save_fig
        )
        plotting_tools.plot_confusion_matrix(
            cnf_matrix, classes=self.classes_names, normalize=True, save_fig=save_fig
        )
        txt_file = open(efficiency_filename, "w+")
        for pid in range(self.get_n_classes() - 1):
            self.efficiency_stats(cnf_matrix, pid, txt_file)
        txt_file.close()

    def generate_plots(self):
        """
        Generate tof, mass2, vars, and pT-rapidity plots
        """
        self._tof_plots()
        self._mass2_plots()
        self._vars_distributions_plots()

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
                    self.particles_df[self.particles_df["xgb_preds"] == pid],
                    self.json_file_name,
                    f"{particle_name} (XGB-selected)",
                )
            except ValueError:
                print(f"No XGB-selected {particle_name}s")

    def _mass2_plots(self):
        """
        Generates mass2 plots.
        """
        kaons_range = (-0.2, 0.6)
        for pid, particle_name in enumerate(self.classes_names):
            plotting_tools.plot_mass2(
                self.particles_df[self.particles_df["xgb_preds"] == pid][
                    self.mass2_variable_name
                ],
                self.particles_df[self.particles_df[self.pid_variable_name] == pid][
                    self.mass2_variable_name
                ],
                particle_name,
                kaons_range,
            )
            plotting_tools.plot_all_particles_mass2(
                self.particles_df[self.particles_df["xgb_preds"] == pid],
                self.mass2_variable_name,
                self.pid_variable_name,
                particle_name,
                kaons_range,
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
                            & (self.particles_df["xgb_preds"] == pid)
                        )
                    ],
                    self.particles_df[
                        (
                            (self.particles_df[self.pid_variable_name] != pid)
                            & (self.particles_df["xgb_preds"] == pid)
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
            plotting_tools.plot_eff_pT_rap(self.particles_df, pid)
            plotting_tools.plot_pt_rapidity(self.particles_df, pid)


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Arguments parser for the main method.

    Args:
        args (List[str]): Arguments from the command line, should be sys.argv[1:].

    Returns:
        argparse.Namespace: argparse.Namespace containg args
    """
    parser = argparse.ArgumentParser(
        prog="ML_PID_CBM ValidateBinaryModel",
        description="Program for validating Binary PID ML models",
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
        nargs=1,
        type=float,
        help="Probability cut value e.g., 0.2",
    )
    proba_group.add_argument(
        "--evaluateproba",
        "-e",
        nargs=3,
        type=float,
        help="""Minimal probability cut, maximal, and number of steps to investigate. 
        Contrary to multi model, the lower the value of probability, the higher probability of signal""",
    )
    parser.add_argument(
        "--nworkers",
        "-n",
        type=int,
        default=1,
        help="Max number of workers for ThreadPoolExecutor which reads Root tree with data.",
    )
    decision_group = parser.add_mutually_exclusive_group()
    decision_group.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode allows selection of probability cuts after evaluating them.",
    )
    decision_group.add_argument(
        "--automatic",
        "-a",
        nargs=1,
        type=float,
        help="""Minimal purity for automatic threshold selection (in percent) e.g., 90.
        Will choose the highest efficiency for purity above this value.
        If max purity is below this value, will choose the highest purity available.""",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    # parser for main class
    args = parse_args(sys.argv[1:])
    # config  arguments to be loaded from args
    json_file_name = args.config[0]
    model_name = args.modelname[0]
    proba_proton = args.probabilitycuts[0] if args.probabilitycuts is not None else -1.0

    n_workers = args.nworkers
    purity_cut = args.automatic[0] if args.automatic is not None else 0.0
    lower_p, upper_p, is_anti = ValidateModel.parse_model_name(model_name)
    # loading test data
    data_file_name = json_tools.load_file_name(json_file_name, "test")

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
    validate = ValidateBinaryModel(
        lower_p, upper_p, is_anti, json_file_name, test_particles.get_data_frame()
    )
    # remap Pid to match output XGBoost format
    validate.remap_names()
    pid_variable_name = json_tools.load_var_name(json_file_name, "pid")
    # set probability cuts
    if args.evaluateproba is not None:
        proba = validate.evaluate_probas(
            args.evaluateproba[0],
            args.evaluateproba[1],
            int(args.evaluateproba[2]),
            purity_cut,
            not args.interactive,
        )
    # if probabilites are set
    # apply probabilty cuts
    print(f"\nApplying probability cuts for kaon: {proba}")
    validate.xgb_preds(proba)
    # graphs
    validate.confusion_matrix_and_stats()
    print("Generating plots...")
    validate.generate_plots()
    # save validated dataset
    validate.save_df()
