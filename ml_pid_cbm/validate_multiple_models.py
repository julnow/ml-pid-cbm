import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from shutil import copy2
from typing import Set

import pandas as pd
from sklearn.metrics import confusion_matrix

import plotting_tools
from load_data import LoadData
from validate_model import ValidateModel
import json_tools


def load_pickles(files_list: Set[str], n_workers: int = 1) -> pd.DataFrame:
    """Loads multiple pickle files produced by validate_model module.

    Args:
        files_list (Set[str]): Files list containg picle files with datasets.
        n_workers (int, optional): Number of workers for multithreading. Defaults to 1.

    Returns:
        pd.DataFrame: Dataframe with merged datasets.
    """
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(pd.read_pickle, files_list))
        whole_df = pd.concat(results, ignore_index=True)
    return whole_df


if __name__ == "__main__":
    # parser for main class
    parser = argparse.ArgumentParser(
        prog="ML_PID_CBM ValidateMultipleModels",
        description="Program for loading multiple validated PID ML models",
    )
    parser.add_argument(
        "--modelnames",
        "-m",
        nargs="+",
        required=True,
        type=str,
        help="Names of folders containing trained and validated ML models.",
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
        type=int,
        default=1,
        help="Max number of workers for ThreadPoolExecutor which reads Root tree with data.",
    )
    args = parser.parse_args()
    # config  arguments to be loaded from args
    json_file_name = args.config[0]
    models = args.modelnames
    n_workers = args.nworkers
    pickle_files = {f"{model}/validated_data.pickle" for model in models}
    all_particles_df = load_pickles(pickle_files, n_workers)
    # new folder for all files
    json_file_path = os.path.join(os.getcwd(), json_file_name)
    if not os.path.exists("all_models"):
        os.makedirs("all_models")
    os.chdir("all_models")
    copy2(json_file_path, os.getcwd())
    # graphs
    # confusion matrix
    pid_variable_name = LoadData.load_var_name(json_file_name, "pid")
    cnf_matrix = confusion_matrix(
        all_particles_df[pid_variable_name], all_particles_df["xgb_preds"]
    )
    plotting_tools.plot_confusion_matrix(cnf_matrix)
    plotting_tools.plot_confusion_matrix(cnf_matrix, normalize=True)
    txt_file = open("efficiency_stats.txt", "w+")
    validate = ValidateModel(-12, 12, False, "", all_particles_df)
    for pid in range(0, 3):
        validate.efficiency_stats(cnf_matrix, pid, pid_variable_name, txt_file)
    txt_file.close()
    print("\nGenerating plots.")
    # tof plots
    # simulated:
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df[pid_variable_name] == 0],
        json_file_name,
        "protons (all simulated)",
    )
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df[pid_variable_name] == 1],
        json_file_name,
        "kaons (all simulated)",
    )
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df[pid_variable_name] == 2],
        json_file_name,
        "pions, muons, electrons (all simulated)",
    )
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df[pid_variable_name] == 3],
        json_file_name,
        "bckgr (all simulated)",
    )
    # xgb selected
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df["xgb_preds"] == 0],
        json_file_name,
        "protons (XGB-selected)",
    )
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df["xgb_preds"] == 1],
        json_file_name,
        "kaons (XGB-selected)",
    )
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df["xgb_preds"] == 2],
        json_file_name,
        "pions, muons, electrons (XGB-selected)",
    )
    # plotting_tools.tof_plot(
    #     all_particles_df[all_particles_df["xgb_preds"] == 3],
    #     json_file_name,
    #     "bckgr (XGB-selected)"
    # )
    # mass2 plots
    mass2_variable_name = json_tools.load_var_name(json_file_name, "mass2")
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
    # pt-rapidity plots
    for i in range(3):
        plotting_tools.plot_eff_pT_rap(all_particles_df, i)
        plotting_tools.plot_pt_rapidity(all_particles_df, i)
