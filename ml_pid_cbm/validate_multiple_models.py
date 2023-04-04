import argparse
import os
import pandas as pd
from shutil import copy2
from sklearn.metrics import confusion_matrix
from load_data import LoadData
import plotting_tools
from validate_model import validate

if __name__ == "__main__":
    # parser for main class
    parser = argparse.ArgumentParser(
        prog="ML_PID_CBM ValidateModel",
        description="Program for validating PID ML models",
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
    args = parser.parse_args()
    # config  arguments to be loaded from args
    json_file_name = args.config[0]
    models = args.modelnames
    pickle_files = {f"{model}/validated_data.pickle" for model in models}
    all_particles_df = pd.concat((pd.read_pickle(f) for f in pickle_files))
    # new folder for all files
    json_file_path = os.path.join(os.getcwd(), json_file_name)
    if not os.path.exists("all_models"):
        os.makedirs("all_models")
    os.chdir("all_models")
    copy2(json_file_path, os.getcwd())
    # graphs
    # confusion matrix
    pid_variable = LoadData.load_var_name(json_file_name, "pid")
    cnf_matrix = confusion_matrix(
        all_particles_df[pid_variable], all_particles_df["xgb_preds"]
    )
    plotting_tools.plot_confusion_matrix(cnf_matrix)
    plotting_tools.plot_confusion_matrix(cnf_matrix, normalize=True)
    txt_file = open("efficiency_stats.txt", "w+")
    for pid in range(0, 3):
        validate.efficiency_stats(cnf_matrix, pid, pid_variable, txt_file, all_particles_df)
    txt_file.close()
    # tof plots
    # simulated:
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df[pid_variable] == 0],
        json_file_name,
        "protons (all simulated)",
    )
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df[pid_variable] == 1],
        json_file_name,
        "kaons (all simulated)",
    )
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df[pid_variable] == 2],
        json_file_name,
        "pions, muons, electrons (all simulated)",
    )
    plotting_tools.tof_plot(
        all_particles_df[all_particles_df[pid_variable] == 3],
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
    mass2_variable_name = LoadData.load_var_name(json_file_name, "mass2")
    plotting_tools.plot_mass2(
        all_particles_df[all_particles_df["xgb_preds"] == 0][mass2_variable_name],
        all_particles_df[all_particles_df[pid_variable] == 0][mass2_variable_name],
        "Protons",
        (-0.1, 1.5),
    )
    plotting_tools.plot_mass2(
        all_particles_df[all_particles_df["xgb_preds"] == 1][mass2_variable_name],
        all_particles_df[all_particles_df[pid_variable] == 1][mass2_variable_name],
        "Kaons",
        (-0.1, 0.4),
    )
    plotting_tools.plot_mass2(
        all_particles_df[all_particles_df["xgb_preds"] == 2][mass2_variable_name],
        all_particles_df[all_particles_df[pid_variable] == 2][mass2_variable_name],
        "Pions (& electrons, muons)",
        (-0.15, 0.15),
    )
    plotting_tools.plot_mass2(
        all_particles_df[all_particles_df["xgb_preds"] == 3][mass2_variable_name],
        all_particles_df[all_particles_df[pid_variable] == 3][mass2_variable_name],
        "Background",
        (-0.15, 0.15),
    )
    # pt-rapidity plots
    for i in range(3):
        plotting_tools.plot_eff_pT_rap(all_particles_df, i)
        plotting_tools.plot_pt_rapidity(all_particles_df, i)
