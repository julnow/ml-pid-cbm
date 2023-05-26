"""
Module with plotting tools
"""
import gc
import itertools
from typing import List, Tuple

import fasttreeshap as shap
import matplotlib as mplt
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hipe4ml.model_handler import ModelHandler
from hipe4ml.plot_utils import plot_corr, plot_distr, plot_output_train_test, plot_roc
from hipe4ml.tree_handler import TreeHandler
from matplotlib import rcParams
from optuna.study import Study
from optuna.visualization import plot_contour, plot_optimization_history
from sklearn.utils import resample

from ml_pid_cbm.tools import json_tools

PARAMS = {
    "axes.titlesize": "22",
    "axes.labelsize": "22",
    "xtick.labelsize": "22",
    "ytick.labelsize": "22",
    "figure.figsize": "10, 7",
    "figure.dpi": "300",
    "legend.fontsize": "20",
}
rcParams.update(PARAMS)


def tof_plot(
    df: pd.DataFrame,
    json_file_name: str,
    particles_title: str,
    file_name: str = "tof_plot",
    x_axis_range: List[int] = [-13, 13],
    y_axis_range: List[str] = [-1, 2],
    save_fig: bool = True,
) -> None:
    """
    Method for creating tof plots.

    Args:
        df (pd.DataFrame): Dataframe with particles to plot
        json_file_name (str): Name of the config.json file
        particles_title (str): Name of the particle type.
        file_name (str, optional): Filename to be created. Defaults to "tof_plot".
            Will add the particles_title after the tof_plot_ when saved.
        x_axis_range (List[int], optional): X-axis range. Defaults to [-13, 13].
        y_axis_range (List[str], optional): Y-axi range. Defaults to [-1, 2].
        save_fig (bool, optional): Where the figure should be saved. Defaults to True.

    Returns:
        None.
    """
    # load variable names
    charge_var_name = json_tools.load_var_name(json_file_name, "charge")
    momentum_var_name = json_tools.load_var_name(json_file_name, "momentum")
    mass2_var_name = json_tools.load_var_name(json_file_name, "mass2")
    # prepare plot variables
    ranges = [x_axis_range, y_axis_range]
    qp = df[charge_var_name] * df[momentum_var_name]
    mass2 = df[mass2_var_name]
    x_axis_name = r"sign($q$) $\cdot p$ (GeV/c)"
    y_axis_name = r"$m^2$ $(GeV/c^2)^2$"
    # plot graph
    fig, _ = plt.subplots(figsize=(15, 10), dpi=300)
    plt.hist2d(qp, mass2, bins=200, norm=matplotlib.colors.LogNorm(), range=ranges)
    plt.xlabel(x_axis_name, fontsize=20, loc="right")
    plt.ylabel(y_axis_name, fontsize=20, loc="top")
    title = f"TOF 2D plot for {particles_title}"
    plt.title(title, fontsize=20)
    fig.tight_layout()
    plt.colorbar()
    title = title.replace(" ", "_")
    # savefig
    if save_fig:
        file_name = particles_title.replace(" ", "_")
        plt.savefig(f"tof_plot_{file_name}.png")
        plt.savefig(f"tof_plot_{file_name}.pdf")
        plt.close()
    else:
        plt.show()
    return fig


def var_distributions_plot(
    vars_to_draw: list,
    data_list: List[TreeHandler],
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
    filename: str = "vars_disitributions",
):
    """
    Plots distributions of given variables using plot_distr from hipe4ml.

    Args:
        vars_to_draw (list): List of variables to draw.
        data_list (List[TreeHandler]): List of TreeHandlers with data.
        leg_labels (List[str], optional): Names of the particles which are given in the list of TreeHandlers.
            Defaults to ["protons", "kaons", "pions"].
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
        filename (str, optional): Name of the plot to be saved. Defaults to "vars_disitributions".
    """
    plot_distr(
        data_list,
        vars_to_draw,
        bins=100,
        labels=leg_labels,
        log=True,
        figsize=(40, 40),
        alpha=0.3,
        grid=False,
    )
    if save_fig:
        plt.savefig(f"{filename}.png")
        plt.savefig(f"{filename}.pdf")
        plt.close()
    else:
        plt.show()


def correlations_plot(
    vars_to_draw: list,
    data_list: List[TreeHandler],
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    """
    Creates correlation plots

    Args:
        vars_to_draw (list): Variables to check correlations.
        data_list (List[TreeHandler]): List of TreeHandlers with data.
        leg_labels (List[str], optional): Names of the particles which are given in the list of TreeHandlers.
            Defaults to ["protons", "kaons", "pions"].
        save_fig (bool, optional):  Whether should save the plot. Defaults to True.
    """
    plt.subplots_adjust(
        left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55
    )
    cor_plots = plot_corr(data_list, vars_to_draw, leg_labels)
    for i, plot in enumerate(cor_plots):
        if save_fig:
            plot.savefig(f"correlations_plot_{i}.png")
            plot.savefig(f"correlations_plot_{i}.pdf")
            plt.close()
        else:
            plot.show()


def opt_history_plot(study: Study, save_fig: bool = True):
    """
    Saves optimization history.

    Args:
        study (Study): optuna.Study to be saved
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
    """
    # for saving python-kaleido package is needed
    fig = plot_optimization_history(study)
    if save_fig:
        fig.write_image("optimization_history.png")
        fig.write_image("optimization_history.pdf")
    else:
        fig.show()
    plt.close()


def opt_contour_plot(study: Study, save_fig: bool = True):
    """
    Saves optimization contour plot

    Args:
        study (Study): optuna.Study to be saved
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
    """
    fig = plot_contour(study)
    if save_fig:
        fig.write_image("optimization_contour.png")
        fig.write_image("optimization_contour.pdf")
        plt.close()
    else:
        plt.show()


def output_train_test_plot(
    model_hdl: ModelHandler,
    train_test_data,
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    logscale: bool = False,
    save_fig: bool = True,
):
    """
    Output traing plot as in hipe4ml.plot_output_train_test

    Args:
        model_hdl (ModelHandler): Model handler to be tested
        train_test_data (_type_): List created by PrepareModel.prepare_train_test_data
        leg_labels (List[str], optional): Names of the classified particles. Defaults to ["protons", "kaons", "pions"].
        logscale (bool, optional): Whether should use logscale. Defaults to False.
        save_fig (bool, optional): Whether should save the plots. Defaults to True.
    """
    ml_out_fig = plot_output_train_test(
        model_hdl,
        train_test_data,
        100,
        False,
        leg_labels,
        logscale=logscale,
        density=False,  # if true histograms are normalized
    )
    if len(leg_labels) > 1:
        for idx, fig in enumerate(ml_out_fig):
            if save_fig:
                fig.savefig(f"output_train_test_plot_{idx}.png")
                fig.savefig(f"output_train_test_plot_{idx}.pdf")
            else:
                fig.show()
    else:
        if save_fig:
            ml_out_fig.savefig(f"output_train_test_plot.png")
            ml_out_fig.savefig(f"output_train_test_plot.pdf")
        else:
            ml_out_fig.show()
    plt.close()


def roc_plot(
    test_df: pd.DataFrame,
    test_labels_array: np.ndarray,
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    """
    Roc plot of the model

    Args:
        test_df (pd.DataFrame): Dataframe containg test_dataset with particles.
        test_labels_array (np.ndarray): Ndarray containig labels of the test_df.
        leg_labels (List[str], optional): Names of the classified particles. Defaults to ["protons", "kaons", "pions"].
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
    """
    plot_roc(test_df, test_labels_array, None, leg_labels, multi_class_opt="ovo")
    if save_fig:
        plt.savefig("roc_plot.png")
        plt.savefig("roc_plot.pdf")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cnf_matrix: np.ndarray,
    classes: List[str] = ["proton", "kaon", "pion", "bckgr"],
    normalize: bool = False,
    title: str = "Confusion matrix",
    cmap=mplt.colormaps["Blues"],
    save_fig: bool = True,
):
    """
    Plot created earlier confusion matrix.

    Args:
        cnf_matrix (np.ndarray): Confusion matrix
        classes (List[str], optional): List of the names of the classes.
            Defaults to ["proton", "kaon", "pion", "bckgr"].
        normalize (bool, optional): Whether should normalize the plot. Defaults to False.
        title (str, optional): Title of the plot. Defaults to "Confusion matrix".
        cmap (_type_, optional): Cmap used for colors. Defaults to mplt.colormaps["Blues"].
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
    """
    filename = "confusion_matrix"
    if normalize:
        cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        title = title + " (normalized)"
        filename = filename + " (norm)"
    else:
        print("Confusion matrix, without normalization")

    print(cnf_matrix)
    np.set_printoptions(precision=2)
    fig, axs = plt.subplots(figsize=(10, 8), dpi=300)
    axs.yaxis.set_label_coords(-0.04, 0.5)
    axs.xaxis.set_label_coords(0.5, -0.005)
    plt.imshow(cnf_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cnf_matrix.max() / 2.0
    for i, j in itertools.product(
        range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            format(cnf_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if cnf_matrix[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    if save_fig:
        plt.savefig(f"{filename}.png")
        plt.savefig(f"{filename}.pdf")
        plt.close()
    else:
        plt.show()


def plot_mass2(
    xgb_mass: pd.Series,
    sim_mass: pd.Series,
    particles_title: str,
    range1: Tuple[float, float],
    y_axis_log: bool = False,
    save_fig: bool = True,
):
    """
    Plots mass^2

    Args:
        xgb_mass (pd.Series): pd.Series containg xgb_selected mass^2
        sim_mass (pd.Series): pd.Series containg MC-true mass^2
        particles_title (str): Name of the plot.
        range1 (tuple[float, float]): Range of the mass2 to be plotted on x-axis.
        y_axis_log (bool, optional): If should use logscale in y-scale. Defaults to False.
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
    """
    # fig, axs = plt.subplots(2, 1,figsize=(15,10), sharex=True,  gridspec_kw={'width_ratios': [10],
    #                            'height_ratios': [8,4]})
    fig, axs = plt.subplots(figsize=(15, 10), dpi=300)

    ns, bins, patches = axs.hist(
        xgb_mass, bins=300, facecolor="red", alpha=0.3, range=range1
    )
    ns1, bins1, patches1 = axs.hist(
        sim_mass, bins=300, facecolor="blue", alpha=0.3, range=range1
    )
    # plt.xlabel("Mass in GeV", fontsize = 15)
    axs.set_ylabel("counts", fontsize=15)
    # axs[0].grid()
    axs.legend(
        ("XGBoost selected " + particles_title, "all simulated " + particles_title),
        loc="upper right",
    )
    if y_axis_log:
        axs.set_yscale("log")
    # plt.rcParams["legend.loc"] = 'upper right'
    title = f"{particles_title} $mass^2$ histogram"
    yName = r"Counts"
    xName = r"$m^2$ $(GeV/c^2)^2$"
    plt.xlabel(xName, fontsize=20, loc="right")
    plt.ylabel(yName, fontsize=20, loc="top")
    axs.set_title(title, fontsize=20)
    axs.grid()
    axs.tick_params(axis="both", which="major", labelsize=18)
    if save_fig:
        plt.savefig(f"mass2_{particles_title}.png")
        plt.savefig(f"mass2_{particles_title}.pdf")
        plt.close()
    else:
        plt.show()


def plot_all_particles_mass2(
    xgb_selected: pd.Series,
    mass2_variable_name: str,
    pid_variable_name: str,
    particles_title: str,
    range1: Tuple[float, float],
    y_axis_log: bool = False,
    save_fig: bool = True,
):
    """
    Plots mc-true particle type in xgb_selected particles

    Args:
        xgb_selected (pd.Series): pd.Series with xgb-selected particles.
        mass2_variable_name (str): Name of the mass2 variable name.
        pid_variable_name (str): Name of the pid variable name.
        particles_title (str): Name of the plot.
        range1 (tuple[float, float]): Range of the x-axis.
        y_axis_log (bool, optional): If should use logscale in y-scale. Defaults to False.
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
    """
    # fig, axs = plt.subplots(2, 1,figsize=(15,10), sharex=True,  gridspec_kw={'width_ratios': [10],
    #                            'height_ratios': [8,4]})
    fig, axs = plt.subplots(figsize=(15, 10), dpi=300)

    selected_protons = xgb_selected[xgb_selected[pid_variable_name] == 0][
        mass2_variable_name
    ]
    selected_kaons = xgb_selected[xgb_selected[pid_variable_name] == 1][
        mass2_variable_name
    ]
    selected_pions = xgb_selected[xgb_selected[pid_variable_name] == 2][
        mass2_variable_name
    ]

    ns, bins, patches = axs.hist(
        selected_protons, bins=300, facecolor="blue", alpha=0.4, range=range1
    )
    ns, bins, patches = axs.hist(
        selected_kaons, bins=300, facecolor="orange", alpha=0.4, range=range1
    )
    ns, bins, patches = axs.hist(
        selected_pions, bins=300, facecolor="green", alpha=0.4, range=range1
    )

    # plt.xlabel("Mass in GeV", fontsize = 15)
    axs.set_ylabel("counts", fontsize=15)
    # axs[0].grid()
    axs.legend(
        (
            f"XGBoost selected true protons",
            "XGBoost selected true kaons",
            "XGBoost selected true pions",
        ),
        loc="upper right",
    )
    if y_axis_log:
        axs.set_yscale("log")
    title = f"ALL XGBoost selected (true and false positive) {particles_title} $mass^2$ histogram"
    yName = r"Counts"
    xName = r"$m^2$ $(GeV/c^2)^2$"
    plt.xlabel(xName, loc="right")
    plt.ylabel(yName, loc="top")
    axs.set_title(title)
    axs.grid()
    axs.tick_params(axis="both", which="major", labelsize=18)
    if save_fig:
        plt.savefig(f"mass2_all_selected_{particles_title}.png")
        plt.savefig(f"mass2_all_selected_{particles_title}.pdf")
        plt.close()
    else:
        plt.show()


def plot_eff_pT_rap(
    df: pd.DataFrame,
    pid: int,
    pid_var_name: str = "Complex_pid",
    rapidity_var_name: str = "Complex_rapidity",
    pT_var_name: str = "Complex_pT",
    ranges: Tuple[Tuple[float, float], Tuple[float, float]] = [[0, 5], [0, 3]],
    nbins: int = 50,
    save_fig: bool = True,
    particle_names: List[str] = ["protons", "kaons", "pions", "bckgr"],
):
    df_true = df[(df[pid_var_name] == pid)]  # simulated
    df_reco = df[(df["xgb_preds"] == pid)]  # reconstructed by xgboost

    x = np.array(df_true[rapidity_var_name])
    y = np.array(df_true[pT_var_name])

    xe = np.array(df_reco[rapidity_var_name])
    ye = np.array(df_reco[pT_var_name])

    fig = plt.figure(figsize=(8, 10), dpi=300)
    plt.title(f"$p_T$-rapidity efficiency for all selected {particle_names[pid]}")
    true, yedges, xedges = np.histogram2d(x, y, bins=nbins, range=ranges)
    reco, _, _ = np.histogram2d(xe, ye, bins=(yedges, xedges), range=ranges)

    eff = np.divide(true, reco, out=np.zeros_like(true), where=reco != 0)  # Efficiency
    eff[eff == 0] = np.nan  # show zeros as white
    img = plt.imshow(
        eff,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=1,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    cbar = fig.colorbar(img, fraction=0.025, pad=0.08)  # above plot H
    cbar.set_label("efficiency (selected/simulated)", rotation=270, labelpad=20)

    plt.xlabel("rapidity")
    plt.ylabel("$p_T$ (GeV/c)")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"eff_pT_rap_{particle_names[pid]}.png")
        plt.savefig(f"eff_pT_rap_{particle_names[pid]}.pdf")
        plt.close()
    else:
        plt.show()


def plot_pt_rapidity(
    df: pd.DataFrame,
    pid: int,
    pid_var_name: str = "Complex_pid",
    rapidity_var_name: str = "Complex_rapidity",
    pT_var_name: str = "Complex_pT",
    ranges: Tuple[Tuple[float, float], Tuple[float, float]] = [[0, 5], [0, 3]],
    nbins=50,
    save_fig: bool = True,
    particle_names: List[str] = ["protons", "kaons", "pions", "bckgr"],
):
    """
    Plots pt-rapidity 2D histogram.

    Args:
        df (pd.DataFrame): Dataframe with input data.
        pid (int): Pid of the variable to be plotted.
        pid_var_name (str, optional): Name of the pid variable. Defaults to "Complex_pid".
        rapidity_var_name (str, optional): Name of the rapidity variable. Defaults to "Complex_rapidity".
        pT_var_name (str, optional): Name of the pT variable. Defaults to "Complex_pT".
        ranges (Tuple[Tuple[float, float], Tuple[float, float]], optional):
            Ranges of the plot. Defaults to [[0, 5], [0, 3]].
        nbins (int, optional): Number of bins in each axis. Defaults to 50.
        save_fig (bool, optional): Whether should save the figute. Defaults to True.
        particle_names (List[str], optional): Names of the particles corresponding to pid.
            Defaults to ["protons", "kaons", "pions", "bckgr"].
    """
    df_true = df[(df[pid_var_name] == pid)]  # simulated

    x = np.array(df_true[rapidity_var_name])
    y = np.array(df_true[pT_var_name])

    fig = plt.figure(figsize=(8, 10), dpi=300)
    plt.title(f"$p_T$-rapidity graph for all simulated {particle_names[pid]}")

    true, yedges, xedges = np.histogram2d(x, y, bins=nbins, range=ranges)
    true[true == 0] = np.nan  # show zeros as white

    img = plt.imshow(
        true,
        interpolation="nearest",
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    cbar = fig.colorbar(img, fraction=0.025, pad=0.08)  # above plot H
    cbar.set_label("counts", rotation=270, labelpad=20)

    plt.xlabel("rapidity")
    plt.ylabel("$p_T$ (GeV/c)")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"plot_pt_rapidity_{particle_names[pid]}.png")
        plt.savefig(f"plot_pt_rapidity_{particle_names[pid]}.pdf")
        plt.close()
    else:
        plt.show()


def _shap_summary(
    shap_values,
    x_train_resampled: pd.DataFrame,
    features_names: List[str],
    particle_name: str,
    save_fig: bool = True,
):
    """
    Internal method for plotting summary shap plots.

    Args:
        shap_values (_type_): Shap values.
        x_train_resampled (pd.DataFrame): Dataframe with X training variables.
        features_names (List[str]): List of the training variables.
        particle_name (str): Name of the particle.
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    shap.summary_plot(
        shap_values,
        x_train_resampled,
        feature_names=features_names,
        show=False,
    )
    w, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(h + 2, h)
    plt.gcf().set_size_inches(w, w * 3 / 4)
    plt.gcf().axes[-1].set_aspect("auto")
    plt.gcf().axes[-1].set_box_aspect(50)
    plt.xlabel(f"SHAP values for  {particle_name}", fontsize=18)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.tick_params(
        axis="both",
        which="major",
        length=10,
        direction="in",
        labelsize=15,
        zorder=4,
    )
    ax.minorticks_on()
    ax.tick_params(
        axis="both", which="minor", length=5, direction="in", labelsize=15, zorder=5
    )
    fig.tight_layout()
    if save_fig:
        plt.savefig(f"shap_summary_{particle_name}.png")
        plt.savefig(f"shap_summary_{particle_name}.pdf")
        plt.close()
    else:
        plt.show()


def _shap_interaction(
    shap_values,
    x_train_resampled,
    features_names,
    particle_name: str,
    save_fig: bool = True,
):
    """
    Internal method for plotting shap interaction plots.

    Args:
        shap_values (_type_): Shap values.
        x_train_resampled (pd.DataFrame): Dataframe with X training variables.
        features_names (List[str]): List of the training variables.
        particle_name (str): Name of the particle.
        save_fig (bool, optional): Whether should save the plot. Defaults to True.
    """
    for feature in features_names:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        shap.dependence_plot(
            feature,
            shap_values,
            x_train_resampled,
            display_features=x_train_resampled,
            show=False,
        )
        w, h = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(h + 2, h)
        plt.gcf().set_size_inches(w, w * 3 / 4)
        plt.gcf().axes[-1].set_aspect("auto")
        plt.gcf().axes[-1].set_box_aspect(50)
        plt.xlabel(f"{feature} for {particle_name}", fontsize=18)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.tick_params(
            axis="both",
            which="major",
            length=10,
            direction="in",
            labelsize=15,
            zorder=4,
        )
        ax.minorticks_on()
        ax.tick_params(
            axis="both", which="minor", length=5, direction="in", labelsize=15, zorder=5
        )
        fig.tight_layout()
        if save_fig:
            plt.savefig(f"shap_{feature}_{particle_name}.png")
            plt.savefig(f"shap_{feature}_{particle_name}.pdf")
            plt.close()
        else:
            plt.show()


def plot_shap_summary(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_hdl: ModelHandler,
    features_names: List[str],
    n_workers: int = 1,
    save_fig: bool = True,
    approximate: bool = False,
    n_samples: int = 50000,
    particle_names: List[str] = ["protons", "kaons", "pions"],
):
    """
    Method for plotting shap plots

    Args:
        x_train (pd.DataFrame): pd.Dataframe with X training dataset.
        y_train (pd.DataFrame): X training dataset labels.
        model_hdl (ModelHandler): Model Handler to be explained.
        features_names (List[str]): List of the training variables.
        n_workers (int, optional): Number of thread for multithreading.
            Note: it uses fastreeshap library, not shap. Defaults to 1.
        save_fig (bool, optional): Whether should save the plots.. Defaults to True.
        approximate (bool, optional): Whether should the approximate values. Defaults to False.
        n_samples (int, optional): Maximal number of samples in each class. Defaults to 50000.
        particle_names (List[str], optional): List of the classified particle names.
            Defaults to ["protons", "kaons", "pions"].
    """
    print("Creating shap plots...")
    explainer = shap.TreeExplainer(
        model_hdl.get_original_model(), n_jobs=n_workers, approximate=approximate
    )
    # Apply n_sanples in each class
    y_train_df = pd.DataFrame(y_train, columns=["true_class"])
    merged_df = pd.concat([x_train, y_train_df], axis=1)
    grouped_df = merged_df.groupby("true_class")
    resampled_df = pd.concat(
        [
            resample(group, n_samples=min(n_samples, len(group)), replace=False)
            for _, group in grouped_df
        ]
    )

    # Split the resampled pd.DataFrame back into input data and label data
    x_train_resampled = resampled_df.iloc[:, :-1]
    y_train_resampled = resampled_df.iloc[:, -1].to_numpy()
    del merged_df, grouped_df, resampled_df
    gc.collect()

    shap_values = explainer.shap_values(
        x_train_resampled, y_train_resampled, check_additivity=False
    )
    num_classes = len(shap_values)  # get the number of classes
    for i in range(num_classes):
        _shap_summary(
            shap_values[i],
            x_train_resampled,
            features_names,
            particle_names[i],
            save_fig=save_fig,
        )
        _shap_interaction(
            shap_values[i],
            x_train_resampled,
            features_names,
            particle_names[i],
            save_fig=save_fig,
        )


def plot_efficiency_purity(
    probas: np.ndarray,
    efficiencies: List[List[float]],
    purities: List[List[float]],
    save_fig: bool = True,
    particle_names: List[str] = ["protons", "kaons", "pions"],
):
    """
    Plots efficiency and purity in function of probability cuts.

    Args:
        probas (np.ndarray): Probability cuts
        efficiencies (List[List[float]]): List of list of efficiencies for each clas.
        purities (List[List[float]]): List of list of purities for each clas.
        save_fig (bool, optional): Whether should save the fig. Defaults to True.
        particle_names (List[str], optional): List of the particle names. Defaults to ["protons", "kaons", "pions"].
    """
    for i, (eff, pur) in enumerate(zip(efficiencies, purities)):
        if save_fig:
            dpi = 300
        else:
            dpi = 100
        fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)
        ax.plot(probas, eff, label="efficiency")
        ax.plot(probas, pur, label="purity")
        ax.set_xlabel("BDT cut")
        ax.set_ylabel("\% ")
        ax.legend(loc="upper right")
        ax.set_title(
            f"Efficiency and purity in function of BDT cut for {particle_names[i]}"
        )
        ax.grid(which="major", linestyle="-")
        ax.minorticks_on()
        ax.grid(which="minor", linestyle="--")
        if save_fig:
            fig.savefig(f"efficiency_purity__{particle_names[i]}.png")
            fig.savefig(f"efficiency_purity_id_{particle_names[i]}.pdf")
            plt.close()
        else:
            plt.show()


# deprecated
def plot_before_after_variables(
    df: pd.DataFrame,
    pid: float,
    pid_variable_name: str,
    training_variables: List[str],
    save_fig: bool = True,
    log_yscale: bool = True,
):
    """
    Plots each variable before and after selection.
    Legacy: var_distributions_plot should rather be used.

    Args:
        df (pd.DataFrame): _description_
        pid (float): _description_
        pid_variable_name (str): _description_
        training_variables (List[str]): _description_
        save_fig (bool, optional): _description_. Defaults to True.
        log_yscale (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    df_true = df[(df[pid_variable_name] == pid)]  # simulated
    df_reco = df[(df["xgb_preds"] == pid)]  # reconstructed by xgboost

    def variable_plot(
        df_true,
        df_reco,
        variable_name: str,
        log_yscale: bool = True,
        leg1="Simulated",
        leg2="XGB-selected",
        bins=100,
    ):
        fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
        ax.hist(
            df_true[variable_name],
            bins=bins,
            facecolor="blue",
            alpha=0.6,
            histtype="step",
            fill=False,
            linewidth=2,
        )
        ax.hist(
            df_reco[variable_name],
            bins=bins,
            facecolor="red",
            alpha=0.7,
            histtype="step",
            fill=False,
            linewidth=2,
        )
        ax.grid()
        ax.set_xlabel(variable_name, fontsize=15, loc="right")
        ax.set_ylim(bottom=1)
        ax.set_ylabel("counts", fontsize=15)
        if log_yscale:
            ax.set_yscale("log")
        ax.legend((leg1, leg2), fontsize=15, loc="upper right")
        ax.set_title(
            f"{variable_name} before and after XGB selection for pid={pid}", fontsize=15
        )
        return fig

    for training_variable in training_variables:
        plot = variable_plot(df_true, df_reco, training_variable, log_yscale)
        if save_fig:
            plot.savefig(f"{training_variable}_before_after_pid_{pid}.png")
            plot.savefig(f"{training_variable}_before_after_{pid}.pdf")
            plt.close()

        else:
            plot.show()
