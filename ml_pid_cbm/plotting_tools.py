from typing import List
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib import rcParams
from pandas import DataFrame
from pandas import Series
import numpy as np
from hipe4ml.plot_utils import plot_distr, plot_corr, plot_output_train_test, plot_roc
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
)
import shap
from load_data import LoadData


def tof_plot(
    df: DataFrame,
    json_file_name: str,
    particles_title: str,
    file_name: str = "tof_plot",
    x_axis_range: List[int] = [-13, 13],
    y_axis_range: List[str] = [-1, 2],
    save_fig: bool = True,
):
    # load variable names
    charge_var_name = LoadData.load_var_name(json_file_name, "charge")
    momentum_var_name = LoadData.load_var_name(json_file_name, "momentum")
    mass2_var_name = LoadData.load_var_name(json_file_name, "mass2")
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
        plt.savefig(f"{title}_{file_name}.png")
        plt.savefig(f"{title}_{file_name}.pdf")
    plt.close()


def var_distributions_plot(
    vars_to_draw: list,
    data_list: List[TreeHandler],
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    params = {
        "axes.titlesize": "22",
        "axes.labelsize": "22",
        "xtick.labelsize": "22",
        "ytick.labelsize": "22",
    }
    rcParams.update(params)
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300

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
        plt.savefig("vars_disitributions.png")
        plt.savefig("vars_disitributions.pdf")
    plt.close()


def correlations_plot(
    vars_to_draw: list,
    data_list: List[TreeHandler],
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300
    plt.subplots_adjust(
        left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55
    )
    plot_corr(data_list, vars_to_draw, leg_labels)
    if save_fig:
        plt.savefig("correlations_plot.png")
        plt.savefig("correlations_plot.pdf")


def opt_history_plot(study, save_fig: bool = True):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300
    plot_optimization_history(study)
    if save_fig:
        plt.savefig("optimization_history.png")
        plt.savefig("optimization_history.pdf")
    plt.close()


def opt_contour_plot(study, save_fig: bool = True):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300
    plot_contour(study)
    if save_fig:
        plt.savefig("optimization_contour.png")
        plt.savefig("optimization_contour.pdf")
    plt.close()


def output_train_test_plot(
    model_hdl: ModelHandler,
    train_test_data,
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300

    ml_out_fig = plot_output_train_test(
        model_hdl, train_test_data, 100, False, leg_labels, True, density=True
    )
    if save_fig:
        for idx, fig in enumerate(ml_out_fig):
            fig.savefig(f"output_train_test_plot_{idx}.png")
            fig.savefig(f"output_train_test_plot_{idx}.pdf")
    plt.close()


def roc_plot(
    test_df: DataFrame,
    test_labels_array,
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300
    plot_roc(test_df, test_labels_array, None, leg_labels, multi_class_opt="ovo")
    if save_fig:
        plt.savefig("roc_plot.png")
        plt.savefig("roc_plot.pdf")
    plt.close()



def plot_confusion_matrix(
    cm,
    classes=["proton", "kaon", "pion", "bckgr"],
    normalize=False,
    title="Confusion matrix",
    cmap=cm.get_cmap("Blues"),
    save_fig: bool = True,
):
    filename = "confusion_matrix"
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        title = title + " (normalized)"
        filename = filename + " (norm)"
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    np.set_printoptions(precision=2)
    fig, axs = plt.subplots(figsize=(10, 8), dpi=300)
    axs.yaxis.set_label_coords(-0.04, 0.5)
    axs.xaxis.set_label_coords(0.5, -0.005)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    if save_fig:
        plt.savefig(f"{filename}.png")
        plt.savefig(f"{filename}.pdf")
    plt.close()



def plot_mass2(
    xgb_mass: Series,
    sim_mass: Series,
    particles_title: str,
    range1,
    y_axis_log: bool = False,
    save_fig: bool = True,
):
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
        fontsize=15,
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



def plot_all_particles_mass2(
    xgb_selected: Series,
    mass2_variable_name: str,
    pid_variable_name: str,
    particles_title: str,
    range1,
    y_axis_log: bool = False,
    save_fig: bool = True,
):
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
        fontsize=15,
        loc="upper right",
    )
    if y_axis_log:
        axs.set_yscale("log")
    # plt.rcParams["legend.loc"] = 'upper right'
    title = f"ALL XGBoost selected (true and false positive) {particles_title} $mass^2$ histogram"
    yName = r"Counts"
    xName = r"$m^2$ $(GeV/c^2)^2$"
    plt.xlabel(xName, fontsize=20, loc="right")
    plt.ylabel(yName, fontsize=20, loc="top")
    axs.set_title(title, fontsize=20)
    axs.grid()
    axs.tick_params(axis="both", which="major", labelsize=18)
    if save_fig:
        plt.savefig(f"mass2_all_selected_{particles_title}.png")
        plt.savefig(f"mass2_all_selected_{particles_title}.pdf")
    plt.close()


def plot_eff_pT_rap(
    df: DataFrame,
    pid: float,
    pid_var_name: str = "Complex_pid",
    rapidity_var_name: str = "Complex_rapidity",
    pT_var_name: str = "Complex_pT",
    ranges=[[0, 5], [0, 3]],
    nbins=50,
    save_fig: bool = True,
):
    df_true = df[(df[pid_var_name] == pid)]  # simulated
    df_reco = df[(df["xgb_preds"] == pid)]  # reconstructed by xgboost

    x = np.array(df_true[rapidity_var_name])
    y = np.array(df_true[pT_var_name])

    xe = np.array(df_reco[rapidity_var_name])
    ye = np.array(df_reco[pT_var_name])

    fig = plt.figure(figsize=(8, 10), dpi=300)
    plt.title(
        f"$p_T$-rapidity efficiency for all selected for pid = {pid}",
        fontsize=16,
    )
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

    plt.xlabel("rapidity", fontsize=18)
    plt.ylabel("$p_T$ (GeV/c)", fontsize=18)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"plot_eff_pT_rap_ID={pid}.png")
        plt.savefig(f"plot_eff_pT_rap_ID={pid}.pdf")
    plt.close()



def plot_pt_rapidity(
    df: DataFrame,
    pid: float,
    pid_var_name: str = "Complex_pid",
    rapidity_var_name: str = "Complex_rapidity",
    pT_var_name: str = "Complex_pT",
    ranges=[[0, 5], [0, 3]],
    nbins=50,
    save_fig: bool = True,
):
    df_true = df[(df[pid_var_name] == pid)]  # simulated

    x = np.array(df_true[rapidity_var_name])
    y = np.array(df_true[pT_var_name])

    fig = plt.figure(figsize=(8, 10), dpi=300)
    plt.title(
        f"$p_T$-rapidity graph for all simulated pid = {pid}",
        fontsize=16,
    )

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

    plt.xlabel("rapidity", fontsize=18)
    plt.ylabel("$p_T$ (GeV/c)", fontsize=18)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"plot_pt_rapidity_ID={pid}.png")
        plt.savefig(f"plot_pt_rapidity_ID={pid}.pdf")
    plt.close()


def plot_shap_summary(
    x_train: DataFrame,
    y_train: DataFrame,
    model_hdl: ModelHandler,
    save_fig: bool = True
):
    explainer = shap.TreeExplainer(model_hdl.get_original_model())
    shap_values = explainer.shap_values(x_train, y_train, check_additivity=False)
    num_classes = len(shap_values)  # get the number of classes
    for i in range(num_classes):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        shap.summary_plot(
            shap_values[i], x_train, feature_names=x_train.columns, plot_size=[10, 15], show=False
        )
        w, h = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(h + 2, h)
        plt.gcf().set_size_inches(w, w * 3 / 4)
        plt.gcf().axes[-1].set_aspect("auto")
        plt.gcf().axes[-1].set_box_aspect(50)
        plt.xlabel(f"SHAP values for class {i}", fontsize=18)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.tick_params(
            axis="both", which="major", length=10, direction="in", labelsize=15, zorder=4
        )
        ax.minorticks_on()
        ax.tick_params(
            axis="both", which="minor", length=5, direction="in", labelsize=15, zorder=5
        )
        fig.tight_layout()
        if save_fig:
            plt.savefig(f"shap_summary_{i}.png")
            plt.savefig(f"shap_summary_{i}.pdf")
        plt.close()
        