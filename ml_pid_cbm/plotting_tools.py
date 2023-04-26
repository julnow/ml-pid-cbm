from typing import List
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib import rcParams
from pandas import DataFrame
from pandas import Series
import numpy as np
from hipe4ml.plot_utils import (
    plot_distr,
    plot_corr,
    plot_output_train_test,
    plot_roc,
    plot_feature_imp,
)
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
)
from optuna.study import Study
from load_data import LoadData

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
        plt.savefig(f"{file_name}.png")
        plt.savefig(f"{file_name}.pdf")
    else:
        plt.show()
    plt.close()


def var_distributions_plot(
    vars_to_draw: list,
    data_list: List[TreeHandler],
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
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
    else:
        plt.show()
    plt.close()


def correlations_plot(
    vars_to_draw: list,
    data_list: List[TreeHandler],
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    plt.subplots_adjust(
        left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55
    )
    cor_plots = plot_corr(data_list, vars_to_draw, leg_labels)
    for i, plot in enumerate(cor_plots):
        if save_fig:
            plot.savefig(f"correlations_plot_{i}.png")
            plot.savefig(f"correlations_plot_{i}.pdf")
    plt.close()


def opt_history_plot(study: Study, save_fig: bool = True):
    # for saving python-kaleido package is needed
    fig = plot_optimization_history(study)
    if save_fig:
        fig.write_image("optimization_history.png")
        fig.write_image("optimization_history.pdf")
    else:
        fig.show()
    plt.close()


def opt_contour_plot(study: Study, save_fig: bool = True):
    fig = plot_contour(study)
    if save_fig:
        fig.write_image("optimization_contour.png")
        fig.write_image("optimization_contour.pdf")
    else:
        plt.show()
    plt.close()


def output_train_test_plot(
    model_hdl: ModelHandler,
    train_test_data,
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    ml_out_fig = plot_output_train_test(
        model_hdl, train_test_data, 100, False, leg_labels, True, density=True
    )

    for idx, fig in enumerate(ml_out_fig):
        if save_fig:
            fig.savefig(f"output_train_test_plot_{idx}.png")
            fig.savefig(f"output_train_test_plot_{idx}.pdf")
        else:
            fig.show()
    plt.close()


def roc_plot(
    test_df: DataFrame,
    test_labels_array,
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True,
):
    plot_roc(test_df, test_labels_array, None, leg_labels, multi_class_opt="ovo")
    if save_fig:
        plt.savefig("roc_plot.png")
        plt.savefig("roc_plot.pdf")
    else:
        plt.show()
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
    else:
        plt.show()
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
    else:
        plt.show()
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
    else:
        plt.show()
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
    plt.title(f"$p_T$-rapidity efficiency for all selected for pid = {pid}")
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
        plt.savefig(f"plot_eff_pT_rap_ID={pid}.png")
        plt.savefig(f"plot_eff_pT_rap_ID={pid}.pdf")
    else:
        plt.show()
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
    plt.title(f"$p_T$-rapidity graph for all simulated pid = {pid}")

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
        plt.savefig(f"plot_pt_rapidity_ID={pid}.png")
        plt.savefig(f"plot_pt_rapidity_ID={pid}.pdf")
    else:
        plt.show()
    plt.close()


def plot_shap_summary(
    x_train: DataFrame,
    y_train: DataFrame,
    model_hdl: ModelHandler,
    n_sample: int = 5000,
    save_fig: bool = True,
    labels: List[str] = ["protons", "kaons", "pions"],
    approximate: bool = False,
):
    print("Creating shapley plots...")
    shap_plots = plot_feature_imp(
        x_train, y_train, model_hdl, labels, n_sample, approximate
    )
    plt.tight_layout()
    for i, shap_plot in enumerate(shap_plots):
        print(i)
        if save_fig:
            shap_plot.savefig(f"shap_plot_{i}.png")
            shap_plot.savefig(f"shap_plot_{i}.pdf")
        else:
            shap_plot.show()
        shap_plot.close()


def plot_before_after_variables(
    df: DataFrame,
    pid: float,
    pid_variable_name: str,
    training_variables: List[str],
    save_fig: bool = True,
    log_yscale: bool = True,
):
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
        ax.set_title(f"{variable_name} before and after XGB selection", fontsize=15)
        return fig

    for training_variable in training_variables:
        plot = variable_plot(df_true, df_reco, training_variable, log_yscale)
        if save_fig:
            plot.savefig(f"{training_variable}_before_after_pid_{pid}.png")
            plot.savefig(f"{training_variable}_before_after.pdf")
        else:
            plot.show()
        plt.close()


def plot_efficiency_purity(
    probas: np.ndarray,
    efficiencies: List[List[float]],
    purities: List[List[float]],
    save_fig: bool = True
):
    for i, (eff, pur) in enumerate(zip(efficiencies, purities)):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        ax.plot(probas, eff, label="efficiency")
        ax.plot(probas, pur, label="purity")
        ax.set_xlabel("BDT cut")
        ax.set_ylabel("\% ")
        ax.legend(loc='upper right')
        ax.set_title(f"Efficiency and purity in function of BDT cut for ID = {i}")
        ax.grid(which='major', linestyle='-')
        ax.minorticks_on()
        ax.grid(which='minor', linestyle='--')
        if save_fig:
            fig.savefig(f"efficiency_purity_id_{i}.png")
            fig.savefig(f"efficiency_purity_id_{i}.pdf")
        else:
            plt.show()
        plt.close()
