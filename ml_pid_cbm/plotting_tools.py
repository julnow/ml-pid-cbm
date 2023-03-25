from typing import List
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import rcParams
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from hipe4ml.plot_utils import plot_distr, plot_corr, plot_output_train_test, plot_roc
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
)
from load_data import LoadData


def tof_plot(
    df: DataFrame,
    json_file_name: str,
    particles_title: str,
    x_axis_range: List[int] = [-13, 13],
    y_axis_range: List[str] = [-1, 2],
    save_fig: bool = True
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
    # savefig
    if save_fig:
        file_name = particles_title.rstrip()
        plt.savefig(f"tof_plot_{file_name}.png")
        plt.savefig(f"tof_plot_{file_name}.pdf")
    else:
        plt.show()


def var_distributions_plot(
    vars_to_draw: list,
    data_list: List[TreeHandler],
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True
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
    else:
        plt.show()


def correlations_plot(
    vars_to_draw: list,
    data_list: List[TreeHandler],
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True
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
    else:
        plt.show()


def opt_history_plot(study, save_fig: bool = True):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300
    plot_optimization_history(study)
    if save_fig:
        plt.savefig("optimization_history.png")
        plt.savefig("optimization_history.pdf")
    else:
        plt.show()


def opt_contour_plot(study, save_fig: bool = True):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300
    plot_contour(study)
    if save_fig:
        plt.savefig("optimization_contour.png")
        plt.savefig("optimization_contour.pdf")
    else:
        plt.show()


def output_train_test_plot(
    model_hdl: ModelHandler,
    train_test_data,
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True
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
    else:
        plt.show()


def roc_plot(
    test_df: DataFrame,
    test_labels_array,
    leg_labels: List[str] = ["protons", "kaons", "pions"],
    save_fig: bool = True
):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 300
    plot_roc(test_df, test_labels_array, None, leg_labels, multi_class_opt="ovo")
    if save_fig:
        plt.savefig("roc_plot.png")
        plt.savefig("roc_plot.pdf")
    else:
        plt.show()


# def confusion_matrix_plot():
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label',fontsize = 15)
#     plt.xlabel('Predicted label',fontsize = 15)

