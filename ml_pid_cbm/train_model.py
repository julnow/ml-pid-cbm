"""
Module for training the model.

"""
import argparse
import gc
import os
import sys
from shutil import copy2
from typing import List

from hipe4ml.model_handler import ModelHandler
from sklearn.utils.class_weight import compute_sample_weight
from tools import json_tools, plotting_tools
from tools.load_data import LoadData
from tools.prepare_model import PrepareModel


class TrainModel:
    """
    Class for training the ml model
    """

    def __init__(self, model_hdl: ModelHandler, model_name: str):
        self.model_hdl = model_hdl
        self.model_name = model_name

    def train_model_handler(
        self, train_test_data, sample_weights, model_hdl: ModelHandler = None
    ):
        """Trains model handler

        Args:
            train_test_data (_type_): Train_test_data generated using a method from prepare_model module.
            sample_weights(List[Float]): ndarray of shape (n_samples,) Array with sample weights.
            To be computed with sklearn.utils.class_weight.compute_sample_weight
            model_hdl (ModelHandler, optional):  Hipe4ml model handler. Defaults to None.
        """
        model_hdl = model_hdl or self.model_hdl
        model_hdl.train_test_model(
            train_test_data, multi_class_opt="ovo", sample_weight=sample_weights
        )
        self.model_hdl = model_hdl

    def save_model(self, model_name: str = None, model_hdl: ModelHandler = None):
        """Saves trained model handler.

        Args:
            model_name (str, optional): Name of the model handler. Defaults to None.
            model_hdl (ModelHandler, optional): Hipe4ml model handler. Defaults to None.
        """
        model_name = model_name or self.model_name
        model_hdl = model_hdl or self.model_hdl
        model_hdl.dump_model_handler(model_name)
        print(f"\nModel saved as {model_name}")


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Arguments parser for the main method.

    Args:
        args (List[str]): Arguments from the command line, should be sys.argv[1:].

    Returns:
        argparse.Namespace: argparse.Namespace containg args
    """
    parser = argparse.ArgumentParser(
        prog="ML_PID_CBM TrainModel", description="Program for training PID ML models"
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
        "--momentum",
        "-p",
        nargs=2,
        required=True,
        type=float,
        help="Lower and upper momentum limit, e.g., 1 3",
    )
    parser.add_argument(
        "--antiparticles",
        action="store_true",
        help="If should train on particles instead of particles with positive charge.",
    )
    parser.add_argument(
        "--hyperparams",
        action="store_true",
        help="If should optimize hyper params instead of using const values from config file. Will use ranges from config file.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="If should use GPU for training. Remember that xgboost-gpu version is needed for this.",
    )
    parser.add_argument(
        "--nworkers",
        "-n",
        type=int,
        default=1,
        help="Max number of workers for ThreadPoolExecutor which reads Root tree with data.",
    )
    graphs_group = parser.add_mutually_exclusive_group()
    graphs_group.add_argument(
        "--printplots",
        action="store_true",
        help="Creates plots and prints them without saving to file.",
    )
    graphs_group.add_argument(
        "--saveplots",
        "-plots",
        action="store_true",
        help="Creates plots and saves them to file, without printing.",
    )
    parser.add_argument(
        "--usevalidation",
        action="store_true",
        help="if should use validation dataset for post-training plots",
    )
    return parser.parse_args(args)


# main method of the training
if __name__ == "__main__":
    # parser for main class
    args = parse_args(sys.argv[1:])
    # config  arguments to be loaded from args
    json_file_name = args.config[0]
    lower_p_cut, upper_p_cut = args.momentum[0], args.momentum[1]
    anti_particles = args.antiparticles
    optimize_hyper_params = args.hyperparams
    use_gpu = args.gpu
    n_workers = args.nworkers
    create_plots = args.printplots or args.saveplots or False
    save_plots = args.saveplots
    use_validation = args.usevalidation
    if anti_particles:
        model_name = f"model_{lower_p_cut:.1f}_{upper_p_cut:.1f}_anti"
    else:
        model_name = f"model_{lower_p_cut:.1f}_{upper_p_cut:.1f}_positive"
    data_file_name = json_tools.load_file_name(json_file_name, "training")

    # loading data
    loader = LoadData(
        data_file_name, json_file_name, lower_p_cut, upper_p_cut, anti_particles
    )
    tree_handler = loader.load_tree(max_workers=n_workers)
    NSIGMA_PROTON = 0
    NSIGMA_KAON = 0
    NSIGMA_PION = 0
    protons, kaons, pions = loader.get_protons_kaons_pions(
        tree_handler,
        nsigma_proton=NSIGMA_PROTON,
        nsigma_kaon=NSIGMA_KAON,
        nsigma_pion=NSIGMA_PION,
    )
    print(f"\nProtons, kaons, and pions loaded using file {data_file_name}\n")
    del tree_handler
    gc.collect()
    # change location to specific folder for this model
    json_file_path = os.path.join(os.getcwd(), json_file_name)
    if not os.path.exists(f"{model_name}"):
        os.makedirs(f"{model_name}")
    os.chdir(f"{model_name}")
    copy2(json_file_path, os.getcwd())
    # pretraining plots
    if create_plots:
        print("Creating pre-training plots...")
        plotting_tools.tof_plot(
            protons,
            json_file_name,
            f"protons ({NSIGMA_PROTON}$\sigma$)",
            save_fig=save_plots,
        )
        plotting_tools.tof_plot(
            kaons, json_file_name, f"kaons ({NSIGMA_KAON}$\sigma$)", save_fig=save_plots
        )
        plotting_tools.tof_plot(
            pions,
            json_file_name,
            f"pions, muons, electrons ({NSIGMA_PION}$\sigma$)",
            save_fig=save_plots,
        )
        vars_to_draw = protons.get_var_names()
        plotting_tools.correlations_plot(
            vars_to_draw, [protons, kaons, pions], save_fig=save_plots
        )
    # loading model handler
    model_hdl = PrepareModel(json_file_name, optimize_hyper_params, use_gpu)
    train_test_data = PrepareModel.prepare_train_test_data([protons, kaons, pions])
    del protons, kaons, pions
    gc.collect()
    features_for_train = json_tools.load_features_for_train(json_file_name)
    print("\nPreparing model handler...")
    model_hdl, study = model_hdl.prepare_model_handler(train_test_data=train_test_data)
    if create_plots and optimize_hyper_params:
        plotting_tools.opt_history_plot(study, save_plots)
        plotting_tools.opt_contour_plot(study, save_plots)
    # train model
    train = TrainModel(model_hdl, model_name)
    sample_weights = compute_sample_weight(
        class_weight=None,  # class_weight="balanced" deleted for now
        y=train_test_data[1],
    )
    train.train_model_handler(train_test_data, sample_weights)
    print("\nModel trained!")
    train.save_model(model_name)
    # loading validation dataset as test dataset for pos-training plots
    if use_validation:
        data_file_name_test = json_tools.load_file_name(json_file_name, "test")
        loader_test = LoadData(
            data_file_name_test,
            json_file_name,
            lower_p_cut,
            upper_p_cut,
            anti_particles,
        )
        tree_handler_test = loader_test.load_tree(max_workers=n_workers)
        protons_test, kaons_test, pions_test = loader_test.get_protons_kaons_pions(
            tree_handler_test,
            nsigma_proton=NSIGMA_PROTON,
            nsigma_kaon=NSIGMA_KAON,
            nsigma_pion=NSIGMA_PION,
        )
        validation_data = PrepareModel.prepare_train_test_data(
            [protons_test, kaons_test, pions_test]
        )
        train_test_data = [
            train_test_data[0],
            train_test_data[1],
            validation_data[0],
            validation_data[1],
        ]
    if create_plots:
        print("Creating post-training plots")
        y_pred_train = model_hdl.predict(train_test_data[0], False)
        y_pred_test = model_hdl.predict(train_test_data[2], False)
        plotting_tools.output_train_test_plot(
            train.model_hdl, train_test_data, save_fig=save_plots, logscale=True
        )

        plotting_tools.roc_plot(train_test_data[3], y_pred_test, save_fig=save_plots)
        # shapleys for each class
        feature_names = [item.replace("Complex_", "") for item in features_for_train]
        plotting_tools.plot_shap_summary(
            train_test_data[0][features_for_train],
            train_test_data[1],
            model_hdl,
            features_for_train,
            n_workers,
        )
