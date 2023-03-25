import os, gc
import argparse
from shutil import copy2
from hipe4ml.model_handler import ModelHandler
from load_data import LoadData
from prepare_model import PrepareModel
import plotting_tools


class TrainModel:
    """
    Class for training the ml model
    """

    def __init__(self, model_hdl: ModelHandler, model_name: str):
        self.model_hdl = model_hdl
        self.model_name = model_name

    def train_model_handler(self, train_test_data, model_hdl: ModelHandler = None):
        """Trains model handler

        Args:
            train_test_data (_type_): Train_test_data generated using a method from prepare_model module.
            model_hdl (ModelHandler, optional):  Hipe4ml model handler. Defaults to None.
        """
        model_hdl = model_hdl or self.model_hdl
        model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")
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


# main method of the training
if __name__ == "__main__":
    # parser for main class
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
        nargs=1,
        default=False,
        type=bool,
        help="If should train on particles instead of particles with positive charge.",
    )
    parser.add_argument(
        "--hyperparams",
        nargs=1,
        default=False,
        type=bool,
        help="If should optimize hyper params instead of using const values from config file. Will use ranges from config file.",
    )
    graphs_group = parser.add_mutually_exclusive_group()
    graphs_group.add_argument(
        "--printplots",
        #TODO action = BooleanOptional something
        nargs=1,
        default=False,
        type=bool,
        help="Creates plots and prints them without saving to file.",
    )
    graphs_group.add_argument(
        "--saveplots",
        "-plots", 
        nargs=1,
        default=False,
        type=bool,
        help="Creates plots and saves them to file, without printing.",
    )
    args = parser.parse_args()
    # config  arguments to be loaded from args
    json_file_name = args.config[0]
    lower_p_cut, upper_p_cut = args.momentum[0], args.momentum[1]
    anti_particles = args.antiparticles
    optimize_hyper_params = args.hyperparams
    create_plots = args.printplots or args.saveplots or False
    save_plots = args.saveplots or False
    if anti_particles:
        model_name = f"model_{lower_p_cut:.1f}_{upper_p_cut:.1f}_anti"
    else:
        model_name = f"model_{lower_p_cut:.1f}_{upper_p_cut:.1f}_positive"
    data_file_name = LoadData.load_file_name(json_file_name, "training")

    # loading data
    print(f"\nLoading data from {data_file_name}\n")
    loader = LoadData(
        data_file_name, json_file_name, lower_p_cut, upper_p_cut, anti_particles
    )
    th = loader.load_tree()
    protons, kaons, pions = loader.get_protons_kaons_pions(th)
    print(f"\nProtons, kaons, and pions loaded using file {data_file_name}\n")
    del th
    gc.collect()
    # change location to specific folder for this model
    json_file_path = os.path.join(os.getcwd(), json_file_name)
    if not os.path.exists(f"{model_name}"):
        os.makedirs(f"{model_name}")
    os.chdir(f"{model_name}")
    copy2(json_file_path, os.getcwd())
    # pretraining plots
    if create_plots:
        plotting_tools.tof_plot(protons, json_file_name, "protons", save_fig=save_plots)
        plotting_tools.tof_plot(kaons, json_file_name, "kaons", save_fig=save_plots)
        plotting_tools.tof_plot(
            pions, json_file_name, "pions, muons, electrons", save_fig=save_plots
        )
        vars_to_draw = protons.get_var_names()
        plotting_tools.var_distributions_plot(
            vars_to_draw, [protons, kaons, pions], save_fig=save_plots
        )
        plotting_tools.correlations_plot(
            vars_to_draw, [protons, kaons, pions], save_fig=save_plots
        )
    # loading model handler
    model_hdl = PrepareModel(json_file_name, optimize_hyper_params)
    train_test_data = model_hdl.prepare_train_test_data(protons, kaons, pions)
    print("\nPreparing model handler\n")
    model_hdl, study = model_hdl.prepare_model_handler(train_test_data=train_test_data)
    if create_plots and optimize_hyper_params:
        plotting_tools.opt_history_plot(study, save_plots)
        plotting_tools.opt_contour_plot(study, save_plots)
    # train model
    train = TrainModel(model_hdl, model_name)
    train.train_model_handler(train_test_data)
    print("\nModela trained!")
    if create_plots:
        y_pred_train = model_hdl.predict(train_test_data[0], False)
        y_pred_test = model_hdl.predict(train_test_data[2], False)
        plotting_tools.output_train_test_plot(
            train.model_hdl, train_test_data, save_fig=save_plots
        )
        plotting_tools.roc_plot(train_test_data[3], y_pred_test, save_fig=save_plots)

    train.save_model(model_name)
