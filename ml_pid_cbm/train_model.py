import gc
from hipe4ml.model_handler import ModelHandler
from load_data import LoadData
from prepare_model import PrepareModel


class TrainModel:
    def __init__(self, model_hdl: ModelHandler, model_name: str):
        self.model_hdl = model_hdl
        self.model_name = model_name

    def train_model_handler(self, train_test_data, model_hdl: ModelHandler = None):
        model_hdl = model_hdl or self.model_hdl
        model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")
        self.model_hdl = model_hdl

    def save_model(self, model_name: str = None, model_hdl: ModelHandler = None):
        model_name = model_name or self.model_name
        model_hdl = model_hdl or self.model_hdl
        model_hdl.dump_model_handler(model_name)


# main method of the training
if __name__ == "__main__":
    # TODO: config  arguments to be loaded from args
    data_file_name = "/Users/julnow/gsi/mgr/trees/PlainTree200k_trdrich_12agev.root"
    model_name = "model1test"
    json_file_name = "config.json"
    lower_p_cut = 0
    upper_p_cut = 6
    anti_particles = False
    optimize_hyper_params = True
    # loading data
    loader = LoadData(
        data_file_name, json_file_name, lower_p_cut, upper_p_cut, anti_particles
    )
    th = loader.load_tree()
    protons, kaons, pions = loader.get_protons_kaons_pions(th)
    print(f"\nProtons, kaons, and pions loaded using file {data_file_name}\n")
    del th
    gc.collect()
    # loading model handler
    model_hdl = PrepareModel(json_file_name, optimize_hyper_params)
    train_test_data = model_hdl.prepare_train_test_data(protons, kaons, pions)
    print("\nPreparing model handler\n")
    model_hdl = model_hdl.prepare_model_handler(train_test_data=train_test_data)
    print(f"\nModelHandler ready using configuration from {json_file_name}")
    # train model
    train = TrainModel(model_hdl, model_name)
    print("\nModela trained!")
    train.save_model(model_name)
    print(f"\nModel saved as {model_name}")
