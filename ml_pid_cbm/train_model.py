
from hipe4ml.model_handler import ModelHandler

class TrainModel():

    def __init__(self, model_hdl: ModelHandler, model_name: str):
        self.model_hdl = model_hdl
        self.model_name = model_name

    def train_model_handler(self, train_test_data, model_hdl: ModelHandler = None):
        model_hdl = model_hdl or self.model_hdl
        model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")

    def save_model(self, model_name: str = None, model_hdl: ModelHandler = None):
        model_name = model_name or self.model_name
        model_hdl = model_hdl or self.model_hdl
        model_hdl.dump_model_handler(model_name)