
from hipe4ml.tree_handler import TreeHandler
import json

class LoadData:

    def __init__(self, data_file_name: str, json_file_name: str, lower_p_cut: float, upper_p_cut: float, anti_particles: bool):
        self.data_file_name = data_file_name 
        self.lower_p_cut    = lower_p_cut
        self.upper_p_cut    = upper_p_cut
        self.anti_particles = anti_particles
        self.json_file_name = json_file_name

    #load file with data into hipe4ml TreeHandler
    def load_file(self, data_file_name: str = None, tree_type: str = 'plain_tree') -> TreeHandler:
        if data_file_name is None:
            data_file_name = self.data_file_name
        return self.load_tree(data_file_name, tree_type)
    
    #load tree using quality cuts defined in json file
    def load_tree(self, data_file_name: str, tree_type: str, json_file_name: str = None):
        if json_file_name is None:
            json_file_name = self.json_file_name
        th = TreeHandler(data_file_name, tree_type)
        quality_cuts = self.load_quality_cuts(json_file_name)
        for cut in quality_cuts:
            th = th.get_subset(cut)
        #include specific momentum cut
        p_cut = self.create_cut_string(self.lower_p_cut, "Complex_p", self.upper_p_cut)
        th = th.get_subset(p_cut)
        return th
    
    #loads qualit cuts from json file
    def load_quality_cuts(self, json_filename: str):
        with open(json_filename, "r") as f:
            cuts = json.load(f)["cuts"]
        quality_cuts = [self.__class__.create_cut_string(cut_data['lower'], cut_data['upper'], cut_name) for cut_name, cut_data in cuts.items()]
        return quality_cuts

    #create string for cuts when loading data with precision of one digit after comma
    @staticmethod
    def create_cut_string(lower: float, upper: float, cut_name: str) -> str:
        cut_string = f"{lower:.1f} < {cut_name} < {upper:.1f}"
        return cut_string
    
