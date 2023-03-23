"""
This module is used for training our ml model
"""
import json
from typing import Dict, List, Tuple


class TrainModel:
    """
    Class for training the ml pid model
    """

    def __init__(self, json_file_name: str, optimize_hyper_params: bool):
        self.json_file_name = json_file_name
        self.optimize_hyper_params =  optimize_hyper_params

    def load_hyper_params_ranges(self, json_file_name: str = None) -> Dict[str, Tuple[float, float]]:
        json_file_name = json_file_name or self.json_file_name
        with open(json_file_name, "r") as json_file:
            hyper_params_ranges = json.load(json_file)["hyper_params"]["ranges"]
        #json accepts only lists, so they need to be transformed back into tuples
        hyper_params_ranges = {k: tuple(v) for k, v in hyper_params_ranges.items()}
        return hyper_params_ranges

    def load_hyper_params_vals(
        self, json_file_name: str = None
    ) -> Tuple[str, str, str]:
        json_file_name = json_file_name or self.json_file_name
        with open(json_file_name, "r") as json_file:
            hyper_params_vals = json.load(json_file)["hyper_params"]["values"]
        n_estimators = hyper_params_vals["n_estimators"]
        max_depth = hyper_params_vals["max_depth"]
        learning_rate = hyper_params_vals["learning_rate"]
        return (n_estimators, max_depth, learning_rate)

    def load_features_for_train(self, json_file_name: str = None) -> List[str]:
        json_file_name = json_file_name or self.json_file_name
        with open(json_file_name, "r") as json_file:
            features_for_train = json.load(json_file)["features_for_train"]
        return features_for_train
