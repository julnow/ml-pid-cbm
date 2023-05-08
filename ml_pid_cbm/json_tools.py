"""
Module for operating the json config file.
"""

import json
from typing import List, Tuple


def create_cut_string(lower: float, upper: float, cut_name: str) -> str:
    """Creates cut string for hipe4ml loader in format "lower_value < cut_name < upper_value"

    Args:
        lower (float): Value of lower cut, 1 decimal place
        upper (float): Value of upper cut, 1 decimal place
        cut_name (str): Name of the cut variable

    Returns:
        str: Formatted string in format "lower_value < cut_name < upper_value"
    """
    cut_string = f"{lower:.1f} <= {cut_name} < {upper:.1f}"
    return cut_string


def load_quality_cuts(json_file_name: str) -> List[str]:
    """Loads quality cuts defined in json file into array of strings

    Args:
        json_filename (str): Name of the json file containg defined cuts

    Returns:
        list[str]: List of strings containg cuts definitions
    """
    with open(json_file_name, "r") as json_file:
        cuts = json.load(json_file)["cuts"]
    quality_cuts = [
        create_cut_string(cut_data["lower"], cut_data["upper"], cut_name)
        for cut_name, cut_data in cuts.items()
    ]
    return quality_cuts


def load_var_name(json_file_name: str, var: str) -> str:
    """Loads physical variable name used in tree from json file.

    Args:
        json_file_name (str): Name of the json file with var_names
        var (str): Physical variable we look for

    Returns:
        str: Name of physical variable in our tree structure loaded from json file
    """
    with open(json_file_name, "r") as json_file:
        var_names = json.load(json_file)["var_names"]
    return var_names[var]


def load_file_name(json_file_name: str, training_or_test: str):
    """Load file names of both training and test dataset

    Args:
        json_file_name (str): Json file containg filenames.
        training_or_test (str): Name of the dataset (e.g., "test", "training") as defined
        in json to load the dataset filename.

    Returns:
        _type_: _description_
    """
    with open(json_file_name, "r") as json_file:
        var_names = json.load(json_file)["file_names"]
    return var_names[training_or_test]


def load_features_for_train(json_file_name: str) -> List[str]:
    """Load names of variables for training from json file.

    Args:
        json_file_name: Name of json file.

    Returns:
        List[str]: List of variables for training.
    """
    with open(json_file_name, "r") as json_file:
        features_for_train = json.load(json_file)["features_for_train"]
    return features_for_train


def load_hyper_params_vals(json_file_name: str) -> Tuple[str, str, str]:
    """Loads XGBoost hyper parameters values from json file to skip optimization.

    Args:
        json_file_name: Name of json file. 

    Returns:
        Tuple[str, str, str]: Tuple containg n_estimators, max_depth, learning_rate.
    """
    with open(json_file_name, "r") as json_file:
        hyper_params_vals = json.load(json_file)["hyper_params"]["values"]
    n_estimators = hyper_params_vals["n_estimators"]
    max_depth = hyper_params_vals["max_depth"]
    learning_rate = hyper_params_vals["learning_rate"]
    return (n_estimators, max_depth, learning_rate)
