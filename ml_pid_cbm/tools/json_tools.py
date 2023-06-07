"""
Module for operating the json config file.
"""

import json
from typing import Dict, List, Tuple


def create_cut_string(lower: float, upper: float, cut_name: str) -> str:
    """
    Creates a cut string for hipe4ml loader in the format "lower_value < cut_name < upper_value".

    Parameters
    ----------
    lower : float
        Value of the lower cut, rounded to 1 decimal place.
    upper : float
        Value of the upper cut, rounded to 1 decimal place.
    cut_name : str
        Name of the cut variable.

    Returns
    -------
    str
        Formatted string in the format "lower_value < cut_name < upper_value".
    """
    cut_string = f"{lower:.1f} <= {cut_name} < {upper:.1f}"
    return cut_string


def load_quality_cuts(json_file_name: str) -> List[str]:
    """
    Loads quality cuts defined in a JSON file into an array of strings.

    Parameters
    ----------
    json_file_name : str
        Name of the JSON file containing defined cuts.

    Returns
    -------
    List[str]
        List of strings containing cuts definitions.
    """
    with open(json_file_name, "r") as json_file:
        data: Dict[str, Dict[str, float]] = json.load(json_file)

    cuts = data["cuts"]
    quality_cuts = [
        create_cut_string(cut_data["lower"], cut_data["upper"], cut_name)
        for cut_name, cut_data in cuts.items()
    ]
    return quality_cuts


def load_var_name(json_file_name: str, var: str) -> str:
    """
    Loads the physical variable name used in the tree from a JSON file.

    Parameters
    ----------
    json_file_name : str
        Name of the JSON file with var_names.
    var : str
        Physical variable we look for.

    Returns
    -------
    str
        Name of the physical variable in our tree structure loaded from the JSON file.
    """
    with open(json_file_name, "r") as json_file:
        var_names: Dict[str, str] = json.load(json_file)["var_names"]
    return var_names[var]


def load_file_name(json_file_name: str, training_or_test: str) -> str:
    """
    Loads the file names of both the training and test datasets.

    Parameters
    ----------
    json_file_name : str
        JSON file containing filenames.
    training_or_test : str
        Name of the dataset (e.g., "test", "training") as defined in the JSON file to load the dataset filename.

    Returns
    -------
    str
        Filename of the specified dataset.
    """
    with open(json_file_name, "r") as json_file:
        var_names: Dict[str, str] = json.load(json_file)["file_names"]
    return var_names[training_or_test]


def load_features_for_train(json_file_name: str) -> List[str]:
    """
    Load names of variables for training from a JSON file.

    Parameters
    ----------
    json_file_name : str
        Name of the JSON file.

    Returns
    -------
    List[str]
        List of variables for training.
    """
    with open(json_file_name, "r") as json_file:
        features_for_train = json.load(json_file)["features_for_train"]
    return features_for_train


def load_vars_to_draw(json_file_name: str) -> List[str]:
    """
    Load names of variables to draw from a JSON file.

    Parameters
    ----------
    json_file_name : str
        Name of the JSON file.

    Returns
    -------
    List[str]
        List of variables to draw.
    """
    with open(json_file_name, "r") as json_file:
        vars_to_draw = json.load(json_file)["vars_to_draw"]
    return vars_to_draw


def load_hyper_params_vals(json_file_name: str) -> Tuple[str, str, str]:
    """
    Loads XGBoost hyperparameters values from a JSON file to skip optimization.

    Parameters
    ----------
    json_file_name : str
        Name of the JSON file.

    Returns
    -------
    Tuple[str, str, str]
        Tuple containing n_estimators, max_depth, and learning_rate.
    """
    with open(json_file_name, "r") as json_file:
        hyper_params_vals = json.load(json_file)["hyper_params"]["values"]
    n_estimators = hyper_params_vals["n_estimators"]
    max_depth = hyper_params_vals["max_depth"]
    learning_rate = hyper_params_vals["learning_rate"]
    return n_estimators, max_depth, learning_rate
