import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from shutil import copy2
from typing import List, Set

import pandas as pd

from .validate_model import ValidateModel


class ValidateMultipleModels(ValidateModel):
    """
    Class for validating data from multiple models.
    Inherits from ValidateModel
    """

    def __init__(self, json_file_name: str, files_list: Set[str], n_workers: int = 1):
        super().__init__(-12, 12, False, json_file_name, None)
        self.particles_df = self.load_pickles(files_list, n_workers)

    @staticmethod
    def load_pickles(files_list: Set[str], n_workers: int = 1) -> pd.DataFrame:
        """Loads multiple pickle files produced by validate_model module.

        Args:
            files_list (Set[str]): Files list containg picle files with datasets.
            n_workers (int, optional): Number of workers for multithreading. Defaults to 1.

        Returns:
            pd.DataFrame: Dataframe with merged datasets.
        """
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(pd.read_pickle, files_list))
            whole_df = pd.concat(results, ignore_index=True)
        return whole_df


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Arguments parser for the main method.

    Args:
        args (List[str]): Arguments from the command line, should be sys.argv[1:].

    Returns:
        argparse.Namespace: argparse.Namespace containg args
    """
    parser = argparse.ArgumentParser(
        prog="ML_PID_CBM ValidateMultipleModels",
        description="Program for loading multiple validated PID ML models",
    )
    parser.add_argument(
        "--modelnames",
        "-m",
        nargs="+",
        required=True,
        type=str,
        help="Names of folders containing trained and validated ML models.",
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
        "--nworkers",
        "-n",
        type=int,
        default=1,
        help="Max number of workers for ThreadPoolExecutor which reads Root tree with data.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # config  arguments to be loaded from args
    json_file_name = args.config[0]
    models = args.modelnames
    n_workers = args.nworkers
    pickle_files = {f"{model}/validated_data.pickle" for model in models}
    validate = ValidateMultipleModels(json_file_name, pickle_files, n_workers)
    # new folder for all files
    json_file_path = os.path.join(os.getcwd(), json_file_name)
    if not os.path.exists("all_models"):
        os.makedirs("all_models")
    os.chdir("all_models")
    copy2(json_file_path, os.getcwd())
    # graphs
    validate.confusion_matrix_and_stats()
    print("Generating plots...")
    validate.generate_plots()
