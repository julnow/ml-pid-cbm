import argparse
import os
import sys
from shutil import copy2
from typing import List

from ml_pid_cbm.gauss.validate_gauss import ValidateGauss
from ml_pid_cbm.validate_multiple_models import ValidateMultipleModels


class ValidateMultipleGauss(ValidateMultipleModels, ValidateGauss):
    """
    Class for validating data from multiple models.
    Inherits from ValidateModel
    """


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Arguments parser for the main method.

    Args:
        args (List[str]): Arguments from the command line, should be sys.argv[1:].

    Returns:
        argparse.Namespace: argparse.Namespace containg args
    """
    parser = argparse.ArgumentParser(
        prog="ML_PID_CBM ValidateMultipleGauss",
        description="Program for loading multiple validated PID Gauss models",
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
    validate = ValidateMultipleGauss(json_file_name, pickle_files, n_workers)
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
