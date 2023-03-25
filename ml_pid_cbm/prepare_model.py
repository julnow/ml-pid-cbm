"""
This module is used for preparing the handler of the ML model.
"""
import json
from typing import Dict, List, Tuple
import xgboost as xgb
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler


class PrepareModel:
    """
    Class for preparing the ML pid ModelHandler
    """

    def __init__(self, json_file_name: str, optimize_hyper_params: bool):
        self.json_file_name = json_file_name
        self.optimize_hyper_params = optimize_hyper_params

    def prepare_model_handler(
        self,
        json_file_name: str = None,
        train_test_data=None,
    ):
        """Prepares model handler for training

        Args:
            json_file_name (str, optional): Name of json file containing list of features
            for training and hyper_params info. Defaults to None.
            train_test_data (_type_, optional): Trainig_test_dataset using hipe4ml.train_test_generator. Only used for
            optimization of hyper params if this option is set to True. Defaults to None.

        Raises:
            TypeError: Raises error if optimize_hyper_params is set to True, but no train_test_data was provided.

        Returns:
            ModelHandler: Hipe4ml model handler ready for training.
        """
        json_file_name = json_file_name or self.json_file_name
        features_for_train = self.load_features_for_train(json_file_name)
        if self.optimize_hyper_params is True and train_test_data is not None:
            model_clf = xgb.XGBClassifier()
            model_hdl = ModelHandler(model_clf, features_for_train)
            hyper_params_ranges = self.load_hyper_params_ranges(json_file_name)
            study = model_hdl.optimize_params_optuna(
                train_test_data,
                hyper_params_ranges,
                cross_val_scoring="roc_auc_ovo",
                timeout=120,
                n_jobs=2,
                n_trials=2,
                direction="maximize",
            )
        elif self.optimize_hyper_params is True and train_test_data is None:
            raise TypeError("train_test_data must be defined to optimize hyper params")
        elif self.optimize_hyper_params is False:
            n_estimators, max_depth, learning_rate = self.load_hyper_params_vals(
                json_file_name
            )
            model_clf = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )
            model_hdl = ModelHandler(model_clf, features_for_train)
        print(f"\nModelHandler ready using configuration from {json_file_name}")
        if self.optimize_hyper_params:
            return model_hdl, study
        else:
            return model_hdl, None

    def load_hyper_params_ranges(
        self, json_file_name: str = None
    ) -> Dict[str, Tuple[float, float]]:
        """Loads XGBoost hyper parameters ranges for optimization from json file.

        Args:
            json_file_name (str, optional): Name of json file. Defaults to None.

        Returns:
            Dict[str, Tuple[float, float]]: Dictionary containg variables names
            for optimization and their ranges.
        """
        json_file_name = json_file_name or self.json_file_name
        with open(json_file_name, "r") as json_file:
            hyper_params_ranges = json.load(json_file)["hyper_params"]["ranges"]
        # json accepts only lists, so they need to be transformed back into tuples
        hyper_params_ranges = {k: tuple(v) for k, v in hyper_params_ranges.items()}
        return hyper_params_ranges

    def load_hyper_params_vals(
        self, json_file_name: str = None
    ) -> Tuple[str, str, str]:
        """Loads XGBoost hyper parameters values from json file to skip optimization.

        Args:
            json_file_name (str, optional): Name of json file. Defaults to None.

        Returns:
            Tuple[str, str, str]: Tuple containg n_estimators, max_depth, learning_rate.
        """
        json_file_name = json_file_name or self.json_file_name
        with open(json_file_name, "r") as json_file:
            hyper_params_vals = json.load(json_file)["hyper_params"]["values"]
        n_estimators = hyper_params_vals["n_estimators"]
        max_depth = hyper_params_vals["max_depth"]
        learning_rate = hyper_params_vals["learning_rate"]
        return (n_estimators, max_depth, learning_rate)

    def load_features_for_train(self, json_file_name: str = None) -> List[str]:
        """Load names of variables for training from json file.

        Args:
            json_file_name (str, optional): Name of json file. Defaults to None.

        Returns:
            List[str]: List of variables for training.
        """
        json_file_name = json_file_name or self.json_file_name
        with open(json_file_name, "r") as json_file:
            features_for_train = json.load(json_file)["features_for_train"]
        return features_for_train

    @staticmethod
    def prepare_train_test_data(
        protons_th: TreeHandler,
        kaons_th: TreeHandler,
        pions_th: TreeHandler,
        test_size: float = 0.1,
    ):
        """Prepares trainig_test_dataset using hipe4ml.train_test_generator

        Args:
            protons_th (TreeHandler): TreeHandler containg protons.
            kaons_th (TreeHandler): TreeHandler containg kaons.
            pions_th (TreeHandler): TreeHandler containg pions.
            test_size(float, optional): Size of created test dataset. Defaults to 0.1.

        Returns:
            List containing respectively training set dataframe,
            training label array, test set dataframe, test label array.
        """
        train_test_data = train_test_generator(
            [protons_th, kaons_th, pions_th], [0, 1, 2], test_size=test_size
        )
        return train_test_data
