"""
This module is used for preparing the handler of the ML model.
"""
import json
from typing import Dict, Tuple, Union

import xgboost as xgb
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from optuna.study import Study
import json_tools


class PrepareModel:
    """
    Class for preparing the ML pid ModelHandler
    """

    def __init__(self, json_file_name: str, optimize_hyper_params: bool, use_gpu: bool):
        self.json_file_name = json_file_name
        self.optimize_hyper_params = optimize_hyper_params
        self.use_gpu = use_gpu

    def prepare_model_handler(
        self,
        json_file_name: str = None,
        train_test_data=None,
    ) -> Tuple[ModelHandler, Union[None, Study]]:
        """
        Prepares model handler for training

        Args:
            json_file_name (str, optional): Name of json file containing list of features. Defaults to None.
            train_test_data (_type_, optional): Trainig_test_dataset using hipe4ml.train_test_generator. Only used for
            optimization of hyper params if this option is set to True. Defaults to None.

        Raises:
            TypeError: Raises error if optimize_hyper_params is set to True, but no train_test_data was provided.

        Returns:
            Tuple[ModelHandler, Union[None, Study]]: Tuple of Hipe4ml model handler ready for training,
            and optuna optuna.study.Study if was performed.
        """
        json_file_name = json_file_name or self.json_file_name
        features_for_train = json_tools.load_features_for_train(json_file_name)
        if self.use_gpu:
            tree_method = "gpu_hist"
        else:
            tree_method = "hist"
        if self.optimize_hyper_params is True and train_test_data is not None:
            model_clf = xgb.XGBClassifier(tree_method=tree_method)
            model_hdl = ModelHandler(model_clf, features_for_train)
            hyper_params_ranges = self.load_hyper_params_ranges(json_file_name)
            study = model_hdl.optimize_params_optuna(
                train_test_data,
                hyper_params_ranges,
                cross_val_scoring="roc_auc_ovo",
                timeout=120,
                n_jobs=4,
                n_trials=3,
                direction="maximize",
            )
        elif self.optimize_hyper_params is True and train_test_data is None:
            raise TypeError("train_test_data must be defined to optimize hyper params")
        elif self.optimize_hyper_params is False:
            n_estimators, max_depth, learning_rate = json_tools.load_hyper_params_vals(
                json_file_name
            )
            model_clf = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                tree_method=tree_method,
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

    @staticmethod
    def prepare_train_test_data(
        protons_th: TreeHandler,
        kaons_th: TreeHandler,
        pions_th: TreeHandler,
        test_size: float = 0.2,
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
