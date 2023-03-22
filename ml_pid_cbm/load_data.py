"""
Module for loading data saved in .tree format into hipe4ml.TreeHandler,
data cleaning and preparing training and test dataset.
"""

import json
from typing import Tuple
from hipe4ml.tree_handler import TreeHandler
from particles_id import ParticlesId as Pid


class LoadData:
    """
    Class for loading data stored in .tree format into hipe4ml.TreeHandler,
    data cleaning and preparing dataset for training and testing of the ML model
    """

    def __init__(
        self,
        data_file_name: str,
        json_file_name: str,
        lower_p_cut: float,
        upper_p_cut: float,
        anti_particles: bool,
    ):
        self.data_file_name = data_file_name
        self.lower_p_cut = lower_p_cut
        self.upper_p_cut = upper_p_cut
        self.anti_particles = anti_particles
        self.json_file_name = json_file_name

    def get_proton_kaons_pions(
        self,
        tree_handler: TreeHandler,
        anti_particles: bool,
        nsigma: int = 3,
        nsigma_proton: int = None,
        nsigma_kaon: int = None,
        nsigma_pion: int = None,
    ) -> Tuple[TreeHandler, TreeHandler, TreeHandler]:
        """Gets protons, kaons, and pions from TreeHandler in  nsigma region.
        In this tof model, pions, muons and electrons are treated the same

        Args:
            tree_handler (TreeHandler): _description_
            anti_particles (bool): _description_

        Returns:
            Tuple[TreeHandler, TreeHandler, TreeHandler]: Tuple of protons, kaons, and pions for training
        """
        # if not defined get default value of nsigma
        nsigma_proton = nsigma_proton or nsigma
        nsigma_kaon = nsigma_kaon or nsigma
        nsigma_pion = nsigma_pion or nsigma

        if anti_particles is False:
            protons = self.get_particles_type(
                tree_handler, Pid.PROTON.value, nsigma_proton
            )
            kaons = self.get_particles_type(
                tree_handler, Pid.POS_KAON.value, nsigma_kaon
            )
            pions = self.get_particles_type(
                tree_handler,
                #pions, muons and electrons impossible to ditinguish in this model
                [Pid.POS_PION.value, Pid.POS_MUON.value, Pid.POSITRON.value],
                nsigma_pion,
            )
        elif anti_particles is True:
            protons = self.get_particles_type(
                tree_handler, Pid.ANTI_PROTON.value, nsigma_proton
            )
            kaons = self.get_particles_type(tree_handler, Pid.NEG_KAON.value, kaons)
            pions = self.get_particles_type(
                tree_handler,
                [Pid.NEG_PION.value, Pid.NEG_MUON.value, Pid.ELECTRON.value],
                nsigma_pion,
            )
        return (protons, kaons, pions)

    def get_particles_type(
        self, tree_handler: TreeHandler, pid: float, nsigma: float = 0.0
    ) -> TreeHandler:
        """Gets particle of given pid in selected sigma region of mass2

        Args:
            tree_handler (TreeHandler): TreeHandler with the data
            pid (float): Pid of given particle type
            nsigma (float, optional): Number of sigma to select sigma region of mass2. Defaults to 0.

        Returns:
            TreeHandler: TreeHandler with given particles type in given sigma region
        """
        pid_var_name = self.__class__.load_var_name(self.json_file_name, "pid")
        mass2_var_name = self.__class__.load_var_name(self.json_file_name, "mass2")
        particles = tree_handler.get_subset(f"{pid_var_name} == {pid}")
        # getting selected nsigma region in the mass2
        if nsigma > 0:
            mass2_column = particles.get_data_frame()[mass2_var_name]
            mean = mass2_column.mean()
            std = mass2_column.std()
            mass2_cut = LoadData.create_cut_string(
                mean - nsigma * std, mean + nsigma * std, mass2_var_name
            )
            particles = particles.get_subset(mass2_cut)
        return particles

    # load file with data into hipe4ml TreeHandler
    def load_tree(
        self, data_file_name: str = None, tree_type: str = "plain_tree"
    ) -> TreeHandler:
        """Loads tree from given file into hipe4ml TreeHandler

        Args:
            data_file_name (str, optional): name of the file with the tree. Defaults to None.
            tree_type (str, optional): type of the tree structure to be loaded.
            Defaults to "plain_tree".

        Returns:
            TreeHandler: hipe4ml structure contatining tree to train and test model on
        """

        if data_file_name is None:
            data_file_name = self.data_file_name
        tree_handler = TreeHandler(data_file_name, tree_type)

        return self.clean_tree(tree_handler, self.json_file_name)

    def clean_tree(
        self, tree_handler: TreeHandler, json_file_name: str = None
    ) -> TreeHandler:
        """Takes already loaded tree from TreeHandler and makes quality cuts
            defined in the json file, and momentum cut defined in class definition

        Args:
            tree_handler (TreeHandler): TreeHandler contating tree to be cleaned
            json_file_name (str, optional): Name of the json file containg
            quality cuts definition. Defaults to None.

        Returns:
            TreeHandler: _description_
        """
        if json_file_name is None:
            json_file_name = self.json_file_name
        quality_cuts = self.load_quality_cuts(json_file_name)
        momemntum_variable_name = self.__class__.load_var_name(
            json_file_name, "momentum"
        )
        charge_variable_name = self.__class__.load_var_name(json_file_name, "charge")

        for cut in quality_cuts:
            tree_handler = tree_handler.get_subset(cut)
        # include specific momentum cut
        p_cut = self.create_cut_string(
            self.lower_p_cut, self.upper_p_cut, momemntum_variable_name
        )
        tree_handler = tree_handler.get_subset(p_cut)
        # include sign of charge
        if self.anti_particles is False:
            tree_handler = tree_handler.get_subset(f"{charge_variable_name} > 0")
        elif self.anti_particles is True:
            tree_handler = tree_handler.get_subset(f"{charge_variable_name} < 0")

        return tree_handler

    def load_quality_cuts(self, json_file_name: str):  # -> list[str]:
        """Loads quality cuts defined in json file into array of strings

        Args:
            json_filename (str): Name of the json file containg defined cuts

        Returns:
            list[str]: List of strings containg cuts definitions
        """
        with open(json_file_name, "r") as json_file:
            cuts = json.load(json_file)["cuts"]
        quality_cuts = [
            self.__class__.create_cut_string(
                cut_data["lower"], cut_data["upper"], cut_name
            )
            for cut_name, cut_data in cuts.items()
        ]
        return quality_cuts

    @staticmethod
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

    @staticmethod
    def create_cut_string(lower: float, upper: float, cut_name: str) -> str:
        """Creates cut string for hipe4ml loader in format "lower_value < cut_name < upper_value"

        Args:
            lower (float): Value of lower cut, 1 decimal place
            upper (float): Value of upper cut, 1 decimal place
            cut_name (str): Name of the cut variable

        Returns:
            str: Formatted string in format "lower_value < cut_name < upper_value"
        """
        cut_string = f"{lower:.1f} < {cut_name} < {upper:.1f}"
        return cut_string
