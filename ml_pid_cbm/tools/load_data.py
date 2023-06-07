"""
Module for loading data saved in .tree format into hipe4ml.TreeHandler,
data cleaning and preparing training and test dataset.
"""

from typing import Tuple

from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler

from . import json_tools
from .particles_id import ParticlesId as Pid


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
        """
        Initializes the LoadDataObject

        Parameters
        ----------
        data_file_name : str
             Name of the data file in .tree format.

        json_file_name : str
             Name of the JSON file containing variable names and cuts definitions.

        lower_p_cut : float
            Value of the lower momentum cut.

        upper_p_cut : float
            Value of the upper momentum cut.

        anti_particles : bool
            Specifies whether to load only antiparticles (True) or positive particles (False).
        """
        self.data_file_name = data_file_name
        self.lower_p_cut = lower_p_cut
        self.upper_p_cut = upper_p_cut
        self.anti_particles = anti_particles
        self.json_file_name = json_file_name

    def get_protons_kaons_pions(
        self,
        tree_handler: TreeHandler,
        nsigma: float = 5,
        anti_particles: bool = None,
        nsigma_proton: float = None,
        nsigma_kaon: float = None,
        nsigma_pion: float = None,
        json_file_name: str = None,
    ) -> Tuple[TreeHandler, TreeHandler, TreeHandler]:
        """
        Gets protons, kaons, and pions from a TreeHandler in the nsigma region.

        In this tof model, pions, muons, and electrons are treated the same.

        Parameters
        ----------
        tree_handler : TreeHandler
            TreeHandler containing the data.

        nsigma : float, optional
            Number of sigma for data cleaning, by default 5.

        anti_particles : bool, optional
            Loads only antiparticles if set to True, positive particles if set to False.
            Defaults to None.

        nsigma_proton : float, optional
            Number of sigma for protons, if not specified uses nsigma.
            Defaults to None.

        nsigma_kaon : float, optional
            Number of sigma for kaons, if not specified uses nsigma.
            Defaults to None.

        nsigma_pion : float, optional
            Number of sigma for pions, if not specified uses nsigma.
            Defaults to None.

        json_file_name : str, optional
            Name of the JSON file containing variable names, by default None.

        Returns
        -------
        Tuple[TreeHandler, TreeHandler, TreeHandler]
            Tuple containing TreeHandlers for protons, kaons, and pions.
        """
        anti_particles = anti_particles or self.anti_particles
        nsigma_proton = nsigma_proton if nsigma_proton is not None else nsigma
        nsigma_kaon = nsigma_kaon if nsigma_kaon is not None else nsigma
        nsigma_pion = nsigma_pion if nsigma_pion is not None else nsigma
        json_file_name = json_file_name or self.json_file_name

        if anti_particles is False:
            protons = self.get_particles_type(
                tree_handler, Pid.PROTON.value, nsigma_proton, json_file_name
            )
            kaons = self.get_particles_type(
                tree_handler, Pid.POS_KAON.value, nsigma_kaon, json_file_name
            )
            pions = self.get_particles_type(
                tree_handler,
                # pions, muons and electrons impossible to ditinguish in this model
                [Pid.POS_PION.value, Pid.POS_MUON.value, Pid.POSITRON.value],
                nsigma_pion,
                json_file_name,
            )
        elif anti_particles is True:
            protons = self.get_particles_type(
                tree_handler, Pid.ANTI_PROTON.value, nsigma_proton, json_file_name
            )
            kaons = self.get_particles_type(
                tree_handler, Pid.NEG_KAON.value, nsigma_kaon
            )
            pions = self.get_particles_type(
                tree_handler,
                [Pid.NEG_PION.value, Pid.NEG_MUON.value, Pid.ELECTRON.value],
                nsigma_pion,
                json_file_name,
            )
        print(
            f"\nNumber of protons: {len(protons)}\nNumber of kaons: {len(kaons)}\nNumber of pions: {len(pions)}"
        )
        return (protons, kaons, pions)

    def get_particles_type(
        self,
        tree_handler: TreeHandler,
        pid: float,
        nsigma: float = 0.0,
        json_file_name: str = None,
    ) -> TreeHandler:
        """
        Gets particles of a given pid in the selected sigma region of mass2.

        Parameters:
            tree_handler (TreeHandler): TreeHandler with the data.

            pid (float): Pid of the given particle type.

            nsigma (float, optional): Number of sigma to select the sigma region of mass2. Defaults to 0.

            json_file_name (str, optional): Name of the JSON file containing variable names. Defaults to None.

        Returns:
            TreeHandler: TreeHandler with the particles of the given type in the specified sigma region.
        """
        json_file_name = json_file_name or self.json_file_name
        pid_var_name = json_tools.load_var_name(json_file_name, "pid")
        mass2_var_name = json_tools.load_var_name(json_file_name, "mass2")
        particles = tree_handler.get_subset(f"{pid_var_name} == {pid}")
        # getting selected nsigma region in the mass2
        if nsigma > 0:
            print(f"Getting particles pid={pid} in {nsigma}-sigma region")
            mass2_column = particles.get_data_frame()[mass2_var_name]
            mean = mass2_column.mean()
            std = mass2_column.std()
            if std > 0:
                mass2_cut = json_tools.create_cut_string(
                    mean - nsigma * std, mean + nsigma * std, mass2_var_name
                )
                particles = particles.get_subset(mass2_cut)

        return particles

    def load_tree(
        self,
        data_file_name: str = None,
        tree_type: str = "plain_tree",
        max_workers: int = 1,
        model_handler: ModelHandler = None,
    ) -> TreeHandler:
        """
        Loads tree from given file into hipe4ml TreeHandler.

        Parameters:
            data_file_name (str, optional): Name of the file with the tree. Defaults to None.

            tree_type (str, optional): Type of the tree structure to be loaded. Defaults to "plain_tree".

            max_workers (int, optional): Number of max_workers for ThreadPoolExecutor used to load data with multithreading.
                Defaults to 1.
                
            model_handler (ModelHandler, optional): ModelHandler to apply if the dataset is validation one. Defaults to None.

        Returns:
            TreeHandler: hipe4ml structure containing the tree to train and test the model on.
        """
        data_file_name = data_file_name or self.data_file_name
        tree_handler = TreeHandler()
        preselection = self.clean_tree()
        tree_handler.get_handler_from_large_file(
            data_file_name,
            tree_type,
            preselection=preselection,
            max_workers=max_workers,
            model_handler=model_handler,
            output_margin=False,
        )
        print(f"\nLoading tree from {data_file_name}...")
        return tree_handler

    def clean_tree(self, json_file_name: str = None) -> str:
        """
        Creates a string with preselections (quality cuts, momentum range, and sign of charge).

        Parameters:
            json_file_name (str, optional): Name of the JSON file containing quality cuts definition
                (if different than in the class). Defaults to None.

        Returns:
            str: Preselection string for the the TreeHandler object.
        """
        preselection = ""
        json_file_name = json_file_name or self.json_file_name
        quality_cuts = json_tools.load_quality_cuts(json_file_name)
        momemntum_variable_name = json_tools.load_var_name(json_file_name, "momentum")
        charge_variable_name = json_tools.load_var_name(json_file_name, "charge")

        for cut in quality_cuts:
            preselection += f"({cut}) and "
        # include specific momentum cut
        p_cut = json_tools.create_cut_string(
            self.lower_p_cut, self.upper_p_cut, momemntum_variable_name
        )
        preselection += f"({p_cut}) and "
        # include sign of charge
        if self.anti_particles is False:
            preselection += f"({charge_variable_name} > 0)"
        elif self.anti_particles is True:
            preselection += f"({charge_variable_name} < 0)"

        return preselection
