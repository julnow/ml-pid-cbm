"""
Moudule with enum of known PID in MONTE CARLO PARTICLE NUMBERING SCHEME
More details: https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
"""

from enum import Enum


class ParticlesId(Enum):
    """
    Enum class containg known PID in Monte Carlo format
    """

    POSITRON = -11.0
    ELECTRON = 11.0
    PROTON = 2212.0
    ANTI_PROTON = -2212.0
    NEG_MUON = -13.0
    POS_MUON = 13.0
    NEG_PION = -211.0
    POS_PION = 211.0
    NEG_KAON = -321.0
    POS_KAON = 321.0

    @classmethod
    def is_known_particle(cls, pid: float) -> bool:
        """hecks if given pid is a known particle in this class

        Args:
            pid (float): Pid to be checked if in list

        Returns:
            bool: Returns True, if given pid belongs to class
        """
        return pid in cls._value2member_map_
