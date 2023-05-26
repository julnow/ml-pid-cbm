import unittest

from .particles_id import ParticlesId


class TestParticlesId(unittest.TestCase):
    def test_is_known_particle(self):
        # proton is in enum class
        self.assertTrue(ParticlesId.is_known_particle(211))
        # Delta (PID=3122) is not in enum class
        self.assertFalse(ParticlesId.is_known_particle(3122))
