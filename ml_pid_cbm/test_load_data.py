
import unittest
from load_data import LoadData
from unittest.mock import patch, mock_open

class TestLoadData(unittest.TestCase):

    def setUp(self):
        self.loader_1 = LoadData(data_file_name = "data.tree", json_file_name = "config.json", lower_p_cut = 0., upper_p_cut = 1., anti_particles = False)


    def test_create_cut_string(self):
        #ideally formatted
        expected_string = "0.1 < test_cut < 13.0"
        self.assertEqual(LoadData.create_cut_string(.1, 13., "test_cut"), expected_string)
        #not ideally formatted
        self.assertEqual(LoadData.create_cut_string(.141, 13, "test_cut"), expected_string)

            
    def test_load_quality_cuts(self):
        json_data = """{"cuts":{"Complex_mass2": {"lower": -1.0,"upper": 2.0},"Complex_pT": {"lower": 0.0,"upper": 2.0}}}"""
        #mocking json file for testing
        with patch("builtins.open", mock_open(read_data=json_data)):
            quality_cuts = self.loader_1.load_quality_cuts("test.json")
            expected_cuts = ["-1.0 < Complex_mass2 < 2.0", "0.0 < Complex_pT < 2.0"]
            self.assertEqual(quality_cuts, expected_cuts)
            
    if __name__ == '__main__':
        unittest.main()
