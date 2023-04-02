import unittest
from validate_model import ValidateModel


class TestValidateModel(unittest.TestCase):
    def test_parse_model_name(self):
        model_name_positive = "model_0_3_positive"
        lower_p, upper_p, anti = ValidateModel.parse_model_name(model_name_positive)
        self.assertEqual([lower_p, upper_p, anti], [0, 3, False])
        model_name_anti = "model_-4_-1_anti"
        lower_p, upper_p, anti = ValidateModel.parse_model_name(model_name_anti)
        self.assertEqual([lower_p, upper_p, anti], [-4, -1, True])
        model_name_incorrect = "model_anti_1_4"
        self.assertRaises(
            ValueError, lambda: ValidateModel.parse_model_name(model_name_incorrect)
        )

    if __name__ == "__main__":
        unittest.main()
