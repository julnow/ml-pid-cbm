import unittest
from validate_model import ValidateModel


class TestValidateModel(unittest.TestCase):
    def test_parse_model_name(self):
        model_name_positive = "model_0.0_6.0_positive"
        lower_p, upper_p, anti = ValidateModel.parse_model_name(model_name_positive)
        self.assertEqual([lower_p, upper_p, anti], [0., 6., False])
        model_name_anti = "model_3.0_6.0_anti"
        lower_p, upper_p, anti = ValidateModel.parse_model_name(model_name_anti)
        self.assertEqual([lower_p, upper_p, anti], [3., 6., True])
        model_name_incorrect = "model_anti_1_4"
        self.assertRaises(
            ValueError, lambda: ValidateModel.parse_model_name(model_name_incorrect)
        )

    if __name__ == "__main__":
        unittest.main()
