import unittest
import numpy as np
from LoadData import load_data, split_data


class TestNN(unittest.TestCase):

    def testload_data(self):
        """Test function load_data"""
        data_shape_expected = (17379, 15)
        csv = "./Bike-Sharing-Dataset/hour.csv"
        data = load_data(csv)
        data_shape_result = data.shape
        self.assertEqual(data_shape_result, data_shape_expected)

    def testsplit_data(self):
        """Test function split_data."""
        input_hour_shape_expected = (10, 14)
        label_hour_shape_expected = (10,)
        data = np.arange(150).reshape((10, 15))
        input_hour, label_hour = split_data(data)
        input_hour_shape_result = input_hour.shape
        label_hour_shape_result = label_hour.shape
        self.assertEqual(input_hour_shape_result, input_hour_shape_expected)
        self.assertEqual(label_hour_shape_result, label_hour_shape_expected)


if __name__ == "__main__":
    unittest.main()
