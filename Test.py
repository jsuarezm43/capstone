import Model
import Dataset

import unittest
import os
import tensorflow as tf
import numpy as np

train_dir = "./SampleTrain"
test_dir = "./SampleTest"


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.train_df = Dataset.load_train_data(train_dir)
        self.test_df = Dataset.load_test_data(test_dir)
        self.model = Model.create_model(200, 200, 3)

    def test_load_train_data(self):
        self.assertEqual(self.train_df.shape[0], count_files(train_dir), "Not all of the training images were loaded.")

    def test_load_test_data(self):
        self.assertEqual(self.test_df.shape[0], count_files(test_dir), "Not all of the testing images were loaded.")


if __name__ == '__main__':
    unittest.main()


def count_files(directory):
    path, dirs, files = next(os.walk(directory))
    file_count = len(files)
    return file_count

