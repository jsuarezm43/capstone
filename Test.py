import Model
import Dataset
import Generator
import Train

import unittest
import os
import tensorflow as tf
import numpy as np

train_dir = "./SampleTrain"
test_dir = "./SampleTest"
image_height = 200
image_width = 200
image_size = (image_width, image_height)
image_channels = 3
epochs = 3


def count_files(directory):
    path, dirs, files = next(os.walk(directory))
    file_count = len(files)
    return file_count


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.df = Dataset.load_train_data(train_dir)
        self.test_df = Dataset.load_test_data(test_dir)
        self.model = Model.create_model(image_width, image_height, image_channels)

        self.df = Dataset.replace_nums_with_animals(self.df)
        self.train_df, self.validate_df = Train.split_data(self.df)

        self.total_train, self.total_valid, self.batch_size = Train.define_steps(self.train_df, self.validate_df, 15)

        self.train_gen = Generator.training_generator(self.train_df, train_dir, image_size, self.batch_size)
        self.validate_gen = Generator.validation_generator(self.validate_df, train_dir, image_size, self.batch_size)
        self.test_gen = Generator.test_generator(self.test_df, test_dir, image_size, self.batch_size)

        self.callbacks = Model.callbacks(10)
        self.history = Train.fit_model(self.model, self.train_gen, self.total_train, self.validate_gen,
                                       self.total_valid, self.batch_size, self.callbacks, epochs)
        Train.evaluate_train(self.history, epochs)
        Model.save_model(self.model, "test_model")

    def test_load_train_data(self):
        self.assertEqual(self.df.shape[0], count_files(train_dir), "Not all of the training images were loaded.")

    def test_load_test_data(self):
        self.assertEqual(self.test_df.shape[0], count_files(test_dir), "Not all of the testing images were loaded.")


if __name__ == '__main__':
    unittest.main()

