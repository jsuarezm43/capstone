# IMPORTS
import pandas as pd
import os

# FUNCTIONS


# Loads the training data from a specified directory and returns a data frame with the classified data in it
#
# Inputs:
#   directory = the specified directory to take the images from
# Outputs:
#   df = the training data as a data frame
def load_train_data(directory):
    filenames = os.listdir(directory)   # the files in the specified directory
    categories = []                     # the categories to classify the images by

    # do the following for every image in the directory
    for filename in filenames:
        category = filename.split('.')[0]   # grabs the first part of the file name that specifies 'cat' or 'dog'

        # classify according to category
        if category == 'dog':
            categories.append(1)    # if dog, classify as a 1
        else:
            categories.append(0)    # if cat, classify as a 0

    # save everything to a data frame
    df = pd.DataFrame({
        'filename': filenames,  # the image
        'category': categories  # whether the image is of a cat or a dog
    })
    return df


# Loads the testing data from a specified directory and returns a data frame with the data in it
# Inputs:
#   directory = the specified directory to take the images from
# Outputs:
#   test_df = the testing data as a data frame
def load_test_data(directory):
    test_filenames = os.listdir(directory)  # the files in the specified directory

    # place the images in a data frame
    test_df = pd.DataFrame({
        'filename': test_filenames  # the image
    })
    return test_df


# replaces the numbers in the category for a data frame with the animal
#
# Inputs:
#   df = the data frame to apply the operation to
# Outputs:
#   df = input df after the transformation is applied
def replace_nums_with_animals(df):
    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
    return df
