# IMPORTS
import Model
import Dataset
import Train
import Generator

from keras.preprocessing.image import load_img
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
IMAGE_WIDTH = 200  # the width to force the image to for the network
IMAGE_HEIGHT = 200  # the height to force the image to for the network
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)  # the size of the image using the above
IMAGE_CHANNELS = 3  # the number of color channels the images will be using (three for RGB)

# GPU configuration
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

train_dir = "./train"  # defines the directory for the training set

df = Dataset.load_train_data(train_dir)  # load the training data

# Creating the network
model = Model.create_model(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)  # create the network as a model
model.summary()  # print out a summary of the network
callbacks = Model.callbacks(patience=10)  # make any optimizations needed for training

df = Dataset.replace_nums_with_animals(df)  # replace numbers with more informative labels

# Splitting the training data
train_df, validate_df = Train.split_data(df)
# split the data into training and validation sets
total_train, total_validate, batch_size = Train.define_steps(train_df, validate_df, num_batch_size=15)
# get numbers for training steps

# Generators
train_generator = Generator.training_generator(train_df, train_dir, IMAGE_SIZE, batch_size)
# create a training generator
validation_generator = Generator.validation_generator(validate_df, train_dir, IMAGE_SIZE, batch_size)
# create a validation generator

# Training step
epochs = 50  # the number of epochs to train on
history = Train.fit_model(model, train_generator, total_train, validation_generator, total_validate, batch_size,
                          callbacks, epochs)
# fit the model
Model.save_model(model, "model")        # save the model
Train.evaluate_train(history, epochs)   # evaluate the training

# Testing step
test_dir = "./test"                         # defines the directory for the testing set
test_df = Dataset.load_test_data(test_dir)  # load the testing data
total_test = test_df.shape[0]               # gives the number of test images
test_generator = Generator.test_generator(test_df, test_dir, IMAGE_SIZE, batch_size)
# create a test generator
predict = model.predict_generator(test_generator, steps=np.ceil(total_test/batch_size))
# returns the probabilities than an image belongs to each category
test_df['category'] = np.argmax(predict, axis=-1)
# sets the category in the test data equal to whichever category has the higher probability
label_map = dict((v, k) for k, v in train_generator.class_indices.items())
# labels from the generator class
test_df['category'] = test_df['category'].replace(label_map)
# replaces the testing data labels with those from the generator class

# Testing evaluation
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("./test/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()
