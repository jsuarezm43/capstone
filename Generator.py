from keras.preprocessing.image import ImageDataGenerator


# FUNCTIONS


# Creates a training generator to make more images using the training set
# Inputs:
#   train_df = the training data set
#   train_dir = the directory where the training data is
#   image_size = the image size to make the transformations for
#   batch_size = the batch size to make the images for
# Outputs:
#   train_generator = the generator for the training data
def training_generator(train_df, train_dir, image_size, batch_size):
    train_datagen = ImageDataGenerator(rotation_range=15, rescale=1. / 255, shear_range=0.1, zoom_range=0.2,
                                       horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    train_generator = train_datagen.flow_from_dataframe(train_df, train_dir, x_col='filename', y_col='category',
                                                        target_size=image_size, class_mode='categorical',
                                                        batch_size=batch_size)
    return train_generator


# Creates a validation generator to make more images using the validation set
# Inputs:
#   validate_df = the validation data set
#   train_dir = the directory where the training data is
#   image_size = the image size to make the transformations for
#   batch_size = the batch size to make the images for
# Outputs:
#   valid_generator = the generator for the validation data
def validation_generator(validate_df, train_dir, image_size, batch_size):
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = validation_datagen.flow_from_dataframe(validate_df, train_dir, x_col='filename',
                                                             y_col='category', target_size=image_size,
                                                             class_mode='categorical', batch_size=batch_size)
    return valid_generator


# Creates a testing generator to make more images using the test set
# Inputs:
#   test_df = the test data set
#   test_dir = the directory where the test data is
#   image_size = the image size to make the transformations for
#   batch_size = the batch size to make the images for
# Outputs:
#   testing_generator = the generator for the test data
def test_generator(test_df, test_dir, image_size, batch_size):
    test_gen = ImageDataGenerator(rescale=1. / 255)
    testing_generator = test_gen.flow_from_dataframe(test_df, test_dir, x_col='filename', y_col=None, class_mode=None,
                                                     target_size=image_size, batch_size=batch_size, shuffle=False)
    return testing_generator
