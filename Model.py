# IMPORTS
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# FUNCTIONS


# Creates a convolutional network with five layers and returns it
# the first layer is the input layer to represent the image data and reshape it into a single-dimensional array
# the second layer is the convolutional layer that will extract the features from the image
# the third layer is the pooling layer that to reduce the spatial volume of the input image
# the fourth layer is the fully connected layer that connects from one layer to another
# the fifth layer is the output layer that will give the prediction
#
# Inputs
#   image_width = the width of the images for the network to take
#   image_height = the height of the images for the network to take
#   image_channels = the number of color channels of the images for the network to take
def create_model(image_width, image_height, image_channels):
    model = Sequential()    # initializes the model

    # input layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, image_channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # pooling layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # fully connected layer
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 2 because we have cat and dog classes

    # output layer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


# Defines callbacks for early stopping and learning rate reduction to optimize the model
# Inputs:
#   patience = the max number of steps the validation loss function can go without decreasing
# Outputs:
#   an array containing the following:
#   - early_stop = stops the model from fitting when the patience number is reached
#   - learning_rate_reduction = reduces the learning rate if the accuracy doesn't increase for two steps
def callbacks(patience):
    early_stop = EarlyStopping(patience=patience)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1,
                                                factor=0.5, min_lr=0.00001)
    return [early_stop, learning_rate_reduction]


# Saves the model and its weights into separate files
# Inputs:
#   model = the model you want to save
#   model_name = what you want to name the saved model
def save_model(model, model_name):
    model.save_weights("models/" + model_name + "_weights.hdf5")
    model.save("models/" + model_name + ".hdf5")
