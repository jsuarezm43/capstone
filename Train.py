# IMPORTS
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# FUNCTIONS


# Splits a data frame into a training set and a validation set
# Inputs:
#   df = the data frame with the data you want to split
# Outputs:
#   train_df = the training set as a data frame
#   validate_df = the validation set as a data frame
def split_data(df):
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    return train_df, validate_df


# Defines the steps for training
# Inputs:
#   train_df = the training data
#   validate_df = the validation data
#   num_batch_size = the desired batch_size
# Outputs:
#   total_train = the number of images in the training set
#   total_validate = the number of images in the validation set
#   batch_size = the size of the batches to use
def define_steps(train_df, validate_df, num_batch_size):
    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    batch_size = num_batch_size
    return total_train, total_validate, batch_size


# Fits the model
# Inputs:
#   model = the model to fit on
#   train_data = the training set
#   total_train = the total number of images in the training set
#   validate_data = the validation set
#   total_validate = the total number of images in the validation set
#   batch_size = the batch size
#   callbacks_to_use = the callbacks to use for optimization purposes
#   epochs = the number of epochs to use for training
# Outputs:
#   history = the fitted model
def fit_model(model, train_data, total_train, validate_data, total_validate, batch_size, callbacks_to_use, epochs):
    history = model.fit_generator(train_data, epochs=epochs, validation_data=validate_data,
                                  validation_steps=total_validate // batch_size,
                                  steps_per_epoch=total_train // batch_size, callbacks=callbacks_to_use)
    return history


# Evaluates training via plotting the loss functions for the model and the model accuracy
# Inputs:
#   history = the fitted model
#   epochs = the number of epochs the model was trained on
def evaluate_train(history, epochs):
    # plotting loss function
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    # plotting accuracy
    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()
