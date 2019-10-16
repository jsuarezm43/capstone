# IMPORTS
import os
import numpy as np

from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename

from keras.models import load_model
from keras.preprocessing import image
from keras import backend as k

# CONSTANTS
MODEL_PATH = "./models/model.hdf5"          # file for the model

IMAGE_WIDTH = 200                           # width to use for images
IMAGE_HEIGHT = 200                          # height to use for images
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)    # size to use for images

# Flask set-up
app = Flask(__name__)

# Load up the model
k.clear_session()
model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Start serving...')


# Uses the model to predict whether an image is of a cat or a dog
# Inputs:
#   img_path = the path where the image file is located
#   my_model = the model to make the prediction with
# Outputs:
#   result[0] = 1 if image is of a dog; 0 if image is of a cat
def model_predict(img_path, my_model):
    img = image.load_img(img_path, False, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = x.reshape((1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    predict = my_model.predict(x, batch_size=1, verbose=0)
    result = np.argmax(predict, axis=-1)
    return result[0]


# Render main page
@app.route('/', methods=['GET'])
def index():
    return render_template('Index.html')


# Perform actions when a file is uploaded
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']                  # get the file from the post request

        # Save the file to the uploads folder
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        result = model_predict(file_path, model)    # make the prediction on the uploaded image

        # Return a result based on the prediction
        if result == 1:
            return "Dog"    # 1 means dog
        else:
            return "Cat"    # 0 means cat
    return None


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3000))
    app.run(host='127.0.0.1', port=port, threaded=False)
