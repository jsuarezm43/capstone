"# capstone" 

This is my captstone project. It is for an image classifier that tells apart cats and dogs. It contains the following Python files:
- App.py - runs the deployed the model
- Config.py - creates, trains, and tests model
- Dataset.py - has all the methods relating to loading and manipulating the images themselves
- Generator.py - has all the methods relating to creating image generators to make transformations for the images
- Model.py - has all the methods relating to creating and using the model
- Train.py - has all the methods relating to training the model
- Test.py - unit tests for the project

In addition, I also have two notebooks: one for data exploration (Cat_and_dog_data_exploration.ipynb) and one for working with the model itself (Cat_and_dog_image_classifier.ipynb).

There are also some html files, a css file, and a js file to help with the running of the deployment.

To use this, first run Config.py to create a image classifier model. Next, run the App.py to create the API to use the image classifier on your own images. There is a Dockerfile that should be able to do this for you.

This image classifier is able to tell the difference between cats and dogs with about 90% accuracy.
