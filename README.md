# fashion-classification
Experimentation with classifying fashion images using [PyTorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/)as the data science backend and [Streamlit](https://www.streamlit.io/) as the frontend. Dataset used to train this comes from [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). I used [this tutorial](https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582) to get me started with PyTorch and [this tutorial](https://www.tensorflow.org/tutorials/images/cnn) to get me up-to-speed on CNNs in Tensorflow. Additional support for arbitrary image as a test-holdout set can come from anywhere - I'll be using images from [Zersten](http://zerstenapparel.com/) as a test of the application and classifier.

## Repo structure
### Model
Contains training and setup for the models themself. There is a `saved_models` folder here containing saved tf and PyTorch models that will be compared in the app.  
### Data
Contains data gathering steps in addition to the raw data. Primarily used for raw data storage.
### App
Contains everything needed for the Streamlit app to test the model and classify arbitrary images.
