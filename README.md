# fashion-classification
Experimentation with classifying fashion images using [PyTorch](https://pytorch.org/) as the data science backend and [Streamlit](https://www.streamlit.io/) as the frontend. Dataset used to train this comes from [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). Additional support for arbitrary image as a test-holdout set can come from anywhere - I'll be using images from [Zersten](http://zerstenapparel.com/) as a test of the application and classifier.

## Repo structure
### Model
Contains training and setup for the model itself.  
### Data
Contains data gathering steps in addition to the raw data.
### App
Contains everything needed for the Streamlit app to test the model and classify arbitrary images.
