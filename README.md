# fashion-classification
Experimentation with classifying fashion images using [PyTorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/)as the data science backend and [Streamlit](https://www.streamlit.io/) as the frontend. Dataset used to train this comes from [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). I used [this tutorial](https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582) to get me started with PyTorch and [this tutorial](https://www.tensorflow.org/tutorials/images/cnn) to get me up-to-speed on CNNs in Tensorflow. Additional support for arbitrary image as a test-holdout set can come from anywhere - I'll be using images from [Zersten](http://zerstenapparel.com/) as a test of the application and classifier. You can use images of clothing from anywhere in the app yourself!

## Repo structure

### Model
Contains training and setup for the models themself. There is a `saved_models` folder here containing saved tf and PyTorch models that will be compared in the app.
### Data
Contains data gathering steps in addition to the raw data. Primarily used for raw data storage. The training data portion is empty in the repo; it's simply available as a convenient location to point Tensorflow and PyTorch at for downloading their train and test datasets.
### App
Contains everything needed for the Streamlit app to test the model and classify arbitrary images. You can run it yourself!

## How to run the Streamlit app
1. Install the required libraries (virtual environment usage recommended) using `pip install -r requirements.txt`. 
2. Open your terminal or Command Prompt and run the app via `streamlit run app/app.py`. Streamlit should open a page in your web browser of choice.
3. Classify your image!

![Example image of the app](https://github.com/G-Benn/fashion-classification/tree/master/app/example_img.JPG)
