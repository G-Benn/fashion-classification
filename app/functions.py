import streamlit
import os
import torch
import torchvision
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import torch.nn as nn
import torch.nn.functional as F

from PIL import Image


# The Torch NN, needed to load it in
class Network(nn.Module):
  def __init__(self):
    super().__init__()

    # define layers
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

    self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)

  # define forward function
  def forward(self, t):
    # conv 1
    t = self.conv1(t) # transform the given tensor on the 1st convolutional layer
    t = F.relu(t) # compute the activation function on the tensor
    t = F.max_pool2d(t, kernel_size=2, stride=2) # pool the layer (it's now 1/2 the size it was (28-4 in each direction))

    # conv 2
    t = self.conv2(t) # repeat above
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # fc1
    t = t.reshape(-1, 12*4*4) # Flatten out the final pooling layer 
    t = self.fc1(t)
    t = F.relu(t)

    # fc2
    t = self.fc2(t)
    t = F.relu(t)

    # output
    t = self.out(t)
    # don't need softmax here since we'll use cross-entropy as activation.

    return t

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Process the image and spit out an image in the format and size expected
def process_image(pil_img):
    gray_image = pil_img.convert('L')
    shrunk_image = gray_image.resize((28,28),resample=Image.LANCZOS)
    img_array = np.array(shrunk_image)
    img_array = img_array.reshape((28, 28, 1)) # Give it 1 color channel
    img_array = img_array / 255. # Convert to between 0 and 1
    
    # Add image to batch where it's the only member for tensorflow
    img_array = (np.expand_dims(img_array,0))
    img_array_tf = img_array.reshape((img_array.shape[0], 28, 28, 1)) # channel last for tf
    img_array_torch = img_array.reshape((img_array.shape[0],1,28,28)) # channel first for torch
    
    return img_array_tf, img_array_torch
    
def plot_image(img, ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Image as seen by the NNs', fontdict={'fontsize':50})
    ax.imshow(img, cmap=plt.cm.binary)
    return ax

def plot_value_array(predictions_array, model, ax):
    predictions_array = predictions_array.reshape((10,))
    predictions_array /= np.max(predictions_array,axis=0)
    ax.grid(False)
    ax.set_xticks(range(10))
    ax.set_xticklabels(labels=class_names,rotation=30, fontdict={'fontsize':28})
    ax.set_yticks([])
    ax.bar(range(10), predictions_array, color="#777777")
    ax.set_ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)

    ax.set_title(f"{model}",fontdict={'fontsize':50})
    
    return ax

def plot_comparisons(img_array, torch_prediction, tf_prediction, tf_best_prediction):
    fig, axs = plt.subplots(2,2, figsize=(32,32))
    
    axs[0][0] = plot_image(img_array.reshape((28,28)), axs[0][0])
    axs[0][1] = plot_value_array(torch_prediction,'PyTorch', axs[0][1])
    axs[1][0] = plot_value_array(tf_prediction,'TensorFlow mimicing PyTorch', axs[1][0])
    axs[1][1] = plot_value_array(tf_best_prediction,'TensorFlow best model', axs[1][1])
    
    return fig

# Change this function directly to change the models loaded
def load_models():
    current_directory = os.getcwd()
    prepath=''
    if 'app' in current_directory: # you're in the app directly and need to navigate up a folder
        prepath = '..'
    else:
        prepath='.'
    torch_model  = Network()
    torch_model.load_state_dict(torch.load(f'{prepath}/model/saved_models/torch-model-3.pth'))
    
    tf_model_1 = tf.keras.models.load_model(f'{prepath}/model/saved_models/tf-mimic-3.h5')
    tf_model_2 = tf.keras.models.load_model(f'{prepath}/model/saved_models/tf-model-9.h5')
    
    return torch_model, tf_model_1, tf_model_2

def get_label_from_predictions(pred_array):
    #print(pred_array.shape)
    #percent_certain = np.max(pred_array) / sum(pred_array)
    return class_names[np.argmax(pred_array)]#, percent_certain

def make_predictions(img, torch_model, tf_model_1, tf_model_2):
    img_array_tf, img_array_torch = process_image(img)
    
    torch_prediction = torch_model(torch.from_numpy(img_array_torch).float()).detach().numpy()
    
    tf_prediction_1 = tf_model_1.predict(img_array_tf)
    tf_prediction_2 = tf_model_2.predict(img_array_tf)
    
    return torch_prediction, tf_prediction_1, tf_prediction_2
    
def make_percentage(predictions_array):
    perc_array = predictions_array / np.sum(predictions_array,axis=0)
    return np.max(perc_array)


    
    

               
