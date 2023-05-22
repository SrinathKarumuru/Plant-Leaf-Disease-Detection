import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras_preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

"""Load the trained model and its labels for prediction."""

# Define the path to the saved model
model_path = 'ram://c12212e2-70a0-4f68-a08b-c9da46edde24'

# Create a LoadOptions object with the io_device set to '/job:localhost'
options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

# Load the saved model with the LoadOptions object
loaded_model = tf.saved_model.load(model_path, options=options)

# Load model
filename = 'cnn_model.pkl'
model = pickle.load(open(filename, 'rb'))

# Load labels
filename = 'label_transform.pkl'
image_labels = pickle.load(open(filename, 'rb'))

DEFAULT_IMAGE_SIZE = tuple((256, 256))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def predict_disease(image_path):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    plt.imshow(plt.imread(image_path))
    result = model.predict_classes(np_image)
    #print((image_labels.classes_[result][0]))
    return (image_labels.classes_[result][0])

print(predict_disease(r"D:\Projects\plant_disease_website\dataset\test\Pepper,_bell___healthy\01fbd010-0cc1-4c48-98bc-49e328bf9bbc___JR_HL 8584.JPG"))