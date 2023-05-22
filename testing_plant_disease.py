
from keras.preprocessing import image
import numpy as np
from IPython.display import SVG, Image
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load the saved model
classifier = tf.keras.models.load_model('my_disease.h5')

img_size = 48
batch_size = 64

# Define the data generators for training and validation
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory(
    r"D:\Projects\plant_disease_website\dataset\train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory(
    r"D:\Projects\plant_disease_website\dataset\test",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# Load a test image and predict its class
path = r"D:\Projects\plant_disease_website\dataset\test\Apple___Cedar_apple_rust\0ce943e7-3fed-41cb-8430-0e0f54ff2bc4___FREC_C.Rust 0014.JPG"
test_image = tf.keras.utils.load_img(path)
plt.imshow(test_image)

test_img = tf.keras.utils.load_img(path, target_size=(img_size, img_size))
test_img = tf.keras.utils.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
result = classifier.predict(test_img)
a = result.argmax()
s = train_generator.class_indices
name = []
for i in s:
    name.append(i)
for i in range(len(s)):
    if i == a:
        p = name[i]
print("Predicted class:", p)
plt.imshow(test_image)
