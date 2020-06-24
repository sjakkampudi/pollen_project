from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image


print("tensorflow version is:", tf.__version__)


traindf = pd.read_csv('/home/agtrivedi/ims.csv', dtype = str)

datagen = ImageDataGenerator(rescale=1./255.)
train_generator = datagen.flow_from_dataframe(
        dataframe = traindf,
        directory = '/home/agtrivedi/.keras/datasets/train/images/',
        x_col = "file name",
        y_col = "label",
        subset = "training")
#print(train_generator.filepaths) #train_generator is a directory iterator obj with attributes filepaths and labels

#model = keras.Sequential([
#    keras.layers.Dense(128, activation = 'relu'),
#    keras.layers.Dense(10)
#    ])

model = models.Sequential()
#model.add(layers.Conv2D(84, (3,3), activation = 'relu'))

train_images = train_generator.filepaths
train_labels = train_generator.labels

#label_array = np.empty((11279, 1))
label_array = np.array(train_labels)
label_array = np.reshape(label_array, (np.size(label_array), 1))
img_array = np.empty((11279, 84, 84, 3))
#img_array = []

for index in range(len(train_images)):
    img = image.load_img(train_images[index])
    img_array[index] = image.img_to_array(img, data_format = 'channels_last')
    #label_array[index] = train_labels[index]

print(np.shape(label_array))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

print(np.shape(img_array))
print(np.shape(label_array))
model.fit(img_array, label_array, epochs = 2)
