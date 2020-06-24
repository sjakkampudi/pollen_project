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

# reading data from csv which has filename, label stored as strings
traindf = pd.read_csv('/home/agtrivedi/repos/pollen_project/anu/segm_ims.csv', dtype = str)
#traindf = pd.read_csv('/home/agtrivedi/repos/pollen_project/anu/obj_ims.csv', dtype = str)
#traindf = pd.read_csv('/home/agtrivedi/repos/pollen_project/anu/mask_ims.csv', dtype = str)

datagen = ImageDataGenerator(rescale=1./255.)
train_generator = datagen.flow_from_dataframe(
        dataframe = traindf,
        directory = '/home/agtrivedi/.keras/datasets/train/images/',
        x_col = "file name",
        y_col = "label",
        subset = "training")

train_images = train_generator.filepaths
train_labels = train_generator.labels

label_array = np.array(train_labels)
label_array = np.reshape(label_array, (np.size(label_array), 1))

# total images: 11,279; each image is 84x84; RGB image so 3 channels
img_array = np.empty((11279, 84, 84, 3))

for index in range(len(train_images)):
    img = image.load_img(train_images[index])
    img_array[index] = image.img_to_array(img, data_format = 'channels_last')
    #label_array[index] = train_labels[index]

model = models.Sequential()
model.add(layers.Conv2D(3, (3,3), activation = 'relu', input_shape = (84, 84, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4))

model.summary()


model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

print(np.shape(img_array))
print(np.shape(label_array))
model.fit(img_array, label_array, epochs = 5)

test_loss, test_acc = model.evaluate(img_array, label_array, verbose=1)
