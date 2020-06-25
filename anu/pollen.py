from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from numpy.random import seed, randint

print("tensorflow version is:", tf.__version__)

# reading data from csv which has filename, label stored as strings
#traindf = pd.read_csv('/home/agtrivedi/repos/pollen_project/anu/segm_ims.csv', dtype = str)
traindf = pd.read_csv('/home/agtrivedi/repos/pollen_project/anu/obj_ims.csv', dtype = str)
#traindf = pd.read_csv('/home/agtrivedi/repos/pollen_project/anu/mask_ims.csv', dtype = str)

datagen = ImageDataGenerator(rescale=1./255.)
train_generator = datagen.flow_from_dataframe(
        dataframe = traindf,
        directory = '/home/agtrivedi/.keras/datasets/train/images/',
        x_col = "file name",
        y_col = "label",
        subset = "training")

images = train_generator.filepaths
labels = train_generator.labels
#labels = [x + 1 for x in labels]

label_array = np.array(labels)
label_array = np.reshape(label_array, (np.size(label_array), 1))

# total images: 11,279; each image is 84x84; RGB image so 3 channels
img_array = np.empty((11279, 84, 84, 3))

for index in range(len(images)):
    img = image.load_img(images[index])
    img_array[index] = image.img_to_array(img, data_format = 'channels_last')

# select 2000 images to train, 2000 images to validate
seed(1)
train_vals = randint(0, len(labels), 8000)
train_vals = np.reshape(train_vals, (len(train_vals), 1))   

validate_vals = randint(0, len(labels), 2000)
validate_vals = np.reshape(validate_vals, (len(validate_vals), 1))

test_vals = randint(0, len(labels), 2000)
test_vals = np.reshape(test_vals, (len(test_vals), 1))

train_labels = np.empty((np.shape(train_vals)))
validate_labels = np.empty((np.shape(validate_vals)))
test_labels = np.empty((np.shape(test_vals)))

train_images = np.empty((np.shape(train_labels)[0], 84, 84, 3))
validate_images = np.empty((np.shape(validate_labels)[0], 84, 84, 3))
test_images = np.empty((np.shape(test_labels)[0], 84, 84, 3))

for i in range(np.shape(train_labels)[0]):
    train_labels[i] = labels[train_vals[i,0]]
    train_images[i] = img_array[train_vals[i,0]]

for j in range(np.shape(validate_labels)[0]):
    validate_labels[j] = labels[validate_vals[j,0]]
    validate_images[j] = img_array[validate_vals[j,0]]

for k in range(np.shape(test_labels)[0]):
    test_labels[k] = labels[test_vals[k,0]]
    test_images[k] = img_array[test_vals[k,0]]

print("CHECK SIZES\n-------------------------------------- \nThe size of training dataset is:", np.shape(train_images), "\nThe size of the validate dataset is:", np.shape(validate_images))    

model = models.Sequential()
model.add(layers.Conv2D(84, (3,3), activation = 'relu', input_shape = (84, 84, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(84, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(84, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(4))

model.summary()

model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 3, validation_data = (validate_images, validate_labels))

test_loss, test_acc = model.evaluate(validate_images, validate_labels, verbose=1)

image_pred = np.array([img_array[9744]])
print(np.shape(image_pred))

label_pred = label_array[9744, 0]
#print(label_pred)

prediction = model.predict(image_pred, batch_size=1, verbose=1)
print("predicted label:", np.argmax(prediction))
print("true label:", label_pred)
