from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from numpy.random import seed, randint
from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets, layers, models


print("tensorflow version is:", tf.__version__)

train_path1 = open('/home/agtrivedi/repos/pollen_project/matthew/Pollen_Classifier/train/images/1/train_OBJ/paths.txt').read().splitlines()
train_path2 = open('/home/agtrivedi/repos/pollen_project/matthew/Pollen_Classifier/train/images/2/train_OBJ/paths.txt').read().splitlines()
train_path3 = open('/home/agtrivedi/repos/pollen_project/matthew/Pollen_Classifier/train/images/3/train_OBJ/paths.txt').read().splitlines()
train_path4 = open('/home/agtrivedi/repos/pollen_project/matthew/Pollen_Classifier/train/images/4/train_OBJ/paths.txt').read().splitlines()

print(len(train_path1))
print(len(train_path2))
print(len(train_path3))
print(len(train_path4))

train_labels = []
for _ in train_path1:
    label = [1]
    train_labels.append(label)
for _ in train_path2:
    label = [2]
    train_labels.append(label)
for _ in train_path3:
    label = [3]
    train_labels.append(label)
for _ in train_path4:
    label = [4]
    train_labels.append(label)
train_labels = np.asarray(train_labels)
print(train_labels.shape)

train_images = []
train_labels = []

# first element for padding, classes begin at 1
label_counts = [len(train_path1), len(train_path2), len(train_path3), len(train_path4)] 

total_labels = label_counts[0] + label_counts[1] + label_counts[2] + label_counts[3]

for i in range(total_labels):
    # Get random label that is still available
    while True:
        random_label = random.randint(1, 4)
        if (label_counts[random_label-1] > 0):
            label_counts[random_label-1] = label_counts[random_label-1] - 1 # decrement the label count
            break
    
    # append the label to the label list and add the corresponding image to the image list
    train_labels.append([random_label - 1]) 
    if random_label == 1:
        path = train_path1.pop(len(train_path1) - 1) # get the path at the end of the list
        image = Image.open(path)
        image_rot = image.rotate(45)
        image = np.asarray(image, dtype=np.float64)
        image_rot = np.asarray(image_rot, dtype=np.float64)
        train_images.append(image)
        train_images.append(image_rot)
        label = [0]
        train_labels.append(label)
    elif random_label == 2:
        path = train_path2.pop(len(train_path2) - 1) # get the path at the end of the list
        image = Image.open(path)
        image_rot = image.rotate(45)
        image = np.asarray(image, dtype=np.float64)
        image_rot = np.asarray(image_rot, dtype=np.float64)
        train_images.append(image)
        train_images.append(image_rot)
        label = [1]
        train_labels.append(label)
    elif random_label == 3:
        path = train_path3.pop(len(train_path3) - 1) # get the path at the end of the list
        image = imread(path)
        image = np.asarray(image, dtype=np.float64)
        train_images.append(image)
    elif random_label == 4:
        path = train_path4.pop(len(train_path4) - 1) # get the path at the end of the list
        image = Image.open(path)
        image_rot = image.rotate(45)
        image = np.asarray(image, dtype=np.float64)
        image_rot = np.asarray(image_rot, dtype=np.float64)
        train_images.append(image)
        train_images.append(image_rot)
        label = [3]
        train_labels.append(label)
    else:
        print("Issue...")

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
print(train_images.shape)
print(train_labels.shape)

(train_images, test_images) = np.split(train_images, [10000], 0)
(train_labels, test_labels) = np.split(train_labels, [10000], 0)

# further break up the test_images to keep back some secret data that is not used to train the model
(secret_images, test_images) = np.split(test_images, [abs(1000 - test_images.shape[0])], 0)
(secret_labels, test_labels) = np.split(test_labels, [abs(1000 - test_labels.shape[0])], 0)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
print(secret_images.shape)
print(secret_labels.shape)

train_images = train_images / 255
test_images = test_images / 255
secret_images = secret_images / 255

model = models.Sequential() # Indeed a model that can be implemented as a CNN
model.add(layers.Conv2D(168, (3, 3), activation='relu', input_shape=(84, 84, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(168, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(168, (3, 3), activation='relu'))
print(model.summary())

model.add(layers.Flatten()) # flatten the 3-D tensor output of the preceding layer into a
                            # 1-D vector to feed to the top Dense layers
model.add(layers.Dropout(0.5))
model.add(layers.Dense(168, activation='relu'))
model.add(layers.Dense(4)) # final Dense layer has 10 neurons representing the 10 classes
print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=2,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
image_number = 2

class_names = ['1', '2', '3', '4']
image = np.array([secret_images[image_number]], dtype=np.float32)
label = class_names[int(secret_labels[image_number])]
print("label: " + label)
print("shape: " + str(image.shape))

# Do the prediction
prediction = model.predict(image, batch_size=1, verbose=1)
print("model predicted: " + class_names[np.argmax(prediction)])
print("ground truth label: " + label)
