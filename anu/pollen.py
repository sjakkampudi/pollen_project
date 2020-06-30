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
from tensorflow.random import set_seed

seed_value = 50
os.environ['PYTHONHASHSEED']=str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)
set_seed(seed_value)


print("tensorflow version is:", tf.__version__)

train_path1 = open('../matthew/Pollen_Classifier/train/images/1/train_OBJ/paths.txt').read().splitlines()
train_path2 = open('../matthew/Pollen_Classifier/train/images/2/train_OBJ/paths.txt').read().splitlines()
train_path3 = open('../matthew/Pollen_Classifier/train/images/3/train_OBJ/paths.txt').read().splitlines()
train_path4 = open('../matthew/Pollen_Classifier/train/images/4/train_OBJ/paths.txt').read().splitlines()

print("Total class 1 images:", len(train_path1))
print("Total class 2 images:", len(train_path2))
print("Total class 3 images:", len(train_path3))
print("Total class 4 images:", len(train_path4))

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

type1, type2, type3, type4 = 0, 0, 0, 0

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
        image_45 = image.rotate(45)
        image_90 = image.rotate(90)
        image = np.asarray(image, dtype=np.float64)
        image_45 = np.asarray(image_45, dtype=np.float64)
        image_90 = np.asarray(image_90, dtype=np.float64)
        train_images.append(image)
        train_images.append(image_45)
        train_images.append(image_90)
        label = [0]
        train_labels.append(label)
        train_labels.append(label)
        type1 += 3
    elif random_label == 2:
        path = train_path2.pop(len(train_path2) - 1) # get the path at the end of the list
        image = Image.open(path)
        image_45 = image.rotate(45)
        image_90 = image.rotate(90)
        image_60 = image.rotate(60)
        image_15 = image.rotate(15)
        image = np.asarray(image, dtype=np.float64)
        image_45 = np.asarray(image_45, dtype=np.float64)
        image_90 = np.asarray(image_90, dtype=np.float64)
        image_60 = np.asarray(image_60, dtype=np.float64)
        image_15 = np.asarray(image_15, dtype=np.float64)
        train_images.append(image)
        train_images.append(image_45)
        train_images.append(image_90)
        train_images.append(image_60)
        train_images.append(image_15)
        label = [1]
        train_labels.append(label)
        train_labels.append(label)
        train_labels.append(label)
        train_labels.append(label)
        type2 += 5
    elif random_label == 3:
        path = train_path3.pop(len(train_path3) - 1) # get the path at the end of the list
        image = Image.open(path)
        image_45 = image.rotate(45)
        image_90 = image.rotate(90)
        image = np.asarray(image, dtype=np.float64)
        image_45 = np.asarray(image_45, dtype=np.float64)
        image_90 = np.asarray(image_90, dtype=np.float64)
        train_images.append(image)
        train_images.append(image_45)
        train_images.append(image_90)
        label = [2]
        train_labels.append(label)
        train_labels.append(label)
        type3 += 3
    elif random_label == 4:
        path = train_path4.pop(len(train_path4) - 1) # get the path at the end of the list
        image = Image.open(path)
        image_45 = image.rotate(45)
        image_90 = image.rotate(90)
        image_60 = image.rotate(60)
        image_15 = image.rotate(15)
        image = np.asarray(image, dtype=np.float64)
        image_45 = np.asarray(image_45, dtype=np.float64)
        image_90 = np.asarray(image_90, dtype=np.float64)
        image_60 = np.asarray(image_60, dtype=np.float64)
        image_15 = np.asarray(image_15, dtype=np.float64)
        train_images.append(image)
        train_images.append(image_45)
        train_images.append(image_90)
        train_images.append(image_60)
        train_images.append(image_15)
        label = [3]
        train_labels.append(label)
        train_labels.append(label)
        train_labels.append(label)
        train_labels.append(label)
        type4 += 5
    else:
        print("Issue...")

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

training_size = 15000

#(train_images, test_images) = np.split(train_images, [size], 0)
#(train_labels, test_labels) = np.split(train_labels, [size], 0)

(train_images_split, secret_images) = np.split(train_images, [training_size], 0)
(train_labels_split, secret_labels) = np.split(train_labels, [training_size], 0)


testing_size = 1000
test_images = np.ones((testing_size, 84, 84, 3))
test_labels = np.ones((testing_size, 1))

for i in range(testing_size):
    rand_array = random.sample(range(len(train_images)), testing_size)
    rand_array = np.asarray(rand_array)

    for j in range(int(testing_size/4)):
        if train_labels[rand_array[i]] == 0:
            test_images[j] = train_images[rand_array[i]]
            test_labels[j] = train_labels[rand_array[i]]
            
    for j in range(int(testing_size/4), 2*int(testing_size/4)):
        if train_labels[rand_array[i]] == 1:
            test_images[j] = train_images[rand_array[i]]
            test_labels[j] = train_labels[rand_array[i]]
            
    for j in range(2*int(testing_size/4), 3*int(testing_size/4)):
        if train_labels[rand_array[i]] == 2:
            test_images[j] = train_images[rand_array[i]]
            test_labels[j] = train_labels[rand_array[i]]
            
    for j in range(3*int(testing_size/4), 4*int(testing_size/4)):
        if train_labels[rand_array[i]] == 3:
            test_images[j] = train_images[rand_array[i]]
            test_labels[j] = train_labels[rand_array[i]]        
    
print(test_images.shape)    

#print("Size of training images array before splitting:", train_images.shape)
#print("Size of training labels array before splitting:", train_labels.shape)
    
train_1_count, train_2_count, train_3_count, train_4_count = 0, 0, 0, 0

for i in range(len(train_labels_split)):
    if train_labels_split[i,0] == 0:
        train_1_count += 1
    elif train_labels_split[i,0] == 1:
        train_2_count += 1
    elif train_labels_split[i,0] == 2:
        train_3_count += 1
    elif train_labels_split[i,0] == 3:
        train_4_count += 1

print("Out of", training_size, "images, there are", train_1_count, "in class 1,", train_2_count, \
      "in class 2,", train_3_count, "in class 3,", "and", train_4_count, "in class 4")

# further break up the test_images to keep back some secret data that is not used to train the model
#(secret_images, test_images) = np.split(test_images, [abs(1000 - test_images.shape[0])], 0)
#(secret_labels, test_labels) = np.split(test_labels, [abs(1000 - test_labels.shape[0])], 0)

print("Train images:", train_images.shape[0])
print("Train labels:", train_labels.shape[0])
print("Test images:", test_images.shape[0])
print("Test labels:", test_labels.shape[0])
print("Secret images:", secret_images.shape[0])
print("Secret labels:", secret_labels.shape[0])

#print("Total type 1 images:", type1, "\nTotal type 2 images:", type2, "\nTotal type 3 images:", type3, "\nTotal type 4 images:", type4)
#print("-------------------------------------------")
#print("Total images:", type1+type2+type3+type4)

train_images = train_images / 255
test_images = test_images / 255
secret_images = secret_images / 255

model = models.Sequential() # Indeed a model that can be implemented as a CNN
model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=(84, 84, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
print(model.summary())

model.add(layers.Flatten()) # flatten the 3-D tensor output of the preceding layer into a
                            # 1-D vector to feed to the top Dense layers
#model.add(layers.Dropout(0.5))
#model.add(layers.SpatialDropout2D(0.25))
model.add(layers.Dense(168, activation='relu'))
model.add(layers.Dense(4)) # final Dense layer has 10 neurons representing the 10 classes
print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images_split, train_labels_split, epochs=2, #batch_size=20,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
image_number = 2

class_names = ['0', '1', '2', '3']
image = np.array([secret_images[image_number]], dtype=np.float32)
label = class_names[int(secret_labels[image_number])]
print("label: " + label)
print("shape: " + str(image.shape))

# Do the prediction
prediction = model.predict(image, batch_size=1, verbose=1)
print("model predicted: " + class_names[np.argmax(prediction)])
print("ground truth label: " + label)
