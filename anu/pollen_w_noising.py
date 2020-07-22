import os
import random
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, optimizers
from sklearn.model_selection import train_test_split
import IPython
import kerastuner as kt

from skimage.util import random_noise
from pathlib import Path
import cv2

NUM_IMAGES_PER_CLASS = 20000
NUM_EPOCHS = 10




seed_value = 1
os.environ['PYTHONHASHSEED']=str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


print("tensorflow version is:", tf.__version__)

train_path1 = open('../segged_data/segmented_pollen/1/paths.txt').read().splitlines()
train_path2 = open('../segged_data/segmented_pollen/2/paths.txt').read().splitlines()
train_path3 = open('../segged_data/segmented_pollen/3/paths.txt').read().splitlines()
train_path4 = open('../segged_data/segmented_pollen/4/paths.txt').read().splitlines()

print("Total class 1 images:", len(train_path1))
print("Total class 2 images:", len(train_path2))
print("Total class 3 images:", len(train_path3))
print("Total class 4 images:", len(train_path4))

CLASS_1_PREFIX = '../segged_data/seg_pollen_rgb/1/'
CLASS_2_PREFIX = '../segged_data/seg_pollen_rgb/2/'
CLASS_3_PREFIX = '../segged_data/seg_pollen_rgb/3/'
CLASS_4_PREFIX = '../segged_data/seg_pollen_rgb/4/'

train_images_1 = []
train_images_2 = []
train_images_3 = []
train_images_4 = []


#*********** READ IN SOME DATA ************#
for train_path in train_path1:
    image_path = CLASS_1_PREFIX + Path(train_path).stem + Path(train_path).suffix
    train_images_1.append(cv2.imread(image_path))
for train_path in train_path2:
    image_path = CLASS_2_PREFIX + Path(train_path).stem + Path(train_path).suffix
    train_images_2.append(cv2.imread(image_path))
for train_path in train_path3:
    image_path = CLASS_3_PREFIX + Path(train_path).stem + Path(train_path).suffix
    train_images_3.append(cv2.imread(image_path))
for train_path in train_path4:
    image_path = CLASS_4_PREFIX + Path(train_path).stem + Path(train_path).suffix
    train_images_4.append(cv2.imread(image_path))
    
    
print("----- AUGMENTING DATA -----")
########### Augmentation for class 1 ###############
i = 0
rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
noise = ['gaussian', 'poisson', 's&p', 'speckle']
original_data_length = len(train_images_1)
#*********** AUGMENT SOME DATA ************#
while len(train_images_1) < NUM_IMAGES_PER_CLASS:
    img = train_images_1[i]
    i = (i + 1) % original_data_length # cycle through original data only
    random_noise_type_index = random.randint(0, 3)
    random_rotation = random.randint(0, 2)
    noise_rotate_or_both = random.randint(0, 2)
    # only add noise
    if noise_rotate_or_both == 0:
        img = random_noise(img, noise[random_noise_type_index])
    # only rotate
    elif noise_rotate_or_both == 1:
        img = cv2.rotate(img, rotations[random_rotation])
    # add noise and rotate
    else:
        img = random_noise(img, noise[random_noise_type_index])
        img = cv2.rotate(img, rotations[random_rotation])
    train_images_1.append(img)

print('New total class 1 images: ' + str(len(train_images_1)))

########### Augmentation for class 2 ###############
i = 0
rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
noise = ['gaussian', 'poisson', 's&p', 'speckle']
original_data_length = len(train_images_2)
#*********** AUGMENT SOME DATA ************#
while len(train_images_2) < NUM_IMAGES_PER_CLASS:
    img = train_images_2[i]
    i = (i + 1) % original_data_length # cycle through original data only
    random_noise_type_index = random.randint(0, 3)
    random_rotation = random.randint(0, 2)
    noise_rotate_or_both = random.randint(0, 2)
    # only add noise
    if noise_rotate_or_both == 0:
        img = random_noise(img, noise[random_noise_type_index])
    # only rotate
    elif noise_rotate_or_both == 1:
        img = cv2.rotate(img, rotations[random_rotation])
    # add noise and rotate
    else:
        img = random_noise(img, noise[random_noise_type_index])
        img = cv2.rotate(img, rotations[random_rotation])
    train_images_2.append(img)

print('New total class 2 images: ' + str(len(train_images_2)))

########### Augmentation for class 3 ###############
i = 0
rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
noise = ['gaussian', 'poisson', 's&p', 'speckle']
original_data_length = len(train_images_3)
#*********** AUGMENT SOME DATA ************#
while len(train_images_3) < NUM_IMAGES_PER_CLASS:
    img = train_images_3[i]
    i = (i + 1) % original_data_length # cycle through original data only
    random_noise_type_index = random.randint(0, 3)
    random_rotation = random.randint(0, 2)
    noise_rotate_or_both = random.randint(0, 2)
    # only add noise
    if noise_rotate_or_both == 0:
        img = random_noise(img, noise[random_noise_type_index])
    # only rotate
    elif noise_rotate_or_both == 1:
        img = cv2.rotate(img, rotations[random_rotation])
    # add noise and rotate
    else:
        img = random_noise(img, noise[random_noise_type_index])
        img = cv2.rotate(img, rotations[random_rotation])
    train_images_3.append(img)

print('New total class 3 images: ' + str(len(train_images_3)))

########### Augmentation for class 4 ###############
i = 0
rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
noise = ['gaussian', 'poisson', 's&p', 'speckle']
original_data_length = len(train_images_4)
#*********** AUGMENT SOME DATA ************#
while len(train_images_4) < NUM_IMAGES_PER_CLASS:
    img = train_images_4[i]
    i = (i + 1) % original_data_length # cycle through original data only
    random_noise_type_index = random.randint(0, 3)
    random_rotation = random.randint(0, 2)
    noise_rotate_or_both = random.randint(0, 2)
    # only add noise
    if noise_rotate_or_both == 0:
        img = random_noise(img, noise[random_noise_type_index])
    # only rotate
    elif noise_rotate_or_both == 1:
        img = cv2.rotate(img, rotations[random_rotation])
    # add noise and rotate
    else:
        img = random_noise(img, noise[random_noise_type_index])
        img = cv2.rotate(img, rotations[random_rotation])
    train_images_4.append(img)

print('New total class 4 images: ' + str(len(train_images_4)))

train_images = []
train_labels = []

label_counts = [len(train_images_1), len(train_images_2),len(train_images_3), len(train_images_4)]
total_labels = label_counts[0] + label_counts[1] + label_counts[2] + label_counts[3]

############# Shuffling the data ##############
print("----- SHUFFLING DATA -----")
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
        img = train_images_1.pop(len(train_images_1) -1) # get the image at the end of the list
        img = np.asarray(img, dtype=np.float64)
        train_images.append(img)
    elif random_label == 2:
        img = train_images_2.pop(len(train_images_2) -1) # get the image at the end of the list
        img = np.asarray(img, dtype=np.float64)
        train_images.append(img)
    elif random_label == 3:
        img = train_images_3.pop(len(train_images_3) -1) # get the image at the end of the list
        img = np.asarray(img, dtype=np.float64)
        train_images.append(img)
    elif random_label == 4:
        img = train_images_4.pop(len(train_images_4) -1) # get the image at the end of the list
        img = np.asarray(img, dtype=np.float64)
        train_images.append(img)
    else:
        print("Issue...")

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

print("New total training images: " + str(len(train_images)))

secret_images, test_images, secret_labels, test_labels = train_test_split(train_images,
                                                                       train_labels,
                                                                       test_size = 0.30,
                                                                       train_size = 0.05, # this is actually the secret category
                                                                       random_state = seed_value)

train_images, temp_images, train_labels, temp_labels = train_test_split(train_images,
        train_labels, train_size = 0.65, random_state = seed_value)

print("----- TRAIN/TEST SPLIT: 65% training, 30% testing, 5% secret -----")

train_1_count, train_2_count, train_3_count, train_4_count = 0, 0, 0, 0

for i in range(len(train_labels)):
    if train_labels[i,0] == 0:
        train_1_count += 1
    elif train_labels[i,0] == 1:
        train_2_count += 1
    elif train_labels[i,0] == 2:
        train_3_count += 1
    elif train_labels[i,0] == 3:
        train_4_count += 1
        
total = train_1_count + train_2_count + train_3_count + train_4_count

print("Out of", total, "training images, there are", train_1_count, "in class 1,", train_2_count, \
      "in class 2,", train_3_count, "in class 3,", "and", train_4_count, "in class 4")

test_1_count, test_2_count, test_3_count, test_4_count = 0, 0, 0, 0

for i in range(len(test_labels)):
    if test_labels[i,0] == 0:
        test_1_count += 1
    elif test_labels[i,0] == 1:
        test_2_count += 1
    elif test_labels[i,0] == 2:
        test_3_count += 1
    elif test_labels[i,0] == 3:
        test_4_count += 1

total = test_1_count + test_2_count + test_3_count + test_4_count

print("Out of", total, "testing images, there are", test_1_count, "in class 1,", test_2_count, \
      "in class 2,", test_3_count, "in class 3,", "and", test_4_count, "in class 4\n\n")

''' # comment this out if you want to use the keras tuner

def model_builder(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape = (84, 84, 3)))

    hp_units = hp.Int('units', min_value = 16, max_value = 512, step = 16)
    model.add(keras.layers.Dense(units = hp_units, activation = 'relu'))
    model.add(keras.layers.Dense(4))

    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

    model.compile(optimizer = optimizers.Adam(learning_rate = hp_learning_rate),
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            metrics = ['accuracy'])

    return model


tuner= kt.Hyperband(model_builder,
        objective = 'val_accuracy',
        max_epochs = 10,
        factor = 3,
        directory = 'my_dir',
        project_name = 'intro_to_kt',
        overwrite = True)

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

tuner.search(train_images,train_labels, epochs = 10, validation_data = (test_images, test_labels), callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print("The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.")

model = tuner.hypermodel.build(best_hps)
model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels))

'''

model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(84, 84, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(168, activation='relu'))
model.add(layers.Dense(4)) # final Dense layer has 4 neurons representing the 4 classes

print(model.summary())

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=NUM_EPOCHS, #batch_size=32,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

prediction = model.predict(secret_images, verbose=1)
#print(prediction)

predicted_label = []
incorrect_labels = []
class_names = [1, 2, 3, 4]

for i in range(len(prediction)):
#    predicted_label.append(class_names[np.argmax(prediction[i])])
    if int(class_names[int(secret_labels[i][0])]) != int(class_names[np.argmax(prediction[i])]):
       # print("The true label was", int(class_names[int(secret_labels[i][0])]), "and the predicted label was", int(class_names[np.argmax(prediction[i])]))
        incorrect_labels.append(int(class_names[int(secret_labels[i][0])]))
        predicted_label.append(class_names[np.argmax(prediction[i])])

print("The number of incorrect labels is:", len(incorrect_labels))

incorrect_1, incorrect_2, incorrect_3, incorrect_4 = 0, 0, 0, 0

for i in range(len(incorrect_labels)):
    if incorrect_labels[i] == 1:
        incorrect_1 += 1
    elif incorrect_labels[i] == 2:
        incorrect_2 += 1
    elif incorrect_labels[i] == 3:
        incorrect_3 += 1
    elif incorrect_labels[i] == 4:
        incorrect_4 += 1

print("Out of", len(incorrect_labels), "incorrectly predicted labels,", incorrect_1, "were actually in class 1,", incorrect_2, "were actually in class 2,", incorrect_3, "were actually in class 3, and", incorrect_4, "were actually in class 4.")

print("Class 1 accuracy:", int(100*(1-incorrect_1/len(incorrect_labels))), "%")
print("Class 2 accuracy:", int(100*(1-incorrect_2/len(incorrect_labels))), "%")
print("Class 3 accuracy:", int(100*(1-incorrect_3/len(incorrect_labels))), "%")
print("Class 4 accuracy:", int(100*(1-incorrect_4/len(incorrect_labels))), "%")

# ''' # comment this quotes set out if you want to use the keras manually added layers






