import os
import random
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from configobj import ConfigObj

def parse_args():
    parser = argparse.ArgumentParser(description="pollen classifier")

    parser.add_argument("config_path", type=str)

    args = parser.parse_args()
    return args

args = parse_args()

cfg = ConfigObj(args.config_path)

os.environ['PYTHONHASHSEED']=str(cfg["seed_value"])

random.seed(cfg["seed_value"])
np.random.seed(cfg["seed_value"])
tf.random.set_seed(cfg["seed_value"])


print("tensorflow version is:", tf.__version__)

train_path1 = open('train/images/1/train_OBJ/paths.txt').read().splitlines()
train_path2 = open('train/images/2/train_OBJ/paths.txt').read().splitlines()
train_path3 = open('train/images/3/train_OBJ/paths.txt').read().splitlines()
train_path4 = open('train/images/4/train_OBJ/paths.txt').read().splitlines()

if cfg["balance_classes"]:
    train_path1 = train_path1[:cfg["class_size"]]
    train_path2 = train_path2[:cfg["class_size"]]
    train_path3 = train_path3[:cfg["class_size"]]
    train_path4 = train_path4[:cfg["class_size"]]

print("Total class 1 images:", len(train_path1))
print("Total class 2 images:", len(train_path2))
print("Total class 3 images:", len(train_path3))
print("Total class 4 images:", len(train_path4))

train_images = []
train_labels = []

# first element for padding, classes begin at 1
label_counts = [len(train_path1), len(train_path2), len(train_path3), len(train_path4)] 

total_labels = label_counts[0] + label_counts[1] + label_counts[2] + label_counts[3]

type1, type2, type3, type4 = 0, 0, 0, 0
counter1, counter2, counter3, counter4 = 0, 0, 0, 0

train_images = [None] * total_labels
train_labels = [None] * total_labels

paths = train_path1 + train_path2 + train_path3 + train_path4

train_labels = [0] * len(train_path1) + [1] * len(train_path2) + [2] * len(train_path3) + [3] * len(train_path4)

for i, path in enumerate(paths):
    train_images[i] = np.asarray(Image.open(path.split("anu/")[1]), dtype=np.float64)

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

test_size = 1 - cfg["train_size"] - cfg["secret_size"]

secret_images, test_images, secret_labels, test_labels = train_test_split(train_images,
                                                                       train_labels,
                                                                       test_size = test_size,
                                                                       train_size = cfg["secret_size"], # this is actually the secret category
                                                                       random_state = cfg["seed_value"])

train_images, temp_images, train_labels, temp_labels = train_test_split(train_images,
        train_labels, train_size = cfg["train_size"], random_state = cfg["seed_value"])

print("----- TRAIN/TEST SPLIT: " + str(cfg["train_size"]*100) + "% training, " + str(test_size*100) + "% validation, " + str(cfg["secret_size"]*100) + "%  test -----")

train_images = train_images / 255
test_images = test_images / 255

print(train_images[1,:,:].shape)

model = Sequential() 

model.add(Conv2D(cfg["num_filters"], (cfg["square_filter_size"], cfg["square_filter_size"]), activation='relu', input_shape=(84, 84, 3)))
model.add(MaxPooling2D((cfg["pool_size"], cfg["pool_size"])))

for _ in range(cfg["conv_layers"]-1):
    model.add(Conv2D(cfg["num_filters"], (cfg["square_filter_size"], cfg["square_filter_size"]), activation='relu'))
    model.add(MaxPooling2D((cfg["pool_size"], cfg["pool_size"])))

model.add(Flatten()) 

model.add(Dense(cfg["dense_layer_dim"], activation='relu'))
model.add(Dense(4)) # final Dense layer has 4 neurons representing the 4 classes

print(model.summary())

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=cfg["epochs"], #batch_size=32,
                    validation_data=(test_images, test_labels))

val_loss, val_acc = model.evaluate(test_images, test_labels, verbose=1)

prediction = model.predict(secret_images, verbose=1)

model.save("checkpoint_dir/latest.h5")

predicted_label = []
incorrect_labels = []

for i in range(len(prediction)):
    if secret_labels[i] != np.argmax(prediction[i]):
        incorrect_labels.append(secret_labels[i])
        predicted_label.append(np.argmax(prediction[i]))

print("The number of incorrect labels is:", len(incorrect_labels))

incorrect_1, incorrect_2, incorrect_3, incorrect_4 = 0, 0, 0, 0

test_acc = 1 - len(incorrect_labels)/len(prediction)

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

print("Test accuracy", int(test_acc*10000)/100, "%")

c1_acc = 1-incorrect_1/len(incorrect_labels)
c2_acc = 1-incorrect_2/len(incorrect_labels)
c3_acc = 1-incorrect_3/len(incorrect_labels)
c4_acc = 1-incorrect_4/len(incorrect_labels)

print("Class 1 accuracy:", int(c1_acc*10000)/100, "%")
print("Class 2 accuracy:", int(c2_acc*10000)/100, "%")
print("Class 3 accuracy:", int(c3_acc*10000)/100, "%")
print("Class 4 accuracy:", int(c4_acc*10000)/100, "%")
