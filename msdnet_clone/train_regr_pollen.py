#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#-----------------------------------------------------------------------
#Copyright 2019 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/msdnet_clone.
#License: MIT
#
#This file is part of msdnet_clone. a Python implementation of the
#Mixed-Scale Dense Convolutional Neural Network.
#-----------------------------------------------------------------------

"""
Example 01: Train a network for regression
==========================================

This script trains a MS-D network for regression (i.e. denoising/artifact removal)
Run generatedata.py first to generate required training data.
"""

# Import code
import msdnet_clone
import glob
from pathlib import Path

NUM_TRAIN = 100  # number of training images to use per class
NUM_VAL = 50    # number of validation images to use per class

# Define dilations in [1,10] as in paper.
dilations = msdnet_clone.dilations.IncrementDilations(10)

# Create main network object for regression, with 100 layers,
# [1,10] dilations, 1 input channel, 1 output channel, using
# the GPU (set gpu=False to use CPU)
n = msdnet_clone.network.MSDNet(100, dilations, 3, 1, gpu=False)

# Initialize network parameters
n.initialize()

# Define training data
flsin_1 = sorted(glob.glob('train/1/*.tiff'))
flsin_1 = flsin_1[:min(NUM_TRAIN, len(flsin_1))]
flstg_1 = []
for _ in range(0, len(flsin_1)):
    flstg_1.append('train/class1_expectation.tiff')
print(len(flsin_1))
print(len(flstg_1))

flsin_2 = sorted(glob.glob('train/2/*.tiff'))
flsin_2 = flsin_2[:min(NUM_TRAIN, len(flsin_2))]
flstg_2 = []
for _ in range(0, len(flsin_2)):
    flstg_2.append('train/class2_expectation.tiff')
print(len(flsin_2))
print(len(flstg_2))

flsin_3 = sorted(glob.glob('train/3/*.tiff'))
flsin_3 = flsin_3[:min(NUM_TRAIN, len(flsin_3))]
flstg_3 = []
for _ in range(0, len(flsin_3)):
    flstg_3.append('train/class3_expectation.tiff')
print(len(flsin_3))
print(len(flstg_3))

flsin_4 = sorted(glob.glob('train/4/*.tiff'))
flsin_4 = flsin_4[:min(NUM_TRAIN, len(flsin_4))]
flstg_4 = []
for _ in range(0, len(flsin_4)):
    flstg_4.append('train/class4_expectation.tiff')
print(len(flsin_4))
print(len(flstg_4))

flsin = flsin_1 + flsin_2 + flsin_3 + flsin_4
flstg = flstg_1 + flstg_2 + flstg_3 + flstg_4
print(len(flsin))
print(len(flstg))

# Create list of datapoints (i.e. input/target pairs)
dats = []
for i in range(len(flsin)):
    # Create datapoint with file names
    d = msdnet_clone.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    # Augment data by rotating and flipping
    d_augm = msdnet_clone.data.RotateAndFlipDataPoint(d)
    # Add augmented datapoint to list
    dats.append(d_augm)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# dats = msdnet_clone.utils.load_simple_data('train/noisy/*.tiff', 'train/noiseless/*.tiff', augment=True)

# Normalize input and output of network to zero mean and unit variance using
# training data images
n.normalizeinout(dats)

# Use image batches of a single image
bprov = msdnet_clone.data.BatchProvider(dats,32)

# Define validation data (not using augmentation)
flsin_1 = sorted(glob.glob('val/1/*.tiff'))
flsin_1 = flsin_1[:min(NUM_VAL, len(flsin_1))]
flstg_1 = []
for _ in range(0, len(flsin_1)):
    flstg_1.append('val/class1_expectation.tiff')
print(len(flsin_1))
print(len(flstg_1))
flsin_2 = sorted(glob.glob('val/2/*.tiff'))
flsin_2 = flsin_2[:min(NUM_VAL, len(flsin_2))]
flstg_2 = []
for _ in range(0, len(flsin_2)):
    flstg_2.append('val/class2_expectation.tiff')
print(len(flsin_2))
print(len(flstg_2))
flsin_3 = sorted(glob.glob('val/3/*.tiff'))
flsin_3 = flsin_3[:min(NUM_VAL, len(flsin_3))]
flstg_3 = []
for _ in range(0, len(flsin_3)):
    flstg_3.append('val/class3_expectation.tiff')
print(len(flsin_3))
print(len(flstg_3))
flsin_4 = sorted(glob.glob('val/4/*.tiff'))
flsin_4 = flsin_4[:min(NUM_VAL, len(flsin_4))]
flstg_4 = []
for _ in range(0, len(flsin_4)):
    flstg_4.append('val/class4_expectation.tiff')
print(len(flsin_4))
print(len(flstg_4))
flsin = flsin_1 + flsin_2 + flsin_3 + flsin_4
flstg = flstg_1 + flstg_2 + flstg_3 + flstg_4
print(len(flsin))
datsv = []
for i in range(len(flsin)):
    d = msdnet_clone.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    datsv.append(d)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# datsv = msdnet_clone.utils.load_simple_data('val/noisy/*.tiff', 'val/noiseless/*.tiff', augment=False)

# Select loss function
l2loss = msdnet_clone.loss.L2Loss()

# Validate with loss function
val = msdnet_clone.validate.LossValidation(datsv, loss=l2loss)

# Use ADAM training algorithms
t = msdnet_clone.train.AdamAlgorithm(n, loss=l2loss)

# Log error metrics to console
consolelog = msdnet_clone.loggers.ConsoleLogger()
# Log error metrics to file
filelog = msdnet_clone.loggers.FileLogger('log_regr.txt')
# Log typical, worst, and best images to image files
imagelog = msdnet_clone.loggers.ImageLogger('log_regr', onlyifbetter=True)

# Train network until program is stopped manually
# Network parameters are saved in regr_params.h5
# Validation is run after every len(datsv) (=25)
# training steps.
msdnet_clone.train.train(n, t, val, bprov, 'regr_params.h5',loggers=[consolelog,filelog,imagelog], val_every=len(datsv))


# In[2]:


import numpy as np


# In[3]:


flsin_4 = sorted(glob.glob('train/4/*.tiff'))
datapoints = []
for i in range(20):
    datapoints.append(msdnet_clone.data.ImageFileDataPoint(str(flsin_4[i]))) # all class 3 images
outputs = []
for i in range(20):
    outputs.append(n.forward(datapoints[i].input))
for i in range(20):
    class_1_space = outputs[i][0][:21, 0:83]
    class_1_score = np.sum(class_1_space)
    class_2_space = outputs[i][0][22:42, 0:83]
    class_2_score = np.sum(class_2_space)
    class_3_space = outputs[i][0][43:63, 0:83]
    class_3_score = np.sum(class_3_space)
    class_4_space = outputs[i][0][64:84, 0:83]
    class_4_score = np.sum(class_4_space)
    print(len(outputs[i]))
    print("class 1 score: " + str(class_1_score))
    print("class 2 score: " + str(class_2_score))
    print("class 3 score: " + str(class_3_score))
    print("class 4 score: " + str(class_4_score))
    print()


# In[ ]:




