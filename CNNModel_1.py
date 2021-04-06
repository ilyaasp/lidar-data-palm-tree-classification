# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 08:21:16 2021

@author: Who
"""

import time
# Waktu mulai setelah import library time
start = time.perf_counter()

import pandas as pd
import numpy as np
import cv2 as cv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle
import tensorflow as tf
from tensorflow.keras import models, layers

def get_data_palm(PATH, NUM, X, label):
    for x in range(1, NUM+1, 1):
        # tidak dinormalisasi
        temp_dem = cv.imread(PATH+'/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        temp_dsm = cv.imread(PATH+'/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        # temp_intensity = cv.imread(PATH+'/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        # temp_r = cv.imread(PATH+'/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        # temp_g = cv.imread(PATH+'/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        # temp_b = cv.imread(PATH+'/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        
        scaler = MinMaxScaler(feature_range=(np.min(temp_dem), np.max(temp_dem)))
    
        #dengan dinormalisasi
        # temp_dem = scaler.fit_transform(cv.imread(PATH+'/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        # temp_dsm = scaler.fit_transform(cv.imread(PATH+'/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        temp_intensity = scaler.fit_transform(cv.imread(PATH+'/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        temp_r = scaler.fit_transform(cv.imread(PATH+'/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        temp_g = scaler.fit_transform(cv.imread(PATH+'/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        temp_b = scaler.fit_transform(cv.imread(PATH+'/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        
        temp_concat = list(np.dstack((temp_dem, temp_dsm, temp_intensity)))
        X.append(temp_concat)
        label.append(1)

def get_data_non_palm(PATH, NUM, X, label):
    for x in range(1, NUM+1, 1):
        # tidak dinormalisasi
        temp_dem = cv.imread(PATH+'/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        temp_dsm = cv.imread(PATH+'/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        # temp_intensity = cv.imread(PATH+'/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        # temp_r = cv.imread(PATH+'/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        # temp_g = cv.imread(PATH+'/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        # temp_b = cv.imread(PATH+'/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)
        
        scaler = MinMaxScaler(feature_range=(np.min(temp_dem), np.max(temp_dem)))
    
        #dengan dinormalisasi
        # temp_dem = scaler.fit_transform(cv.imread(PATH+'/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        # temp_dsm = scaler.fit_transform(cv.imread(PATH+'/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        temp_intensity = scaler.fit_transform(cv.imread(PATH+'/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        temp_r = scaler.fit_transform(cv.imread(PATH+'/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        temp_g = scaler.fit_transform(cv.imread(PATH+'/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        temp_b = scaler.fit_transform(cv.imread(PATH+'/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED))
        
        temp_concat = list(np.dstack((temp_dem, temp_dsm, temp_intensity)))
        X.append(temp_concat)
        label.append(0)

num_oil_palm1 = 586
num_non_oil_palm1 = 1210
num_oil_palm2 = 709
num_non_oil_palm2 = 381

X = []
label = []

PATH_PALM_OIL1 = 'dataset/img_rgb3000_1500/kelas_sawit'
PATH_PALM_OIL2 = 'dataset/img_rgb0_4500/kelas_sawit'
PATH_NON_PALM_OIL1 = 'dataset/img_rgb3000_1500/bukan_sawit'
PATH_NON_PALM_OIL2 = 'dataset/img_rgb3000_1500/bukan_sawit'

get_data_palm(PATH_PALM_OIL1, num_oil_palm1, X, label)
get_data_palm(PATH_PALM_OIL2, num_oil_palm2, X, label)
get_data_non_palm(PATH_NON_PALM_OIL1, num_non_oil_palm1, X, label)
get_data_non_palm(PATH_NON_PALM_OIL2, num_non_oil_palm2, X, label)

# SPLIT TEST, TRAIN, VALIDATION SET
# train_X, test_X, train_label, test_label = train_test_split(X, label, stratify=label, test_size=0.2, random_state=1)
# train_X, val_X, train_label, val_label = train_test_split(train_X, train_label, stratify=train_label, test_size=0.2, random_state=1)
# print(len(train_X), 'train examples')
# print(len(val_X), 'validation examples')
# print(len(test_X), 'test examples')

# train_X = tf.cast(tf.constant(train_X), dtype="float32")
# test_X = tf.cast(tf.constant(test_X), dtype="float32")
# val_X = tf.cast(tf.constant(val_X), dtype="float32")

# train_label = tf.cast(tf.constant(train_label), dtype="float32")
# test_label = tf.cast(tf.constant(test_label), dtype="float32")
# val_label = tf.cast(tf.constant(val_label), dtype="float32")

SAMPLE = len(label)
X = tf.cast(tf.constant(X), dtype="float32")
label = tf.cast(tf.constant(label), dtype="float32")
train_X, test_X, val_X = tf.split(X, [int(SAMPLE*0.7), int(SAMPLE*0.1)+1, int(SAMPLE*0.2)], 0)
train_label, test_label, val_label = tf.split(label, [int(SAMPLE*0.7), int(SAMPLE*0.1)+1, int(SAMPLE*0.2)], 0)

print(len(train_X), 'train examples')
print(len(val_X), 'validation examples')
print(len(test_X), 'test examples')

# A VGG like model, conv conv maxpool
def model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(60, 60, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(1))
    return model

model = model()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_X, train_label, epochs=50, batch_size=32,
                        validation_data=(val_X, val_label), verbose=1)
    

total = time.perf_counter() - start
print("Waktu yang dibutuhkan: ", total) # dalam detik

# Save model
# model.save("model_cnn.h5")
