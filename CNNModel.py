# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:18:21 2021

@author: Who
"""

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import random
import os
from shutil import copyfile

def model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(60, 60, 3)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

num_oil_palm1 = 586
num_non_oil_palm1 = 1210
num_oil_palm2 = 709
num_non_oil_palm2 = 381

print(len(os.listdir('dataset/img_rgb3000_1500/kelas_sawit/rgb/')))
print(len(os.listdir('dataset/img_rgb3000_1500/bukan_sawit/rgb/')))
print(len(os.listdir('dataset/img_rgb0_4500/kelas_sawit/rgb/')))
print(len(os.listdir('dataset/img_rgb0_4500/bukan_sawit/rgb/')))

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
  files = []
  for filename in os.listdir(SOURCE):
    file = SOURCE + filename
    if os.path.getsize(file) > 0:
      files.append(filename)      
    else:
      print(filename + " is zero length, so ignoring.")

  training_length = int(len(files) * SPLIT_SIZE)
  testing_length = int(len(files) - training_length)
  shuffled_set = random.sample(files, len(files))
  training_set = shuffled_set[0:training_length]
  testing_set = shuffled_set[:testing_length]

  for filename in training_set:
    this_file = SOURCE + filename
    dest_file = TRAINING + filename
    copyfile(this_file, dest_file)
  
  for filename in testing_set:
    this_file = SOURCE + filename
    dest_file = TESTING + filename
    copyfile(this_file, dest_file)


SAWIT_SOURCE = 'dataCNN/kelas_sawit/'
TRAINING_SAWIT = 'dataCNN/training/kelas_sawit/'
TESTING_SAWIT = 'dataCNN/testing/kelas_sawit/'
NON_SAWIT_SOURCE = 'dataCNN/bukan_sawit/'
TRAINING_NON_SAWIT = 'dataCNN/training/bukan_sawit/'
TESTING_NON_SAWIT = 'dataCNN/testing/bukan_sawit/'

split_size = 0.9
split_data(SAWIT_SOURCE, TRAINING_SAWIT, TESTING_SAWIT, split_size)
split_data(NON_SAWIT_SOURCE, TRAINING_NON_SAWIT, TESTING_NON_SAWIT, split_size)

print(len(os.listdir('dataCNN/training/kelas_sawit/')))
print(len(os.listdir('dataCNN/training/bukan_sawit/')))
print(len(os.listdir('dataCNN/testing/kelas_sawit/')))
print(len(os.listdir('dataCNN/testing/bukan_sawit/')))
    
model = model()
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

TRAINING_DIR = 'dataCNN/training/'
TESTING_DIR = 'dataCNN/testing/'

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255.)

BATCH_SIZE = 100 # 50, 250

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=BATCH_SIZE, 
    class_mode='binary',
    target_size=(60, 60)
)

validation_generator = train_datagen.flow_from_directory(
    TESTING_DIR,
    batch_size=BATCH_SIZE, 
    class_mode='binary',
    target_size=(60, 60)
)



if __name__ == '__main__':
    history = model.fit(train_generator,
                        epochs=15,
                        #steps_per_epoch=225,
                        validation_data=validation_generator,
                        #validation_steps=25,
                        verbose=1)





