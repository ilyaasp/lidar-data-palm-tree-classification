# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:30:40 2021

@author: Who
"""
import kerastuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


SAMPLE = len(label)
X = tf.cast(tf.constant(X), dtype="float32")
label = tf.cast(tf.constant(label), dtype="float32")
train_X, test_X, val_X = tf.split(X, [int(SAMPLE*0.7), int(SAMPLE*0.1)+1, int(SAMPLE*0.2)], 0)
train_label, test_label, val_label = tf.split(label, [int(SAMPLE*0.7), int(SAMPLE*0.1)+1, int(SAMPLE*0.2)], 0)

print(len(train_X), 'train examples')
print(len(val_X), 'validation examples')
print(len(test_X), 'test examples')


def model_builder(hp):
    model = keras.Sequential()
    
    model.add(layers.Conv2D(hp.Int("units_1", min_value=16, max_value=512, step=16), (3, 3), activation=hp.Choice("act_1", ['relu', 'tanh']), padding=hp.Choice("pad_1", ['same', 'valid']), input_shape=(60, 60, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(hp.Int("units_2", min_value=16, max_value=512, step=16), (3, 3), activation=hp.Choice("act_2", ['relu', 'tanh'])))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    for i in range(hp.Int('num_hidden_layers', 1, 5)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i+2), min_value=16, max_value=512, step=16), activation=hp.Choice("act_" + str(i+2), ['relu', 'tanh'])))
    model.add(layers.Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.9, step=0.1)))
    model.add(layers.Dense(1))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='kelapa_sawit_clf_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(train_X, train_label, epochs=50, validation_data=(val_X, val_label), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the first Conv2D layer is {best_hps.get('units_1')} , activation {best_hps.get('act_1')} and padding {best_hps.get('pad_1')}. 
Second Conv2D layer best unit is {best_hps.get('units_2')}, activation {best_hps.get('act_2')}. The optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}. And the best Dropout is {best_hps.get('dropout_1')}
""")

best_num_hidden_layers = best_hps.get('num_hidden_layers')

if best_num_hidden_layers == 1:
    print(f"""
    The optimal number of units in the first dense layer is {best_hps.get('units_3')} , activation {best_hps.get('act_3')}.
    """)
elif best_num_hidden_layers == 2:
    print(f"""
    The optimal number of units in the first dense layer is {best_hps.get('units_3')} , activation {best_hps.get('act_3')}.
    Second dense layer units is {best_hps.get('units_4')} , activation {best_hps.get('act_4')}
    """)
elif best_num_hidden_layers == 3:
    print(f"""
    The optimal number of units in the first dense layer is {best_hps.get('units_3')} , activation {best_hps.get('act_3')}.
    Second dense layer units is {best_hps.get('units_4')} , activation {best_hps.get('act_4')}
    Third dense layer units is {best_hps.get('units_5')} , activation {best_hps.get('act_5')}
    """)
elif best_num_hidden_layers == 4:
    print(f"""
    The optimal number of units in the first dense layer is {best_hps.get('units_3')} , activation {best_hps.get('act_3')}.
    Second dense layer units is {best_hps.get('units_4')} , activation {best_hps.get('act_4')}
    Third dense layer units is {best_hps.get('units_5')} , activation {best_hps.get('act_5')}
    Third dense layer units is {best_hps.get('units_6')} , activation {best_hps.get('act_6')}
    """)


# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_X, train_label, epochs=100, validation_data=(val_X, val_label))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(train_X, train_label, epochs=best_epoch, validation_data=(val_X, val_label))


eval_result = hypermodel.evaluate(test_X, test_label)
print("[test loss, test accuracy]:", eval_result)

# Save model
hypermodel.save("model/model_cnn_kerastuner.h5")