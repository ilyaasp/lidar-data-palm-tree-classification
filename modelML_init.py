# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:37:24 2021

@author: HP
"""

import numpy as np
import cv2 as cv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle

num_oil_palm1 = 586
num_non_oil_palm1 = 1210
num_oil_palm2 = 709
num_non_oil_palm2 = 381

X = []
label = []


for x in range(1, num_oil_palm1+1, 1):
    #tidak dinormalisasi
    temp_dem = np.reshape(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    temp_dsm = np.reshape(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_intensity = np.reshape(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_r = np.reshape(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_g = np.reshape(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_b = np.reshape(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    
    scaler = MinMaxScaler(feature_range=(min(temp_dem), max(temp_dem)))
    
    #dengan dinormalisasi
    # temp_dem = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    # temp_dsm = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_intensity = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_r = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_g = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_b = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/kelas_sawit/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    
    temp_concat = list(np.concatenate((temp_dem, temp_dsm, temp_intensity), axis=0))
    X.append(temp_concat)
    label.append(1)

for x in range(1, num_oil_palm2+1, 1):
    #tidak dinormalisasi
    temp_dem = np.reshape(cv.imread('dataset/img_rgb0_4500/kelas_sawit/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    temp_dsm = np.reshape(cv.imread('dataset/img_rgb0_4500/kelas_sawit/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_intensity = np.reshape(cv.imread('dataset/img_rgb0_4500/kelas_sawit/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_r = np.reshape(cv.imread('dataset/img_rgb0_4500/kelas_sawit/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_g = np.reshape(cv.imread('dataset/img_rgb0_4500/kelas_sawit/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_b = np.reshape(cv.imread('dataset/img_rgb0_4500/kelas_sawit/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    
    scaler = MinMaxScaler(feature_range=(min(temp_dem), max(temp_dem)))
    
    #dengan dinormalisasi
    # temp_dem = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb0_4500/kelas_sawit/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    # temp_dsm = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb0_4500/kelas_sawit/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_intensity = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb0_4500/kelas_sawit/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_r = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb0_4500/kelas_sawit/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_g = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb0_4500/kelas_sawit/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_b = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb0_4500/kelas_sawit/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    
    temp_concat = list(np.concatenate((temp_dem, temp_dsm, temp_intensity), axis=0))
    X.append(temp_concat)
    label.append(1)
    
for x in range(1, num_non_oil_palm1+1, 1):
    #tidak dinormalisasi
    temp_dem = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    temp_dsm = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_intensity = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_r = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_g = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_b = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    
    scaler = MinMaxScaler(feature_range=(min(temp_dem), max(temp_dem)))
    
    #dengan dinormalisasi
    # temp_dem = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    # temp_dsm = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_intensity = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_r = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_g = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_b = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    
    
    temp_concat = list(np.concatenate((temp_dem, temp_dsm, temp_intensity), axis=0))
    X.append(temp_concat)
    label.append(0)

for x in range(1, num_non_oil_palm2+1, 1):
    #tidak dinormalisasi
    temp_dem = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    temp_dsm = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_intensity = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_r = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_g = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    # temp_b = np.reshape(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED), 3600)
    
    scaler = MinMaxScaler(feature_range=(min(temp_dem), max(temp_dem)))
    
    #dengan dinormalisasi
    # temp_dem = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dem/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    # temp_dsm = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dsm/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_intensity = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/intensity/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_r = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/r/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_g = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/g/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    temp_b = np.reshape(scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/b/'+str(x)+'.tif', cv.IMREAD_UNCHANGED)), 3600)
    
    
    temp_concat = list(np.concatenate((temp_dem, temp_dsm, temp_intensity), axis=0))
    X.append(temp_concat)
    label.append(0)


X_train, X_test, y_train, y_test = train_test_split(X, label, stratify=label,  test_size=0.2, random_state=1)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))
    
pred = clf.predict(X_test)
print(confusion_matrix(y_test, pred))

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))