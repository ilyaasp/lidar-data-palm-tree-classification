# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:58:30 2021

@author: HP
"""


import numpy as np
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf

filename = 'model/cnn_percobaan_5.h5'

model = load_model(filename)

r = 30
img_gt = cv.imread('Region/Region/0_4500/img_rgb0_4500.tif', cv.IMREAD_UNCHANGED) # Ground truth
gr_oil_palm = img_gt[:,:,2];
gr_non_oil_palm = img_gt[:,:,0];

img_rgb = cv.imread('Region/Region/0_4500/rgb_ROI.tif', 1) # ROI
img_grayscale = cv.imread('Region/Region/0_4500/grayscale_ROI.tif', cv.IMREAD_UNCHANGED)
img_b = cv.imread('Region/Region/0_4500/b_ROI.tif', cv.IMREAD_UNCHANGED)
img_g = cv.imread('Region/Region/0_4500/g_ROI.tif', cv.IMREAD_UNCHANGED)
img_r = cv.imread('Region/Region/0_4500/r_ROI.tif', cv.IMREAD_UNCHANGED)
img_dsm = cv.imread('Region/Region/0_4500/dsm_ROI.tif', cv.IMREAD_UNCHANGED)
img_dem = cv.imread('Region/Region/0_4500/dem_ROI.tif', cv.IMREAD_UNCHANGED)
img_intensity = cv.imread('Region/Region/0_4500/intensity_ROI.tif', cv.IMREAD_UNCHANGED)
[baris, kolom, channel] = img_rgb.shape
temp_inc_oil=1;
temp_inc_non_oil=1;
for x in range(1, baris, 20):
    for y in range(1, kolom, 20):
        if ((x+((r*2)))<=baris and (y+((r*2)))<=kolom):
            temp_dem = img_dem[x:x+60,y:y+60]
            temp_dsm = img_dsm[x:x+60,y:y+60]
            scaler = MinMaxScaler(feature_range=(np.min(np.min(temp_dem)), np.max(np.max(temp_dem)))) 
            temp_intensity = scaler.fit_transform(img_intensity[x:x+60,y:y+60])
            
            temp_concat = list(np.dstack((temp_dem, temp_dsm, temp_intensity)))
            temp_concat = tf.expand_dims(temp_concat, axis=0)
            # pred = model.predict_classes(temp_concat) # predict.classes for tf model
            # np.argmax(model.predict(x), axis=-1) for multi class
            pred = (model.predict(temp_concat) > 0.5).astype("int32") # kalo true jadi 1 dan false jadi 0
            if (pred==1):
                img_rgb =cv.circle(img_rgb, (x+30,y+30), 3, (0,0,255),cv.FILLED)
cv.imwrite("output/output_cnn_percobaan_5_2.tif", img_rgb)
cv.namedWindow('Image RGB', cv.WINDOW_AUTOSIZE)
cv.imshow('Image RGB', img_rgb)
cv.waitKey(0)
cv.destroyAllWindows()