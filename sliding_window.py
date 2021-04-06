# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:58:30 2021

@author: HP
"""


import numpy as np
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler
import pickle
from keras.models import load_model

filename = 'cnn_model.h5'

model = load_model(filename)
r = 30
img_gt = cv.imread('Region/Region/0_4500/img_rgb0_4500.tif', cv.IMREAD_UNCHANGED)
gr_oil_palm = img_gt[:,:,2];
gr_non_oil_palm = img_gt[:,:,0];

img_rgb = cv.imread('Region/Region/0_4500/rgb_ROI.tif', 1)
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
for x in range(1, baris, 15):
    for y in range(1, kolom, 15):
        if ((x+((r*2)))<=baris and (y+((r*2)))<=kolom):
            #temp_dem = np.reshape(img_dem[x:x+60,y:y+60], 3600).reshape(1,-1)
            #temp_dsm = np.reshape(img_dsm[x:x+60,y:y+60], 3600).reshape(1,-1)
            #scaler = MinMaxScaler(feature_range=(min(min(temp_dem)), max(max(temp_dem))))
            #temp_intensity = np.reshape(scaler.fit_transform(img_intensity[x:x+60,y:y+60]),3600).reshape(1,-1)
            
            #temp_concat = list(np.concatenate((temp_dem, temp_dsm, temp_intensity), axis=1))
            
            img_rgb_x = img_rgb[x:(x+((r*2))), y:(y+((r*2)))]
            #img_rgb_x = np.reshape(img_rgb_x, [1, 60, 60, 3])
            img_rgb_x = np.expand_dims(img_rgb_x, axis=0)
            image = np.vstack([img_rgb_x])
            
            pred = model.predict_classes(image)
            if (pred==1):
                img_rgb =cv.circle(img_rgb, (x+30,y+30), 3, (0,0,255),cv.FILLED)
cv.imwrite("output_cnn_new_model.tif", img_rgb)
cv.namedWindow('Image RGB', cv.WINDOW_AUTOSIZE)
cv.imshow('Image RGB', img_rgb)
cv.waitKey(0)
cv.destroyAllWindows()