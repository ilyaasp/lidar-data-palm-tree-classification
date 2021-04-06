# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:06:07 2021

@author: Who
"""

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


temp_dem = cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dem/1.tif', cv.IMREAD_UNCHANGED) # dibelakang -1 dia kebelakang, dan sebaliknya
temp_dsm = cv.imread('dataset/img_rgb3000_1500/bukan_sawit/dsm/1.tif', cv.IMREAD_UNCHANGED)

scaler = MinMaxScaler(feature_range=(np.min(temp_dem), np.max(temp_dem))) 

temp_intensity = scaler.fit_transform(cv.imread('dataset/img_rgb3000_1500/bukan_sawit/intensity/1.tif', cv.IMREAD_UNCHANGED))

#x = np.concatenate((temp_dem, temp_dsm, temp_intensity), axis=1).reshape(1, -1) # concate axis=1 equals to hstack, concate axis=0 equals to
                                                                # vstack
temp = np.dstack((temp_dem, temp_dsm, temp_intensity))
x = tf.cast(tf.constant(temp), dtype='float32')

y = tf.constant([1., 2.])
