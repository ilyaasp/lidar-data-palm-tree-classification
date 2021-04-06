# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:39:30 2020

@author: HP
"""

import numpy as np
import cv2 as cv

img_rgb = cv.imread('rgb.tif', 1)
imgGrayscale = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
img_dsm = cv.imread('dsm.tif', cv.IMREAD_UNCHANGED)
img_dem = cv.imread('dem.tif', cv.IMREAD_UNCHANGED)
img_intensity = cv.imread('intensity.tif', cv.IMREAD_UNCHANGED)
img_intensity = img_intensity[:,:,0]
img_blue = img_rgb[:,:,0]
img_green = img_rgb[:,:,1]
img_red = img_rgb[:,:,2]

#setting
x1= 3000
x2= 4500
y1= 1500
y2= 3000

rgbROI = img_rgb[x1:x2, y1:y2]
imgGrayscaleROI = imgGrayscale[x1:x2, y1:y2]
bROI = img_blue[x1:x2, y1:y2]
gROI = img_green[x1:x2, y1:y2]
rROI = img_red[x1:x2, y1:y2]
dsmROI = img_dsm[x1:x2, y1:y2]
demROI = img_dem[x1:x2, y1:y2]
intensityROI = img_intensity[x1:x2, y1:y2]

cv.imwrite("Region/Region/3000_1500/rgb_ROI.tif", rgbROI)
cv.imwrite("Region/Region/3000_1500/grayscale_ROI.tif", imgGrayscaleROI)
cv.imwrite("Region/Region/3000_1500/b_ROI.tif", bROI)
cv.imwrite("Region/Region/3000_1500/g_ROI.tif", gROI)
cv.imwrite("Region/Region/3000_1500/r_ROI.tif", rROI)
cv.imwrite("Region/Region/3000_1500/dsm_ROI.tif", dsmROI)
cv.imwrite("Region/Region/3000_1500/dem_ROI.tif", demROI)
cv.imwrite("Region/Region/3000_1500/intensity_ROI.tif", intensityROI)



cv.namedWindow('Image b', cv.WINDOW_NORMAL)
cv.imshow('Image b', bROI)
cv.namedWindow('Image g', cv.WINDOW_NORMAL)
cv.imshow('Image g', gROI)
cv.namedWindow('Image r', cv.WINDOW_NORMAL)
cv.imshow('Image r', rROI)
cv.waitKey(0)
cv.destroyAllWindows()