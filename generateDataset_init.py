# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:39:30 2020

@author: HP
"""

import numpy as np
import cv2 as cv
r = 30
img_gt = cv.imread('Region/Region/3000_1500/img_rgb3000_1500.tif', cv.IMREAD_UNCHANGED)
gr_oil_palm = img_gt[:,:,2];
gr_non_oil_palm = img_gt[:,:,0];

img_rgb = cv.imread('Region/Region/3000_1500/rgb_ROI.tif', 1)
img_grayscale = cv.imread('Region/Region/3000_1500/grayscale_ROI.tif', cv.IMREAD_UNCHANGED)
img_b = cv.imread('Region/Region/3000_1500/b_ROI.tif', cv.IMREAD_UNCHANGED)
img_g = cv.imread('Region/Region/3000_1500/g_ROI.tif', cv.IMREAD_UNCHANGED)
img_r = cv.imread('Region/Region/3000_1500/r_ROI.tif', cv.IMREAD_UNCHANGED)
img_dsm = cv.imread('Region/Region/3000_1500/dsm_ROI.tif', cv.IMREAD_UNCHANGED)
img_dem = cv.imread('Region/Region/3000_1500/dem_ROI.tif', cv.IMREAD_UNCHANGED)
img_intensity = cv.imread('Region/Region/3000_1500/intensity_ROI.tif', cv.IMREAD_UNCHANGED)
[baris, kolom, channel] = img_rgb.shape
temp_inc_oil=1;
temp_inc_non_oil=1;
for x in range(1, baris, 1):
    for y in range(1, kolom, 1):
        if ((x+((r*2)))<=baris and (y+((r*2)))<=kolom):
            if (gr_oil_palm[x+r,y+r]==255):
                cv.imwrite("dataset/img_rgb3000_1500/kelas_sawit/b/"+str(temp_inc_oil)+".tif", img_b[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/kelas_sawit/g/"+str(temp_inc_oil)+".tif", img_g[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/kelas_sawit/r/"+str(temp_inc_oil)+".tif", img_r[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/kelas_sawit/dem/"+str(temp_inc_oil)+".tif", img_dem[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/kelas_sawit/dsm/"+str(temp_inc_oil)+".tif", img_dsm[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/kelas_sawit/intensity/"+str(temp_inc_oil)+".tif", img_intensity[x:(x+((r*2))), y:(y+((r*2)))])
                temp_inc_oil+=1
            if (gr_non_oil_palm[x+r,y+r]==255):
                cv.imwrite("dataset/img_rgb3000_1500/bukan_sawit/b/"+str(temp_inc_non_oil)+".tif", img_b[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/bukan_sawit/g/"+str(temp_inc_non_oil)+".tif", img_g[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/bukan_sawit/r/"+str(temp_inc_non_oil)+".tif", img_r[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/bukan_sawit/dem/"+str(temp_inc_non_oil)+".tif", img_dem[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/bukan_sawit/dsm/"+str(temp_inc_non_oil)+".tif", img_dsm[x:(x+((r*2))), y:(y+((r*2)))])
                cv.imwrite("dataset/img_rgb3000_1500/bukan_sawit/intensity/"+str(temp_inc_non_oil)+".tif", img_intensity[x:(x+((r*2))), y:(y+((r*2)))])
                temp_inc_non_oil+=1



cv.namedWindow('Image RGB', cv.WINDOW_AUTOSIZE)
cv.imshow('Image RGB', img_b[1:60,1:60])
cv.waitKey(0)
cv.destroyAllWindows()