# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:37:47 2020

@author: HP
"""

import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

img_g = cv.imread('g.tif', cv.IMREAD_UNCHANGED)
template = cv.imread('rgb_ROI_51_671.tif',cv.IMREAD_UNCHANGED)
w, h = template.shape[::-1]

res = cv.matchTemplate(img_g,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_g, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv.imwrite('res.png',img_g)