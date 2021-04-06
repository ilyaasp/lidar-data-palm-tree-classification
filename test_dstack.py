# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:40:00 2021

@author: Who
"""

import numpy as np

image = np.random.randint(100, size=(100, 100, 3))

r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

result = np.dstack((r, g, b))

print("image shape", image.shape)
print("result shape", result.shape)