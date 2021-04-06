# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:19:32 2021

@author: Who
"""

from os import listdir
from PIL import Image
   
count = 0
for filename in listdir('dataCNN/training/kelas_sawit/'):
  if filename.endswith('.tif'):
    try:
      img = Image.open('dataCNN/training/kelas_sawit/'+filename) # open the image file
      img.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
      count = count + 1
      print('Bad file:', filename) # print out the names of corrupt files
      
print(count)