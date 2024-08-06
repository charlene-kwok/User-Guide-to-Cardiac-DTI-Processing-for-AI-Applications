#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:34:50 2024

@author: charlene
"""

import import_functions
import helper_functions
import unwrap_functions
import local_SD_analysis_functions

from import_functions import get_, pp,rotate_imgs_auto, crop, mask
from helper_functions import to_dist, plot_dists, helper_cmaps, plot_ims, pims
from unwrap_functions import uw2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import random
import pandas as pd
# from local_SD_analysis_functions import calc_var
from sklearn.model_selection import train_test_split
import pickle
import os

# Specify the path to your Excel file
file_path = '/Users/charlene/Desktop/UROP/Data/insertion_points.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

x = df['x']
y = df['y']

param = 'E2A'
method = 'STEAM'
hs = get_('Healthy',method,'SYSTOLE',param,'myo')
hd = get_('Healthy',method,'DIASTOLE',param,'myo')
uhs = get_('HCM',method,'SYSTOLE',param,'myo')
uhd = get_('HCM',method,'DIASTOLE',param,'myo')

hd = rotate_imgs_auto(hd)

plt.figure(1)
img2 = hd[0][5]
x1 = 41
y1 = 45
plt.imshow(img2, **helper_cmaps(img2))
plt.grid(True)
plt.plot(x1, y1, 'kx')

hd2 = pp(hd)

hd_imgs = hd[0] #retrieve only images not maps
hs_imgs = hs[0] #retrieve only images not maps
uhd_imgs = uhd[0] #retrieve only images not maps
uhs_imgs = uhs[0] #retrieve only images not maps

# #hd image 1
# plt.figure(1)
# img1 = hd[0][0]
# x = 29
# y = 50
# plt.imshow(img1, **helper_cmaps(img1))
# plt.grid(True)
# plt.plot(x, y, 'kx')


# diff = []

# for i in range(0,15):
#     maxx = (hd[0][i]).shape[1]
#     maxx_cropped = (hd2[0][i]).shape[1]
#     diff.append(maxx-maxx_cropped)
# for i in range(0,15):
#     x[i] = x[i]-diff[i]
    
# hd image 2
plt.figure(2)
img2 = hd2[0][5]
plt.imshow(img2, **helper_cmaps(img2))
plt.grid(True)
plt.plot(x[5], y[5], 'kx')
    
# #hd image 3
# plt.figure(3)
# img3 = hd[0][2]
# x = 35
# y = 48
# plt.imshow(img3, **helper_cmaps(img3))
# plt.grid(True)
# plt.plot(x, y, 'kx')

# #hd image 4
# plt.figure(4)
# img4 = hd[0][3]
# x = 36
# y = 37
# plt.imshow(img4, **helper_cmaps(img4))
# plt.grid(True)
# plt.plot(x, y, 'kx')

#hd image 5
# plt.figure(1)
# img = hd[0][14]
# x = 42
# y = 41
# plt.imshow(img, **helper_cmaps(img))
# plt.grid(True)
# plt.plot(x, y, 'kx')

# plt.figure(3)

# # Create a figure with a grid of 3x5 subplots
# fig, axes = plt.subplots(3, 5, figsize=(15, 9))

# # Loop over the subplots and plot the images
# for i in range(15):
#     row = i // 5
#     col = i % 5
#     img = hd2[0][i]
#     ax = axes[row, col]
#     ax.imshow(img, **helper_cmaps(img))
#     ax.plot(x[i], y[i], 'kx')
#     ax.grid(True)
#     ax.set_xticks([])
#     ax.set_yticks([])

# # Adjust layout
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# pims(hd_imgs)
# pims(hs_imgs)
# pims(uhd_imgs)
# pims(uhs_imgs)

# h_imgs = hd[0] #retrieve only images not maps
# uh_imgs = uhd[0] #retrieve only images not maps
# X_train,X_test,y_train,y_test = gen_data_split(h_imgs,uh_imgs,5) #create data split, taking 5 from healthy and 5 from unhealthy
# X_train,X_test = normalise(X_train,X_test)
# # X_train,X_test = standardise(X_train,X_test)
# X_train,X_test = resize_images(X_train,X_test)

# data = X_train, y_train, X_test, y_test
