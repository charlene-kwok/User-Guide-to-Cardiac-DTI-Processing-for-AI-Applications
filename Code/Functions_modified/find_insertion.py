#find_insertion

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


# Read all sheets into a dictionary of DataFrames
df_hd = pd.read_excel(file_path, sheet_name='hd_STEAM')
df_hs = pd.read_excel(file_path, sheet_name='hs_STEAM')
df_uhd = pd.read_excel(file_path, sheet_name='uhd_STEAM')
df_uhs = pd.read_excel(file_path, sheet_name='uhs_STEAM')

x = df_uhd['x']
y = df_uhd['y']

param = 'E2A'
method = 'STEAM'
hs = get_('Healthy',method,'SYSTOLE',param,'myo')
hd = get_('Healthy',method,'DIASTOLE',param,'myo')
uhd = get_('HCM',method,'DIASTOLE',param,'myo')
uhs = get_('HCM',method,'SYSTOLE',param,'myo')


uhs_SE = get_('HCM','SE','DIASTOLE',param,'myo')
pims(uhs_SE[0])
pims(uhs[0])

uhd = rotate_imgs_auto(uhd)

plt.figure(1)
img2 = uhd[0][10]
x1 = 43
y1 = 48
plt.imshow(img2, **helper_cmaps(img2))
plt.grid(True)
plt.plot(x1, y1, 'kx')

uhd2 = pp(uhd)


# Create a figure with a grid of 3x5 subplots
fig, axes = plt.subplots(2, 6, figsize=(15, 9))

# Loop over the subplots and plot the images
for i in range(11):
    row = i // 6
    col = i % 6
    img = uhs[0][i]
    ax = axes[row, col]
    ax.imshow(img, **helper_cmaps(img))
    ax.plot(x[i], y[i], 'kx')
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust layout
plt.grid(True)
plt.tight_layout()
plt.show()


diff = []

for i in range(0,11):
    maxx = (uhd[0][i]).shape[1]
    maxx_cropped = (uhd2[0][i]).shape[1]
    diff.append(maxx-maxx_cropped)
for i in range(0,11):
    x[i] = x[i]-diff[i]


# Create a figure with a grid of 3x5 subplots
fig, axes = plt.subplots(2,6, figsize=(15, 9))

# Loop over the subplots and plot the images
for i in range(11):
    row = i // 6
    col = i % 6
    img = uhd2[0][i]
    ax = axes[row, col]
    ax.imshow(img, **helper_cmaps(img))
    ax.plot(x[i], y[i], 'kx',markersize=14, markeredgewidth=4)
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust layout
plt.grid(True)
plt.tight_layout()
plt.show()
