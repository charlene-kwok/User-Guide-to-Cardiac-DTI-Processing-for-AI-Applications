#Helper Functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def helper_cmaps(imgs): #Configures plt.imshow() cmaps for images.
    '''
    Configures plt.imshow() cmap settings for plotting. Matches the vmax and vmin across all images
    Use like this: plt.imshow(img, **helper_cmaps(imgs))
    Or if single image: plt.imshow(img, **helper_cmaps([img]))

    Inputs:
    imgs: list of images
    
    '''
    imgs = np.concatenate([np.array(img).flatten() for img in imgs])
    min_val = np.nanmin(imgs)
    max_val = np.nanmax(imgs)
    # pos = dict(cmap=mpl.colormaps['Greys'],vmin=0,vmax=np.nanmax(imgs))
    pos = dict(cmap=mpl.colormaps['viridis'],vmin=0,vmax=np.nanmax(imgs))
    pos_and_neg = dict(cmap=mpl.colormaps['bwr'],vmin=min_val,vmax=max_val)
    if min_val<0:
        return pos_and_neg
    else:
        return pos
    
def pims(imgs,title=None,figsize = (10,6)):
    '''
    Plotting Helper Function for plotting multiple figures at once
    Plots up to 15 images from a list

    Inputs:
    imgs: list of images
    title: title of plot
    figsize: figure size of plot

    '''
    fig, axs = plt.subplots(nrows=3,ncols=5,layout='constrained',figsize=figsize)
    fig.patch.set_facecolor((211/255,238/255,251/255,1))
    for ax in axs.ravel():
        ax.set_axis_off()
    imgs = imgs[0:15]
    for i,img in enumerate(imgs):
        # axs.ravel()[i].set_axis_on()
        im = axs.ravel()[i].imshow(img,**helper_cmaps(imgs))
        axs.ravel()[i].set_title(str(i+1))
        fig.suptitle(title)
    fig.colorbar(im, ax = axs.ravel().tolist(),shrink=0.2,orientation='horizontal')
    plt.show(block=False)

def to_dist(imgs):
    '''
    Converts a list of images to a distribution

    Inputs:
    imgs: list of images

    Outputs:
    dist: 1d distribution of the values in the image.
    '''
    oup = []
    for im in imgs:
        image = im.copy()
        image = image[image!=0]
        image = image[~np.isnan(image)]
        image = image.flatten()
        oup.append(image)
    dist = np.concatenate(oup)
    return dist

def plot_dists(dist_h,dist_uh,bin_n,labels=['Healthy','HCM'],ax = None):
    '''
    Plot Distributions obtained from to_dist(imgs)

    Inputs
    dist_h,dist_uh: the two distributions you want to plot. (N,) Numpy array of pixel values
    bin_n: number of bins to use in histogram
    labels: labels assigned to dist_h,dist_uh respectively
    ax: axis to plot to, if no axis set, new axis created
    '''
    if ax is None:
        fig,ax = plt.subplots()
    # fig.patch.set_facecolor((211/255,238/255,251/255,1))
    bins = np.linspace(min(dist_h.min(),dist_uh.min()),max(dist_h.max(),dist_uh.max()),bin_n,endpoint=True)
    dist_hy, h_bin_edges = np.histogram(dist_h,bins,density=True)
    dist_uhy, uh_bin_edges = np.histogram(dist_uh,bins,density=True)
    bincenters = 0.5 * (h_bin_edges[1:] + h_bin_edges[:-1])
    ax.hist(dist_h,bins=bins,color='g',alpha=0.3,density=True,label=labels[0])
    ax.hist(dist_uh,bins=bins,color='r',alpha=0.3,density=True,label=labels[1])
    ax.plot(bincenters,dist_hy,'-g',lw=1)
    ax.plot(bincenters,dist_uhy,'-r',lw=1)
    # ax.set_xlabel('Local Variance')
    # ax.set_ylabel('Density Frequency')
    ax.legend()

def plot_ims(set_nan=True, *args):
    '''
    Another Plotting Function Helper

    Inputs:
    set_nan: sets the 0 values in the image to nan so no background is present.
    args: list of images.
    '''
    N = len(args)
    if N < 3:
        ncols = N
    else:
        ncols = 3
    fig,axs = plt.subplots(nrows = (N-1)//3+1,ncols = ncols)
    # fig.patch.set_facecolor((211/255,238/255,251/255,1))
    # fig.patch.set_facecolor((0,0,0,1))
    if N > 1:
        for ax in axs.ravel():
            ax.set_axis_off()
        for ax,img in zip(axs.ravel(),args):
            ax.set_axis_off()
            img = img.copy()
            img = img.astype(float)
            if set_nan: img[img==0] = np.nan
            im = ax.imshow(img,**helper_cmaps(args))

        cbar = fig.colorbar(im, ax = axs.ravel().tolist(),shrink=0.3,orientation='horizontal')
        cbar.ax.tick_params(labelsize=16)
    else:
        img = args[0]
        axs.set_axis_off()
        img = img.copy()
        img = img.astype(float)
        if set_nan: img[img==0] = np.nan
        im = axs.imshow(img,**helper_cmaps([img]))
        cbar = fig.colorbar(im, ax = axs,shrink=0.5,orientation='horizontal')
        cbar.ax.tick_params(labelsize=16)
    plt.show()

'''
Example code

import import_functions
# import helper_functions
# import unwrap_functions
import local_SD_analysis_functions

from import_functions import get_, pp,rotate_imgs_auto, crop, mask
# from helper_functions import to_dist, plot_dists, helper_cmaps, plot_ims, pims
# from unwrap_functions import uw2
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

param = 'E2A'
health = 'HCM'
method = 'STEAM'
hs = get_('Healthy',method,'SYSTOLE',param,'myo')
hd = get_('Healthy',method,'DIASTOLE',param,'myo')
uhs = get_('HCM',method,'SYSTOLE',param,'myo')
uhd = get_('HCM',method,'DIASTOLE',param,'myo')
hs = pp(hs)
hd = pp(hd)
uhs = pp(uhs)
uhd = pp(uhd)
h_imgs = hd[0] #retrieve only images not maps
uh_imgs = uhd[0]
dist_h = to_dist(h_imgs)
dist_uh = to_dist(uh_imgs)

bin_n = 30
plot_dists(dist_h,dist_uh,bin_n,labels=['Healthy','HCM'],ax = None)
plot_ims(True,*h_imgs)
pims(h_imgs)

'''

    