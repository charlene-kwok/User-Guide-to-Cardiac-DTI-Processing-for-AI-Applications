#Unwrap Analysis Functions
import import_functions
import helper_functions

from import_functions import get_, pp
from helper_functions import helper_cmaps,pims,to_dist,plot_dists,plot_ims
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import skimage
import pandas as pd
from scipy.signal import convolve2d
from skimage.segmentation import flood_fill
import cv2
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.set_option('future.no_silent_downcasting', True)

def pick_centre_mean_(alg_mask):
    '''
    private function
    Calculates the centre of the image using the centre of mass
    '''
    x,y = np.nonzero(1-alg_mask)
    x_centre=int(x.mean())
    y_centre=int(y.mean())
    return x_centre,y_centre

def pick_centre_middle_(alg_mask):
    '''
    private function
    Calculates the centre of the image by taking the central point
    '''
    x_mid = int(alg_mask.shape[1]//2)
    y_mid = int(alg_mask.shape[0]//2)
    return x_mid,y_mid
    
def unwrap_inner_(img1,mask1,x,y,pick_centre_algorithm=pick_centre_mean_):
    '''
    private function
    Computes the inner unwrap of an image
    Inputs: Image
            Mask
            (x,y) coordinates of insertion point
    output: list of numpy arrays containing information on the pixels in each layer. Each numpy array is of size Nx4. 
            N: number of points in layer i, ouput[i]
            4: holds the pixel value, pixel angle, pixel i and j coordinate
    '''
    img1 = img1.copy()
    mask1 = mask1.copy()
    bordered_imag = cv2.copyMakeBorder(img1, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    bordered_mask = cv2.copyMakeBorder(mask1, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    alg_mask = flood_fill(bordered_mask,(0,0),1)
    alg_mask = np.ascontiguousarray(alg_mask,dtype=np.uint8)
    x_coords,y_coords = np.meshgrid(np.arange(bordered_imag.shape[0]),np.arange(bordered_imag.shape[1]),indexing='ij')
    i_coords,j_coords = np.meshgrid(np.arange(bordered_imag.shape[0]),np.arange(bordered_imag.shape[1]),indexing='ij')
    
    x_centre,y_centre = pick_centre_algorithm(bordered_mask)
    x_coords_c = -(x_coords-x_centre)
    y_coords_c = (y_coords-y_centre)
    angle_pixel = np.arctan2(x-x_centre,y-y_centre)
    angles = np.arctan2(x_coords_c,y_coords_c)
    angles = angles-angle_pixel

    angles[np.where(angles<0)] = 2*np.pi+angles[np.where(angles<0)]
    angles[np.where((angles==0)&bordered_mask)] = 2*np.pi

    def get_contour_values(alg_mask,bordered_mask,bordered_imag,angles):
        contour_mask = np.zeros_like(bordered_mask,dtype=np.uint8)
        contours,_ = cv2.findContours(np.ascontiguousarray(alg_mask,dtype=np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        contour_mask = cv2.drawContours(contour_mask,contours,1,1,1)

        def add_diagonal_elements_contour_mask(contour_mask):
            kernel = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
            neighbors_count = convolve2d(contour_mask, kernel, mode='same', boundary='fill', fillvalue=0)
            outside = flood_fill(contour_mask,(0,0),1,connectivity=1,tolerance=0)
            outside[np.nonzero(contour_mask)] = 0
            neighbors_count[np.where(outside==0)] = 0
            oup = contour_mask.copy()
            oup[np.where(neighbors_count>1)] = 1
            return oup
        contour_mask = add_diagonal_elements_contour_mask(contour_mask)

        num_overlapping_points = int(np.sum(contour_mask*bordered_mask))
        if num_overlapping_points == 0:
            return False
        contour_indxs = np.nonzero(contour_mask*bordered_mask)
        imag_values = bordered_imag[contour_indxs]
        imag_angles = angles[contour_indxs]
        imag_i = i_coords[contour_indxs]-20
        imag_j = j_coords[contour_indxs]-20
        alg_mask[np.nonzero(contour_mask)] = 0
        return alg_mask,np.dstack((imag_values,imag_angles,imag_i,imag_j))[0]
    

    contour_values = []
    while(True):
        oup = get_contour_values(alg_mask,bordered_mask,bordered_imag,angles)
        if not oup:
            break
        else:
            alg_mask = oup[0]
            contour_values.append(oup[1])
    return contour_values

def unwrap_outer_(img1,mask1,x,y,pick_centre_algorithm = pick_centre_mean_):
    '''
    private function
    Computes the outer unwrap of an image
    Inputs: Image
            Mask
            (x,y) coordinates of insertion point
    output: list of numpy arrays containing information on the pixels in each layer. Each numpy array is of size Nx4. 
            N: number of points in layer i, ouput[i]m
            4: holds the pixel value, pixel angle, pixel i and j coordinate
            this is referred to as contour_values frequently
    '''
    img1 = img1.copy()
    mask1 = mask1.copy()
    bordered_imag = cv2.copyMakeBorder(img1, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    bordered_mask = cv2.copyMakeBorder(mask1, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)


    x_mid = int(mask1.shape[1]//2+20)
    y_mid = int(mask1.shape[0]//2+20)
    alg_mask = flood_fill(bordered_mask,(x_mid,y_mid),1)
    alg_mask = np.ascontiguousarray(alg_mask,dtype=np.uint8)

    x_centre,y_centre = pick_centre_algorithm(bordered_mask)
    x_coords,y_coords = np.meshgrid(np.arange(bordered_imag.shape[0]),np.arange(bordered_imag.shape[1]),indexing='ij')
    i_coords,j_coords = np.meshgrid(np.arange(bordered_imag.shape[0]),np.arange(bordered_imag.shape[1]),indexing='ij')

    x_coords_c = -(x_coords-x_centre)
    y_coords_c = (y_coords-y_centre)
    angle_pixel = np.arctan2(x-x_centre,y-y_centre)
    angles = np.arctan2(x_coords_c,y_coords_c)
    angles = angles-angle_pixel

    angles[np.where(angles<0)] = 2*np.pi+angles[np.where(angles<0)]
    angles[np.where((angles==0)&bordered_mask)] = 2*np.pi

    def get_contour_values(alg_mask,bordered_mask,bordered_imag,angles):
        contour_mask = np.zeros_like(bordered_mask,dtype=np.uint8)
        contours,_ = cv2.findContours(np.ascontiguousarray(alg_mask,dtype=np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        contour_mask = cv2.drawContours(contour_mask,contours,0,1,1)

        def add_diagonal_elements_contour_mask(contour_mask):
            kernel = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
        
            neighbors_count = convolve2d(contour_mask, kernel, mode='same', boundary='fill', fillvalue=0)
            outside = flood_fill(contour_mask,(0,0),1,connectivity=1,tolerance=0)
            inside = 1-outside
            neighbors_count[np.where(inside==0)] = 0
            oup = contour_mask.copy()
            oup[np.where(neighbors_count>1)] = 1
            return oup
        contour_mask = add_diagonal_elements_contour_mask(contour_mask)

        num_overlapping_points = int(np.sum(contour_mask*bordered_mask))
        if num_overlapping_points == 0:
            return False
        contour_indxs = np.nonzero(contour_mask*bordered_mask)
        imag_values = bordered_imag[contour_indxs]
        imag_angles = angles[contour_indxs]
        imag_i = i_coords[contour_indxs]-20
        imag_j = j_coords[contour_indxs]-20
        alg_mask[np.nonzero(contour_mask)] = 0
        return alg_mask,np.dstack((imag_values,imag_angles,imag_i,imag_j))[0]

    contour_values = []
    while(True):
        oup = get_contour_values(alg_mask,bordered_mask,bordered_imag,angles)
        if not oup:
            break
        else:
            alg_mask = oup[0]
            contour_values.append(oup[1])

    return contour_values

def correct_end_vals_(out):
    '''
    private function
    Correction for the number of layers in the image
    This is the algorithm responsible for correcting the layers if the last layers have too few values in them'''
    sum_vals = 0
    for i in range(len(out)-1):
        num_vals = len(out[::-1][i])
        sum_vals += num_vals
        next_num_vals = len(out[::-1][i+1])
        if next_num_vals >= sum_vals:
            oup = i
    full_oup = [np.empty((0,4))]
    for i in range(len(out)):
        if i <= oup:
            full_oup[0] = np.vstack((full_oup[0],out[::-1][i]))
        else:
            full_oup.append(out[::-1][i])
    full_oup = full_oup[::-1]
    return full_oup

def correct_N_(inn,out):
    '''
    private function
    get the Number of layers estimated in an image after correction has been applied
    '''
    len_inner = len(correct_end_vals_(inn))
    len_outer = len(correct_end_vals_(out))
    # return (len(inn)+len(out)+1)//2-0.0001
    return (len_inner+len_outer+1.5)//2-0.0001
def get_val_layer_(i,j,layers):
    '''
    private function
    returns the pixel value and angle of a pixel at position i,j in the original image from the 
    list of numpy arrays'''
    for layer_idx,layer in enumerate(layers):
        idx = ((layer[:,[2,3]] == [i,j]).all(axis=1))
        if sum(idx) > 0:
            return layer[idx, [0,1]].copy(),layer_idx
        
def vals_to_layer_map_(image,inn,out,custom_tol=None):
    '''
    private function
    This takes the list of numpy arrays that contain the layer information and builds the final depth layer map
    '''
    x,y = np.nonzero(image)
    new_img = image.copy()
    new_img[new_img==0] = np.nan
    new_img[~np.isnan(new_img)] = 0

    beta = 1 #penalisation factor
    for i,j in zip(x,y):
        val,layer_inn = get_val_layer_(i,j,inn)
        val,layer_out = get_val_layer_(i,j,out)
        inn_perc = layer_inn/(len(inn)-1)
        out_perc = layer_out/(len(out)-1)
        vals = (1-out_perc)**beta-(1-inn_perc)**beta
        if layer_inn == 0:
            vals = -1
        elif layer_out == 0:
            vals =1
        new_img[i,j] = custom_tol*((vals+1)/2)//1
    return new_img

def layer_map_to_contour_values_(image,layer_map,inn):
    '''
    private function
    from the final layer_map, calculates the contour_values from the layer_map and image
    '''
    new_img = layer_map.copy()
    x,y = np.nonzero(image)
    tol = int(np.nanmax(layer_map))+1
    contour_values = [np.empty((0,2)) for i in range(tol)]
    for i,j in zip(x,y):
        val,_ = get_val_layer_(i,j,inn)
        layer = int(new_img[i,j])
        contour_values[layer] = np.vstack((contour_values[layer],val))
    # contour_values = [i for i in contour_values if len(i)>0]
    return contour_values

def vals_to_barcode_updated_(contour_values,tol):
    '''
    private function
    converts the contour_values into a barcode Method 1
    '''
    all_angles = np.empty((0,))
    for values in contour_values:
        values = np.array(values)
        angles_in_shape = values[:,1].flatten()
        all_angles = np.append(all_angles,angles_in_shape)

    all_angles = [np.round(tol*i)/tol for i in all_angles]
    all_angles = list(set(all_angles))
    all_angles.sort()
    df = pd.DataFrame(index=list(range(len(contour_values)))[::-1],columns=all_angles)

    for idx,values in enumerate(contour_values):
        for point in values:
            df.loc[idx,np.round(tol*point[1])/tol] = point[0]
    df = df.apply(pd.to_numeric)

    df = df.ffill(axis=1,limit=1)
    df = df.interpolate('linear',axis=0,limit=1)
    df = df.interpolate('linear',axis=1,limit=1)
    df = df.ffill(axis=1,limit=1)
    df = df.ffill(axis=0)
    unwrapped_img = df.to_numpy(dtype=float)
    return unwrapped_img

def vals_to_barcode_updated_method2_(contour_values,N=None):
    '''
    private function
    converts the contour_values into a barcode Method 2
    '''
    if N is None:
        N = max([len(i) for i in contour_values])
    angles = np.linspace(0,2*np.pi,N,endpoint=False)
    df = pd.DataFrame(index=list(range(len(contour_values))),columns=angles)
    for idx,values in enumerate(contour_values):
        for point in values:
            df.iloc[idx,int((angles<point[1]).sum()-1)] = point[0]
    
    
    df = df.copy().astype(float)
    df = df.apply(pd.to_numeric)
    df = df.iloc[::-1,:].copy()
    df = df.ffill(axis=1,limit=2)
    df = df.bfill(axis=1,limit=2)
    df = df.bfill(axis=0,limit=2)
    df = df.ffill(axis=0)
    df = df.interpolate('linear',axis=0)
    mid_val = df.shape[0]//2
    for i in range(1,mid_val):
        df.iloc[i,:] = df.iloc[i,:].fillna(df.iloc[i-1,:])
    for i in range(mid_val,df.shape[0]-1):
        df.iloc[i,:] = df.iloc[i,:].fillna(df.iloc[i+1,:])

    df = df.apply(pd.to_numeric)
    unwrapped_img = df.to_numpy(dtype=float)
    return unwrapped_img

def get_inner_outer_unwraps(data):
    '''
    Returns the inner and outer unwraps of the data
    data: list of images and masks
    oups_inner,oups_outer: returns the contour_values of the inner and outer unwrap
    '''
    oups_inner = []
    oups_outer = []
    for img,mask in zip(*data):
        val_inner = unwrap_inner_(img,mask)
        val_outer = unwrap_outer_(img,mask)
        val_inner = correct_end_vals_(val_inner)
        val_outer = correct_end_vals_(val_outer)
        oups_inner.append(val_inner)
        oups_outer.append(val_outer)
    return oups_inner,oups_outer

def uw2(data,x,y,num_M=None,num_layers=None,return_contours=False,return_layer_maps=False):
    '''
    this is the main unwrap function responsible for most of the work. It takes a bit long
    on a large amount of images so I recommend unwrapping before doing heavy computations. 
    Or use jupyter notebook.

    Inputs:
    data: list of images and masks
    num_M: Number of points in the circumfrential direction of barcode. If left blank, will take the value of the largest
            by pixel count. Doesnt need to be set if return_contours or return_layer_maps == True

    num_layers: number of layers to have in the final image. set to 3 to retrieve inner, middle, outer sections
                if left as None, will automatically select it based on the corrected lengths of the inner and outer unwrap

    return_contours: returns the contour_values of the image after final depth calculation

    return_layer_maps: returns the layer_maps of the image

    if both return_contours and return_layer_maps is set to False, the unwrapped barcode is returned
    '''
    oups = []
    for i in range(len(x)):
        imgs = data[0]
        img = imgs[i]
        masks = data[1]
        mask = masks[i]
        val_inner = unwrap_inner_(img, mask, x[i], y[i])
        val_outer = unwrap_outer_(img, mask, x[i], y[i])
        N = correct_N_(val_inner, val_outer)
        if num_layers is None:
            custom_tol = N
        else:
            custom_tol = num_layers-0.001
        layer_map = vals_to_layer_map_(img,val_inner,val_outer,custom_tol=custom_tol)
        oup = layer_map_to_contour_values_(img,layer_map,val_inner)
        if return_contours:
            oups.append(oup)
        elif return_layer_maps:
            oups.append(layer_map)
        else:
            oups.append(vals_to_barcode_updated_method2_(oup,num_M))
    return oups


def get_layer_maps(data,num_layers=None):
    '''
    Retrieves the layer_maps of all images in data

    Inputs:
    data: list of images and masks
    num_layers: number of layers to have in the final image. set to 3 to retrieve inner, middle, outer sections
                if left as None, will automatically select it based on the corrected lengths of the inner and outer unwrap
    
    Outputs:
    oups
    '''
    layer_maps = []
    for img,mask in zip(*data):
        val_inner = unwrap_inner_(img,mask)
        val_outer = unwrap_outer_(img,mask)
        N = correct_N_(val_inner,val_outer)
        if num_layers is None:
            custom_tol = N
        else:
            custom_tol = num_layers-0.001
        layer_map = vals_to_layer_map_(img,val_inner,val_outer,custom_tol=custom_tol)
        layer_maps.append(layer_map)
    return layer_maps


def get_num_layers(data):
    '''
    returns the number of layers in an image after correction. Use for approximating size
    
    Inputs:
    data: list of images and masks

    Outputs: Number of layers in each image
    '''
    oups = []
    for img,mask in zip(*data):
        val_inner = unwrap_inner_(img,mask)
        val_outer = unwrap_outer_(img,mask)
        N = correct_N_(val_inner,val_outer)
        oups.append(N)
    return oups

def get_angles1_(img1,mask1,pick_centre_algorithm=pick_centre_mean_):
    '''
    private function
    returns angle maps of the image
    '''
    img1 = img1.copy()
    mask1 = mask1.copy()
    alg_mask = flood_fill(mask1,(0,0),1)
    alg_mask = np.ascontiguousarray(alg_mask,dtype=np.uint8)
    x_centre,y_centre = pick_centre_algorithm(mask1)
    x_coords,y_coords = np.meshgrid(np.arange(img1.shape[0]),np.arange(img1.shape[1]),indexing='ij')
    x_coords_c = -(x_coords-x_centre)
    y_coords_c = (y_coords-y_centre)
    angles = np.arctan2(x_coords_c,y_coords_c)
    
    angles[np.where(angles<0)] = 2*np.pi+angles[np.where(angles<0)]
    angles[np.where((angles==0)&mask1)] = 2*np.pi
    return angles*mask1

def get_angles(data,minn=None,maxx=None):
    '''
    returns the angle maps of all images in data, can set minn and maxx to only select specific regions.(120,300 roughly for septum)

    Inputs:
    data: list of images and masks
    minn,maxx: minimum and maximum angles to retrieve

    Outputs:

    '''
    angle_maps = []
    for img,mask in zip(*data):
        img = img.copy()
        mask = mask.copy()
        angles = get_angles1_(img,mask)
        angles = np.rad2deg(angles.copy())
        if (minn is not None) and (maxx is not None):
            mask2 = (angles<=maxx)&(angles>=minn)
            angles = np.where(mask2,angles,0)
        angles[np.where(angles)] = 1
        img = img*angles
        angle_maps.append(img)
    return angle_maps

def get_angle_maps(data):
    '''
    returns the angle maps of all images in data, can set minn and maxx to only select specific regions.(120,300 roughly for septum)

    Inputs:
    data: list of images and masks
    minn,maxx: minimum and maximum angles to retrieve

    Outputs:

    '''
    angle_maps = []
    for img,mask in zip(*data):
        img = img.copy()
        mask = mask.copy()
        angles = get_angles1_(img,mask)
        angles = np.rad2deg(angles.copy())

        angle_maps.append(angles)
    return angle_maps
def calculate_average_angle(imgs):
    ''' 
    given a set of images, calculates the periodic average angle in it
    used for Figure of diastolic and systolic E2A in septum
    '''
    oups = []
    for img in imgs:
        img = img.copy()
        img = img[img!=0]
        img = img[~np.isnan(img)]
        radians = np.deg2rad(img)
        complex_numbers = np.exp(2j*radians)
        mean_vector = np.mean(complex_numbers)
        average_angle = np.rad2deg(np.angle(mean_vector))/2
        oups.append(np.abs(average_angle))
    return oups
'''
Example code
'''
import import_functions
import helper_functions
import local_SD_analysis_functions

from import_functions import get_, pp,rotate_imgs_auto
from helper_functions import to_dist, plot_dists, helper_cmaps, plot_ims, pims
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
import copy

# Specify the path to your Excel file
file_path = '/Users/charlene/Desktop/UROP/Data/insertion_points.xlsx'


# Read all sheets into a dictionary of DataFrames
df_hd = pd.read_excel(file_path, sheet_name='hd_STEAM')
df_hs = pd.read_excel(file_path, sheet_name='hs_STEAM')
df_uhd = pd.read_excel(file_path, sheet_name='uhd_STEAM')
df_uhs = pd.read_excel(file_path, sheet_name='uhs_STEAM')

#find corresponding data setm eg df_hd or df_uhd
x_df = df_hd['x']
y_df = df_hd['y']

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

hd = copy.deepcopy(uw2(hd,x_df,x_df,14,3,True))

# img1 = hd[0][1]
# mask1 = hd[1][1]


# x = x_df[1]
# y = y_df[1]

# plt.figure(1)
# plt.imshow(img1, **helper_cmaps(img1))
# plt.grid(True)
# plt.plot(x, y, 'kx')


# plt.figure(2)
# oups = []
# val_inner = unwrap_inner_(img1,mask1,x,y,pick_centre_algorithm)
# val_outer = unwrap_outer_(img1,mask1,x,y,pick_centre_algorithm)
# N = correct_N_(val_inner,val_outer)
# custom_tol = N
# layer_map = vals_to_layer_map_(img1,val_inner,val_outer,custom_tol=custom_tol)
# oup = layer_map_to_contour_values_(img1,layer_map,val_inner)
# num_M = 180
# oups.append(vals_to_barcode_updated_method2_(oup,num_M))
# plt.imshow(oups[0], **helper_cmaps(oups[0]))



# data = hd
# x = x_df
# y = y_df
# oups = []

# for i in range(len(x)):
#     imgs = data[0]
#     masks = data[1]
#     val_inner = unwrap_inner_(imgs[i], masks[i], x[i], y[i])
#     val_outer = unwrap_outer_(imgs[i], masks[i], x[i], y[i])
#     N = correct_N_(val_inner, val_outer)

#     if num_layers is None:
#         custom_tol = N
#     else:
#         custom_tol = num_layers - 0.001

#     layer_map = vals_to_layer_map_(imgs[i], val_inner, val_outer, custom_tol=custom_tol)
#     oup = layer_map_to_contour_values_(imgs[i], layer_map, val_inner)

#     if return_contours:
#         oups.append(oup)
#     elif return_layer_maps:
#         oups.append(layer_map)
#     else:
#         oups.append(vals_to_barcode_updated_method2_(oup, num_M))

# print(oups)



