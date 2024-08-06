#Import and Preprocessing Functions
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import distance_transform_edt as sdf
def get_(health,technique,heart_mode,param,mask='myo'):
    '''
    Main method for pulling data from csv files

    Inputs:
    health: 'Healthy' or 'HCM'
    technique: 'SE' or 'STEAM'
    heart_mode: 'Systole' or 'Diastole'
    param: params2d = ['CR', 'E2A', 'E2A_alt', 'E3A', 'FA', 'HA', 'IA', 'MD', 'mode', 'negative_eigenvalues', 'norm', 'S0', 'TA']
    mask: any of the masks available in the mask folders
    
    Outputs:
    Returns: list with two entries. The first entry is a list of the images, the second entry is a list of the masks
    Dont unpack the output, if you see a function has a data input, it means you need to pass in this list of two entries
    '''
    if param == 'IA':
        param = 'imbrication_angle'
    dir = f"/Users/charlene/Desktop/UROP/Data/{health}/{technique}_{heart_mode}"
   
    # Check if the directory exists
    if not os.path.exists(dir):
        raise FileNotFoundError(f"The directory {dir} does not exist.")

    oup_imgs = []
    oup_masks = []
    if health =='Healthy':
         pre = 'h'
    else:
         pre = 'p'
    for i in range(len(os.listdir(dir))-1):
    # for i in range(1,15):   
        idx = i+1
        img = np.load(f"{dir}/{pre}{idx}/DTI/{param}.npy")
        mask_file = np.load(f"{dir}/{pre}{idx}/Mask/{mask}.npy")

        # if param == 'MD':
        #     outliers = np.where(img>5)
        #     img[outliers] = 0
        #     mask_file[outliers] = 0
        oup_imgs.append(img)
        oup_masks.append(mask_file)
    return oup_imgs, oup_masks
    #dir = f"./Data/{health}/{technique.upper()}_{heart_mode.upper()}"
def rotate_imgs_auto(data):
    '''
    Function for automatically rotating images
    data: list of images and masks
    returns: list of images and masks rotated
    '''
    imgs,masks = data
    oup_imgs = []
    oup_masks = []
    for img,mask in zip(imgs,masks):
        distance_from_mask = sdf(1-mask)
        img_mask = img!=0
        outside_region_distance = distance_from_mask*img_mask*(1-mask)
        max_pos = np.argmax(outside_region_distance)
        max_pos = np.unravel_index(max_pos, outside_region_distance.shape)
        lv_direction = ''
        if max_pos[0]/outside_region_distance.shape[0] < 0.5:
            #top region
            lv_direction+= ('T')
        else:
            #bottom region
            lv_direction+=('B')
        if max_pos[1]/outside_region_distance.shape[1] < 0.5:
            #left region
            lv_direction+=('L')
        else:
            #right region
            lv_direction+=('R')
        if lv_direction == 'BL':
            oup_imgs.append(img)
            oup_masks.append(mask)
        elif lv_direction == 'TL':
            img1 = np.rot90(img,1)
            mask1 = np.rot90(mask,1)
            oup_imgs.append(img1)
            oup_masks.append(mask1)
        elif lv_direction == 'TR':
            img1 = np.rot90(img,2)
            mask1 = np.rot90(mask,2)
            oup_imgs.append(img1)
            oup_masks.append(mask1)
        elif lv_direction == 'BR':
            img1 = np.rot90(img,3)
            mask1 = np.rot90(mask,3)
            oup_imgs.append(img1)
            oup_masks.append(mask1)
    return oup_imgs,oup_masks

def mask(data):
    '''
    Masks the images with the mask obtained from get_

    Inputs:
    data: list of images and masks

    Outputs:
    ouptut: list of masked images and masks
    '''
    imgs,masks = data
    output = [img*mask for img,mask in zip(imgs,masks)],masks
    return output
def crop(data):
    '''
    Crops the images and masks by removing all columns and rows that contain only 0's

    Inputs:
    data: list of images and masks
    Outputs:
    output: list of cropped images and masks
    '''
    imgs,masks = data
    oup_imgs = []
    oup_masks = []
    for img,mask in zip(imgs,masks):
        x, y = np.nonzero(mask)
        xl,xr = x.min(),x.max()
        yl,yr = y.min(),y.max()
        oup_imgs.append(img[xl:xr+1, yl:yr+1])
        oup_masks.append(mask[xl:xr+1, yl:yr+1])
    output = oup_imgs,oup_masks
    return output
def pp(data):
    '''
    Pre-Processing:Rotates, Masks, Crops the input images and masks

    Inputs:
    data: list of images and masks

    Outputs:
    ouptut: list of rotated, masked, cropped images and masks
    '''
    return crop(mask(rotate_imgs_auto(data)))


