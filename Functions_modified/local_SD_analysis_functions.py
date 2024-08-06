#Local SD Analysis Functions
import numpy as np
from scipy.ndimage import generic_filter

def local_SD_(window,min_points):
    '''
    private function
    calculates the non-periodic standard deviation of a window in the image
    min_points: minimum number of points required in the window to calculate the SD,
                if less, oup is set to 0
    '''
    wdw = window[~np.isnan(window)]
    if len(wdw) < min_points:
        return 0
    return np.sqrt(wdw.var())

def circular_SD_(window,min_points):
    '''
    private funtion
    calculates the periodic standard deviation of a window in the image
    min_points: minimum number of points required in the window to calculate the SD,
                if less, oup is set to 0
    '''
    wdw = window[~np.isnan(window)]
    if len(wdw) < min_points:
        return 0
    radians = np.deg2rad(wdw)
    complex_numbers = np.exp(2j * radians)
    mean_vector = np.mean(complex_numbers)
    circ_variance = 1 - np.abs(mean_vector)

    circ_sd = np.sqrt(-2*np.log(np.abs(mean_vector)))
    return circ_sd
    # return np.sqrt(circ_variance) 

    #both returns are valid SDs, although you should be using the one with the log as its more mathematically rigerous for
    #circular data. It does behave badly with noise however. Be ready for big positive values if you have some pixels outside
    #of the LV
    
def image_SD_calc_function_(img,mask,n,periodic_var=False,min_num_points=5):
    '''
    private function
    calculates the local SD over a single image
    '''
    img = img.copy()
    mask = mask.copy()
    # img = np.clip(img,0,3)
    padded_img = np.pad(img,n,'constant',constant_values=0)
    padded_mask = np.pad(mask,n,'constant',constant_values=0)
    img[np.where(mask==0)] = np.nan
    
    # local_variance_image = generic_filter(img, local_variance, size=n)
    if periodic_var:
        local_variance_image = generic_filter(padded_img,lambda x: circular_SD_(x,min_points=min_num_points), size=n)
    else:
        local_variance_image = generic_filter(padded_img, lambda x: local_SD_(x,min_points=min_num_points), size=n)
    final_image = local_variance_image*padded_mask
    return final_image[n:-n,n:-n]

def calc_SD(data, n, periodic_SD=False,min_num_points=5):
    '''
    calculates the local variance over all images in data
    data: list of images and masks
    n: size of filter (nxn)
    periodic_var: bool, if True, will calculate the periodic SD, if False, will calculate the classic SD
    min_num_points=5: minimum number of pixels required in a filter to compute the SD.
    '''
    oups = []
    masks = data[1]
    for img,mask in zip(*data):
        img = img.copy()
        mask = mask.copy()
        val = image_SD_calc_function_(img,mask,n,periodic_var=periodic_SD,min_num_points=min_num_points)
        oups.append(val)
    # return np.array(oups)
    return oups,masks