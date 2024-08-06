#Cross Correlation Functions
import import_functions
import helper_functions
import unwrap_functions
import pandas as pd

from import_functions import get_, pp
from helper_functions import to_dist, plot_dists, helper_cmaps, plot_ims
from unwrap_functions import uw2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import copy

def group_unwraps_(countour_value):
    '''
    helper_function for when youre working with the inner, middle outer region. The default uw2 method
    returns the contour_values output as a list of Nx3, where N is the number of images and 3 is the number of regions
    This just converts it to a 3XN list so that extracting all images for a particular region can be done through oup[i]

    Inputs:
    contour_value: contour_value output from uw2 function
    '''
    oup = [[],[],[]]
    for i in range(len(countour_value)):
        for j in range(len(countour_value[0])):
            oup[j].append(countour_value[i][j])
    return oup

def split_unwraps(countour_value,test_idx):
    '''
    Split the unwraps into train and a singular test image for LOOCV.
    
    Inputs:
    uws: output of unwraps
    test_idx: the index individual image you want to take out to put into the test
    '''
    uws = group_unwraps_(countour_value)
    train = [[],[],[]]
    test = [[],[],[]]
    num_vals = len(uws[0])
    for i in range(len(uws)):
        test_idx2 = [test_idx]
        train_idx = [j for j in range(num_vals) if j not in test_idx2]
        if test_idx is None:
            test_temp = []
        else:
            test_temp = uws[i][test_idx2[0]]
        train_temp = [uws[i][j] for j in train_idx]
        train[i] = np.vstack(train_temp) #combines training data into one singular contour_values
        test[i]= test_temp
    return train,test


def cross_correlate_images(img1,img2,change_angles,filter_settings=None,plot_vals=False):
    '''
    Note when the contour_values are talked about, it may seem difficult to retrieve them
    They are ultimately just a numpy array where each row is a pixel_value / angle pair
    you can just use the unwrap algorithm and set it the number of layer to 1 to retrieve it for the full images
    or you can unwrap into multiple regions and then split_unwraps to retrieve the training or testing data to use for cross correlation



    This is the cross correlation of two images. Only really applicable to E2A or IA parameter
    img1: contour_values of first image to compare: generally put systole here
    img2: contour_values of second image to compare: generally put diastole here
    change_angles: [bool1,bool2], changes the range of the angle parameters of [img1,img2]
                    respectively from -90-90 to 0-90
    filter_settings: savgol filter was implemented but not mentioned in report, can be used to
                    smoothen the distributions in the columnwise direction
                    filter_settings = [window_length,polyorder]
    plot_vals: if False, doesn't plot anything. If given a value, will plot the:
                 histograms of the distributions of img1,img2
                 2d PDFs of img1 and img2
                 Angle shift over the circumfrential position of image
                 Titles are set to the value of plot_vals

    Returns: (hist1,hist2),(oup1,oup2,oup3,(x,angle_shifts))
            hist1, hist2: 2D histograms of img1 and img2
            oup1: full columnwise crosscorrelation
            oup2: only maximum values of cross correllations
            oup3: maximum values but set to 1
            x: x positions of angle_shift
            angle_shifts: angle_shift at each discrete point along the circumfrential position
    '''
    a1,a2 = change_angles
    img1 = img1.copy()
    img2 = img2.copy()
    img1x = img1[:,1]*180/np.pi
    img1y = img1[:,0]
    range1 = [-90,90]
    range2 = [-90,90]
    if a1:
        img1y[np.where(img1y<0)] = img1y[np.where(img1y<0)] + 180 
        range1 = [0,180]
    img2x = img2[:,1]*180/np.pi
    img2y = img2[:,0]
    if a2:
        img2y[np.where(img2y<0)] = img2y[np.where(img2y<0)] + 180
        range2 = [0,180]
    if plot_vals:
        plot_dists(img1y,img2y,20,['SYS','DIA'])
        # plt.gca().set_title('HCM Middle E2A')
        plt.show()
    bin_size_x = 30
    bin_size_y = 15
    hist1, xedges1, yedges1 = np.histogram2d(img1x, img1y, bins=[bin_size_x, bin_size_y],range=[[0,360],range1])
    hist2, xedges2, yedges2 = np.histogram2d(img2x, img2y, bins=[bin_size_x, bin_size_y],range=[[0,360],range2])
    hist1,hist2 = hist1.T,hist2.T
    if plot_vals:    
        plt.imshow(hist1,origin='lower',extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
        cbar = plt.colorbar(label='Count',shrink=0.3,orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='Count',size=12)
        plt.xlabel('Circumfrential Position (Degrees)',fontsize=14)
        plt.title('Systole')
        plt.ylabel(f'{plot_vals}' ,fontsize=14) 
        plt.gca().tick_params(axis='both', which='major', labelsize=12)
        plt.gca().set_yticks(np.arange(range1[0],range1[1]+45,45))
        plt.gca().set_xticks(np.arange(0,420,60))
        plt.show()
        plt.imshow(hist2,origin='lower',extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])
        cbar = plt.colorbar(label='Count',shrink=0.3,orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='Count',size=12)
        plt.xlabel('Circumfrential Position (Degrees)',fontsize=14)
        plt.ylabel(f'{plot_vals}' ,fontsize=14) 
        plt.gca().tick_params(axis='both', which='major', labelsize=12)
        plt.gca().set_yticks(np.arange(range2[0],range2[1]+45,45))
        plt.gca().set_xticks(np.arange(0,420,60))
        plt.title('Diastole')
        plt.show()
    if not (filter_settings is None):
        window_length,polyorder = filter_settings
        hist1 = savgol_filter(hist1,window_length,polyorder,axis=0)
        hist2 = savgol_filter(hist2,window_length,polyorder,axis=0)

    oup1 = np.zeros_like(hist1)
    oup2 = np.zeros_like(hist1)
    oup3 = np.zeros_like(hist1)
    angle_shifts = np.zeros(hist1.shape[1])
    for i in range(oup1.shape[1]):
        row1 = hist1[:,i]
        row2 = hist2[:,i]
        cc = np.correlate(row1,row2,mode='same')
        idxmax = int(np.argmax(cc))
        oup3[idxmax,i] = 1
        oup1[:,i] = cc
        oup2[idxmax,i] = np.nanmax(cc)
        shift = idxmax - (len(cc) // 2)
        angle_shift = shift * (yedges1[1] - yedges1[0])
        angle_shift2 = angle_shift+range1[1]-range2[1]
        if angle_shift2>90: # angle correction to E2A mobility not greater than 90. Converts a shift of 120 to 60
            angle_shift2 = 180-angle_shift2
        angle_shifts[i] = np.abs(angle_shift2)
        
    if plot_vals:    
        plt.figure(figsize=(10, 5))
        plt.plot(xedges1[:-1] + (xedges1[1] - xedges1[0]) / 2, angle_shifts)
        plt.xlabel('Circumferential direction (degrees)')
        plt.ylabel('Angle shift (degrees)')
        plt.title('Angle Shifts Across Circumferential Direction')
        plt.show()
    return (hist1,hist2),(oup1,oup2,oup3,(xedges1[:-1] + (xedges1[1] - xedges1[0]) / 2,angle_shifts))

def cross_correlate_images_circular(img1, img2, N=30):
    '''
    This function calculates the mean angle shifts between two sets of contour values using circular statistics.
    
    Parameters:
    img1, img2: Arrays of contour values to compare (each row should represent a point [y, x]).
    N: Number of bins to use in the circumferential direction. Default is 30.
    
    Returns:
    (x, angle_shifts), (angle_systole, angle_diastole):
        x: Central position of bins in degrees.
        angle_shifts: Angle shifts at each bin in degrees.
        angle_systole: Average systolic (img1) angle at each bin in degrees.
        angle_diastole: Average diastolic (img2) angle at each bin in degrees.
    '''
    img1 = img1.copy()
    img2 = img2.copy()
    
    # Extract x and y coordinates
    img1x = img1[:, 1]
    img1y = img1[:, 0]
    img2x = img2[:, 1]
    img2y = img2[:, 0]
    
    # Define bin edges
    bins_start = np.linspace(0, 2 * np.pi, N + 1, endpoint=True)
    
    # Digitize the x-coordinates into bins
    bin_indices_img1 = np.digitize(img1x, bins_start) - 1
    bin_indices_img2 = np.digitize(img2x, bins_start) - 1
    
    # Ensure bin indices are within the valid range
    bin_indices_img1 = np.clip(bin_indices_img1, 0, N - 1)
    bin_indices_img2 = np.clip(bin_indices_img2, 0, N - 1)
    
    # Group y-values by bin indices
    grouped_values_img1 = [[] for _ in range(N)]
    grouped_values_img2 = [[] for _ in range(N)]
    
    for i, value in enumerate(img1y):
        bin_index = bin_indices_img1[i]
        grouped_values_img1[bin_index].append(value)
    
    for i, value in enumerate(img2y):
        bin_index = bin_indices_img2[i]
        grouped_values_img2[bin_index].append(value)
    
    angle_shifts = []
    angle_systole = []
    angle_diastole = []
    
    for systole, diastole in zip(grouped_values_img1, grouped_values_img2):
        if systole and diastole:
            # Convert to radians
            radians_systole = np.deg2rad(systole)
            radians_diastole = np.deg2rad(diastole)
            
            # Calculate circular mean
            complex_numbers_systole = np.exp(2j * radians_systole)
            complex_numbers_diastole = np.exp(2j * radians_diastole)
            
            mean_vector_systole = np.mean(complex_numbers_systole)
            mean_vector_diastole = np.mean(complex_numbers_diastole)
            
            mean_angle_systole = np.rad2deg(np.angle(mean_vector_systole)) / 2
            mean_angle_diastole = np.rad2deg(np.angle(mean_vector_diastole)) / 2
            
            # Calculate shift
            x1, y1 = mean_vector_systole.real, mean_vector_systole.imag
            x2, y2 = mean_vector_diastole.real, mean_vector_diastole.imag
            magnitude = np.abs(mean_vector_systole) * np.abs(mean_vector_diastole)
            shift = np.rad2deg(np.arccos((x1 * x2 + y1 * y2) / magnitude)) / 2
            
            if shift >= 90:
                shift = 180 - shift
            
            angle_shifts.append(abs(shift))
            angle_systole.append(mean_angle_systole)
            angle_diastole.append(mean_angle_diastole)
        else:
            angle_shifts.append(0)
            angle_systole.append(0)
            angle_diastole.append(0)
    
    # Calculate bin centers in degrees
    bin_centers = (bins_start[:-1] + (bins_start[1] - bins_start[0]) / 2) * (180 / np.pi)
    
    return (bin_centers, angle_shifts), (angle_systole, angle_diastole)


'''
Example Code:
'''
param = 'E2A'
health = 'HCM'
method = 'STEAM'
hs = get_('Healthy',method,'SYSTOLE',param,'myo')
hd = get_('Healthy',method,'DIASTOLE',param,'myo')
uhs = get_('HCM',method,'SYSTOLE',param,'myo')
uhd = get_('HCM',method,'DIASTOLE',param,'myo')
# plot_ims(True,df[0][0],df[0][1])
hs = pp(hs)
hd = pp(hd)
uhs = pp(uhs)
uhd = pp(uhd)

hs = copy.deepcopy(uw2(hs,14,3,True))
hd = copy.deepcopy(uw2(hd,14,3,True))
uhs = copy.deepcopy(uw2(uhs,14,3,True))
uhd = copy.deepcopy(uw2(uhd,14,3,True))

hsys_train,hsys_test = split_unwraps(hs,None)
hdia_train,hdia_test = split_unwraps(hd,None)
uhsys_train,uhsys_test = split_unwraps(uhs,None)
uhdia_train,uhdia_test = split_unwraps(uhd,None)
region = 1 #0:inner, 1:middle, 2:outer
hsys = hsys_train[region]
hdia = hdia_train[region]
hsys2 = hsys_test[region]
hdia2 = hsys_test[region]

fullsysh = np.concatenate(hsys_train)
fullsysuh = np.concatenate(uhsys_train)
fulldiah = np.concatenate(hdia_train)
fulldiauh = np.concatenate(uhdia_train)
uhsys = uhsys_train[region]
uhdia = uhdia_train[region]
uhsys2 = uhsys_test[region]
uhdia2 = uhsys_test[region]

(x,angle_shifts),(_,_) = cross_correlate_images_circular(fullsysh,fulldiah)
plt.plot(x,angle_shifts)
plt.show()
x,angle_shifts = cross_correlate_images_circular(fullsysuh,fulldiauh)
plt.plot(x,angle_shifts)
plt.show()

filter_settings = None
plot_vals = False
region = 1 
angle_shifth = []
angle_shiftuh = []
N = 40
for i in range(len(hd)):
    # print('a')
    hsys_train,hsys_test = split_unwraps(hs,i) #set these if you want to use the full image and not just the region
    hdia_train,hdia_test = split_unwraps(hd,i)
    hsyst = hsys_test[region]
    hdiat = hdia_test[region]
    # hsyst = np.concatenate(hsys_test)
    # hdiat = np.concatenate(hdia_test)
    # (hist1,hist2),(oup1,oup2,oup3,(x,angle_shift)) = cross_correlate_images(hsyst,hdiat,[True,False],filter_settings,plot_vals=False)
    (x,angle_shift),(angle_sys,angle_dia) = cross_correlate_images_circular(hsyst,hdiat,N)
    angle_shifth.append(angle_shift)

for i in range(len(uhs)):
    uhsys_train,uhsys_test = split_unwraps(uhs,i)
    uhdia_train,uhdia_test = split_unwraps(uhd,i)
    uhsyst = uhsys_test[region]
    uhdiat = uhdia_test[region]
    # uhsyst = np.concatenate(uhsys_test) #set these if you want to use the full image and not just the region
    # uhdiat = np.concatenate(uhdia_test)
    # (hist1,hist2),(oup1,oup2,oup3,(x,angle_shift)) = cross_correlate_images(uhsyst,uhdiat,[True,True],filter_settings,plot_vals=False)
    (x,angle_shift),(angle_sys,angle_dia) = cross_correlate_images_circular(uhsyst,uhdiat,N)
    angle_shiftuh.append(angle_shift)

    
plt.figure(figsize=(10, 5))
wdw = 3 
averaged_shifth = np.array(angle_shifth).mean(axis=0)
averaged_shiftuh = np.array(angle_shiftuh).mean(axis=0)
for angle_shift in angle_shifth:
    # print(len(angle_shift), end=' ')
    # plt.plot(x, angle_shift,'--g',lw=0.7)
    plt.plot(x, pd.DataFrame(angle_shift).rolling(window=wdw).mean().fillna(pd.DataFrame(angle_shift)),'--g',lw=0.7)
    # break
for angle_shift in angle_shiftuh:
    # print(len(angle_shift), end=' ')
    # plt.plot(x, angle_shift,'--r',lw=0.7)
    plt.plot(x, pd.DataFrame(angle_shift).rolling(window=wdw).mean().fillna(pd.DataFrame(angle_shift)),'--r',lw=0.7)
    # break
plt.plot(x,pd.DataFrame(averaged_shifth).rolling(window=wdw).mean().fillna(pd.DataFrame(averaged_shifth)),'-g',lw=2)
plt.plot(x,pd.DataFrame(averaged_shiftuh).rolling(window=wdw).mean().fillna(pd.DataFrame(averaged_shiftuh)),'-r',lw=2)

# plt.plot(x,(averaged_shifth),'-g',lw=2)
# plt.plot(x,(averaged_shiftuh),'-r',lw=2)
plt.xlabel('Circumferential Position (degrees)',fontsize=14)
plt.ylabel('E2A Mobility (degrees)',fontsize=14)
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.grid()
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='g', lw=0.7, linestyle='--'),
                Line2D([0], [0], color='r', lw=0.7, linestyle='--')]

plt.legend(custom_lines, ['Healthy Average', 'HCM Average', 'Individual Healthy', 'Individual HCM'],loc='upper right',fontsize=12)
# plt.title('Angle Shifts Across Circumferential Direction')
plt.show()