#stats functions
import import_functions
import helper_functions
import unwrap_functions
import local_SD_analysis_functions
import TwoDimensionAnalysis

from import_functions import get_, pp
from helper_functions import to_dist, plot_dists, helper_cmaps, plot_ims
from unwrap_functions import uw2,get_angles1_
from TwoDimensionAnalysis import group_unwraps_
from local_SD_analysis_functions import calc_SD

import numpy as np
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import random
from copy import deepcopy


'''
The first four functions in this file are responsible for computing which distribution an image matches better
They have a general format that needs to be followed to be able to include your own.

So the general format is for inputs:
img: the input image, can be a distribution or a 2d image
health: the health of this image. 0: Healthy 1: HCM
h_dist: healthy distribution to compare to
uh_dist: unhealthy distribution to compare to
n_bins: number of bins to use if needed

then the ouputs are: [oup1,oup2]
oup2 is always the prediction of the model. 0:healthy 1:HCM
oup1 has a set format. The first entry is 1 if the prediction is correct. The last entry is always 1 to count the number of times
the image was used in the dataset. All entries in the middle handle any test statistics that want to be stored per image
'''
def log_like(img,health,h_dist,uh_dist,n_bins):
    img = img[img!=0]
    img = img[~np.isnan(img)]
    bins = np.linspace(min(img.min(),h_dist.min(),uh_dist.min()),max(img.max(),h_dist.max(),uh_dist.max()),n_bins,endpoint=True)
    dist_img,_ = np.histogram(img, bins=bins,density=False)
    dist_h,_ = np.histogram(h_dist, bins=bins,density=False)
    dist_uh,_ = np.histogram(uh_dist, bins=bins,density=False)
    L1 = 0
    L2 = 0
    dist_hsum = sum(dist_h)
    dist_uhsum = sum(dist_uh)
    for num,dist1,dist2 in zip(dist_img,dist_h,dist_uh):

        if dist1 == 0:
            L1 += num*np.log(0.01)
        else:
            L1 += num*np.log(dist1/dist_hsum)
        if dist2 == 0: 
            L2 += num*np.log(0.01)
        else:
            L2 += num*np.log(dist2/dist_uhsum)
    oup =  L2/(L1+L2)
    oup2 = int(oup<0.5)
    if health == 1:
        oup1 = [int(oup<0.5),oup,1]
    elif health == 0:
        oup1 = [int(oup>0.5),oup,1]
    oup1 = [float(i) for i in oup1]
    return [oup1,oup2]
    
def custom_r2(img,health,h_dist,uh_dist,n_bins):
    img = img[img!=0]
    img = img[~np.isnan(img)]
    bins = np.linspace(min(img.min(),h_dist.min(),uh_dist.min()),max(img.max(),h_dist.max(),uh_dist.max()),n_bins,endpoint=True)
    dist_img,_ = np.histogram(img, bins=n_bins,density=False)
    dist_h,_ = np.histogram(h_dist, bins=n_bins,density=False)
    dist_uh,_ = np.histogram(uh_dist, bins=n_bins,density=False)
    _,_,r_h,p_h,_= stats.linregress(dist_img,dist_h)
    _,_,r_uh,p_uh,_= stats.linregress(dist_img,dist_uh)
    oup2 = int(r_h**2<r_uh**2)
    if health == 1:
        oup1 = [int(r_h**2<r_uh**2),r_h**2,r_uh**2,p_h,p_uh,1]
    elif health == 0:
        oup1 = [int(r_h**2>r_uh**2),r_h**2,r_uh**2,p_h,p_uh,1]
    oup1 = [float(i) for i in oup1]
    return [oup1,oup2]

def mann_whitney(img,health,h_dist,uh_dist):
    def calculate_zscore(imgs,dist,U):
        n1 = len(imgs)
        n2 = len(dist)
        return (U-n1*n2/2)/(np.sqrt(n1*n2/12)*np.sqrt(n1+n2+1))
    img = img[img!= 0]
    img = img[~np.isnan(img)]
    U_h,pnorm_h = stats.mannwhitneyu(img,h_dist)
    U_uh,pnorm_uh = stats.mannwhitneyu(img,uh_dist)
    z_h = calculate_zscore(img,h_dist,U_h)
    z_uh = calculate_zscore(img,uh_dist,U_uh)

    oup2 = int(abs(z_h)>abs(z_uh))
    if health == 1:
        oup1 = [int(abs(z_h)>abs(z_uh)),z_h,z_uh,pnorm_h,pnorm_uh,1]
    elif health == 0:
        oup1 = [int(abs(z_h)<abs(z_uh)),z_h,z_uh,pnorm_h,pnorm_uh,1]
    oup1 = [float(i) for i in oup1]
    return [oup1,oup2]
    

def KS2(img,health,h_dist,uh_dist):
    img = img[img != 0]
    img = img[~np.isnan(img)]
    res_h = stats.ks_2samp(img,h_dist,method="asymp")
    res_uh = stats.ks_2samp(img,uh_dist,method="asymp")
    stat_h = res_h.statistic
    stat_uh = res_uh.statistic
    p_h = res_h.pvalue
    p_uh = res_uh.pvalue
    oup2 = int(p_h<p_uh)
    if health == 1:
        oup1 = [bool(p_h<p_uh),stat_h,stat_uh,p_h,p_uh,1]
    elif health == 0:
        oup1 = [bool(p_h>p_uh),stat_h,stat_uh,p_h,p_uh,1]
    oup1 = [float(i) for i in oup1]
    return [oup1,oup2]

def run_all_tests_loocv(imgs_h,imgs_uh,bin_n=30):
    '''
    This is the main function used to run LOOCV statistical tests on each image in imgs_h and imgs_uh
    imgs_h,imgs_uh = list of healthy and unhealthy images
    bin_n = number of bins for LL and R2 test
    '''

    len_h = len(imgs_h)
    len_uh = len(imgs_uh)
    
    '''
    initialisation of info dfs that store test statistics, accuracy and the number of times the test is run.
    '''
    h_r_mw = pd.DataFrame(data = 0,columns=["acc (%)","z_h","z_uh","p_h","p_uh","n"],index=list(range(len_h)),dtype=float)
    uh_r_mw = pd.DataFrame(data = 0,columns=["acc (%)","z_h","z_uh","p_h","p_uh","n"],index=list(range(len_uh)),dtype=float)
    h_r_ks = pd.DataFrame(data = 0,columns=["acc (%)","stat_h","stat_uh","p_h","p_uh","n"],index=list(range(len_h)),dtype=float)
    uh_r_ks = pd.DataFrame(data = 0,columns=["acc (%)","stat_h","stat_uh","p_h","p_uh","n"],index=list(range(len_uh)),dtype=float)
    h_r_r2 = pd.DataFrame(data = 0,columns=["acc (%)","r2_h","r2_uh","p_h","p_uh","n"],index=list(range(len_h)),dtype=float)
    uh_r_r2 = pd.DataFrame(data = 0,columns=["acc (%)","r2_h","r2_uh","p_h","p_uh","n"],index=list(range(len_uh)),dtype=float)
    h_r_ll = pd.DataFrame(data = 0,columns=["acc (%)","p-val","n"],index=list(range(len_h)),dtype=float)
    uh_r_ll = pd.DataFrame(data = 0,columns=["acc (%)","p-val","n"],index=list(range(len_uh)),dtype=float)
    pm_mw = pd.DataFrame(data=0,columns=['Healthy','HCM'],index=['Negative','Positive'])
    pm_ks = pd.DataFrame(data=0,columns=['Healthy','HCM'],index=['Negative','Positive'])
    pm_r2 = pd.DataFrame(data=0,columns=['Healthy','HCM'],index=['Negative','Positive'])
    pm_ll = pd.DataFrame(data=0,columns=['Healthy','HCM'],index=['Negative','Positive'])
    full_uh_dist = np.concatenate([i.flatten() for i in imgs_uh])
    full_uh_dist = full_uh_dist[full_uh_dist!=0]
    full_uh_dist = full_uh_dist[~np.isnan(full_uh_dist)]
    full_h_dist = np.concatenate([i.flatten() for i in imgs_h])
    full_h_dist = full_h_dist[full_h_dist!=0]
    full_h_dist = full_h_dist[~np.isnan(full_h_dist)]

    for healthy_idx in range(len_h):
        '''
        for each healthy individual, retrieve that individual image. 
        Combine all the rest into distributions and perform the comparison
        The data is then stored in the info dfs initialised above
        '''
        h_img = imgs_h[healthy_idx].copy()
        train_idx = [i for i in range(len_h) if i != healthy_idx]
        train_h = [imgs_h[i] for i in train_idx]
        h_dist = np.concatenate([i.flatten() for i in train_h])
        h_dist = h_dist[h_dist!=0]
        h_dist = h_dist[~np.isnan(h_dist)]
        h_img = h_img[h_img!=0]
        h_img = h_img[~np.isnan(h_img)]

        oup = deepcopy(mann_whitney(h_img,0,h_dist,full_uh_dist))
        h_r_mw.loc[healthy_idx,:] += oup[0]
        predict = oup[1]
        pm_mw.iloc[predict,0] += 1

        
        oup = deepcopy(KS2(h_img,0,h_dist,full_uh_dist))
        h_r_ks.loc[healthy_idx,:] += oup[0]
        predict = oup[1]
        pm_ks.iloc[predict,0] += 1
        
        oup = deepcopy(custom_r2(h_img,0,h_dist,full_uh_dist,bin_n))
        h_r_r2.loc[healthy_idx,:] += oup[0]
        predict = oup[1]
        pm_r2.iloc[predict,0] += 1

        
        oup = deepcopy(log_like(h_img,0,h_dist,full_uh_dist,bin_n))
        h_r_ll.loc[healthy_idx,:] += oup[0]
        predict = oup[1]
        pm_ll.iloc[predict,0] += 1

    for unhealthy_idx in range(len_uh):
        uh_img = imgs_uh[unhealthy_idx].copy()
        train_idx = [i for i in range(len_uh) if i != unhealthy_idx]
        train_uh = [imgs_uh[i] for i in train_idx]
        uh_dist = np.concatenate([i.flatten() for i in train_uh])
        uh_dist = uh_dist[uh_dist!=0]
        uh_dist = uh_dist[~np.isnan(uh_dist)]
        uh_img = uh_img[uh_img!=0]
        uh_img = uh_img[~np.isnan(uh_img)]

        oup = deepcopy(mann_whitney(uh_img,1,full_h_dist,uh_dist))
        uh_r_mw.loc[unhealthy_idx,:] += oup[0]
        predict = oup[1]
        pm_mw.iloc[predict,1] += 1

        
        oup = deepcopy(KS2(uh_img,1,full_h_dist,uh_dist))
        uh_r_ks.loc[unhealthy_idx,:] += oup[0]
        predict = oup[1]
        pm_ks.iloc[predict,1] += 1

        
        oup = deepcopy(custom_r2(uh_img,1,full_h_dist,uh_dist,bin_n))
        uh_r_r2.loc[unhealthy_idx,:] += oup[0]
        predict = oup[1]
        pm_r2.iloc[predict,1] += 1

        
        oup = deepcopy(log_like(uh_img,1,full_h_dist,uh_dist,bin_n))
        uh_r_ll.loc[unhealthy_idx,:] += oup[0]
        predict = oup[1]
        pm_ll.iloc[predict,1] += 1

    #account for class imbalance by setting sums of healthy and unhealthy individuals to the same number
    pm_mw = pm_mw.copy().div(pm_mw.sum())
    pm_ks = pm_ks.copy().div(pm_ks.sum())
    pm_r2 = pm_r2.copy().div(pm_r2.sum())
    pm_ll = pm_ll.copy().div(pm_ll.sum())
    returnval = [(h_r_mw,uh_r_mw),(h_r_ks,uh_r_ks),(h_r_r2,uh_r_r2),(h_r_ll,uh_r_ll),(pm_mw,pm_ks,pm_r2,pm_ll)]
    def set_acc(arrs):
        oup = []
        for arr in arrs:
            h_arr = arr[0]
            uh_arr = arr[1]
            h_arr.loc[:,"acc (%)"] = np.round(100*h_arr.loc[:,"acc (%)"]/h_arr.loc[:,'n'],2)
            uh_arr.loc[:,"acc (%)"] = np.round(100*uh_arr.loc[:,"acc (%)"]/uh_arr.loc[:,'n'],2)
            oup.append([h_arr,uh_arr])
        return oup
    #The accuracy is actually only set at the end by doing acc(%)/n as acc stores how many times it was correctly predicted
    returnval[:-1] = set_acc(returnval[:-1])
    return returnval


def run_all_tests_kfold(imgs_h,imgs_uh,test_size,num_runs,bin_n):
    '''
    Similar run_all_tests_loocv but selects multiple individuals for the test set. 
    imgs_h,imgs_uh: list of healthy and HCM images
    test_size: Number of individuals to put into test set [healthy_test_number,HCM_test_number]
    num_runs: how many times to rerun the train-test split and statistical test workflow
    bin_n: number of bins for LL and R2 test
    '''
    len_h = len(imgs_h)
    len_uh = len(imgs_uh)
    
    h_r_mw = pd.DataFrame(data = 0,columns=["acc (%)","z_h","z_uh","p_h","p_uh","n"],index=list(range(len_h)),dtype=float)
    uh_r_mw = pd.DataFrame(data = 0,columns=["acc (%)","z_h","z_uh","p_h","p_uh","n"],index=list(range(len_uh)),dtype=float)
    h_r_ks = pd.DataFrame(data = 0,columns=["acc (%)","stat_h","stat_uh","p_h","p_uh","n"],index=list(range(len_h)),dtype=float)
    uh_r_ks = pd.DataFrame(data = 0,columns=["acc (%)","stat_h","stat_uh","p_h","p_uh","n"],index=list(range(len_uh)),dtype=float)
    h_r_r2 = pd.DataFrame(data = 0,columns=["acc (%)","r2_h","r2_uh","p_h","p_uh","n"],index=list(range(len_h)),dtype=float)
    uh_r_r2 = pd.DataFrame(data = 0,columns=["acc (%)","r2_h","r2_uh","p_h","p_uh","n"],index=list(range(len_uh)),dtype=float)
    h_r_ll = pd.DataFrame(data = 0,columns=["acc (%)","p-val","n"],index=list(range(len_h)),dtype=float)
    uh_r_ll = pd.DataFrame(data = 0,columns=["acc (%)","p-val","n"],index=list(range(len_uh)),dtype=float)
    pm_mw = pd.DataFrame(data=0,columns=['Healthy','HCM'],index=['Negative','Positive'])
    pm_ks = pd.DataFrame(data=0,columns=['Healthy','HCM'],index=['Negative','Positive'])
    pm_r2 = pd.DataFrame(data=0,columns=['Healthy','HCM'],index=['Negative','Positive'])
    pm_ll = pd.DataFrame(data=0,columns=['Healthy','HCM'],index=['Negative','Positive'])
    for run_num in range(num_runs):
        
        test_h_idx = np.random.choice(len_h,size=test_size[0],replace=False)
        test_uh_idx = np.random.choice(len_uh,size=test_size[1],replace=False)
        train_h_idx = [i for i in range(len_h) if i not in test_h_idx]
        train_uh_idx = [i for i in range(len_uh) if i not in test_uh_idx]

        test_h = [imgs_h[i] for i in test_h_idx]
        train_h = [imgs_h[i] for i in train_h_idx]
        test_uh = [imgs_uh[i] for i in test_uh_idx]
        train_uh = [imgs_uh[i] for i in train_uh_idx]

        h_dist = np.concatenate([i.flatten() for i in train_h])
        uh_dist = np.concatenate([i.flatten() for i in train_uh])
        h_dist = h_dist[h_dist!=0]
        uh_dist = uh_dist[uh_dist!=0]
        h_dist = h_dist[~np.isnan(h_dist)]
        uh_dist = uh_dist[~np.isnan(uh_dist)]

        for h_idx,h_img in zip(test_h_idx,test_h):
            oup = mann_whitney(h_img,0,h_dist,uh_dist)
            h_r_mw.loc[h_idx,:] += oup[0]
            predict = oup[1]
            pm_mw.iloc[predict,0] += 1

            
            oup = KS2(h_img,0,h_dist,uh_dist)
            h_r_ks.loc[h_idx,:] += oup[0]
            predict = oup[1]
            pm_ks.iloc[predict,0] += 1
            
            oup = custom_r2(h_img,0,h_dist,uh_dist,bin_n)
            h_r_r2.loc[h_idx,:] += oup[0]
            predict = oup[1]
            pm_r2.iloc[predict,0] += 1

            
            oup = log_like(h_img,0,h_dist,uh_dist,bin_n)
            h_r_ll.loc[h_idx,:] += oup[0]
            predict = oup[1]
            pm_ll.iloc[predict,0] += 1
            
            
        for uh_idx,uh_img in zip(test_uh_idx,test_uh):
            oup = mann_whitney(uh_img,1,h_dist,uh_dist)
            uh_r_mw.loc[uh_idx,:] += oup[0]
            predict = oup[1]
            pm_mw.iloc[predict,1] += 1

            
            oup = KS2(uh_img,1,h_dist,uh_dist)
            uh_r_ks.loc[uh_idx,:] += oup[0]
            predict = oup[1]
            pm_ks.iloc[predict,1] += 1

            
            oup = custom_r2(uh_img,1,h_dist,uh_dist,bin_n)
            uh_r_r2.loc[uh_idx,:] += oup[0]
            predict = oup[1]
            pm_r2.iloc[predict,1] += 1

            
            oup = log_like(uh_img,1,h_dist,uh_dist,bin_n)
            uh_r_ll.loc[uh_idx,:] += oup[0]
            predict = oup[1]
            pm_ll.iloc[predict,1] += 1
            

        pm_mw = pm_mw.copy().div(pm_mw.sum())
        pm_ks = pm_ks.copy().div(pm_ks.sum())
        pm_r2 = pm_r2.copy().div(pm_r2.sum())
        pm_ll = pm_ll.copy().div(pm_ll.sum())
    returnval = [(h_r_mw,uh_r_mw),(h_r_ks,uh_r_ks),(h_r_r2,uh_r_r2),(h_r_ll,uh_r_ll),(pm_mw,pm_ks,pm_r2,pm_ll)]
    def set_acc(arrs):
        oup = []
        for arr in arrs:
            h_arr = arr[0]
            uh_arr = arr[1]
            h_arr.loc[:,"acc (%)"] = np.round(100*h_arr.loc[:,"acc (%)"]/h_arr.loc[:,'n'],2)
            uh_arr.loc[:,"acc (%)"] = np.round(100*uh_arr.loc[:,"acc (%)"]/uh_arr.loc[:,'n'],2)
            oup.append([h_arr,uh_arr])
        return oup
    returnval[:-1] = set_acc(returnval[:-1])
    return returnval

def get_performance_metrics(df):
    '''
    private
    Used to calculate performance metrics
    '''
    b = df.at['Positive','Healthy'] #b #False Positive
    d = df.at['Negative','Healthy'] #d #True Negative
    a = df.at['Positive','HCM'] #a #True Positive
    c = df.at['Negative','HCM'] #c #False Negative
    precision = np.round(a/(a+b)*100,3)
    recall = np.round(a/(a+c)*100,3)
    performance_metrics = {"Sensitivity": np.round(a/(a+c)*100,3),"Specificity" : np.round(d/(b+d)*100,3), "PPV" : np.round(a/(a+b)*100,3), "NPV" : np.round(d/(c+d)*100,3),"F1 Score": 2*precision*recall/(precision+recall),"Total" : np.round((a+d)/(a+b+c+d)*100,3)}
    return performance_metrics

def get_metrics(output,test,pri=None):
    '''
    retrieves the image level data and performance matrix from the big output array of run_all_tests_loocv or run_all_test_kfold
    output: output of run_all_tests_loocv or run_all_test_kfold (returnval)
    test: one of the tests below to specifically retrieve
    pri: set to something to print the image level data and performance matrix
    '''
    tests = ['MW','KS','R2','LL']
    test = tests.index(test)
    if pri is not None:
        print("Healthy\n",output[test][0])
        print("Unhealthy\n", output[test][1])
        print("Performance Matrix\n", output[-1][test])
    return get_performance_metrics(output[-1][test])

'''
Example code to generate the large tables found in the appendix
This example computes the F1 Scores and NPV's for the middle region of the local variance images


techniques = ['SE','STEAM']
params = ['MD','FA','IA','E2A']
methods = ['Systole', 'Diastole']
bin_n = 30
n = 3

df_f1 = pd.DataFrame(columns=['MW','KS','R2','LL']) #store the performance of each test per subset of data
df_npv = pd.DataFrame(columns=['MW','KS','R2','LL'])
for param in params: # for each subset of data effectively
    print(param)
    for technique in techniques:
        print(technique)
        for method in methods:
            h = get_('Healthy',technique,method,param,'myo') #get the data
            uh = get_('HCM',technique,method,param,'myo')
            h = deepcopy(pp(h)) #preprocess
            uh = deepcopy(pp(uh))

            masksh = h[1] #retrieve masks
            masksuh = uh[1]

            if param in ['MD','FA']: #if param is MD or FA, use non-periodic classic SD, if its an angle param use periodic SD
                angle_param = False
            else:
                angle_param = True

            h = deepcopy(calc_SD(h,n,angle_param))
            uh = deepcopy(calc_SD(uh,n,angle_param))

            h = [h,masksh] #images and masks need to be recombined to pass into uw2
            uh = [uh,masksuh]
            
            h = deepcopy(uw2(h,num_layers=3,return_contours=True)) #return the contour_values for the three regions
            uh = deepcopy(uw2(uh,num_layers=3,return_contours=True))

            h = deepcopy(group_unwraps_(h)) #group unwraps to make them shape 3XN instead of Nx3
            uh = deepcopy(group_unwraps_(uh))

            h = h[1] #take the middle region
            uh = uh[1]

            h_ = [i[:,0] for i in h] #retrieve only the pixel values from the countour_values, i[:,1] would be the circumfrential position
            uh_ = [i[:,0] for i in uh]

            oup = run_all_tests_loocv(h_,uh_,30) #perform loocv tests on images
            mw = get_metrics(oup,'MW') #retrieve performances of tests
            ks = get_metrics(oup,'KS')
            r2 = get_metrics(oup,'R2')
            ll = get_metrics(oup,'LL')
            # print(mw)
            # print(ks)
            # print(r2)
            # print(ll)
            df = pd.DataFrame(columns = mw.keys()) #all performance metrics of all tests
            df.loc[len(df)] = mw.values()
            df.loc[len(df)] = ks.values()
            df.loc[len(df)] = r2.values()
            df.loc[len(df)] = ll.values()
            df.index = ['MW','KS','R2','LL']
            df_f1.loc[len(df_f1)] = df.loc[:,'F1 Score'] #add to performance table
            df_npv.loc[len(df_npv)] = df.loc[:,'NPV']

print(df_f1)
print(df_npv)
df_f1 = np.round(df_f1,2) #round and save
df_npv = np.round(df_npv,2)
df_f1.to_csv('!F1Scores.csv')
df_npv.to_csv('!NPV.csv')


#If you instead just wanted to test the full images, it would be as simple as this:


techniques = ['SE','STEAM']
params = ['MD','FA','IA','E2A']
bin_n = 30
n = 3
df_f1 = pd.DataFrame(columns=['MW','KS','R2','LL']) #store the performance of each test per subset of data
df_npv = pd.DataFrame(columns=['MW','KS','R2','LL'])
for param in params: # for each subset of data effectively
    print(param)
    for technique in techniques:
        print(technique)
        for method in methods:
            h = get_('Healthy',technique,method,param,'myo') #get the data
            uh = get_('HCM',technique,method,param,'myo')
            h = deepcopy(pp(h)) #preprocess
            uh = deepcopy(pp(uh))

            h_ = h[0] #retrieve images from data
            uh_ = uh[0] 

            oup = run_all_tests_loocv(h_,uh_,30) #perform loocv tests on images
            mw = get_metrics(oup,'MW') #retrieve performances of tests
            ks = get_metrics(oup,'KS')
            r2 = get_metrics(oup,'R2')
            ll = get_metrics(oup,'LL')
            # print(mw)
            # print(ks)
            # print(r2)
            # print(ll)
            df = pd.DataFrame(columns = mw.keys()) #all performance metrics of all tests
            df.loc[len(df)] = mw.values()
            df.loc[len(df)] = ks.values()
            df.loc[len(df)] = r2.values()
            df.loc[len(df)] = ll.values()
            df.index = ['MW','KS','R2','LL']
            df_f1.loc[len(df_f1)] = df.loc[:,'F1 Score'] #add to performance table
            df_npv.loc[len(df_npv)] = df.loc[:,'NPV']
print(df_f1)
print(df_npv)
df_f1 = np.round(df_f1,2)# round and save
df_npv = np.round(df_npv,2)
df_f1.to_csv('!F1Scores.csv')
df_npv.to_csv('!NPV.csv')

'''