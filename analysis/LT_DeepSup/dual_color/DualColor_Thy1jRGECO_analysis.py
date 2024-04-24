import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import copy
import os
import pickle
import seaborn as sns
from scipy import stats

def clean_traces(ogSignal, sigma = 6, sig_up = 4, sig_down = 12, peak_th=0.1):
    lowpassSignal = uniform_filter1d(ogSignal, size = 4000, axis = 0)
    signal = gaussian_filter1d(ogSignal, sigma = sigma, axis = 0)
    for nn in range(signal.shape[1]):
        baseSignal = np.histogram(ogSignal[:,nn], 100)
        baseSignal = baseSignal[1][np.argmax(baseSignal[0])]
        baseSignal = baseSignal + lowpassSignal[:,nn] - np.min(lowpassSignal[:,nn]) 
        cleanSignal = signal[:,nn]-baseSignal
        cleanSignal = cleanSignal/np.max(cleanSignal,axis = 0)
        cleanSignal[cleanSignal<0] = 0
        signal[:,nn] = cleanSignal

    biSignal = np.zeros(signal.shape)

    gaus = lambda x,sig,amp,vo: amp*np.exp(-(((x)**2)/(2*sig**2)))+vo;
    x = np.arange(-5*sig_down, 5*sig_down,1);
    upGaus = gaus(x,sig_up, 1, 0); 
    upGaus[5*sig_down+1:] = 0
    downGaus = gaus(x,sig_down, 1, 0); 
    downGaus[:5*sig_down+1] = 0
    finalGaus = downGaus + upGaus;

    for nn in range(signal.shape[1]):
        peakSignal,_ =find_peaks(signal[:,nn],height=peak_th)
        biSignal[peakSignal, nn] = signal[peakSignal, nn]
        if finalGaus.shape[0]<signal.shape[0]:
            biSignal[:, nn] = np.convolve(biSignal[:, nn],finalGaus, 'same')
    return biSignal

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nnDist < np.percentile(nnDist, 20)
    return noiseIdx

def get_centroids(input_A, input_B, label_A, label_B, dir_A = None, dir_B = None, ndims = 2, nCentroids = 20):
    input_A = input_A[:,:ndims]
    input_B = input_B[:,:ndims]
    if label_A.ndim>1:
        label_A = label_A[:,0]
    if label_B.ndim>1:
        label_B = label_B[:,0]
    #compute label max and min to divide into centroids
    total_label = np.hstack((label_A, label_B))
    labelLimits = np.array([(np.percentile(total_label,5), np.percentile(total_label,95))]).T[:,0] 
    #find centroid size
    centSize = (labelLimits[1] - labelLimits[0]) / (nCentroids)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    centEdges = np.column_stack((np.linspace(labelLimits[0],labelLimits[0]+centSize*(nCentroids),nCentroids),
                                np.linspace(labelLimits[0],labelLimits[0]+centSize*(nCentroids),nCentroids)+centSize))

    if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
        centLabel_A = np.zeros((nCentroids,ndims))
        centLabel_B = np.zeros((nCentroids,ndims))
        
        ncentLabel_A = np.zeros((nCentroids,))
        ncentLabel_B = np.zeros((nCentroids,))
        for c in range(nCentroids):
            points_A = input_A[np.logical_and(label_A >= centEdges[c,0], label_A<centEdges[c,1]),:]
            centLabel_A[c,:] = np.median(points_A, axis=0)
            ncentLabel_A[c] = points_A.shape[0]
            
            points_B = input_B[np.logical_and(label_B >= centEdges[c,0], label_B<centEdges[c,1]),:]
            centLabel_B[c,:] = np.median(points_B, axis=0)
            ncentLabel_B[c] = points_B.shape[0]
    else:
        input_A_left = copy.deepcopy(input_A[dir_A[:,0]==1,:])
        label_A_left = copy.deepcopy(label_A[dir_A[:,0]==1])
        input_A_right = copy.deepcopy(input_A[dir_A[:,0]==2,:])
        label_A_right = copy.deepcopy(label_A[dir_A[:,0]==2])
        
        input_B_left = copy.deepcopy(input_B[dir_B[:,0]==1,:])
        label_B_left = copy.deepcopy(label_B[dir_B[:,0]==1])
        input_B_right = copy.deepcopy(input_B[dir_B[:,0]==2,:])
        label_B_right = copy.deepcopy(label_B[dir_B[:,0]==2])
        
        centLabel_A = np.zeros((2*nCentroids,ndims))
        centLabel_B = np.zeros((2*nCentroids,ndims))
        ncentLabel_A = np.zeros((2*nCentroids,))
        ncentLabel_B = np.zeros((2*nCentroids,))
        
        for c in range(nCentroids):
            points_A_left = input_A_left[np.logical_and(label_A_left >= centEdges[c,0], label_A_left<centEdges[c,1]),:]
            centLabel_A[2*c,:] = np.median(points_A_left, axis=0)
            ncentLabel_A[2*c] = points_A_left.shape[0]
            points_A_right = input_A_right[np.logical_and(label_A_right >= centEdges[c,0], label_A_right<centEdges[c,1]),:]
            centLabel_A[2*c+1,:] = np.median(points_A_right, axis=0)
            ncentLabel_A[2*c+1] = points_A_right.shape[0]

            points_B_left = input_B_left[np.logical_and(label_B_left >= centEdges[c,0], label_B_left<centEdges[c,1]),:]
            centLabel_B[2*c,:] = np.median(points_B_left, axis=0)
            ncentLabel_B[2*c] = points_B_left.shape[0]
            points_B_right = input_B_right[np.logical_and(label_B_right >= centEdges[c,0], label_B_right<centEdges[c,1]),:]
            centLabel_B[2*c+1,:] = np.median(points_B_right, axis=0)
            ncentLabel_B[2*c+1] = points_B_right.shape[0]

    del_cent_nan = np.all(np.isnan(centLabel_A), axis= 1)+ np.all(np.isnan(centLabel_B), axis= 1)
    del_cent_num = (ncentLabel_A<20) + (ncentLabel_B<20)
    del_cent = del_cent_nan + del_cent_num
    
    centLabel_A = np.delete(centLabel_A, del_cent, 0)
    centLabel_B = np.delete(centLabel_B, del_cent, 0)

    return centLabel_A, centLabel_B

def find_rotation(data_A, data_B, v):
    angles = np.linspace(-np.pi,np.pi,100)
    error = list()
    for angle in angles:
        #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        #https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
        a = np.cos(angle/2)
        b = np.sin(angle/2)*v[0,0]
        c = np.sin(angle/2)*v[1,0]
        d = np.sin(angle/2)*v[2,0]
        R = np.array([
                [a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d),a**2-b**2+c**2-d**2, 2*(c*d - a*b)],
                [2*(b*d - a*c), 2*(c*d + a*b), a**2-b**2-c**2+d**2]
            ])

        new_data =np.matmul(R, data_A.T).T
        error.append(np.sum(np.linalg.norm(new_data - data_B, axis=1)))

    return error

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

base_dir = '/home/julio/Documents/DeepSup_project/DualColor/Thy1jRGECO/'
###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################           PRE DATA         #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################
pre_data_dict = {}

#Thy1jRGECO11
pre_green_cells = [0,1,4,6,7,8,12,13,15,16,19,20,26,27,28,29,31,35,38,39,40,41,42,43,45,46,48,52,54,57,58,59,60,61,62,63,69,70,71,72,73,75,76,77,79,80,81,82,84,85,86,88,89,92,94,95,96,97,98,99,101,102,104,105,106,107,108,109,110,111,113,121,124,127,129,130,134,138,139,142,143,144,146,147,148,149,151,153,154,155,160,161,162,163,164,173,184,185,187,189,190,193,194,199,203,205,206,211,217,220,222,223,224,234]
pre_red_cells = [0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,21,22,23,24,25,26,27,28,29,30,34,38,39,40,41,42,43,44,45,46,47,48,51,53,55,56,57,58,59,60,61,62,63,65,66,67,68,70,72,76,77,78,79,80,81,83,84,85,89,91,92,93,94,95,96,97,98,99,100,102,103,104,106,108,109,110,111,112,114,115,116,117,118,119,120,121,122,123,124,126,128,129,130,131,132,134,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,155,157,158,160,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,184,185,187,189,197,198,200,201,202,203,204,205,207,210,211,216,217,218,221,223,225,226,227,228,229,231,232,233,237,238,239,240,243,245,246,247,249,250,251,252,253,255,256,257,258,260,261,262,263,264,266,267,268,269,270,271,274,275,277,280,281,282,283,285,286,287,288,289,291,292,294,296,299,300,301,302,303,305,306,307,308,309,310,311,312,314,315,316,323,330,331,332,334,340,342,343,345,355,356,357,358,359,360,361,362,363,364,366,367,368,369,370,371,373,374,375,376,377,378,379,381,384,385,387,388,389,390,391,392,393,395,396,397,398,399,400,401,404,405,407,408,409,410,414,415,416,417,418,419,420,421,423,424,425,431,432,433,435,436,437,438,439,442,444,445,447,449,450,451,452,453,454,455,456,459,460,461,462,463,465,467,470,471,472,473,474,477,478,479,486,490,491,492,493,494,495,496,497,498,499,500,503,504,505,506,509,513,514,515,516,517,518,519,522,523,524,525,529,533,536,539,541,542,543,544,547,550,551,553,554,558,559,560,562,563,564,565,566,567,569,573,574,575,576,577,578,580]
rot_green_cells = [1,3,4,7,8,6,2,5,10,11,12,9,20,16,15,13,14,18,21,22,26,33,23,29,30,28,31,24,27,32,37,154,34,36,39,38,46,43,49,48,56,50,44,45,57,42,51,55,59,64,67,63,52,62,68,66,58,61,60,65,79,69,71,78,75,70,76,74,80,82,84,83,87,90,91,93,95,101,99,100,105,110,102,104,111,107,103,109,114,115,113,116,117,118,119,120,131,125,128,132,133,135,136,138,139,140,141,143,145,144,146,149,147,35]
rot_red_cells = [0,2,3,4,16,5,6,17,9,18,11,12,91,14,15,7,109,107,105,20,19,26,40,497,24,27,22,21,28,30,31,487,44,34,36,35,33,37,43,32,182,48,45,47,63,50,53,55,52,59,56,60,58,51,62,57,65,494,66,68,67,197,75,74,76,93,80,97,206,77,79,78,85,87,81,95,90,92,83,89,94,82,86,96,88,100,222,103,101,102,121,108,230,106,112,116,110,114,111,113,104,117,99,98,120,119,130,125,181,141,145,134,136,140,135,137,170,133,138,144,149,151,127,131,148,163,166,169,164,165,155,159,158,299,178,180,176,160,168,171,172,188,191,175,177,183,154,153,187,190,300,193,304,184,194,196,203,211,199,205,207,213,209,208,210,499,219,216,215,489,218,217,239,212,252,253,496,254,226,225,227,234,229,232,235,249,223,236,238,233,250,241,247,243,245,248,255,242,251,283,259,268,264,267,279,275,263,272,280,273,266,276,257,256,500,284,269,270,261,262,287,291,288,293,289,292,294,295,296,285,301,303,306,302,308,309,307,314,311,312,315,322,317,316,340,319,318,325,334,324,328,348,326,327,332,335,333,336,338,329,323,415,331,370,493,342,343,344,368,341,345,354,349,356,357,355,382,375,352,361,360,363,364,359,346,386,371,384,385,373,372,376,377,379,381,378,387,388,389,391,394,392,393,407,396,390,399,400,404,403,454,401,408,406,405,491,402,414,411,463,416,425,420,421,423,424,419,427,428,430,429,449,447,434,437,433,436,469,440,441,439,438,442,445,444,435,432,450,451,453,456,457,459,475,455,461,460,462,464,465,466,468,467,472,471,473,470,474,477,476,495,481,480,483,485,484,486,492,418,380,129,122,124,224,152,412,185]
pre_data_dict['Thy1jRGECO11'] = {
    'mouse': 'Thy1jRGECO11',
    'n_neigh': 120,
    'dim': 3,
    'vel_th': 5,
    'green_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO11/Inscopix_data/Thy1jRGECO11_lt_green_raw.csv'),
    'green_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO11/Inscopix_data/Thy1jRGECO11_rot_green_raw.csv'),
    'red_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO11/Inscopix_data/Thy1jRGECO11_lt_red_raw.csv'),
    'red_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO11/Inscopix_data/Thy1jRGECO11_rot_red_raw.csv'),
    'pos_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO11/Inscopix_data/Thy1jRGECO11_lt_position.mat'),
    'pos_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO11/Inscopix_data/Thy1jRGECO11_rot_position.mat'),
    'pre_green_cells': copy.deepcopy(pre_green_cells),
    'pre_red_cells': copy.deepcopy(pre_red_cells),
    'rot_green_cells': copy.deepcopy(rot_green_cells),
    'rot_red_cells': copy.deepcopy(rot_red_cells)
}

#Thy1jRGECO22
pre_green_cells = [2,10,11,15,17,18,21,22,23,24,25,26,27,28,30,31,32,38,39,40,41,42,50,51,52,53,54,55,56,60,61,63,64,66,68,69,71,72,73,74,75,76,77,78,79,80,81,82,83,84,86,90,92,94,95,97,98,100,101,107,108,112,115,116,117,118,119,120,121,122,123,124,126,127,128,129,130,131,132,133,134,135,136,137,139,140,144,145,150,153,155,156,158,159,160,161,162,163,164,167,168,171,175,176,177,178,180,181,182,183,184,185,187,189,190,191,192,198,199,200,201,203,204,205,206,207,208,209,212,214,216,217,220,221,222,223,224,225,226,227,228,230,236,237,238,240,242,243,244,245,246,247,249,251,252,253,254,255,257,258,259,260,261,262,263,264,265,266,269,274,280,288,289,290,291,292,293,294,295,296,299,300,301,302,303,304,305,306,308,309,310,311,315,316,317,318,319,321,322,324,325,326,331,332,333,334,335,336,337]
pre_green_cells = [x-1 for x in pre_green_cells]
pre_red_cells = [1,2,3,4,5,6,7,9,10,16,17,22,23,25,26,27,28,30,33,37,41,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,65,68,70,71,77,79,80,81,82,83,84,86,87,89,91,92,96,99,100,102,103,105,106,107,109,110,111,112,113,114,117,120,122,124,129,130,131,133,135,136,139,140,142,144,146,147,148,151,152,153,154,155,156,160,161,163,165,166,170,172,173,175,178,179,180,182,183,188,189,190,191,196,197,198,200,201,202,203,208,209,210,213,214,215,216,217,219,220,221,222,223,224,225,226,228,229,234,236,239,243,244,245,246,247,248,250,251,252,253,254,255,256,258,259,260,261,263,266,267,269,270,271,272,273,276,279,282,283,285,286,287,288,289,290,291,293,297,298,301,303,306,307,308,311,313,314,315,316,319,324,331,332,333,334,335,336,341,343,344,349,351,352,354,355,356,357,358,359,360,362,364,365,367,368,369,371,372,374,375,378,383,385,386,387,388,389,390,391,392,393,394,395,396,397,398,400,401,403,405,408,421,422,423,424,427,428,430,432,433,437,439,440,441,442,443,445,447,449,450,451,452,453,454,455,456,465,466,469,470,471,476,479,480,481]
pre_red_cells = [x-1 for x in pre_red_cells]
rot_green_cells = [0,1,3,4,10,7,9,11,12,22,17,14,18,21,13,15,16,28,287,29,25,24,23,35,48,36,38,42,47,39,43,45,49,34,290,53,50,52,54,55,59,71,56,64,67,60,121,58,63,70,66,75,61,65,68,77,80,82,76,84,94,89,86,88,95,104,99,108,103,105,116,114,101,96,97,113,109,110,115,112,117,118,125,119,131,120,288,130,127,126,128,129,132,147,134,140,145,139,141,136,142,135,138,146,148,149,144,152,151,150,153,154,158,156,159,155,162,157,165,234,174,163,164,168,173,236,176,167,172,170,175,166,192,178,246,201,183,185,188,180,179,182,181,177,193,195,220,198,211,203,200,217,219,196,205,213,210,199,218,207,197,216,194,269,221,206,227,222,228,229,235,238,239,241,243,245,244,248,247,242,250,251,253,252,255,267,257,256,268,262,261,265,263,266,254,270,274,275,276,277,281,282,285,286,186,32,90,74,40]
rot_green_cells = [x-1 for x in rot_green_cells]
rot_red_cells = [3,4,2,10,0,9,8,1,64,7,11,14,23,15,13,89,20,21,18,22,65,24,12,25,38,27,36,26,33,37,28,41,29,34,35,32,31,40,339,30,42,43,44,143,51,48,47,49,50,54,59,53,62,69,60,66,63,67,57,68,46,80,70,76,71,79,77,78,84,337,73,83,86,82,95,93,74,81,72,85,92,343,97,98,122,121,102,103,99,120,101,115,104,107,100,112,118,114,190,117,116,119,131,109,128,130,123,124,126,133,145,138,148,139,141,142,144,147,140,134,136,135,149,170,163,155,154,152,158,160,157,173,175,176,161,177,169,164,249,156,171,151,174,165,172,153,184,179,178,183,181,193,186,194,192,182,202,196,276,201,180,187,197,203,191,200,188,228,209,207,216,340,189,217,227,206,211,210,223,225,229,230,231,232,233,235,236,234,252,237,254,241,246,242,251,240,248,250,243,244,253,238,239,245,273,257,258,270,262,261,259,264,260,275,266,269,271,265,277,274,267,255,272,296,280,279,287,283,284,295,281,278,288,291,290,286,289,297,329,298,299,300,333,304,308,309,310,306,320,312,314,313,316,318,311,319,322,325,321,324,323,326,327,330,332,338,16,17,105,150,137,162]
rot_red_cells = [x-1 for x in rot_red_cells]
pre_data_dict['Thy1jRGECO22'] = {
    'mouse': 'Thy1jRGECO22',
    'n_neigh': 120,
    'dim': 3,
    'vel_th': 5,
    'green_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO22/Inscopix_data/Thy1jRGECO22_lt_green_raw.csv'),
    'green_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO22/Inscopix_data/Thy1jRGECO22_rot_green_raw.csv'),
    'red_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO22/Inscopix_data/Thy1jRGECO22_lt_red_raw.csv'),
    'red_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO22/Inscopix_data/Thy1jRGECO22_rot_red_raw.csv'),
    'pos_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO22/Inscopix_data/Thy1jRGECO22_lt_position.mat'),
    'pos_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO22/Inscopix_data/Thy1jRGECO22_rot_position.mat'),
    'pre_green_cells': copy.deepcopy(pre_green_cells),
    'pre_red_cells': copy.deepcopy(pre_red_cells),
    'rot_green_cells': copy.deepcopy(rot_green_cells),
    'rot_red_cells': copy.deepcopy(rot_red_cells)
}

#Thy1jRGECO23
pre_green_cells = [4,5,6,7,8,9,10,11,14,16,17,18,21,25,26,27,28,30,32,33,34,35,36,41,42,44,45,46,47,48,49,51,52,53,54,56,57,58,60,61,62,63,64,65,66,67,70,71,72,73,75,76,77,78,79,80,81,83,84,86,87,88,89,90,93,95,96,97,98,104,105,107,108,110,112,113,114,115,119,120,122,123,124,125,127,129,130,131,134,136,137,138,139,140,141,144,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,172,173,176,177,178,180,182,184,185,186,188,189,191,193,194,195,196,198,199,201,202,207,208,209,210,211,212,218,219,220,222,223,224,225,226,227,229,232,236,237,238,240,241,242,243,245,247,248,249,250,251,252,253,254,255,256,257,258,259,260,262,263,264,267,268,269,270,272,273,274,278,280,281,282,283,285,286,287,288,291,292,293,294,295,296,300,301,302,306,307,309,310,314,315,316,318]
pre_red_cells = [1,3,4,8,9,10,11,12,13,14,15,16,21,22,23,24,27,28,30,31,37,43,45,46,47,48,50,51,52,53,54,57,59,61,64,65,66,67,68,69,70,71,72,73,75,76,77,78,79,82,83,91,92,93,94,95,98,99,101,102,103,104,106,108,110,116,122,123,124,126,129,130,131,132,133,137,138,139,140,141,145,146,147,148,149,150,152,153,154,156,157,159,160,162,164,165,166,167,168,169,170,172,173,174,178,179,180,181,182,192,193,194,196,197,199,201,202,205,207,208,210,214,215,218,229,230,231,232,233,234,235,236,237,241,242,244,245,252,254,255,256,257,258,260,261,263,264,265,267,268,270,271,272,273,276,277,278,279,280,281,282,283,284,286,288,289,293,294,299,300,305,306,307,308,310,311,313,314,315,316,318,319,320,321,323,328,330,331,333,334,342,343,347,349,351,354,355,357,359,361,362,363,364,365,368,369,370,371,372,376,377,378,380,381,383,384,385,386,387,389,395,396,398,399,400,403,404,406,407,409,410,411,412,413,414,415,416,417,418,419,422,423,425,426,427,428,432,435,438,440,441,443,446,447,448,449,451,452,453,456,458,459,461,462]
rot_green_cells = [12,3,4,5,14,20,9,13,19,16,15,17,18,21,23,24,32,25,37,26,36,27,0,28,35,34,39,40,64,38,46,52,51,43,60,44,53,48,58,63,67,41,62,45,66,59,65,47,42,68,82,78,72,80,77,73,81,74,76,75,83,85,86,88,333,91,92,94,93,95,97,96,22,100,99,102,101,106,107,112,134,113,160,119,120,127,129,125,135,131,130,128,114,117,136,159,157,70,146,148,142,151,158,137,144,332,145,155,138,140,161,164,166,163,165,168,167,172,179,178,181,192,180,190,188,185,191,182,193,184,208,196,214,205,206,201,210,202,204,199,213,197,195,118,227,224,221,215,219,233,249,245,242,238,234,240,231,259,253,258,256,254,260,251,257,252,250,267,274,176,263,268,266,270,264,265,269,272,271,276,275,283,285,281,279,220,288,292,289,230,297,299,247,301,303,304,309,306,305,307,311,308,310,316,313,320,321,319,330,323,200,8,116,132]
rot_red_cells = [1,2,0,24,23,6,9,8,7,22,12,11,10,14,17,32,26,27,29,28,34,37,39,51,42,52,45,41,47,57,395,43,3,40,59,74,66,85,63,80,84,13,76,67,81,79,61,77,73,82,64,70,87,62,83,71,69,58,89,105,96,95,99,98,94,104,90,397,109,116,114,112,111,117,118,121,120,122,123,124,128,126,36,145,138,130,136,133,134,135,140,144,139,147,129,173,149,68,153,152,171,158,174,154,164,168,156,160,155,169,172,165,178,182,180,103,177,194,193,200,186,196,191,197,199,190,179,198,202,214,203,207,206,204,113,208,215,219,220,221,226,240,230,224,237,238,233,231,141,241,146,236,235,229,228,225,244,246,243,264,258,256,249,253,254,266,259,263,247,270,275,272,273,278,280,281,267,284,289,288,299,297,293,291,295,303,201,287,305,306,314,310,313,315,317,318,320,321,319,323,325,322,327,326,227,345,332,328,329,331,336,252,337,349,339,338,348,340,347,346,350,290,352,357,355,363,356,282,351,371,359,364,375,361,362,366,378,368,370,369,373,365,367,381,374,380,382,386,389,391,383,392,390,151,5,170,279,248,344,4,38,250,101,161]
pre_data_dict['Thy1jRGECO23'] = {
    'mouse': 'Thy1jRGECO23',
    'n_neigh': 120,
    'dim': 3,
    'vel_th': 5,
    'green_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO23/Inscopix_data/Thy1jRGECO23_lt_green_raw.csv'),
    'green_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO23/Inscopix_data/Thy1jRGECO23_rot_green_raw.csv'),
    'red_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO23/Inscopix_data/Thy1jRGECO23_lt_red_raw.csv'),
    'red_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO23/Inscopix_data/Thy1jRGECO23_rot_red_raw.csv'),
    'pos_name_pre': os.path.join(base_dir, 'data/Thy1jRGECO23/Inscopix_data/Thy1jRGECO23_lt_position.mat'),
    'pos_name_rot': os.path.join(base_dir, 'data/Thy1jRGECO23/Inscopix_data/Thy1jRGECO23_rot_position.mat'),
    'pre_green_cells': copy.deepcopy(pre_green_cells),
    'pre_red_cells': copy.deepcopy(pre_red_cells),
    'rot_green_cells': copy.deepcopy(rot_green_cells),
    'rot_red_cells': copy.deepcopy(rot_red_cells)
}

with open(os.path.join(base_dir, 'processed_data', f"dual_color_pre_data_dict.pkl"), "wb") as file:
    pickle.dump(pre_data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################      PROCESS ALL DATA      #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################

for mouse in list(pre_data_dict.keys()):
    print(f"Working on mouse {mouse}: ")
    mouse_dict = {
        'params': copy.deepcopy(pre_data_dict[mouse])
    }
    save_dir = os.path.join(base_dir, 'processed_data', mouse)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    ######################
    #       PARAMS       #
    ######################
    n_neigh = mouse_dict['params']['n_neigh']
    dim = mouse_dict['params']['dim']
    vel_th = mouse_dict['params']['vel_th']

    green_name_pre = mouse_dict['params']['green_name_pre']
    green_name_rot = mouse_dict['params']['green_name_rot']
    red_name_pre = mouse_dict['params']['red_name_pre']
    red_name_rot = mouse_dict['params']['red_name_rot']
    pos_name_pre = mouse_dict['params']['pos_name_pre']
    pos_name_rot = mouse_dict['params']['pos_name_rot']

    ######################
    #     LOAD SIGNAL    #
    ######################
    signal_green_pre = pd.read_csv(green_name_pre).to_numpy()[1:,1:].astype(np.float64)
    signal_green_rot = pd.read_csv(green_name_rot).to_numpy()[1:,1:].astype(np.float64)
    signal_red_pre = pd.read_csv(red_name_pre).to_numpy()[1:,1:].astype(np.float64)
    signal_red_rot = pd.read_csv(red_name_rot).to_numpy()[1:,1:].astype(np.float64)

    ######################
    #      LOAD POS      #
    ######################
    pos_pre = scipy.io.loadmat(pos_name_pre)['Position']
    pos_pre = pos_pre[::2,:]/10
    pos_rot = scipy.io.loadmat(pos_name_rot)['Position']
    pos_rot = pos_rot[::2,:]/10

    ######################
    #     DELETE NANs    #
    ######################
    nan_idx = np.where(np.sum(np.isnan(signal_green_pre),axis=1)>0)[0]
    nan_idx = np.concatenate((nan_idx,np.where(np.sum(np.isnan(signal_red_pre),axis=1)>0)[0]),axis=0)
    nan_idx = np.concatenate((nan_idx,np.where(np.sum(np.isnan(pos_pre),axis=1)>0)[0]),axis=0)
    signal_green_pre = np.delete(signal_green_pre,nan_idx, axis=0)
    signal_red_pre = np.delete(signal_red_pre,nan_idx, axis=0)
    pos_pre = np.delete(pos_pre,nan_idx, axis=0)

    nan_idx = np.where(np.sum(np.isnan(signal_green_rot),axis=1)>0)[0]
    nan_idx = np.concatenate((nan_idx,np.where(np.sum(np.isnan(signal_red_rot),axis=1)>0)[0]),axis=0)
    nan_idx = np.concatenate((nan_idx,np.where(np.sum(np.isnan(pos_rot),axis=1)>0)[0]),axis=0)
    signal_green_rot = np.delete(signal_green_rot,nan_idx, axis=0)
    signal_red_rot = np.delete(signal_red_rot,nan_idx, axis=0)
    pos_rot = np.delete(pos_rot,nan_idx, axis=0)

    ######################
    #    MATCH LENGTH    #
    ######################
    if signal_green_pre.shape[0]>signal_red_pre.shape[0]:
        signal_green_pre = signal_green_pre[:signal_red_pre.shape[0],:]
    elif signal_red_pre.shape[0]>signal_green_pre.shape[0]:
        signal_red_pre = signal_red_pre[:signal_green_pre.shape[0],:]

    if pos_pre.shape[0]>signal_green_pre.shape[0]:
        pos_pre = pos_pre[:signal_green_pre.shape[0],:]
    else:
        signal_green_pre = signal_green_pre[:pos_pre.shape[0],:]
        signal_red_pre = signal_red_pre[:pos_pre.shape[0],:]

    if signal_green_rot.shape[0]>signal_red_rot.shape[0]:
        signal_green_rot = signal_green_rot[:signal_red_rot.shape[0],:]
    elif signal_red_rot.shape[0]>signal_green_rot.shape[0]:
        signal_red_rot = signal_red_rot[:signal_green_rot.shape[0],:]

    if pos_rot.shape[0]>signal_green_rot.shape[0]:
        pos_rot = pos_rot[:signal_green_rot.shape[0],:]
    else:
        signal_green_rot = signal_green_rot[:pos_rot.shape[0],:]
        signal_red_rot = signal_red_rot[:pos_rot.shape[0],:]


    ######################
    #     COMPUTE DIR    #
    ######################
    vel_pre = np.diff(pos_pre[:,0]).reshape(-1,1)*10
    vel_pre = np.concatenate((vel_pre[0].reshape(-1,1), vel_pre), axis=0)
    dir_pre = np.zeros((vel_pre.shape))
    dir_pre[vel_pre>0] = 1
    dir_pre[vel_pre<0] = -1

    vel_rot = np.diff(pos_rot[:,0]).reshape(-1,1)*10
    vel_rot = np.concatenate((vel_rot[0].reshape(-1,1), vel_rot), axis=0)
    dir_rot = np.zeros((vel_rot.shape))
    dir_rot[vel_rot>0] = 1
    dir_rot[vel_rot<0] = -1

    ######################
    #     COMPUTE VEL    #
    ######################
    vel_pre = np.abs(np.diff(pos_pre[:,0]).reshape(-1,1)*10)
    vel_pre = np.concatenate((vel_pre[0].reshape(-1,1), vel_pre), axis=0)
    vel_pre = gaussian_filter1d(vel_pre, sigma = 5, axis = 0)


    vel_rot = np.abs(np.diff(pos_rot[:,0]).reshape(-1,1)*10)
    vel_rot = np.concatenate((vel_rot[0].reshape(-1,1), vel_rot), axis=0)
    vel_rot = gaussian_filter1d(vel_rot, sigma = 5, axis = 0)

    mouse_dict['original_signals'] = {
        'signal_green_pre': copy.deepcopy(signal_green_pre),
        'signal_green_rot': copy.deepcopy(signal_green_rot),
        'signal_red_pre': copy.deepcopy(signal_red_pre),
        'signal_red_rot': copy.deepcopy(signal_red_rot),
        'pos_pre': copy.deepcopy(pos_pre),
        'pos_rot': copy.deepcopy(pos_rot),
        'dir_pre': copy.deepcopy(dir_pre),
        'dir_rot': copy.deepcopy(dir_rot),
        'vel_pre': copy.deepcopy(vel_pre),
        'vel_rot': copy.deepcopy(vel_rot)
    }
    ######################
    #  DELETE LOW SPEED  #
    ######################
    low_speed_idx_pre = np.where(vel_pre<vel_th)[0]
    signal_green_pre = np.delete(signal_green_pre,low_speed_idx_pre, axis=0)
    signal_red_pre = np.delete(signal_red_pre,low_speed_idx_pre, axis=0)
    pos_pre = np.delete(pos_pre,low_speed_idx_pre, axis=0)
    vel_pre = np.delete(vel_pre,low_speed_idx_pre, axis=0)
    dir_pre = np.delete(dir_pre,low_speed_idx_pre, axis=0)


    low_speed_idx_rot = np.where(vel_rot<vel_th)[0]
    signal_green_rot = np.delete(signal_green_rot,low_speed_idx_rot, axis=0)
    signal_red_rot = np.delete(signal_red_rot,low_speed_idx_rot, axis=0)
    pos_rot = np.delete(pos_rot,low_speed_idx_rot, axis=0)
    vel_rot = np.delete(vel_rot,low_speed_idx_rot, axis=0)
    dir_rot = np.delete(dir_rot,low_speed_idx_rot, axis=0)

    ######################
    #    CREATE TIME     #
    ######################
    time_pre = np.arange(pos_pre.shape[0])
    time_rot = np.arange(pos_rot.shape[0])

    mouse_dict['speed_filtered_signals'] = {
        'low_speed_idx_pre': copy.deepcopy(low_speed_idx_pre),
        'low_speed_idx_rot': copy.deepcopy(low_speed_idx_rot),
        'signal_green_pre': copy.deepcopy(signal_green_pre),
        'signal_green_rot': copy.deepcopy(signal_green_rot),
        'signal_red_pre': copy.deepcopy(signal_red_pre),
        'signal_red_rot': copy.deepcopy(signal_red_rot),
        'pos_pre': copy.deepcopy(pos_pre),
        'pos_rot': copy.deepcopy(pos_rot),
        'dir_pre': copy.deepcopy(dir_pre),
        'dir_rot': copy.deepcopy(dir_rot),
        'vel_pre': copy.deepcopy(vel_pre),
        'vel_rot': copy.deepcopy(vel_rot),
        'time_pre': copy.deepcopy(time_pre),
        'time_rot': copy.deepcopy(time_rot)
    }

    ######################
    #    CLEAN TRACES    #
    ######################
    signal_green_pre = clean_traces(signal_green_pre)
    signal_red_pre = clean_traces(signal_red_pre)
    signal_green_rot = clean_traces(signal_green_rot)
    signal_red_rot = clean_traces(signal_red_rot)

    mouse_dict['clean_traces_all'] = {
        'signal_green_pre': copy.deepcopy(signal_green_pre),
        'signal_green_rot': copy.deepcopy(signal_green_rot),
        'signal_red_pre': copy.deepcopy(signal_red_pre),
        'signal_red_rot': copy.deepcopy(signal_red_rot),
    }
    ############################
    #     REGISTER SIGNALS     #
    ############################

    pre_green_cells = mouse_dict['params']['pre_green_cells']
    pre_red_cells = mouse_dict['params']['pre_red_cells']
    rot_green_cells = mouse_dict['params']['rot_green_cells']
    rot_red_cells = mouse_dict['params']['rot_red_cells']
    signal_green_pre = signal_green_pre[:,pre_green_cells]
    signal_red_pre = signal_red_pre[:, pre_red_cells]
    signal_green_rot = signal_green_rot[:,rot_green_cells]
    signal_red_rot = signal_red_rot[:, rot_red_cells]

    mouse_dict['registered_clean_traces'] = {
        'pre_green_cells': copy.deepcopy(pre_green_cells),
        'pre_red_cells': copy.deepcopy(pre_red_cells),
        'rot_green_cells': copy.deepcopy(rot_green_cells),
        'rot_red_cells': copy.deepcopy(rot_red_cells),
        'signal_green_pre': copy.deepcopy(signal_green_pre),
        'signal_green_rot': copy.deepcopy(signal_green_rot),
        'signal_red_pre': copy.deepcopy(signal_red_pre),
        'signal_red_rot': copy.deepcopy(signal_red_rot)
    }

    #############################
    # REGISTERED CELLS TOGETHER #
    #############################
    #%%all data
    index = np.vstack((np.zeros((signal_green_pre.shape[0],1)),np.ones((signal_green_rot.shape[0],1))))
    concat_signal_green = np.vstack((signal_green_pre, signal_green_rot))
    model = umap.UMAP(n_neighbors =n_neigh, n_components =dim, min_dist=0.1)
    model.fit(concat_signal_green)
    emb_concat_green = model.transform(concat_signal_green)
    emb_green_pre = emb_concat_green[index[:,0]==0,:]
    emb_green_rot = emb_concat_green[index[:,0]==1,:]

    #%%all data
    index = np.vstack((np.zeros((signal_red_pre.shape[0],1)),np.ones((signal_red_rot.shape[0],1))))
    concat_signal_red = np.vstack((signal_red_pre, signal_red_rot))
    model = umap.UMAP(n_neighbors=n_neigh, n_components =dim, min_dist=0.1)
    model.fit(concat_signal_red)
    emb_concat_red = model.transform(concat_signal_red)
    emb_red_pre = emb_concat_red[index[:,0]==0,:]
    emb_red_rot = emb_concat_red[index[:,0]==1,:]


    D = pairwise_distances(emb_green_pre)
    noise_idx_green_pre = filter_noisy_outliers(emb_green_pre,D=D)
    csignal_green_pre = signal_green_pre[~noise_idx_green_pre,:]
    cemb_green_pre = emb_green_pre[~noise_idx_green_pre,:]
    cpos_green_pre = pos_pre[~noise_idx_green_pre,:]
    cdir_green_pre = dir_pre[~noise_idx_green_pre]

    D = pairwise_distances(emb_red_pre)
    noise_idx_red_pre = filter_noisy_outliers(emb_red_pre,D=D)
    csignal_red_pre = signal_red_pre[~noise_idx_red_pre,:]
    cemb_red_pre = emb_red_pre[~noise_idx_red_pre,:]
    cpos_red_pre = pos_pre[~noise_idx_red_pre,:]
    cdir_red_pre = dir_pre[~noise_idx_red_pre]

    D = pairwise_distances(emb_green_rot)
    noise_idx_green_rot = filter_noisy_outliers(emb_green_rot,D=D)
    csignal_green_rot = signal_green_rot[~noise_idx_green_rot,:]
    cemb_green_rot = emb_green_rot[~noise_idx_green_rot,:]
    cpos_green_rot = pos_rot[~noise_idx_green_rot,:]
    cdir_green_rot = dir_rot[~noise_idx_green_rot]

    D = pairwise_distances(emb_red_rot)
    noise_idx_red_rot = filter_noisy_outliers(emb_red_rot,D=D)
    csignal_red_rot = signal_red_rot[~noise_idx_red_rot,:]
    cemb_red_rot = emb_red_rot[~noise_idx_red_rot,:]
    cpos_red_rot = pos_rot[~noise_idx_red_rot,:]
    cdir_red_rot = dir_rot[~noise_idx_red_rot]

    mouse_dict['registered_clean_emb'] = {

        'deep_signal_pre': signal_green_pre,
        'deep_signal_rot': signal_green_rot,
        'all_signal_pre': signal_red_pre,
        'all_signal_rot': signal_red_rot,

        'deep_umap_pre': emb_green_pre,
        'deep_umap_rot': emb_green_rot,
        'all_umap_rot': emb_red_rot,
        'all_umap_rot': emb_red_rot,

        'deep_noise_idx_pre': noise_idx_green_pre,
        'all_noise_idx_pre': noise_idx_red_pre,
        'deep_noise_idx_rot': noise_idx_green_rot,
        'all_noise_idx_rot': noise_idx_red_rot,

        'cdeep_umap_pre': cemb_green_pre,
        'cdeep_umap_rot': cemb_green_rot,
        'call_umap_pre': cemb_red_pre,
        'call_umap_rot': cemb_red_rot,

        'cdeep_pos_pre': cpos_green_pre,
        'cdeep_pos_rot': cpos_green_rot,
        'call_pos_pre': cpos_red_pre,
        'call_pos_rot': cpos_red_rot,

        'cdeep_dir_pre': cdir_green_pre,
        'cdeep_dir_rot': cdir_green_rot,
        'call_dir_pre': cdir_red_pre,
        'call_dir_rot': cdir_red_rot,

    }

    dir_color_green_pre = np.zeros((cdir_green_pre.shape[0],3))
    for point in range(cdir_green_pre.shape[0]):
        if cdir_green_pre[point]==0:
            dir_color_green_pre[point] = [14/255,14/255,143/255]
        elif cdir_green_pre[point]==-1:
            dir_color_green_pre[point] = [12/255,136/255,249/255]
        else:
            dir_color_green_pre[point] = [17/255,219/255,224/255]

    dir_color_red_pre = np.zeros((cdir_red_pre.shape[0],3))
    for point in range(cdir_red_pre.shape[0]):
        if cdir_red_pre[point]==0:
            dir_color_red_pre[point] = [14/255,14/255,143/255]
        elif cdir_red_pre[point]==-1:
            dir_color_red_pre[point] = [12/255,136/255,249/255]
        else:
            dir_color_red_pre[point] = [17/255,219/255,224/255]

    dir_color_green_rot = np.zeros((cdir_green_rot.shape[0],3))
    for point in range(cdir_green_rot.shape[0]):
        if cdir_green_rot[point]==0:
            dir_color_green_rot[point] = [14/255,14/255,143/255]
        elif cdir_green_rot[point]==-1:
            dir_color_green_rot[point] = [12/255,136/255,249/255]
        else:
            dir_color_green_rot[point] = [17/255,219/255,224/255]

    dir_color_red_rot = np.zeros((cdir_red_rot.shape[0],3))
    for point in range(cdir_red_rot.shape[0]):
        if cdir_red_rot[point]==0:
            dir_color_red_rot[point] = [14/255,14/255,143/255]
        elif cdir_red_rot[point]==-1:
            dir_color_red_rot[point] = [12/255,136/255,249/255]
        else:
            dir_color_red_rot[point] = [17/255,219/255,224/255]


    plt.figure()
    ax = plt.subplot(2,3,1, projection = '3d')
    ax.scatter(*cemb_green_pre[:,:3].T, color ='b', s=10)
    ax.scatter(*cemb_green_rot[:,:3].T, color = 'r', s=10)
    ax.set_title('Deep')
    ax = plt.subplot(2,3,2, projection = '3d')
    ax.scatter(*cemb_green_pre[:,:3].T, c = cpos_green_pre[:,0], s=10, cmap = 'magma')
    ax.scatter(*cemb_green_rot[:,:3].T, c = cpos_green_rot[:,0], s=10, cmap = 'magma')
    ax = plt.subplot(2,3,3, projection = '3d')
    ax.scatter(*cemb_green_pre[:,:3].T, color=dir_color_green_pre, s=10)
    ax.scatter(*cemb_green_rot[:,:3].T, color=dir_color_green_rot, s=10)

    ax = plt.subplot(2,3,4, projection = '3d')
    ax.scatter(*cemb_red_pre[:,:3].T, color ='b', s=10)
    ax.scatter(*cemb_red_rot[:,:3].T, color = 'r', s=10)
    ax.set_title('All')
    ax = plt.subplot(2,3,5, projection = '3d')
    ax.scatter(*cemb_red_pre[:,:3].T, c = cpos_red_pre[:,0], s=10, cmap = 'magma')
    ax.scatter(*cemb_red_rot[:,:3].T, c = cpos_red_rot[:,0], s=10, cmap = 'magma')
    ax = plt.subplot(2,3,6, projection = '3d')
    ax.scatter(*cemb_red_pre[:,:3].T, color=dir_color_red_pre, s=10)
    ax.scatter(*cemb_red_rot[:,:3].T, color=dir_color_red_rot, s=10)
    plt.suptitle(f"{mouse}")
    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb.png'), dpi = 400,bbox_inches="tight")

    with open(os.path.join(save_dir, f"{mouse}_data_dict.pkl"), "wb") as file:
        pickle.dump(mouse_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################        ADD CALB DATA       #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################


for mouse in ['Thy1jRGECO22', 'Thy1jRGECO23']:
    save_dir = os.path.join(base_dir, 'processed_data', mouse)
    mouse_dict = load_pickle(save_dir, mouse+'_data_dict.pkl')

    og_red_signal = mouse_dict['original_signals']['signal_red_pre']
    signal_length = og_red_signal.shape[0]
    color_dir = os.path.join(base_dir, f'data/{mouse}/Inscopix_data/color_registration/')

    matched_signal = pd.read_csv(os.path.join(color_dir,mouse+'_matched_raw.csv')).to_numpy()[1:,1:].astype(np.float64)
    unmatched_signal = pd.read_csv(os.path.join(color_dir,mouse+'_unmatched_raw.csv')).to_numpy()[1:,1:].astype(np.float64)
    uncertain_signal = pd.read_csv(os.path.join(color_dir,mouse+'_uncertain_raw.csv')).to_numpy()[1:,1:].astype(np.float64)

    matched_indexes = np.zeros((matched_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_red_signal.shape[1]).astype(int)
    for cell_matched in range(matched_signal.shape[1]):
        for cell_red in cells_to_check:
            corr_coeff = np.corrcoef(matched_signal[:signal_length,cell_matched], og_red_signal[:,cell_red])[0,1]
            if corr_coeff > 0.999:
                matched_indexes[cell_matched] = cell_red;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_red)[0])
                break;
    matched_indexes = matched_indexes.astype(int)

    unmatched_indexes = np.zeros((unmatched_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_red_signal.shape[1]).astype(int)
    for cell_unmatched in range(unmatched_signal.shape[1]):
        for cell_red in cells_to_check:
            corr_coeff = np.corrcoef(unmatched_signal[:signal_length,cell_unmatched], og_red_signal[:,cell_red])[0,1]
            if corr_coeff > 0.999:
                unmatched_indexes[cell_unmatched] = cell_red;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_red)[0])
                break;
    unmatched_indexes = unmatched_indexes.astype(int)

    uncertain_indexes = np.zeros((uncertain_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_red_signal.shape[1]).astype(int)
    for cell_uncertain in range(uncertain_signal.shape[1]):
        for cell_red in cells_to_check:
            corr_coeff = np.corrcoef(uncertain_signal[:signal_length,cell_uncertain], og_red_signal[:,cell_red])[0,1]
            if corr_coeff > 0.999:
                uncertain_indexes[cell_uncertain] = cell_red;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_red)[0])
                break;
    uncertain_indexes = uncertain_indexes.astype(int)

    registered_red_cells = mouse_dict['registered_clean_traces']['pre_red_cells']
    registered_matched_indexes = [];
    for matched_index in matched_indexes:
        if matched_index in registered_red_cells:
            new_index = np.where(registered_red_cells==matched_index)[0][0]
            registered_matched_indexes.append(new_index)

    registered_unmatched_indexes = [];
    for unmatched_index in unmatched_indexes:
        if unmatched_index in registered_red_cells:
            new_index = np.where(registered_red_cells==unmatched_index)[0][0]
            registered_unmatched_indexes.append(new_index)

    registered_uncertain_indexes = [];
    for uncertain_index in uncertain_indexes:
        if uncertain_index in registered_red_cells:
            new_index = np.where(registered_red_cells==uncertain_index)[0][0]
            registered_uncertain_indexes.append(new_index)


    sup_signal_pre = mouse_dict['registered_clean_traces']['signal_red_pre'][:,registered_unmatched_indexes+registered_uncertain_indexes]
    sup_signal_rot = mouse_dict['registered_clean_traces']['signal_red_rot'][:,registered_unmatched_indexes+registered_uncertain_indexes]
    mouse_dict['registered_clean_traces']['sup_signal_pre'] = sup_signal_pre
    mouse_dict['registered_clean_traces']['sup_signal_rot'] = sup_signal_rot


    pos_pre = mouse_dict['speed_filtered_signals']['pos_pre']
    dir_pre = mouse_dict['speed_filtered_signals']['dir_pre']
    pos_rot = mouse_dict['speed_filtered_signals']['pos_rot']
    dir_rot = mouse_dict['speed_filtered_signals']['dir_rot']

    #%%all data
    index = np.vstack((np.zeros((sup_signal_pre.shape[0],1)),np.ones((sup_signal_rot.shape[0],1))))
    sup_signal_concat = np.vstack((sup_signal_pre, sup_signal_rot))
    model = umap.UMAP(n_neighbors=120, n_components =3, min_dist=0.1)
    model.fit(sup_signal_concat)
    emb_concat_sup = model.transform(sup_signal_concat)
    emb_sup_pre = emb_concat_sup[index[:,0]==0,:]
    emb_sup_rot = emb_concat_sup[index[:,0]==1,:]

    D = pairwise_distances(emb_sup_pre)
    noise_idx_sup_pre = filter_noisy_outliers(emb_sup_pre,D=D)
    csup_signal_pre = sup_signal_pre[~noise_idx_sup_pre,:]
    cemb_sup_pre = emb_sup_pre[~noise_idx_sup_pre,:]
    cpos_sup_pre = pos_pre[~noise_idx_sup_pre,:]
    cdir_sup_pre = dir_pre[~noise_idx_sup_pre]

    D = pairwise_distances(emb_sup_rot)
    noise_idx_sup_rot = filter_noisy_outliers(emb_sup_rot,D=D)
    csup_signal_rot = sup_signal_rot[~noise_idx_sup_rot,:]
    cemb_sup_rot = emb_sup_rot[~noise_idx_sup_rot,:]
    cpos_sup_rot = pos_rot[~noise_idx_sup_rot,:]
    cdir_sup_rot = dir_rot[~noise_idx_sup_rot]

    new_dict = {
        'sup_signal_pre': sup_signal_pre,
        'sup_signal_rot': sup_signal_rot,

        'sup_umap_pre': emb_sup_pre,
        'sup_umap_rot': emb_sup_rot,

        'sup_noise_idx_pre': noise_idx_sup_pre,
        'sup_noise_idx_rot': noise_idx_sup_rot,

        'csup_umap_pre': cemb_sup_pre,
        'csup_umap_rot': cemb_sup_rot,

        'csup_pos_pre': cpos_sup_pre,
        'csup_pos_rot': cpos_sup_rot,

        'csup_dir_pre': cdir_sup_pre,
        'csup_dir_rot': cdir_sup_rot,
    }

    mouse_dict['registered_clean_emb'].update(copy.deepcopy(new_dict))



    dir_color_sup_pre = np.zeros((cdir_sup_pre.shape[0],3))
    for point in range(cdir_sup_pre.shape[0]):
        if cdir_sup_pre[point]==0:
            dir_color_sup_pre[point] = [14/255,14/255,143/255]
        elif cdir_sup_pre[point]==-1:
            dir_color_sup_pre[point] = [12/255,136/255,249/255]
        else:
            dir_color_sup_pre[point] = [17/255,219/255,224/255]

    dir_color_sup_rot = np.zeros((cdir_sup_rot.shape[0],3))
    for point in range(cdir_sup_rot.shape[0]):
        if cdir_sup_rot[point]==0:
            dir_color_sup_rot[point] = [14/255,14/255,143/255]
        elif cdir_sup_rot[point]==-1:
            dir_color_sup_rot[point] = [12/255,136/255,249/255]
        else:
            dir_color_sup_rot[point] = [17/255,219/255,224/255]

    plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*cemb_sup_pre[:,:3].T, color ='b', s=10)
    ax.scatter(*cemb_sup_rot[:,:3].T, color = 'r', s=10)
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*cemb_sup_pre[:,:3].T, c = cpos_sup_pre[:,0], s=10, cmap = 'magma')
    ax.scatter(*cemb_sup_rot[:,:3].T, c = cpos_sup_rot[:,0], s=10, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*cemb_sup_pre[:,:3].T, color=dir_color_sup_pre, s=10)
    ax.scatter(*cemb_sup_rot[:,:3].T, color=dir_color_sup_rot, s=10)
    plt.suptitle(f"{mouse} sup")
    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_sup.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_sup.png'), dpi = 400,bbox_inches="tight")

    with open(os.path.join(save_dir, f"{mouse}_data_dict.pkl"), "wb") as file:
        pickle.dump(mouse_dict, file, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################       COMPUTE ROTATION     #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################
def rotate_cloud_around_axis(point_cloud, angle, v):
    cloud_center = point_cloud.mean(axis=0)
    a = np.cos(angle/2)
    b = np.sin(angle/2)*v[0]
    c = np.sin(angle/2)*v[1]
    d = np.sin(angle/2)*v[2]
    R = np.array([
            [a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c)],
            [2*(b*c+a*d),a**2-b**2+c**2-d**2, 2*(c*d - a*b)],
            [2*(b*d - a*c), 2*(c*d + a*b), a**2-b**2-c**2+d**2]
        ])
    return  np.matmul(R, (point_cloud-cloud_center).T).T+cloud_center

def get_centroids(cloud_A, cloud_B, label_A, label_B, dir_A = None, dir_B = None, num_centroids = 20):
    dims = cloud_A.shape[1]
    if label_A.ndim>1:
        label_A = label_A[:,0]
    if label_B.ndim>1:
        label_B = label_B[:,0]
    #compute label max and min to divide into centroids
    total_label = np.hstack((label_A, label_B))
    label_lims = np.array([(np.percentile(total_label,5), np.percentile(total_label,95))]).T[:,0] 
    #find centroid size
    cent_size = (label_lims[1] - label_lims[0]) / (num_centroids)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    cent_edges = np.column_stack((np.linspace(label_lims[0],label_lims[0]+cent_size*(num_centroids),num_centroids),
                                np.linspace(label_lims[0],label_lims[0]+cent_size*(num_centroids),num_centroids)+cent_size))


    if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
        cent_A = np.zeros((num_centroids,dims))
        cent_B = np.zeros((num_centroids,dims))
        cent_label = np.mean(cent_edges,axis=1).reshape(-1,1)

        num_centroids_A = np.zeros((num_centroids,))
        num_centroids_B = np.zeros((num_centroids,))
        for c in range(num_centroids):
            points_A = cloud_A[np.logical_and(label_A >= cent_edges[c,0], label_A<cent_edges[c,1]),:]
            cent_A[c,:] = np.median(points_A, axis=0)
            num_centroids_A[c] = points_A.shape[0]
            
            points_B = cloud_B[np.logical_and(label_B >= cent_edges[c,0], label_B<cent_edges[c,1]),:]
            cent_B[c,:] = np.median(points_B, axis=0)
            num_centroids_B[c] = points_B.shape[0]
    else:
        cloud_A_left = copy.deepcopy(cloud_A[dir_A==-1,:])
        label_A_left = copy.deepcopy(label_A[dir_A==-1])
        cloud_A_right = copy.deepcopy(cloud_A[dir_A==1,:])
        label_A_right = copy.deepcopy(label_A[dir_A==1])
        
        cloud_B_left = copy.deepcopy(cloud_B[dir_B==-1,:])
        label_B_left = copy.deepcopy(label_B[dir_B==-1])
        cloud_B_right = copy.deepcopy(cloud_B[dir_B==1,:])
        label_B_right = copy.deepcopy(label_B[dir_B==1])
        
        cent_A = np.zeros((2*num_centroids,dims))
        cent_B = np.zeros((2*num_centroids,dims))
        num_centroids_A = np.zeros((2*num_centroids,))
        num_centroids_B = np.zeros((2*num_centroids,))
        
        cent_dir = np.zeros((2*num_centroids, ))
        cent_label = np.tile(np.mean(cent_edges,axis=1),(2,1)).T.reshape(-1,1)
        for c in range(num_centroids):
            points_A_left = cloud_A_left[np.logical_and(label_A_left >= cent_edges[c,0], label_A_left<cent_edges[c,1]),:]
            cent_A[2*c,:] = np.median(points_A_left, axis=0)
            num_centroids_A[2*c] = points_A_left.shape[0]
            points_A_right = cloud_A_right[np.logical_and(label_A_right >= cent_edges[c,0], label_A_right<cent_edges[c,1]),:]
            cent_A[2*c+1,:] = np.median(points_A_right, axis=0)
            num_centroids_A[2*c+1] = points_A_right.shape[0]

            points_B_left = cloud_B_left[np.logical_and(label_B_left >= cent_edges[c,0], label_B_left<cent_edges[c,1]),:]
            cent_B[2*c,:] = np.median(points_B_left, axis=0)
            num_centroids_B[2*c] = points_B_left.shape[0]
            points_B_right = cloud_B_right[np.logical_and(label_B_right >= cent_edges[c,0], label_B_right<cent_edges[c,1]),:]
            cent_B[2*c+1,:] = np.median(points_B_right, axis=0)
            num_centroids_B[2*c+1] = points_B_right.shape[0]

            cent_dir[2*c] = -1
            cent_dir[2*c+1] = 1

    del_cent_nan = np.all(np.isnan(cent_A), axis= 1)+ np.all(np.isnan(cent_B), axis= 1)
    del_cent_num = (num_centroids_A<15) + (num_centroids_B<15)
    del_cent = del_cent_nan + del_cent_num
    
    cent_A = np.delete(cent_A, del_cent, 0)
    cent_B = np.delete(cent_B, del_cent, 0)

    cent_label = np.delete(cent_label, del_cent, 0)

    if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
        return cent_A, cent_B, cent_label
    else:
        cent_dir = np.delete(cent_dir, del_cent, 0)
        return cent_A, cent_B, cent_label, cent_dir

def align_vectors(norm_vector_A, cloud_center_A, norm_vector_B, cloud_center_B):

    def find_rotation_align_vectors(a,b):
        v = np.cross(a,b)
        s = np.linalg.norm(v)
        c = np.dot(a,b)

        sscp = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]], [-v[1],v[0],0]])
        sscp2 = np.matmul(sscp,sscp)
        R = np.eye(3)+sscp+(sscp2*(1-c)/s**2)
        return R

    def check_norm_vector_direction(norm_vector, cloud_center, goal_point):
        og_dir = cloud_center+norm_vector
        op_dir = cloud_center+(-1*norm_vector)

        og_distance = np.linalg.norm(og_dir-goal_point)
        op_distance = np.linalg.norm(op_dir-goal_point)
        if og_distance<op_distance:
            return norm_vector
        else:
            return -norm_vector

    norm_vector_A = check_norm_vector_direction(norm_vector_A,cloud_center_A, cloud_center_B)
    norm_vector_B = check_norm_vector_direction(norm_vector_B,cloud_center_B, cloud_center_A)

    align_mat = find_rotation_align_vectors(norm_vector_A,-norm_vector_B)  #minus sign to avoid 180 flip
    align_angle = np.arccos(np.clip(np.dot(norm_vector_A, -norm_vector_B), -1.0, 1.0))*180/np.pi

    return align_angle, align_mat

def apply_rotation_to_cloud(point_cloud, rotation,center_of_rotation):
    return np.dot(point_cloud-center_of_rotation, rotation) + center_of_rotation

def parametrize_plane(point_cloud):
    '''
    point_cloud.shape = [p,d] (points x dimensions)
    based on Ger reply: https://stackoverflow.com/questions/35070178/
    fit-plane-to-a-set-of-points-in-3d-scipy-optimize-minimize-vs-scipy-linalg-lsts
    '''
    #get point cloud center
    cloud_center = point_cloud.mean(axis=0)
    # run SVD
    u, s, vh = np.linalg.svd(point_cloud - cloud_center)
    # unitary normal vector
    norm_vector = vh[2, :]
    return norm_vector, cloud_center

def project_onto_plane(point_cloud, norm_plane, point_plane):
    return point_cloud- np.multiply(np.tile(np.dot(norm_plane, (point_cloud-point_plane).T),(3,1)).T,norm_plane)

def find_rotation(data_A, data_B, v):
    center_A = data_A.mean(axis=0)
    angles = np.linspace(-np.pi,np.pi,200)
    error = list()
    for angle in angles:
        #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        #https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
        a = np.cos(angle/2)
        b = np.sin(angle/2)*v[0]
        c = np.sin(angle/2)*v[1]
        d = np.sin(angle/2)*v[2]
        R = np.array([
                [a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d),a**2-b**2+c**2-d**2, 2*(c*d - a*b)],
                [2*(b*d - a*c), 2*(c*d + a*b), a**2-b**2-c**2+d**2]
            ])

        new_data =np.matmul(R, (data_A-center_A).T).T + center_A
        error.append(np.sum(np.linalg.norm(new_data - data_B, axis=1)))

    return error

def plot_rotation(cloud_A, cloud_B, pos_A, pos_B, dir_A, dir_B, cent_A, cent_B, cent_pos, plane_cent_A, plane_cent_B, aligned_plane_cent_B, rotated_aligned_cent_rot, angles, error, rotation_angle):
    def process_axis(ax):
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_aspect('equal', adjustable='box')


    fig = plt.figure()
    ax = plt.subplot(3,3,1, projection = '3d')
    ax.scatter(*cloud_A[:,:3].T, color ='b', s= 10)
    ax.scatter(*cloud_B[:,:3].T, color = 'r', s= 10)
    process_axis(ax)

    ax = plt.subplot(3,3,4, projection = '3d')
    ax.scatter(*cloud_A[:,:3].T, c = dir_A, s= 10, cmap = 'tab10')
    ax.scatter(*cloud_B[:,:3].T, c = dir_B, s= 10, cmap = 'tab10')
    process_axis(ax)

    ax = plt.subplot(3,3,7, projection = '3d')
    ax.scatter(*cloud_A[:,:3].T, c = pos_A[:,0], s= 10, cmap = 'viridis')
    ax.scatter(*cloud_B[:,:3].T, c = pos_B[:,0], s= 10, cmap = 'magma')
    process_axis(ax)

    ax = plt.subplot(3,3,2, projection = '3d')
    ax.scatter(*cent_A.T, color ='b', s= 30)
    ax.scatter(*plane_cent_A.T, color ='cyan', s= 30)

    ax.scatter(*cent_B.T, color = 'r', s= 30)
    ax.scatter(*plane_cent_B.T, color = 'orange', s= 30)
    ax.scatter(*aligned_plane_cent_B.T, color = 'khaki', s= 30)
    process_axis(ax)



    ax = plt.subplot(3,3,5, projection = '3d')
    ax.scatter(*plane_cent_A.T, c = cent_pos[:,0], s= 30, cmap = 'viridis')
    ax.scatter(*aligned_plane_cent_B[:,:3].T, c = cent_pos[:,0], s= 30, cmap = 'magma')
    for idx in range(cent_pos.shape[0]):
        ax.plot([plane_cent_A[idx,0], aligned_plane_cent_B[idx,0]], 
                [plane_cent_A[idx,1], aligned_plane_cent_B[idx,1]], 
                [plane_cent_A[idx,2], aligned_plane_cent_B[idx,2]], 
                color='gray', linewidth=0.5)
    process_axis(ax)


    ax = plt.subplot(3,3,8, projection = '3d')
    ax.scatter(*cent_A.T, c = cent_pos[:,0], s= 30, cmap = 'viridis')
    ax.scatter(*rotated_aligned_cent_rot.T, c = cent_pos[:,0], s= 30, cmap = 'magma')
    for idx in range(cent_pos.shape[0]):
        ax.plot([plane_cent_A[idx,0], rotated_aligned_cent_rot[idx,0]], 
                [plane_cent_A[idx,1], rotated_aligned_cent_rot[idx,1]], 
                [plane_cent_A[idx,2], rotated_aligned_cent_rot[idx,2]], 
                color='gray', linewidth=0.5)
    process_axis(ax)

    ax = plt.subplot(1,3,3)
    ax.plot(error, angles*180/np.pi)
    ax.plot([np.min(error),np.max(error)], [rotation_angle]*2, '--r')
    ax.set_yticks([-180, -90, 0 , 90, 180])
    ax.set_xlabel('Error')
    ax.set_ylabel('Angle')
    plt.tight_layout()

    return fig

for mouse in ['Thy1jRGECO22', 'Thy1jRGECO23']:
    mouse_dict = load_pickle(os.path.join(base_dir, 'processed_data', mouse), mouse+'_data_dict.pkl')
    save_dir = os.path.join(base_dir, 'rotation', mouse)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    rotation_dict = {}
    #__________________________________________________________________________
    #|                                                                        |#
    #|                                  DEEP                                  |#
    #|________________________________________________________________________|#

    deep_emb_pre = mouse_dict['registered_clean_emb']['cdeep_umap_pre']
    deep_pos_pre = mouse_dict['registered_clean_emb']['cdeep_pos_pre']
    deep_dir_pre = mouse_dict['registered_clean_emb']['cdeep_dir_pre'][:,0]



    deep_emb_rot = mouse_dict['registered_clean_emb']['cdeep_umap_rot']
    deep_pos_rot = mouse_dict['registered_clean_emb']['cdeep_pos_rot']
    deep_dir_rot = mouse_dict['registered_clean_emb']['cdeep_dir_rot'][:,0]


    #compute centroids
    deep_cent_pre, deep_cent_rot, deep_cent_pos, deep_cent_dir = get_centroids(deep_emb_pre, deep_emb_rot, deep_pos_pre[:,0], deep_pos_rot[:,0], 
                                                    deep_dir_pre, deep_dir_rot, num_centroids=40) 

    #project into planes
    deep_norm_vec_pre, deep_cloud_center_pre = parametrize_plane(deep_emb_pre)
    plane_deep_emb_pre = project_onto_plane(deep_emb_pre, deep_norm_vec_pre, deep_cloud_center_pre)

    deep_norm_vec_rot, deep_cloud_center_rot = parametrize_plane(deep_emb_rot)
    plane_deep_emb_rot = project_onto_plane(deep_emb_rot, deep_norm_vec_rot, deep_cloud_center_rot)

    plane_deep_cent_pre, plane_deep_cent_rot, plane_deep_cent_pos, plane_deep_cent_dir = get_centroids(plane_deep_emb_pre, plane_deep_emb_rot, 
                                                                                        deep_pos_pre[:,0], deep_pos_rot[:,0], 
                                                                                        deep_dir_pre, deep_dir_rot, num_centroids=40) 
    #align them
    deep_align_angle, deep_align_mat = align_vectors(deep_norm_vec_pre, deep_cloud_center_pre, deep_norm_vec_rot, deep_cloud_center_rot)

    aligned_deep_emb_rot =  apply_rotation_to_cloud(deep_emb_rot, deep_align_mat, deep_cloud_center_rot)
    aligned_plane_deep_emb_rot =  apply_rotation_to_cloud(plane_deep_emb_rot, deep_align_mat, deep_cloud_center_rot)

    aligned_deep_cent_rot =  apply_rotation_to_cloud(deep_cent_rot, deep_align_mat, deep_cloud_center_rot)
    aligned_plane_deep_cent_rot =  apply_rotation_to_cloud(plane_deep_cent_rot, deep_align_mat, deep_cloud_center_rot)

    #compute angle of rotation
    deep_angles = np.linspace(-np.pi,np.pi,200)
    deep_error = find_rotation(plane_deep_cent_pre, plane_deep_cent_rot, -deep_norm_vec_pre)
    norm_deep_error = (np.array(deep_error)-np.min(deep_error))/(np.max(deep_error)-np.min(deep_error))
    signed_deep_rotation_angle = deep_angles[np.argmin(norm_deep_error)]*180/np.pi
    deep_rotation_angle = np.abs(signed_deep_rotation_angle)
    print(f"\tDeep: {signed_deep_rotation_angle:2f} degrees")

    rotated_aligned_deep_cent_rot = rotate_cloud_around_axis(aligned_deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_plane_deep_cent_rot = rotate_cloud_around_axis(aligned_plane_deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_deep_emb_rot = rotate_cloud_around_axis(aligned_deep_emb_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_plane_deep_emb_rot = rotate_cloud_around_axis(aligned_plane_deep_emb_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)

    rotated_deep_cent_rot = rotate_cloud_around_axis(deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)

    fig = plot_rotation(deep_emb_pre, deep_emb_rot, deep_pos_pre, deep_pos_rot, deep_dir_pre, deep_dir_rot, 
                deep_cent_pre, deep_cent_rot, deep_cent_pos, plane_deep_cent_pre, plane_deep_cent_rot, 
                aligned_plane_deep_cent_rot, rotated_aligned_plane_deep_cent_rot, deep_angles, deep_error, signed_deep_rotation_angle)
    plt.suptitle(f"{mouse} deep")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_rotation_plot.png'), dpi = 400,bbox_inches="tight")


    rotation_dict['deep'] = {
        #initial data
        'deep_emb_pre': deep_emb_pre,
        'deep_pos_pre': deep_pos_pre,
        'deep_dir_pre': deep_dir_pre,

        'deep_emb_rot': deep_emb_rot,
        'deep_pos_rot': deep_pos_rot,
        'deep_dir_rot': deep_dir_rot,
        #centroids
        'deep_cent_pre': deep_cent_pre,
        'deep_cent_rot': deep_cent_rot,
        'deep_cent_pos': deep_cent_pos,
        'deep_cent_dir': deep_cent_dir,

        #project into plane
        'deep_norm_vec_pre': deep_norm_vec_pre,
        'deep_cloud_center_pre': deep_cloud_center_pre,
        'plane_deep_emb_pre': plane_deep_emb_pre,

        'deep_norm_vec_rot': deep_norm_vec_rot,
        'deep_cloud_center_rot': deep_cloud_center_rot,
        'plane_deep_emb_rot': plane_deep_emb_rot,

        #plane centroids
        'plane_deep_cent_pre': plane_deep_cent_pre,
        'plane_deep_cent_rot': plane_deep_cent_rot,
        'plane_deep_cent_pos': plane_deep_cent_pos,
        'plane_deep_cent_dir': plane_deep_cent_dir,

        #align planes
        'deep_align_angle': deep_align_angle,
        'deep_align_mat': deep_align_mat,

        'aligned_deep_emb_rot': aligned_deep_emb_rot,
        'aligned_plane_deep_emb_rot': aligned_plane_deep_emb_rot,
        'aligned_deep_cent_rot': aligned_deep_cent_rot,
        'aligned_plane_deep_cent_rot': aligned_plane_deep_cent_rot,

        #compute angle of rotation
        'deep_angles': deep_angles,
        'deep_error': deep_error,
        'norm_deep_error': norm_deep_error,
        'signed_deep_rotation_angle': signed_deep_rotation_angle,
        'deep_rotation_angle': deep_rotation_angle,

        #rotate post session
        'rotated_deep_cent_rot': rotated_deep_cent_rot,
        'rotated_aligned_deep_cent_rot': rotated_aligned_deep_cent_rot,
        'rotated_aligned_plane_deep_cent_rot': rotated_aligned_plane_deep_cent_rot,
        'rotated_aligned_deep_emb_rot': rotated_aligned_deep_emb_rot,
        'rotated_aligned_plane_deep_emb_rot': rotated_aligned_plane_deep_emb_rot,
    }


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                  SUP                                  |#
    #|________________________________________________________________________|#

    sup_emb_pre = mouse_dict['registered_clean_emb']['csup_umap_pre']
    sup_pos_pre = mouse_dict['registered_clean_emb']['csup_pos_pre']
    sup_dir_pre = mouse_dict['registered_clean_emb']['csup_dir_pre'][:,0]



    sup_emb_rot = mouse_dict['registered_clean_emb']['csup_umap_rot']
    sup_pos_rot = mouse_dict['registered_clean_emb']['csup_pos_rot']
    sup_dir_rot = mouse_dict['registered_clean_emb']['csup_dir_rot'][:,0]

    #compute centroids
    sup_cent_pre, sup_cent_rot, sup_cent_pos, sup_cent_dir = get_centroids(sup_emb_pre, sup_emb_rot, sup_pos_pre[:,0], sup_pos_rot[:,0], 
                                                    sup_dir_pre, sup_dir_rot, num_centroids=30) 

    #project into planes
    sup_norm_vec_pre, sup_cloud_center_pre = parametrize_plane(sup_emb_pre)
    plane_sup_emb_pre = project_onto_plane(sup_emb_pre, sup_norm_vec_pre, sup_cloud_center_pre)

    sup_norm_vec_rot, sup_cloud_center_rot = parametrize_plane(sup_emb_rot)
    plane_sup_emb_rot = project_onto_plane(sup_emb_rot, sup_norm_vec_rot, sup_cloud_center_rot)

    plane_sup_cent_pre, plane_sup_cent_rot, plane_sup_cent_pos, plane_sup_cent_dir = get_centroids(plane_sup_emb_pre, plane_sup_emb_rot, 
                                                                                        sup_pos_pre[:,0], sup_pos_rot[:,0], 
                                                                                        sup_dir_pre, sup_dir_rot, num_centroids=30) 
    #align them
    sup_align_angle, sup_align_mat = align_vectors(sup_norm_vec_pre, sup_cloud_center_pre, sup_norm_vec_rot, sup_cloud_center_rot)

    aligned_sup_emb_rot =  apply_rotation_to_cloud(sup_emb_rot, sup_align_mat, sup_cloud_center_rot)
    aligned_plane_sup_emb_rot =  apply_rotation_to_cloud(plane_sup_emb_rot, sup_align_mat, sup_cloud_center_rot)

    aligned_sup_cent_rot =  apply_rotation_to_cloud(sup_cent_rot, sup_align_mat, sup_cloud_center_rot)
    aligned_plane_sup_cent_rot =  apply_rotation_to_cloud(plane_sup_cent_rot, sup_align_mat, sup_cloud_center_rot)

    #compute angle of rotation
    sup_angles = np.linspace(-np.pi,np.pi,200)
    sup_error = find_rotation(plane_sup_cent_pre, aligned_plane_sup_cent_rot, -sup_norm_vec_pre)
    norm_sup_error = (np.array(sup_error)-np.min(sup_error))/(np.max(sup_error)-np.min(sup_error))
    signed_sup_rotation_angle = sup_angles[np.argmin(norm_sup_error)]*180/np.pi
    sup_rotation_angle = np.abs(signed_sup_rotation_angle)
    print(f"\tsup: {signed_sup_rotation_angle:2f} degrees")

    rotated_aligned_sup_cent_rot = rotate_cloud_around_axis(aligned_sup_cent_rot, (np.pi/180)*signed_sup_rotation_angle,sup_norm_vec_pre)
    rotated_aligned_plane_sup_cent_rot = rotate_cloud_around_axis(aligned_plane_sup_cent_rot, (np.pi/180)*signed_sup_rotation_angle,sup_norm_vec_pre)
    rotated_aligned_sup_emb_rot = rotate_cloud_around_axis(aligned_sup_emb_rot, (np.pi/180)*signed_sup_rotation_angle,sup_norm_vec_pre)
    rotated_aligned_plane_sup_emb_rot = rotate_cloud_around_axis(aligned_plane_sup_emb_rot, (np.pi/180)*signed_sup_rotation_angle,sup_norm_vec_pre)


    fig = plot_rotation(sup_emb_pre, sup_emb_rot, sup_pos_pre, sup_pos_rot, sup_dir_pre, sup_dir_rot, 
                sup_cent_pre, sup_cent_rot, sup_cent_pos, plane_sup_cent_pre, plane_sup_cent_rot, 
                aligned_plane_sup_cent_rot, rotated_aligned_sup_cent_rot, sup_angles, sup_error, signed_sup_rotation_angle)

    plt.suptitle(f"{mouse} sup")

    plt.savefig(os.path.join(save_dir,f'{mouse}_sup_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_sup_rotation_plot.png'), dpi = 400,bbox_inches="tight")


    rotation_dict['sup'] = {
        #initial data
        'sup_emb_pre': sup_emb_pre,
        'sup_pos_pre': sup_pos_pre,
        'sup_dir_pre': sup_dir_pre,

        'sup_emb_rot': sup_emb_rot,
        'sup_pos_rot': sup_pos_rot,
        'sup_dir_rot': sup_dir_rot,
        #centroids
        'sup_cent_pre': sup_cent_pre,
        'sup_cent_rot': sup_cent_rot,
        'sup_cent_pos': sup_cent_pos,
        'sup_cent_dir': sup_cent_dir,

        #project into plane
        'sup_norm_vec_pre': sup_norm_vec_pre,
        'sup_cloud_center_pre': sup_cloud_center_pre,
        'plane_sup_emb_pre': plane_sup_emb_pre,

        'sup_norm_vec_rot': sup_norm_vec_rot,
        'sup_cloud_center_rot': sup_cloud_center_rot,
        'plane_sup_emb_rot': plane_sup_emb_rot,

        #plane centroids
        'plane_sup_cent_pre': plane_sup_cent_pre,
        'plane_sup_cent_rot': plane_sup_cent_rot,
        'plane_sup_cent_pos': plane_sup_cent_pos,
        'plane_sup_cent_dir': plane_sup_cent_dir,

        #align planes
        'sup_align_angle': sup_align_angle,
        'sup_align_mat': sup_align_mat,

        'aligned_sup_emb_rot': aligned_sup_emb_rot,
        'aligned_plane_sup_emb_rot': aligned_plane_sup_emb_rot,
        'aligned_sup_cent_rot': aligned_sup_cent_rot,
        'aligned_plane_sup_cent_rot': aligned_plane_sup_cent_rot,

        #compute angle of rotation
        'sup_angles': sup_angles,
        'sup_error': sup_error,
        'norm_sup_error': norm_sup_error,
        'signed_sup_rotation_angle': signed_sup_rotation_angle,
        'sup_rotation_angle': sup_rotation_angle,


        #rotate post session
        'rotated_aligned_sup_cent_rot': rotated_aligned_sup_cent_rot,
        'rotated_aligned_plane_sup_cent_rot': rotated_aligned_plane_sup_cent_rot,
        'rotated_aligned_sup_emb_rot': rotated_aligned_sup_emb_rot,
        'rotated_aligned_plane_sup_emb_rot': rotated_aligned_plane_sup_emb_rot

    }


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                  all                                  |#
    #|________________________________________________________________________|#


    all_emb_pre = mouse_dict['registered_clean_emb']['call_umap_pre']
    all_pos_pre = mouse_dict['registered_clean_emb']['call_pos_pre']
    all_dir_pre = mouse_dict['registered_clean_emb']['call_dir_pre'][:,0]



    all_emb_rot = mouse_dict['registered_clean_emb']['call_umap_rot']
    all_pos_rot = mouse_dict['registered_clean_emb']['call_pos_rot']
    all_dir_rot = mouse_dict['registered_clean_emb']['call_dir_rot'][:,0]


    #compute centroids
    all_cent_pre, all_cent_rot, all_cent_pos, all_cent_dir = get_centroids(all_emb_pre, all_emb_rot, all_pos_pre[:,0], all_pos_rot[:,0], 
                                                    all_dir_pre, all_dir_rot, num_centroids=40) 

    #project into planes
    all_norm_vec_pre, all_cloud_center_pre = parametrize_plane(all_emb_pre)
    plane_all_emb_pre = project_onto_plane(all_emb_pre, all_norm_vec_pre, all_cloud_center_pre)

    all_norm_vec_rot, all_cloud_center_rot = parametrize_plane(all_emb_rot)
    plane_all_emb_rot = project_onto_plane(all_emb_rot, all_norm_vec_rot, all_cloud_center_rot)

    plane_all_cent_pre, plane_all_cent_rot, plane_all_cent_pos, plane_all_cent_dir = get_centroids(plane_all_emb_pre, plane_all_emb_rot, 
                                                                                        all_pos_pre[:,0], all_pos_rot[:,0], 
                                                                                        all_dir_pre, all_dir_rot, num_centroids=40) 
    #align them
    all_align_angle, all_align_mat = align_vectors(all_norm_vec_pre, all_cloud_center_pre, all_norm_vec_rot, all_cloud_center_rot)

    aligned_all_emb_rot =  apply_rotation_to_cloud(all_emb_rot, all_align_mat, all_cloud_center_rot)
    aligned_plane_all_emb_rot =  apply_rotation_to_cloud(plane_all_emb_rot, all_align_mat, all_cloud_center_rot)

    aligned_all_cent_rot =  apply_rotation_to_cloud(all_cent_rot, all_align_mat, all_cloud_center_rot)
    aligned_plane_all_cent_rot =  apply_rotation_to_cloud(plane_all_cent_rot, all_align_mat, all_cloud_center_rot)

    #compute angle of rotation
    all_angles = np.linspace(-np.pi,np.pi,200)
    all_error = find_rotation(plane_all_cent_pre, aligned_plane_all_cent_rot, -all_norm_vec_pre)
    norm_all_error = (np.array(all_error)-np.min(all_error))/(np.max(all_error)-np.min(all_error))
    signed_all_rotation_angle = all_angles[np.argmin(norm_all_error)]*180/np.pi
    all_rotation_angle = np.abs(signed_all_rotation_angle)
    print(f"\tall: {signed_all_rotation_angle:2f} degrees")

    rotated_aligned_all_cent_rot = rotate_cloud_around_axis(aligned_all_cent_rot, (np.pi/180)*signed_all_rotation_angle,all_norm_vec_pre)
    rotated_aligned_plane_all_cent_rot = rotate_cloud_around_axis(aligned_plane_all_cent_rot, (np.pi/180)*signed_all_rotation_angle,all_norm_vec_pre)
    rotated_aligned_all_emb_rot = rotate_cloud_around_axis(aligned_all_emb_rot, (np.pi/180)*signed_all_rotation_angle,all_norm_vec_pre)
    rotated_aligned_plane_all_emb_rot = rotate_cloud_around_axis(aligned_plane_all_emb_rot, (np.pi/180)*signed_all_rotation_angle,all_norm_vec_pre)


    fig = plot_rotation(all_emb_pre, all_emb_rot, all_pos_pre, all_pos_rot, all_dir_pre, all_dir_rot, 
                all_cent_pre, all_cent_rot, all_cent_pos, plane_all_cent_pre, plane_all_cent_rot, 
                aligned_plane_all_cent_rot, rotated_aligned_all_cent_rot, all_angles, all_error, signed_all_rotation_angle)

    plt.suptitle(f"{mouse} all")

    plt.savefig(os.path.join(save_dir,f'{mouse}_all_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_all_rotation_plot.png'), dpi = 400,bbox_inches="tight")

    rotation_dict['all'] = {
        #initial data
        'all_emb_pre': all_emb_pre,
        'all_pos_pre': all_pos_pre,
        'all_dir_pre': all_dir_pre,

        'all_emb_rot': all_emb_rot,
        'all_pos_rot': all_pos_rot,
        'all_dir_rot': all_dir_rot,
        #centroids
        'all_cent_pre': all_cent_pre,
        'all_cent_rot': all_cent_rot,
        'all_cent_pos': all_cent_pos,
        'all_cent_dir': all_cent_dir,

        #project into plane
        'all_norm_vec_pre': all_norm_vec_pre,
        'all_cloud_center_pre': all_cloud_center_pre,
        'plane_all_emb_pre': plane_all_emb_pre,

        'all_norm_vec_rot': all_norm_vec_rot,
        'all_cloud_center_rot': all_cloud_center_rot,
        'plane_all_emb_rot': plane_all_emb_rot,

        #plane centroids
        'plane_all_cent_pre': plane_all_cent_pre,
        'plane_all_cent_rot': plane_all_cent_rot,
        'plane_all_cent_pos': plane_all_cent_pos,
        'plane_all_cent_dir': plane_all_cent_dir,

        #align planes
        'all_align_angle': all_align_angle,
        'all_align_mat': all_align_mat,

        'aligned_all_emb_rot': aligned_all_emb_rot,
        'aligned_plane_all_emb_rot': aligned_plane_all_emb_rot,
        'aligned_all_cent_rot': aligned_all_cent_rot,
        'aligned_plane_all_cent_rot': aligned_plane_all_cent_rot,

        #compute angle of rotation
        'all_angles': all_angles,
        'all_error': all_error,
        'norm_all_error': norm_all_error,
        'signed_all_rotation_angle': signed_all_rotation_angle,
        'all_rotation_angle': all_rotation_angle,


        #rotate post session
        'rotated_aligned_all_cent_rot': rotated_aligned_all_cent_rot,
        'rotated_aligned_plane_all_cent_rot': rotated_aligned_plane_all_cent_rot,
        'rotated_aligned_all_emb_rot': rotated_aligned_all_emb_rot,
        'rotated_aligned_plane_all_emb_rot': rotated_aligned_plane_all_emb_rot,

    }

    with open(os.path.join(save_dir, mouse+"_rotation_dict.pkl"), "wb") as file:
        pickle.dump(rotation_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

#__________________________________________________________________________
#|                                                                        |#
#|                          PLOT ROTATION ANGLES                          |#
#|________________________________________________________________________|#

load_dir = os.path.join(base_dir, 'rotation')
save_dir = '/home/julio/Documents/DeepSup_project/DualColor/figures/'
rot_angle = list()
mouse = list()
channel = list()

rotation_dict = load_pickle(os.path.join(load_dir, 'Thy1jRGECO22'), 'Thy1jRGECO22_rotation_dict.pkl')
rot_angle.append(rotation_dict['deep']['deep_rotation_angle'])
channel.append('deep')
mouse.append('Thy1jRGECO22')
rot_angle.append(rotation_dict['sup']['sup_rotation_angle'])
channel.append('sup')
mouse.append('Thy1jRGECO22')
rot_angle.append(rotation_dict['all']['all_rotation_angle'])
channel.append('all')
mouse.append('Thy1jRGECO22')


rotation_dict = load_pickle(os.path.join(load_dir, 'Thy1jRGECO23'), 'Thy1jRGECO23_rotation_dict.pkl')
rot_angle.append(rotation_dict['deep']['deep_rotation_angle'])
channel.append('deep')
mouse.append('Thy1jRGECO23')
rot_angle.append(rotation_dict['sup']['sup_rotation_angle'])
channel.append('sup')
mouse.append('Thy1jRGECO23')
rot_angle.append(rotation_dict['all']['all_rotation_angle'])
channel.append('all')
mouse.append('Thy1jRGECO23')


load_dir = '/home/julio/Documents/DeepSup_project/DualColor/ThyCalbRCaMP/rotation/'
rotation_dict = load_pickle(os.path.join(load_dir,'ThyCalbRCaMP2'), 'ThyCalbRCaMP2_rotation_dict.pkl')
rot_angle.append(rotation_dict['deep']['deep_rotation_angle'])
channel.append('deep')
mouse.append('ThyCalbRCaMP2')
rot_angle.append(rotation_dict['sup']['sup_rotation_angle'])
channel.append('sup')
mouse.append('ThyCalbRCaMP2')
rot_angle.append(rotation_dict['all']['all_rotation_angle'])
channel.append('all')
mouse.append('ThyCalbRCaMP2')


pdAngle = pd.DataFrame(data={'mouse': mouse,
                     'channel': channel,
                     'angle': rot_angle})    

fig, ax = plt.subplots(1, 1, figsize=(6,6))

b = sns.boxplot(x='channel', y='angle', data=pdAngle,
            linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='channel', y='angle', data=pdAngle,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylim([-2.5, 180.5])
ax.set_yticks([0,45,90,135,180]);
plt.savefig(os.path.join(save_dir,f'dual_rot_angle.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dual_rot_angle.png'), dpi = 400,bbox_inches="tight")



from scipy import stats
deepAngle = pdAngle.loc[pdAngle['channel']=='deep']['angle']
calbAngle = pdAngle.loc[pdAngle['channel']=='sup']['angle']
allAngle = pdAngle.loc[pdAngle['channel']=='all']['angle']

deepAngle_norm = stats.shapiro(deepAngle)
calbAngle_norm = stats.shapiro(calbAngle)
allAngle_norm = stats.shapiro(allAngle)

if deepAngle_norm.pvalue<=0.05 or calbAngle_norm.pvalue<=0.05:
    print('deepAngle vs calbAngle:',stats.ks_2samp(deepAngle, calbAngle))
else:
    print('deepAngle vs calbAngle:', stats.ttest_rel(deepAngle, calbAngle))

if deepAngle_norm.pvalue<=0.05 or allAngle_norm.pvalue<=0.05:
    print('deepAngle vs allAngle:',stats.ks_2samp(deepAngle, allAngle))
else:
    print('deepAngle vs allAngle:',stats.ttest_rel(deepAngle, allAngle))

if calbAngle_norm.pvalue<=0.05 or allAngle_norm.pvalue<=0.05:
    print('calbAngle vs allAngle:',stats.ks_2samp(calbAngle, allAngle))
else:
    print('calbAngle vs allAngle:', stats.ttest_rel(calbAngle, allAngle))


from bioinfokit.analys import stat

res = stat()
res.anova_stat(df=pdAngle, res_var='angle', anova_model='angle~C(channel)+C(mouse)+C(channel):C(mouse)')
res.anova_summary


from statsmodels.formula.api import ols
import statsmodels.api as sm
#perform two-way ANOVA
model = ols('angle ~ C(channel) + C(mouse) + C(channel):C(mouse)', data=pdAngle).fit()
sm.stats.anova_lm(model, typ=2)
#__________________________________________________________________________
#|                                                                        |#
#|                           PLOT EMB EXAMPLE                             |#
#|________________________________________________________________________|#

def personalize_ax(ax, ax_view = None):
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if ax_view:
        ax.view_init(ax_view[0], ax_view[1])


save_dir = '/home/julio/Documents/DeepSup_project/DualColor/figures/'
mouse= 'Thy1jRGECO23'
mouse_dict = load_pickle(os.path.join(base_dir, 'processed_data', mouse), mouse+'_data_dict.pkl')

cemb_sup_pre = mouse_dict['registered_clean_emb']['csup_umap_pre']
cemb_sup_rot = mouse_dict['registered_clean_emb']['csup_umap_rot']
cpos_sup_pre = mouse_dict['registered_clean_emb']['csup_pos_pre']
cpos_sup_rot = mouse_dict['registered_clean_emb']['csup_pos_rot']
cdir_sup_pre = mouse_dict['registered_clean_emb']['csup_dir_pre']
cdir_sup_rot = mouse_dict['registered_clean_emb']['csup_dir_rot']



dir_color_sup_pre = np.zeros((cdir_sup_pre.shape[0],3))
for point in range(cdir_sup_pre.shape[0]):
    if cdir_sup_pre[point]==-1:
        dir_color_sup_pre[point] = [14/255,14/255,143/255]
    elif cdir_sup_pre[point]==1:
        dir_color_sup_pre[point] = [12/255,136/255,249/255]
    else:
        dir_color_sup_pre[point] = [17/255,219/255,224/255]

dir_color_sup_rot = np.zeros((cdir_sup_rot.shape[0],3))
for point in range(cdir_sup_rot.shape[0]):
    if cdir_sup_rot[point]==-1:
        dir_color_sup_rot[point] = [14/255,14/255,143/255]
    elif cdir_sup_rot[point]==1:
        dir_color_sup_rot[point] = [12/255,136/255,249/255]
    else:
        dir_color_sup_rot[point] = [17/255,219/255,224/255]

plt.figure(figsize=((13,9)))
ax = plt.subplot(1,3,1, projection = '3d')
ax.scatter(*cemb_sup_pre[:,:3].T, color ='b', s=10)
ax.scatter(*cemb_sup_rot[:,:3].T, color = 'r', s=10)
personalize_ax(ax, [28,120])
ax = plt.subplot(1,3,2, projection = '3d')
ax.scatter(*cemb_sup_pre[:,:3].T, c = cpos_sup_pre[:,0], s=10, cmap = 'magma')
ax.scatter(*cemb_sup_rot[:,:3].T, c = cpos_sup_rot[:,0], s=10, cmap = 'magma')
personalize_ax(ax, [28,120])
ax = plt.subplot(1,3,3, projection = '3d')
ax.scatter(*cemb_sup_pre[:,:3].T, color=dir_color_sup_pre, s=10)
ax.scatter(*cemb_sup_rot[:,:3].T, color=dir_color_sup_rot, s=10)
personalize_ax(ax, [28,120])

plt.tight_layout()
plt.suptitle(mouse)
plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_calb.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_calb.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE DISTANCE                            |#
#|________________________________________________________________________|#


from sklearn.decomposition import PCA

def fit_ellipse(cloud_A, norm_vector):

    def find_rotation_align_vectors(a,b):
        v = np.cross(a,b)
        s = np.linalg.norm(v)
        c = np.dot(a,b)

        sscp = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]], [-v[1],v[0],0]])
        sscp2 = np.matmul(sscp,sscp)
        R = np.eye(3)+sscp+(sscp2*(1-c)/s**2)
        return R

    rot_2D = find_rotation_align_vectors([0,0,1], norm_vector)
    cloud_A_2D = (apply_rotation_to_cloud(cloud_A,rot_2D, cloud_A.mean(axis=0)) - cloud_A.mean(axis=0))[:,:2]
    X = cloud_A_2D[:,0:1]
    Y = cloud_A_2D[:,1:]

    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

    long_axis = ((x[0]+x[2])/2) + np.sqrt(((x[0]-x[2])/2)**2 + x[1]**2)
    short_axis = ((x[0]+x[2])/2) - np.sqrt(((x[0]-x[2])/2)**2 + x[1]**2)

    x_coord = np.linspace(2*np.min(cloud_A_2D[:,0]),2*np.max(cloud_A_2D[:,0]),1000)
    y_coord = np.linspace(2*np.min(cloud_A_2D[:,1]),2*np.max(cloud_A_2D[:,1]),1000)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord

    a = np.where(np.abs(Z_coord-1)<10e-4)
    fit_ellipse_points = np.zeros((a[0].shape[0],2))
    for point in range(a[0].shape[0]):
        x_coord = X_coord[a[0][point], a[1][point]]
        y_coord = Y_coord[a[0][point], a[1][point]]
        fit_ellipse_points[point,:] = [x_coord, y_coord]

    long_axis = np.max(pairwise_distances(fit_ellipse_points))/2
    short_axis = np.mean(pairwise_distances(fit_ellipse_points))/2


    R_ellipse = find_rotation_align_vectors(norm_vector, [0,0,1])

    fit_ellipse_points_3D = np.hstack((fit_ellipse_points, np.zeros((fit_ellipse_points.shape[0],1))))
    fit_ellipse_points_3D = apply_rotation_to_cloud(fit_ellipse_points_3D,R_ellipse, fit_ellipse_points_3D.mean(axis=0)) + cloud_A.mean(axis=0)


    return x, long_axis, short_axis, fit_ellipse_points,fit_ellipse_points_3D

def plot_distance(cent_A, cent_B, cent_pos, cent_dir, plane_cent_A, plane_cent_B, plane_cent_pos, plane_cent_dir, ellipse_A, ellipse_B):

    def process_axis(ax):
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_aspect('equal', adjustable='box')

    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(2,3,1, projection = '3d')
    ax.scatter(*cent_A[:,:3].T, color ='b', s= 20)
    ax.scatter(*cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*cent_B[:,:3].T, color = 'r', s= 20)
    ax.scatter(*cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([cent_A.mean(axis=0)[0], cent_B.mean(axis=0)[0]],
        [cent_A.mean(axis=0)[1], cent_B.mean(axis=0)[1]],
        [cent_A.mean(axis=0)[2], cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    process_axis(ax)

    ax = plt.subplot(2,3,2, projection = '3d')
    ax.scatter(*cent_A[:,:3].T, c = cent_dir, s= 20, cmap = 'tab10')
    ax.scatter(*cent_B[:,:3].T, c = cent_dir, s= 20, cmap = 'tab10')
    ax.scatter(*cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([cent_A.mean(axis=0)[0], cent_B.mean(axis=0)[0]],
        [cent_A.mean(axis=0)[1], cent_B.mean(axis=0)[1]],
        [cent_A.mean(axis=0)[2], cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    process_axis(ax)

    ax = plt.subplot(2,3,3, projection = '3d')
    ax.scatter(*cent_A[:,:3].T, c = cent_pos[:,0], s= 20, cmap = 'viridis')
    ax.scatter(*cent_B[:,:3].T, c = cent_pos[:,0], s= 20, cmap = 'magma')
    ax.scatter(*cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([cent_A.mean(axis=0)[0], cent_B.mean(axis=0)[0]],
        [cent_A.mean(axis=0)[1], cent_B.mean(axis=0)[1]],
        [cent_A.mean(axis=0)[2], cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    process_axis(ax)

    ax = plt.subplot(2,3,4, projection = '3d')
    ax.scatter(*plane_cent_A[:,:3].T, color ='b', s= 20)
    ax.scatter(*plane_cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*plane_cent_B[:,:3].T, color = 'r', s= 20)
    ax.scatter(*plane_cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([plane_cent_A.mean(axis=0)[0], plane_cent_B.mean(axis=0)[0]],
        [plane_cent_A.mean(axis=0)[1], plane_cent_B.mean(axis=0)[1]],
        [plane_cent_A.mean(axis=0)[2], plane_cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    ax.scatter(*ellipse_A[:,:3].T, color ='cyan', s= 5,alpha=0.3)
    ax.scatter(*ellipse_B[:,:3].T, color ='orange', s= 5,alpha=0.3)
    process_axis(ax)

    ax = plt.subplot(2,3,5, projection = '3d')
    ax.scatter(*plane_cent_A[:,:3].T, c = plane_cent_dir, s= 20, cmap = 'tab10')
    ax.scatter(*plane_cent_B[:,:3].T, c = plane_cent_dir, s= 20, cmap = 'tab10')
    ax.scatter(*plane_cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*plane_cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([plane_cent_A.mean(axis=0)[0], plane_cent_B.mean(axis=0)[0]],
        [plane_cent_A.mean(axis=0)[1], plane_cent_B.mean(axis=0)[1]],
        [plane_cent_A.mean(axis=0)[2], plane_cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    ax.scatter(*ellipse_A[:,:3].T, color ='cyan', s= 5,alpha=0.3)
    ax.scatter(*ellipse_B[:,:3].T, color ='orange', s= 5,alpha=0.3)
    process_axis(ax)

    ax = plt.subplot(2,3,6, projection = '3d')
    ax.scatter(*plane_cent_A[:,:3].T, c = plane_cent_pos[:,0], s= 20, cmap = 'viridis')
    ax.scatter(*plane_cent_B[:,:3].T, c = plane_cent_pos[:,0], s= 20, cmap = 'magma')
    ax.scatter(*plane_cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*plane_cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([plane_cent_A.mean(axis=0)[0], plane_cent_B.mean(axis=0)[0]],
        [plane_cent_A.mean(axis=0)[1], plane_cent_B.mean(axis=0)[1]],
        [plane_cent_A.mean(axis=0)[2], plane_cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    ax.scatter(*ellipse_A[:,:3].T, color ='cyan', s= 5,alpha=0.3)
    ax.scatter(*ellipse_B[:,:3].T, color ='orange', s= 5,alpha=0.3)
    process_axis(ax)
    plt.tight_layout()
    return fig


for mouse in ['Thy1jRGECO22','Thy1jRGECO23']:
    print(f"Working on {mouse}:")
    load_dir = os.path.join(base_dir,'rotation',mouse)
    save_dir = os.path.join(base_dir, 'distance', mouse)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    rotation_dict = load_pickle(load_dir, mouse+'_rotation_dict.pkl')
    distance_dict = dict()

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                   DEEP                                  |#
    #|________________________________________________________________________|#

    deep_cent_pre = rotation_dict['deep']['deep_cent_pre']
    deep_cent_rot = rotation_dict['deep']['deep_cent_rot']
    deep_cent_pos = rotation_dict['deep']['deep_cent_pos']
    deep_cent_dir = rotation_dict['deep']['deep_cent_dir']

    deep_inter_dist = np.linalg.norm(deep_cent_pre.mean(axis=0)-deep_cent_rot.mean(axis=0))
    deep_intra_dist_pre = np.percentile(pairwise_distances(deep_cent_pre),95)/2
    deep_intra_dist_rot = np.percentile(pairwise_distances(deep_cent_rot),95)/2
    deep_remap_dist = deep_inter_dist/np.mean((deep_intra_dist_pre, deep_intra_dist_rot))

    plane_deep_cent_pre = rotation_dict['deep']['plane_deep_cent_pre']
    plane_deep_cent_rot = rotation_dict['deep']['plane_deep_cent_rot']
    deep_norm_vector_pre = rotation_dict['deep']['deep_norm_vec_pre']
    plane_deep_cent_pos = rotation_dict['deep']['plane_deep_cent_pos']
    plane_deep_cent_dir = rotation_dict['deep']['plane_deep_cent_dir']
    deep_norm_vector_rot = rotation_dict['deep']['deep_norm_vec_rot']


    plane_deep_inter_dist = np.linalg.norm(plane_deep_cent_pre.mean(axis=0)-plane_deep_cent_rot.mean(axis=0))
    deep_ellipse_pre_params, deep_ellipse_pre_long_axis, deep_ellipse_pre_short_axis, deep_ellipse_pre_fit, deep_ellipse_pre_fit_3D = fit_ellipse(plane_deep_cent_pre, deep_norm_vector_pre)
    deep_ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(deep_ellipse_pre_long_axis+deep_ellipse_pre_short_axis)**2)

    deep_ellipse_rot_params, deep_ellipse_rot_long_axis, deep_ellipse_rot_short_axis, deep_ellipse_rot_fit, deep_ellipse_rot_fit_3D = fit_ellipse(plane_deep_cent_rot, deep_norm_vector_rot)
    deep_ellipse_rot_perimeter = 2*np.pi*np.sqrt(0.5*(deep_ellipse_rot_long_axis+deep_ellipse_rot_short_axis)**2)

    plane_deep_remap_dist = plane_deep_inter_dist/np.mean((deep_ellipse_pre_perimeter, deep_ellipse_rot_perimeter))

    print(f"\tdeep: {deep_remap_dist:.2f} remap dist | {plane_deep_remap_dist:.2f} remap dist plane")

    fig = plot_distance(deep_cent_pre,deep_cent_rot,deep_cent_pos,deep_cent_dir,
            plane_deep_cent_pre,plane_deep_cent_rot, plane_deep_cent_pos, plane_deep_cent_dir,
            deep_ellipse_pre_fit_3D, deep_ellipse_rot_fit_3D)
    plt.suptitle(f"{mouse} deep")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_distance_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_distance_plot.png'), dpi = 400,bbox_inches="tight")

    distance_dict['deep'] = {

        #cent
        'deep_cent_pre': deep_cent_pre,
        'deep_cent_rot': deep_cent_rot,
        'deep_cent_pos': deep_cent_pos,
        'noise_deep_pre': deep_cent_dir,
        #distance og
        'deep_inter_dist': deep_inter_dist,
        'deep_intra_dist_pre': deep_intra_dist_pre,
        'deep_intra_dist_rot': deep_intra_dist_rot,
        'deep_remap_dist': deep_remap_dist,

        #plane
        'plane_deep_cent_pre': deep_cent_pre,
        'deep_norm_vector_pre': deep_norm_vector_pre,
        'plane_deep_cent_rot': plane_deep_cent_rot,
        'deep_norm_vector_rot': deep_norm_vector_rot,
        'plane_deep_cent_pos': plane_deep_cent_pos,
        'plane_deep_cent_dir': plane_deep_cent_dir,

        #ellipse
        'deep_ellipse_pre_params': deep_ellipse_pre_params,
        'deep_ellipse_pre_long_axis': deep_ellipse_pre_long_axis,
        'deep_ellipse_pre_short_axis': deep_ellipse_pre_short_axis,
        'deep_ellipse_pre_fit': deep_ellipse_pre_fit,
        'deep_ellipse_pre_fit_3D': deep_ellipse_pre_fit_3D,

        'deep_ellipse_rot_params': deep_ellipse_rot_params,
        'deep_ellipse_rot_long_axis': deep_ellipse_rot_long_axis,
        'deep_ellipse_rot_short_axis': deep_ellipse_rot_short_axis,
        'deep_ellipse_rot_fit': deep_ellipse_rot_fit,
        'deep_ellipse_rot_fit_3D': deep_ellipse_rot_fit_3D,

        #distance ellipse
        'plane_deep_inter_dist': plane_deep_inter_dist,
        'deep_ellipse_pre_perimeter': deep_ellipse_pre_perimeter,
        'deep_ellipse_rot_perimeter': deep_ellipse_rot_perimeter,
        'plane_deep_remap_dist': plane_deep_remap_dist,
    }


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                   SUP                                  |#
    #|________________________________________________________________________|#

    sup_cent_pre = rotation_dict['sup']['sup_cent_pre']
    sup_cent_rot = rotation_dict['sup']['sup_cent_rot']
    sup_cent_pos = rotation_dict['sup']['sup_cent_pos']
    sup_cent_dir = rotation_dict['sup']['sup_cent_dir']

    sup_inter_dist = np.linalg.norm(sup_cent_pre.mean(axis=0)-sup_cent_rot.mean(axis=0))
    sup_intra_dist_pre = np.percentile(pairwise_distances(sup_cent_pre),95)/2
    sup_intra_dist_rot = np.percentile(pairwise_distances(sup_cent_rot),95)/2
    sup_remap_dist = sup_inter_dist/np.mean((sup_intra_dist_pre, sup_intra_dist_rot))

    plane_sup_cent_pre = rotation_dict['sup']['plane_sup_cent_pre']
    plane_sup_cent_rot = rotation_dict['sup']['plane_sup_cent_rot']
    sup_norm_vector_pre = rotation_dict['sup']['sup_norm_vec_pre']
    plane_sup_cent_pos = rotation_dict['sup']['plane_sup_cent_pos']
    plane_sup_cent_dir = rotation_dict['sup']['plane_sup_cent_dir']
    sup_norm_vector_rot = rotation_dict['sup']['sup_norm_vec_rot']


    plane_sup_inter_dist = np.linalg.norm(plane_sup_cent_pre.mean(axis=0)-plane_sup_cent_rot.mean(axis=0))
    sup_ellipse_pre_params, sup_ellipse_pre_long_axis, sup_ellipse_pre_short_axis, sup_ellipse_pre_fit, sup_ellipse_pre_fit_3D = fit_ellipse(plane_sup_cent_pre, sup_norm_vector_pre)
    sup_ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(sup_ellipse_pre_long_axis+sup_ellipse_pre_short_axis)**2)

    sup_ellipse_rot_params, sup_ellipse_rot_long_axis, sup_ellipse_rot_short_axis, sup_ellipse_rot_fit, sup_ellipse_rot_fit_3D = fit_ellipse(plane_sup_cent_rot, sup_norm_vector_rot)
    sup_ellipse_rot_perimeter = 2*np.pi*np.sqrt(0.5*(sup_ellipse_rot_long_axis+sup_ellipse_rot_short_axis)**2)

    plane_sup_remap_dist = plane_sup_inter_dist/np.mean((sup_ellipse_pre_perimeter, sup_ellipse_rot_perimeter))

    print(f"\tsup: {sup_remap_dist:.2f} remap dist | {plane_sup_remap_dist:.2f} remap dist plane")

    fig = plot_distance(sup_cent_pre,sup_cent_rot,sup_cent_pos,sup_cent_dir,
            plane_sup_cent_pre,plane_sup_cent_rot, plane_sup_cent_pos, plane_sup_cent_dir,
            sup_ellipse_pre_fit_3D, sup_ellipse_rot_fit_3D)
    plt.suptitle(f"{mouse} sup")
    plt.savefig(os.path.join(save_dir,f'{mouse}_sup_distance_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_sup_distance_plot.png'), dpi = 400,bbox_inches="tight")

    distance_dict['sup'] = {

        #cent
        'sup_cent_pre': sup_cent_pre,
        'sup_cent_rot': sup_cent_rot,
        'sup_cent_pos': sup_cent_pos,
        'noise_sup_pre': sup_cent_dir,
        #distance og
        'sup_inter_dist': sup_inter_dist,
        'sup_intra_dist_pre': sup_intra_dist_pre,
        'sup_intra_dist_rot': sup_intra_dist_rot,
        'sup_remap_dist': sup_remap_dist,

        #plane
        'plane_sup_cent_pre': sup_cent_pre,
        'sup_norm_vector_pre': sup_norm_vector_pre,
        'plane_sup_cent_rot': plane_sup_cent_rot,
        'sup_norm_vector_rot': sup_norm_vector_rot,
        'plane_sup_cent_pos': plane_sup_cent_pos,
        'plane_sup_cent_dir': plane_sup_cent_dir,

        #ellipse
        'sup_ellipse_pre_params': sup_ellipse_pre_params,
        'sup_ellipse_pre_long_axis': sup_ellipse_pre_long_axis,
        'sup_ellipse_pre_short_axis': sup_ellipse_pre_short_axis,
        'sup_ellipse_pre_fit': sup_ellipse_pre_fit,
        'sup_ellipse_pre_fit_3D': sup_ellipse_pre_fit_3D,

        'sup_ellipse_rot_params': sup_ellipse_rot_params,
        'sup_ellipse_rot_long_axis': sup_ellipse_rot_long_axis,
        'sup_ellipse_rot_short_axis': sup_ellipse_rot_short_axis,
        'sup_ellipse_rot_fit': sup_ellipse_rot_fit,
        'sup_ellipse_rot_fit_3D': sup_ellipse_rot_fit_3D,

        #distance ellipse
        'plane_sup_inter_dist': plane_sup_inter_dist,
        'sup_ellipse_pre_perimeter': sup_ellipse_pre_perimeter,
        'sup_ellipse_rot_perimeter': sup_ellipse_rot_perimeter,
        'plane_sup_remap_dist': plane_sup_remap_dist,
    }

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                   ALL                                  |#
    #|________________________________________________________________________|#

    all_cent_pre = rotation_dict['all']['all_cent_pre']
    all_cent_rot = rotation_dict['all']['all_cent_rot']
    all_cent_pos = rotation_dict['all']['all_cent_pos']
    all_cent_dir = rotation_dict['all']['all_cent_dir']

    all_inter_dist = np.linalg.norm(all_cent_pre.mean(axis=0)-all_cent_rot.mean(axis=0))
    all_intra_dist_pre = np.percentile(pairwise_distances(all_cent_pre),95)/2
    all_intra_dist_rot = np.percentile(pairwise_distances(all_cent_rot),95)/2
    all_remap_dist = all_inter_dist/np.mean((all_intra_dist_pre, all_intra_dist_rot))

    plane_all_cent_pre = rotation_dict['all']['plane_all_cent_pre']
    plane_all_cent_rot = rotation_dict['all']['plane_all_cent_rot']
    all_norm_vector_pre = rotation_dict['all']['all_norm_vec_pre']
    plane_all_cent_pos = rotation_dict['all']['plane_all_cent_pos']
    plane_all_cent_dir = rotation_dict['all']['plane_all_cent_dir']
    all_norm_vector_rot = rotation_dict['all']['all_norm_vec_rot']

    plane_all_inter_dist = np.linalg.norm(plane_all_cent_pre.mean(axis=0)-plane_all_cent_rot.mean(axis=0))
    all_ellipse_pre_params, all_ellipse_pre_long_axis, all_ellipse_pre_short_axis, all_ellipse_pre_fit, all_ellipse_pre_fit_3D = fit_ellipse(plane_all_cent_pre, all_norm_vector_pre)
    all_ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(all_ellipse_pre_long_axis+all_ellipse_pre_short_axis)**2)

    all_ellipse_rot_params, all_ellipse_rot_long_axis, all_ellipse_rot_short_axis, all_ellipse_rot_fit, all_ellipse_rot_fit_3D = fit_ellipse(plane_all_cent_rot, all_norm_vector_rot)
    all_ellipse_rot_perimeter = 2*np.pi*np.sqrt(0.5*(all_ellipse_rot_long_axis+all_ellipse_rot_short_axis)**2)

    plane_all_remap_dist = plane_all_inter_dist/np.mean((all_ellipse_pre_perimeter, all_ellipse_rot_perimeter))

    print(f"\tall: {all_remap_dist:.2f} remap dist | {plane_all_remap_dist:.2f} remap dist plane")

    fig = plot_distance(all_cent_pre,all_cent_rot,all_cent_pos,all_cent_dir,
            plane_all_cent_pre,plane_all_cent_rot, plane_all_cent_pos, plane_all_cent_dir,
            all_ellipse_pre_fit_3D, all_ellipse_rot_fit_3D)
    plt.suptitle(f"{mouse} all")
    plt.savefig(os.path.join(save_dir,f'{mouse}_all_distance_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_all_distance_plot.png'), dpi = 400,bbox_inches="tight")

    distance_dict['all'] = {

        #cent
        'all_cent_pre': all_cent_pre,
        'all_cent_rot': all_cent_rot,
        'all_cent_pos': all_cent_pos,
        'noise_all_pre': all_cent_dir,
        #distance og
        'all_inter_dist': all_inter_dist,
        'all_intra_dist_pre': all_intra_dist_pre,
        'all_intra_dist_rot': all_intra_dist_rot,
        'all_remap_dist': all_remap_dist,

        #plane
        'plane_all_cent_pre': all_cent_pre,
        'all_norm_vector_pre': all_norm_vector_pre,
        'plane_all_cent_rot': plane_all_cent_rot,
        'all_norm_vector_rot': all_norm_vector_rot,
        'plane_all_cent_pos': plane_all_cent_pos,
        'plane_all_cent_dir': plane_all_cent_dir,

        #ellipse
        'all_ellipse_pre_params': all_ellipse_pre_params,
        'all_ellipse_pre_long_axis': all_ellipse_pre_long_axis,
        'all_ellipse_pre_short_axis': all_ellipse_pre_short_axis,
        'all_ellipse_pre_fit': all_ellipse_pre_fit,
        'all_ellipse_pre_fit_3D': all_ellipse_pre_fit_3D,

        'all_ellipse_rot_params': all_ellipse_rot_params,
        'all_ellipse_rot_long_axis': all_ellipse_rot_long_axis,
        'all_ellipse_rot_short_axis': all_ellipse_rot_short_axis,
        'all_ellipse_rot_fit': all_ellipse_rot_fit,
        'all_ellipse_rot_fit_3D': all_ellipse_rot_fit_3D,

        #distance ellipse
        'plane_all_inter_dist': plane_all_inter_dist,
        'all_ellipse_pre_perimeter': all_ellipse_pre_perimeter,
        'all_ellipse_rot_perimeter': all_ellipse_rot_perimeter,
        'plane_all_remap_dist': plane_all_remap_dist,
    }


    with open(os.path.join(save_dir, mouse+"_distance_dict.pkl"), "wb") as file:
        pickle.dump(distance_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                              PLOT DISTANCE                             |#
#|________________________________________________________________________|#

save_dir = '/home/julio/Documents/DeepSup_project/DualColor/figures/'

emb_distance_list = list()
mouse_list = list()
channel_list = list()

for mouse in ['Thy1jRGECO22','Thy1jRGECO23']:
    load_dir = os.path.join(base_dir, 'distance', mouse)
    distance_dict = load_pickle(load_dir, mouse+'_distance_dict.pkl')

    mouse_list += [mouse]*3
    emb_distance_list.append(distance_dict['deep']['plane_deep_remap_dist'])
    channel_list.append('deep')
    emb_distance_list.append(distance_dict['sup']['plane_sup_remap_dist'])
    channel_list.append('sup')
    emb_distance_list.append(distance_dict['all']['plane_all_remap_dist'])
    channel_list.append('all')

mouse = 'ThyCalbRCaMP2'
distance_dict = load_pickle('/home/julio/Documents/DeepSup_project/DualColor/ThyCalbRCaMP/distance/ThyCalbRCaMP2/', mouse+'_distance_dict.pkl')
mouse_list += [mouse]*3
emb_distance_list.append(distance_dict['deep']['plane_deep_remap_dist'])
channel_list.append('deep')
emb_distance_list.append(distance_dict['sup']['plane_sup_remap_dist'])
channel_list.append('sup')
emb_distance_list.append(distance_dict['all']['plane_all_remap_dist'])
channel_list.append('all')


pd_distance = pd.DataFrame(data={'mouse': mouse_list,
                     'channel': channel_list,
                     'distance': emb_distance_list})    

fig, ax = plt.subplots(1, 1, figsize=(6,6))

b = sns.barplot(x='channel', y='distance', data=pd_distance,
            linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='channel', y='distance', data=pd_distance,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
# ax.set_yticks([0,45,90,135,180])
plt.savefig(os.path.join(save_dir,f'dual_dynamic_distance.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dual_dynamic_distance.png'), dpi = 400,bbox_inches="tight")


deep_dist = pd_distance.loc[pd_distance['channel']=='deep']['distance']
sup_dist = pd_distance.loc[pd_distance['channel']=='sup']['distance']
all_dist = pd_distance.loc[pd_distance['channel']=='all']['distance']

deep_dist_norm = stats.shapiro(deep_dist)
sup_dist_norm = stats.shapiro(sup_dist)
all_dist_norm = stats.shapiro(all_dist)

if deep_dist_norm.pvalue<=0.05 or sup_dist_norm.pvalue<=0.05:
    print('deep_dist vs sup_dist:',stats.ks_2samp(deep_dist, sup_dist))
else:
    print('deep_dist vs sup_dist:', stats.ttest_rel(deep_dist, sup_dist))

if deep_dist_norm.pvalue<=0.05 or all_dist_norm.pvalue<=0.05:
    print('deep_dist vs all_dist:',stats.ks_2samp(deep_dist, all_dist))
else:
    print('deep_dist vs all_dist:',stats.ttest_rel(deep_dist, all_dist))

if sup_dist_norm.pvalue<=0.05 or all_dist_norm.pvalue<=0.05:
    print('sup_dist vs all_dist:',stats.ks_2samp(sup_dist, all_dist))
else:
    print('sup_dist vs all_dist:', stats.ttest_rel(sup_dist, all_dist))


from statsmodels.formula.api import ols
import statsmodels.api as sm
#perform two-way ANOVA
model = ols('distance ~ C(channel) + C(mouse) + C(channel):C(mouse)', data=pd_distance).fit()
sm.stats.anova_lm(model, typ=2)


#__________________________________________________________________________
#|                                                                        |#
#|                       PLOT DEEP/SUP CORRELATION                        |#
#|________________________________________________________________________|#
save_dir = '/home/julio/Documents/SP_project/LT_DualColor/processed_data/'
remap_dist_dict = load_pickle(save_dir, 'remap_distance_dict.pkl')


rot_list = list()
mouse_list = list()
channel_list = list()
distance_list = list()
proportion_list = list()
for mouse in ['Thy1jRGECO22', 'Thy1jRGECO23', 'ThyCalbRCaMP2']:
    if 'Thy1jRGECO' in mouse:
        mouse_dict = load_pickle(save_dir, mouse+'_data_dict.pkl')
        rot_list.append(mouse_dict['alignment']['rotRAngle'])
        channel_list.append('all')
        mouse_list += [mouse]

        num_cells = mouse_dict['registered_clean_emb']['signal_red_pre'].shape[1];
        num_sup = mouse_dict['registered_clean_emb']['sup_signal_pre'].shape[1];
        #proportion_list.append((num_cells-num_sup)/num_sup)
        proportion_list.append(num_sup/num_cells)
    else:
        mouse_dict = load_pickle(os.path.join(save_dir,mouse), mouse+'_alignment_dict.pkl')
        rot_list.append(mouse_dict['rotBAngle'])
        channel_list.append('all')
        mouse_list += [mouse]

        num_deep = mouse_dict['csignal_green_pre'].shape[1];
        num_sup = mouse_dict['csignal_red_pre'].shape[1];
        # proportion_list.append(num_deep/num_sup)
        proportion_list.append(num_sup/(num_sup+num_cells))

    distance_list.append(remap_dist_dict[mouse]['remap_dist_all'])


pd_dualcolor = pd.DataFrame(data={'mouse': mouse_list,
                     'channel': channel_list,
                     'angle': rot_list,
                     'perc_sup': proportion_list,
                     'distance': distance_list})    

fig, ax = plt.subplots(1, 1, figsize=(6,6))

b = sns.boxplot(x='channel', y='angle', data=pdAngle,
            linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='channel', y='angle', data=pdAngle,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylim([-2.5, 180.5])
ax.set_yticks([0,45,90,135,180])
plt.savefig(os.path.join(save_dir,f'dual_dynamic_rot_angle_2.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dual_dynamic_rot_angle_2.png'), dpi = 400,bbox_inches="tight")



from scipy import stats
deepAngle = pdAngle.loc[pdAngle['channel']=='deep']['angle']
calbAngle = pdAngle.loc[pdAngle['channel']=='calb']['angle']
allAngle = pdAngle.loc[pdAngle['channel']=='all']['angle']

deepAngle_norm = stats.shapiro(deepAngle)
calbAngle_norm = stats.shapiro(calbAngle)
allAngle_norm = stats.shapiro(allAngle)

if deepAngle_norm.pvalue<=0.05 or calbAngle_norm.pvalue<=0.05:
    print('deepAngle vs calbAngle:',stats.ks_2samp(deepAngle, calbAngle))
else:
    print('deepAngle vs calbAngle:', stats.ttest_rel(deepAngle, calbAngle))

if deepAngle_norm.pvalue<=0.05 or allAngle_norm.pvalue<=0.05:
    print('deepAngle vs allAngle:',stats.ks_2samp(deepAngle, allAngle))
else:
    print('deepAngle vs allAngle:',stats.ttest_rel(deepAngle, allAngle))

if calbAngle_norm.pvalue<=0.05 or allAngle_norm.pvalue<=0.05:
    print('calbAngle vs allAngle:',stats.ks_2samp(calbAngle, allAngle))
else:
    print('calbAngle vs allAngle:', stats.ttest_rel(calbAngle, allAngle))


from bioinfokit.analys import stat

res = stat()
res.anova_stat(df=pdAngle, res_var='angle', anova_model='angle~C(channel)+C(mouse)+C(channel):C(mouse)')
res.anova_summary


from statsmodels.formula.api import ols
import statsmodels.api as sm
#perform two-way ANOVA
model = ols('angle ~ C(channel) + C(mouse) + C(channel):C(mouse)', data=pdAngle).fit()
sm.stats.anova_lm(model, typ=2)


plt.figure()
plt.scatter(pd_dualcolor['perc_sup'], pd_dualcolor['distance'])