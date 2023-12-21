
###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################       G11 REGISTERED       #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################

import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import copy
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


######################
#       PARAMS       #
######################
nNeigh = 120
dim = 3
velTh = 5

greenNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG11/Inscopix_data/ThyG1_lt_green_raw.csv'
greenNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG11/Inscopix_data/ThyG1_rot_green_raw.csv'
redNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG11/Inscopix_data/ThyG1_lt_red_raw.csv'
redNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG11/Inscopix_data/ThyG1_rot_red_raw.csv'
posNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG11/Inscopix_data/ThyG11_lt_position.mat'
posNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG11/Inscopix_data/ThyG11_rot_position.mat'


######################
#     LOAD SIGNAL    #
######################
signalGPre = pd.read_csv(greenNamePre).to_numpy()[1:,1:].astype(np.float64)
signalGRot = pd.read_csv(greenNameRot).to_numpy()[1:,1:].astype(np.float64)
signalRPre = pd.read_csv(redNamePre).to_numpy()[1:,1:].astype(np.float64)
signalRRot = pd.read_csv(redNameRot).to_numpy()[1:,1:].astype(np.float64)

######################
#      LOAD POS      #
######################
posPre = scipy.io.loadmat(posNamePre)['Position']
posPre = posPre[::2,:]/10
posRot = scipy.io.loadmat(posNameRot)['Position']
posRot = posRot[::2,:]/10

######################
#     DELETE NANs    #
######################
nanIdx = np.where(np.sum(np.isnan(signalGPre),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRPre),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posPre),axis=1)>0)[0]),axis=0)
signalGPre = np.delete(signalGPre,nanIdx, axis=0)
signalRPre = np.delete(signalRPre,nanIdx, axis=0)
posPre = np.delete(posPre,nanIdx, axis=0)

nanIdx = np.where(np.sum(np.isnan(signalGRot),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRRot),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posRot),axis=1)>0)[0]),axis=0)
signalGRot = np.delete(signalGRot,nanIdx, axis=0)
signalRRot = np.delete(signalRRot,nanIdx, axis=0)
posRot = np.delete(posRot,nanIdx, axis=0)

######################
#    MATCH LENGTH    #
######################
if signalGPre.shape[0]>signalRPre.shape[0]:
    signalGPre = signalGPre[:signalRPre.shape[0],:]
elif signalRPre.shape[0]>signalGPre.shape[0]:
    signalRPre = signalRPre[:signalGPre.shape[0],:]

if posPre.shape[0]>signalGPre.shape[0]:
    posPre = posPre[:signalGPre.shape[0],:]
else:
    signalGPre = signalGPre[:posPre.shape[0],:]
    signalRPre = signalRPre[:posPre.shape[0],:]

if signalGRot.shape[0]>signalRRot.shape[0]:
    signalGRot = signalGRot[:signalRRot.shape[0],:]
elif signalRRot.shape[0]>signalGRot.shape[0]:
    signalRRot = signalRRot[:signalGRot.shape[0],:]

if posRot.shape[0]>signalGRot.shape[0]:
    posRot = posRot[:signalGRot.shape[0],:]
else:
    signalGRot = signalGRot[:posRot.shape[0],:]
    signalRRot = signalRRot[:posRot.shape[0],:]


######################
#     COMPUTE DIR    #
######################
velPre = np.diff(posPre[:,0]).reshape(-1,1)*10
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
dirPre = np.zeros((velPre.shape))
dirPre[velPre>0] = 1
dirPre[velPre<0] = 2

velRot = np.diff(posRot[:,0]).reshape(-1,1)*10
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
dirRot = np.zeros((velRot.shape))
dirRot[velRot>0] = 1
dirRot[velRot<0] = 2

######################
#     COMPUTE VEL    #
######################
velPre = np.abs(np.diff(posPre[:,0]).reshape(-1,1)*10)
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)


velRot = np.abs(np.diff(posRot[:,0]).reshape(-1,1)*10)
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
velRot = gaussian_filter1d(velRot, sigma = 5, axis = 0)

######################
#  DELETE LOW SPEED  #
######################
lowSpeedIdxPre = np.where(velPre<velTh)[0]
signalGPre = np.delete(signalGPre,lowSpeedIdxPre, axis=0)
signalRPre = np.delete(signalRPre,lowSpeedIdxPre, axis=0)
posPre = np.delete(posPre,lowSpeedIdxPre, axis=0)
velPre = np.delete(velPre,lowSpeedIdxPre, axis=0)
dirPre = np.delete(dirPre,lowSpeedIdxPre, axis=0)


lowSpeedIdxRot = np.where(velRot<velTh)[0]
signalGRot = np.delete(signalGRot,lowSpeedIdxRot, axis=0)
signalRRot = np.delete(signalRRot,lowSpeedIdxRot, axis=0)
posRot = np.delete(posRot,lowSpeedIdxRot, axis=0)
velRot = np.delete(velRot,lowSpeedIdxRot, axis=0)
dirRot = np.delete(dirRot,lowSpeedIdxRot, axis=0)

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])
timeRot = np.arange(posRot.shape[0])

######################
#    CLEAN TRACES    #
######################
signalGPre = clean_traces(signalGPre)
signalRPre = clean_traces(signalRPre)
signalGRot = clean_traces(signalGRot)
signalRRot = clean_traces(signalRRot)

############################
#  ALL CELLS INDIVIDUAL    #
############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)


plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'All Cells - Individual {velTh}')

############################
#     REGISTER SIGNALS     #
############################
preGreenCells = [0,1,4,6,7,8,12,13,15,16,19,20,26,27,28,29,31,35,38,39,40,41,42,43,45,46,48,52,54,57,58,59,60,61,62,63,69,70,71,72,73,75,76,77,79,80,81,82,84,85,86,88,89,92,94,95,96,97,98,99,101,102,104,105,106,107,108,109,110,111,113,121,124,127,129,130,134,138,139,142,143,144,146,147,148,149,151,153,154,155,160,161,162,163,164,173,184,185,187,189,190,193,194,199,203,205,206,211,217,220,222,223,224,234]
preGreenCells = [x-1 for x in preGreenCells]
preRedCells = [0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,21,22,23,24,25,26,27,28,29,30,34,38,39,40,41,42,43,44,45,46,47,48,51,53,55,56,57,58,59,60,61,62,63,65,66,67,68,70,72,76,77,78,79,80,81,83,84,85,89,91,92,93,94,95,96,97,98,99,100,102,103,104,106,108,109,110,111,112,114,115,116,117,118,119,120,121,122,123,124,126,128,129,130,131,132,134,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,155,157,158,160,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,184,185,187,189,197,198,200,201,202,203,204,205,207,210,211,216,217,218,221,223,225,226,227,228,229,231,232,233,237,238,239,240,243,245,246,247,249,250,251,252,253,255,256,257,258,260,261,262,263,264,266,267,268,269,270,271,274,275,277,280,281,282,283,285,286,287,288,289,291,292,294,296,299,300,301,302,303,305,306,307,308,309,310,311,312,314,315,316,323,330,331,332,334,340,342,343,345,355,356,357,358,359,360,361,362,363,364,366,367,368,369,370,371,373,374,375,376,377,378,379,381,384,385,387,388,389,390,391,392,393,395,396,397,398,399,400,401,404,405,407,408,409,410,414,415,416,417,418,419,420,421,423,424,425,431,432,433,435,436,437,438,439,442,444,445,447,449,450,451,452,453,454,455,456,459,460,461,462,463,465,467,470,471,472,473,474,477,478,479,486,490,491,492,493,494,495,496,497,498,499,500,503,504,505,506,509,513,514,515,516,517,518,519,522,523,524,525,529,533,536,539,541,542,543,544,547,550,551,553,554,558,559,560,562,563,564,565,566,567,569,573,574,575,576,577,578,580]
preRedCells = [x-1 for x in preRedCells]

rotGreenCells = [1,3,4,7,8,6,2,5,10,11,12,9,20,16,15,13,14,18,21,22,26,33,23,29,30,28,31,24,27,32,37,154,34,36,39,38,46,43,49,48,56,50,44,45,57,42,51,55,59,64,67,63,52,62,68,66,58,61,60,65,79,69,71,78,75,70,76,74,80,82,84,83,87,90,91,93,95,101,99,100,105,110,102,104,111,107,103,109,114,115,113,116,117,118,119,120,131,125,128,132,133,135,136,138,139,140,141,143,145,144,146,149,147,35]
rotGreenCells = [x-1 for x in rotGreenCells]
rotRedCells = [0,2,3,4,16,5,6,17,9,18,11,12,91,14,15,7,109,107,105,20,19,26,40,497,24,27,22,21,28,30,31,487,44,34,36,35,33,37,43,32,182,48,45,47,63,50,53,55,52,59,56,60,58,51,62,57,65,494,66,68,67,197,75,74,76,93,80,97,206,77,79,78,85,87,81,95,90,92,83,89,94,82,86,96,88,100,222,103,101,102,121,108,230,106,112,116,110,114,111,113,104,117,99,98,120,119,130,125,181,141,145,134,136,140,135,137,170,133,138,144,149,151,127,131,148,163,166,169,164,165,155,159,158,299,178,180,176,160,168,171,172,188,191,175,177,183,154,153,187,190,300,193,304,184,194,196,203,211,199,205,207,213,209,208,210,499,219,216,215,489,218,217,239,212,252,253,496,254,226,225,227,234,229,232,235,249,223,236,238,233,250,241,247,243,245,248,255,242,251,283,259,268,264,267,279,275,263,272,280,273,266,276,257,256,500,284,269,270,261,262,287,291,288,293,289,292,294,295,296,285,301,303,306,302,308,309,307,314,311,312,315,322,317,316,340,319,318,325,334,324,328,348,326,327,332,335,333,336,338,329,323,415,331,370,493,342,343,344,368,341,345,354,349,356,357,355,382,375,352,361,360,363,364,359,346,386,371,384,385,373,372,376,377,379,381,378,387,388,389,391,394,392,393,407,396,390,399,400,404,403,454,401,408,406,405,491,402,414,411,463,416,425,420,421,423,424,419,427,428,430,429,449,447,434,437,433,436,469,440,441,439,438,442,445,444,435,432,450,451,453,456,457,459,475,455,461,460,462,464,465,466,468,467,472,471,473,470,474,477,476,495,481,480,483,485,484,486,492,418,380,129,122,124,224,152,412,185]
rotRedCells = [x-1 for x in rotRedCells]

signalGPre = signalGPre[:,preGreenCells]
signalRPre = signalRPre[:, preRedCells]
signalGRot = signalGRot[:,rotGreenCells]
signalRRot = signalRRot[:, rotRedCells]

###############################
# REGISTERED CELLS INDIVIDUAL #
###############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)

plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'Reg Cells - Individual {velTh}')

#############################
# REGISTERED CELLS TOGETHER #
#############################
#%%all data
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]

#%%all data
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]


D = pairwise_distances(embGPre)
noiseIdx = filter_noisy_outliers(embGPre,D=D)
csignalGPre = signalGPre[~noiseIdx,:]
cembGPre = embGPre[~noiseIdx,:]
cposGPre = posPre[~noiseIdx,:]
cdirGPre = dirPre[~noiseIdx]


D = pairwise_distances(embRPre)
noiseIdx = filter_noisy_outliers(embRPre,D=D)
csignalRPre = signalRPre[~noiseIdx,:]
cembRPre = embRPre[~noiseIdx,:]
cposRPre = posPre[~noiseIdx,:]
cdirRPre = dirPre[~noiseIdx]

D = pairwise_distances(embGRot)
noiseIdx = filter_noisy_outliers(embGRot,D=D)
csignalGRot = signalGRot[~noiseIdx,:]
cembGRot = embGRot[~noiseIdx,:]
cposGRot = posRot[~noiseIdx,:]
cdirGRot = dirRot[~noiseIdx]

D = pairwise_distances(embRRot)
noiseIdx = filter_noisy_outliers(embRRot,D=D)
csignalRRot = signalRRot[~noiseIdx,:]
cembRRot = embRRot[~noiseIdx,:]
cposRRot = posRot[~noiseIdx,:]
cdirRRot = dirRot[~noiseIdx]


dirColorGPre = np.zeros((cdirGPre.shape[0],3))
for point in range(cdirGPre.shape[0]):
    if cdirGPre[point]==0:
        dirColorGPre[point] = [14/255,14/255,143/255]
    elif cdirGPre[point]==1:
        dirColorGPre[point] = [12/255,136/255,249/255]
    else:
        dirColorGPre[point] = [17/255,219/255,224/255]

dirColorRPre = np.zeros((cdirRPre.shape[0],3))
for point in range(cdirRPre.shape[0]):
    if cdirRPre[point]==0:
        dirColorRPre[point] = [14/255,14/255,143/255]
    elif cdirRPre[point]==1:
        dirColorRPre[point] = [12/255,136/255,249/255]
    else:
        dirColorRPre[point] = [17/255,219/255,224/255]

dirColorGRot = np.zeros((cdirGRot.shape[0],3))
for point in range(cdirGRot.shape[0]):
    if cdirGRot[point]==0:
        dirColorGRot[point] = [14/255,14/255,143/255]
    elif cdirGRot[point]==1:
        dirColorGRot[point] = [12/255,136/255,249/255]
    else:
        dirColorGRot[point] = [17/255,219/255,224/255]

dirColorRRot = np.zeros((cdirRRot.shape[0],3))
for point in range(cdirRRot.shape[0]):
    if cdirRRot[point]==0:
        dirColorRRot[point] = [14/255,14/255,143/255]
    elif cdirRRot[point]==1:
        dirColorRRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRRot[point] = [17/255,219/255,224/255]


plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.set_title('Green')
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color=dirColorGPre, s=10)
ax.scatter(*cembGRot[:,:3].T, color=dirColorGRot, s=10)


ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color=dirColorRPre, s=10)
ax.scatter(*cembRRot[:,:3].T, color=dirColorRRot, s=10)
plt.suptitle()

#compute centroids
centGPre, centGRot = get_centroids(cembGPre, cembGRot, cposGPre[:,0], cposGRot[:,0], 
                                                cdirGPre, cdirGRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midGPre = np.median(centGPre, axis=0).reshape(-1,1)
midGRot = np.median(centGRot, axis=0).reshape(-1,1)
normVector =  midGPre - midGRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centGPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centGPre-midGPre.T, centGRot-midGRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotGAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Green: {rotGAngle:2f} degrees")


#compute centroids
centRPre, centRRot = get_centroids(cembRPre, cembRRot, cposRPre[:,0], cposRRot[:,0], 
                                                cdirRPre, cdirRRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midRPre = np.median(centRPre, axis=0).reshape(-1,1)
midRRot = np.median(centRRot, axis=0).reshape(-1,1)
normVector =  midRPre - midRRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centRPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centRPre-midRPre.T, centRRot-midRRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotRAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Red: {rotRAngle:2f} degrees")



###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################       G21 REGISTERED       #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################

import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import copy
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


######################
#       PARAMS       #
######################
nNeigh = 120
dim = 3
velTh = 5

greenNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_lt_green_raw.csv'
greenNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_green_raw.csv'
redNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_lt_red_raw.csv'
redNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_red_raw.csv'

posNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_lt_position.mat'
posNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_position.mat'


######################
#     LOAD SIGNAL    #
######################
signalGPre = pd.read_csv(greenNamePre).to_numpy()[1:,1:].astype(np.float64)
signalGRot = pd.read_csv(greenNameRot).to_numpy()[1:,1:].astype(np.float64)
signalRPre = pd.read_csv(redNamePre).to_numpy()[1:,1:].astype(np.float64)
signalRRot = pd.read_csv(redNameRot).to_numpy()[1:,1:].astype(np.float64)

######################
#      LOAD POS      #
######################
posPre = scipy.io.loadmat(posNamePre)['Position']
posPre = posPre[::2,:]/10
posRot = scipy.io.loadmat(posNameRot)['Position']
posRot = posRot[::2,:]/10

######################
#     DELETE NANs    #
######################
nanIdx = np.where(np.sum(np.isnan(signalGPre),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRPre),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posPre),axis=1)>0)[0]),axis=0)
signalGPre = np.delete(signalGPre,nanIdx, axis=0)
signalRPre = np.delete(signalRPre,nanIdx, axis=0)
posPre = np.delete(posPre,nanIdx, axis=0)

nanIdx = np.where(np.sum(np.isnan(signalGRot),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRRot),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posRot),axis=1)>0)[0]),axis=0)
signalGRot = np.delete(signalGRot,nanIdx, axis=0)
signalRRot = np.delete(signalRRot,nanIdx, axis=0)
posRot = np.delete(posRot,nanIdx, axis=0)

######################
#    MATCH LENGTH    #
######################
if signalGPre.shape[0]>signalRPre.shape[0]:
    signalGPre = signalGPre[:signalRPre.shape[0],:]
elif signalRPre.shape[0]>signalGPre.shape[0]:
    signalRPre = signalRPre[:signalGPre.shape[0],:]

if posPre.shape[0]>signalGPre.shape[0]:
    posPre = posPre[:signalGPre.shape[0],:]
else:
    signalGPre = signalGPre[:posPre.shape[0],:]
    signalRPre = signalRPre[:posPre.shape[0],:]

if signalGRot.shape[0]>signalRRot.shape[0]:
    signalGRot = signalGRot[:signalRRot.shape[0],:]
elif signalRRot.shape[0]>signalGRot.shape[0]:
    signalRRot = signalRRot[:signalGRot.shape[0],:]

if posRot.shape[0]>signalGRot.shape[0]:
    posRot = posRot[:signalGRot.shape[0],:]
else:
    signalGRot = signalGRot[:posRot.shape[0],:]
    signalRRot = signalRRot[:posRot.shape[0],:]


######################
#     COMPUTE DIR    #
######################
velPre = np.diff(posPre[:,0]).reshape(-1,1)*10
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
dirPre = np.zeros((velPre.shape))
dirPre[velPre>0] = 1
dirPre[velPre<0] = 2

velRot = np.diff(posRot[:,0]).reshape(-1,1)*10
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
dirRot = np.zeros((velRot.shape))
dirRot[velRot>0] = 1
dirRot[velRot<0] = 2

######################
#     COMPUTE VEL    #
######################
velPre = np.abs(np.diff(posPre[:,0]).reshape(-1,1)*10)
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)


velRot = np.abs(np.diff(posRot[:,0]).reshape(-1,1)*10)
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
velRot = gaussian_filter1d(velRot, sigma = 5, axis = 0)

######################
#  DELETE LOW SPEED  #
######################
lowSpeedIdxPre = np.where(velPre<velTh)[0]
signalGPre = np.delete(signalGPre,lowSpeedIdxPre, axis=0)
signalRPre = np.delete(signalRPre,lowSpeedIdxPre, axis=0)
posPre = np.delete(posPre,lowSpeedIdxPre, axis=0)
velPre = np.delete(velPre,lowSpeedIdxPre, axis=0)
dirPre = np.delete(dirPre,lowSpeedIdxPre, axis=0)


lowSpeedIdxRot = np.where(velRot<velTh)[0]
signalGRot = np.delete(signalGRot,lowSpeedIdxRot, axis=0)
signalRRot = np.delete(signalRRot,lowSpeedIdxRot, axis=0)
posRot = np.delete(posRot,lowSpeedIdxRot, axis=0)
velRot = np.delete(velRot,lowSpeedIdxRot, axis=0)
dirRot = np.delete(dirRot,lowSpeedIdxRot, axis=0)

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])
timeRot = np.arange(posRot.shape[0])

######################
#    CLEAN TRACES    #
######################
signalGPre = clean_traces(signalGPre)
signalRPre = clean_traces(signalRPre)
signalGRot = clean_traces(signalGRot)
signalRRot = clean_traces(signalRRot)

############################
#  ALL CELLS INDIVIDUAL    #
############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)


plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'All Cells - Individual {velTh}')

############################
#     REGISTER SIGNALS     #
############################
preGreenCells = [1,2,3,6,7,12,13,15,16,17,19,20,21,22,23,25,26,27,29,30,32,33,34,35,37,39,41,43,44,46,47,48,50,51,52,55,66,67,68,69,73,75,76,77,78,79,80,82,84,86,87,89,91,92,93,94,97,98,99,100,101,102,103,106,108,110,111,112,114,115,117,119,120,122,123,124,125,126,127,128,129,130,131,133,134,136,137,138,139,141,145,146,147,148,149,151,152,153,155,156,157,158,165,166,167,170,171,172,173,174,175,176,177,178,179,180,182,183,185,189,190,191,192,193,194,195,196,197,200,202,206,208,209,210,211,214,216,218,219,224,225,226,227,228, 229,231,232,234,235,236,239,240,248,249,251,254,258,259,261,262,266,267,273,277,278,279,282,284,285,288,290,293]
# preGreenCells = [x-1 for x in preGreenCells]
preRedCells = [2,3,4,6,7,10,12,13,18,22,23,24,25,27,28,29,30,31,32,34,36,37,40,41,42,43,45,46,47,49,50,51,53,54,55,56,57,58,61,62,63,64,65,67,68,69,70,72,74,75,79,81,82,83,84,85,86,87,88,89,94,95,97,98,99,100,101,102,103,105,107,109,112,113,114,117,118,119,120,121,122,124,125,126,127,128,129,130,131,133,135,137,138,139,140,141,142,143,144,145,146,149,152,153,154,155,157,158,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,182,184,185,186,187,188,191,192,196,197,198,199,200,201,203,204,206,207,208,210,211,212,214,215,216,217,218,219,220,221,225,226,227,232,233,234,235,236,237,238,239,240,241,242,244,245,251,252,257,258,259,260,261,262,263,264,266,269,270,271,275,276,277,278,279,280,281,282,284,285,286,287,290,291,292,293,294,295,296,297,299,302,303,304,305,306,307,308,309,310,312,313,314,315,316,319,320,321,324,325,326,329,330,331,332,334,336,337,343,344,346,347,348,350,351,353,354,355,357,361,364,366,367,368,369,375,376,378,379,381,382,383,384,387,389,390,393,395,402,405,406,407,409,410,412,413,415,416,418,422,426,427,432,433,434,435,436,437,438,439,440,441,442,443,445,447,449,450,454,455,457,460,461,462,463,464,465,466,468,471,475,484,486]
# preRedCells = [x-1 for x in preRedCells]

rotRedCells = [6,7,0,1,2,3,4,8,9,10,17,24,16,28,12,11,15,13,18,22,23,29,32,21,27,30,5,34,36,37,40,39,38,41,42,48,43,20,51,47,158,46,44,57,56,54,170,63,65,64,59,60,61,55,58,62,53,68,71,69,73,72,207,86,67,89,74,77,83,85,82,80,76,88,91,75,94,96,95,92,93,97,125,115,101,100,90,99,103,236,111,113,109,116,105,106,104,108,121,114,122,119,107,120,117,123,126,130,128,133,259,159,146,127,135,141,150,154,138,151,152,134,147,144,131,137,140,160,149,153,142,129,156,136,161,157,145,165,163,164,162,277,167,174,173,172,186,176,168,171,182,181,184,179,169,178,180,177,188,187,183,192,185,166,190,367,301,195,200,198,197,196,199,307,201,202,204,205,219,209,206,208,214,212,215,217,216,210,220,228,242,221,247,223,224,244,230,249,315,226,229,234,245,232,235,241,240,227,238,237,246,368,231,233,252,251,263,250,276,274,270,264,254,256,269,258,261,273,260,266,255,262,268,265,222,267,275,253,279,281,283,282,280,278,291,290,284,288,293,287,300,285,286,298,294,299,303,304,311,302,305,309,363,306,364,310,314,313,329,322,318,317,323,327,319,326,320,328,324,321,312,316,351,330,350,331,332,344,337,343,339,335,333,338,345,342,346,349,336,355,356,360,361,352,357,362,354,353,365,132,257]
# rotRedCells=[x-1 for x in rotRedCells]
rotGreenCells = [2,0,4,45,6,1,3,5,8,18,13,11,15,16,10,19,12,20,23,22,24,21,26,27,30,25,37,101,32,36,39,33,35,34,38,40,43,123,42,47,46,50,54,56,61,66,59,73,63,60,138,64,221,71,58,70,57,77,74,87,79,85,80,225,78,76,89,148,113,88,86,83,82,75,103,96,90,97,99,100,94,92,98,95,106,104,110,220,105,112,108,93,191,120,179,115,116,114,186,118,224,119,223,121,124,133,122,128,127,129,126,134,135,143,137,144,139,159,140,136,153,146,149,147,151,152,154,158,160,157,150,166,174,164,162,168,169,172,167,165,171,178,163,161,183,180,189,182,181,188,184,187,185,192,190,193,195,194,201,199,206,210,205,200,208,212,209,213,216,218,68,203]
# rotGreenCells = [x-1 for x in rotGreenCells]

signalGPre = signalGPre[:,preGreenCells]
signalRPre = signalRPre[:, preRedCells]
signalGRot = signalGRot[:,rotGreenCells]
signalRRot = signalRRot[:, rotRedCells]

###############################
# REGISTERED CELLS INDIVIDUAL #
###############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)

plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'Reg Cells - Individual {velTh}')

#############################
# REGISTERED CELLS TOGETHER #
#############################
#%%all data
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]

#%%all data
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]


D = pairwise_distances(embGPre)
noiseIdx = filter_noisy_outliers(embGPre,D=D)
csignalGPre = signalGPre[~noiseIdx,:]
cembGPre = embGPre[~noiseIdx,:]
cposGPre = posPre[~noiseIdx,:]
cdirGPre = dirPre[~noiseIdx]


D = pairwise_distances(embRPre)
noiseIdx = filter_noisy_outliers(embRPre,D=D)
csignalRPre = signalRPre[~noiseIdx,:]
cembRPre = embRPre[~noiseIdx,:]
cposRPre = posPre[~noiseIdx,:]
cdirRPre = dirPre[~noiseIdx]

D = pairwise_distances(embGRot)
noiseIdx = filter_noisy_outliers(embGRot,D=D)
csignalGRot = signalGRot[~noiseIdx,:]
cembGRot = embGRot[~noiseIdx,:]
cposGRot = posRot[~noiseIdx,:]
cdirGRot = dirRot[~noiseIdx]

D = pairwise_distances(embRRot)
noiseIdx = filter_noisy_outliers(embRRot,D=D)
csignalRRot = signalRRot[~noiseIdx,:]
cembRRot = embRRot[~noiseIdx,:]
cposRRot = posRot[~noiseIdx,:]
cdirRRot = dirRot[~noiseIdx]


dirColorGPre = np.zeros((cdirGPre.shape[0],3))
for point in range(cdirGPre.shape[0]):
    if cdirGPre[point]==0:
        dirColorGPre[point] = [14/255,14/255,143/255]
    elif cdirGPre[point]==1:
        dirColorGPre[point] = [12/255,136/255,249/255]
    else:
        dirColorGPre[point] = [17/255,219/255,224/255]

dirColorRPre = np.zeros((cdirRPre.shape[0],3))
for point in range(cdirRPre.shape[0]):
    if cdirRPre[point]==0:
        dirColorRPre[point] = [14/255,14/255,143/255]
    elif cdirRPre[point]==1:
        dirColorRPre[point] = [12/255,136/255,249/255]
    else:
        dirColorRPre[point] = [17/255,219/255,224/255]

dirColorGRot = np.zeros((cdirGRot.shape[0],3))
for point in range(cdirGRot.shape[0]):
    if cdirGRot[point]==0:
        dirColorGRot[point] = [14/255,14/255,143/255]
    elif cdirGRot[point]==1:
        dirColorGRot[point] = [12/255,136/255,249/255]
    else:
        dirColorGRot[point] = [17/255,219/255,224/255]

dirColorRRot = np.zeros((cdirRRot.shape[0],3))
for point in range(cdirRRot.shape[0]):
    if cdirRRot[point]==0:
        dirColorRRot[point] = [14/255,14/255,143/255]
    elif cdirRRot[point]==1:
        dirColorRRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRRot[point] = [17/255,219/255,224/255]


plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.set_title('Green')
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color=dirColorGPre, s=10)
ax.scatter(*cembGRot[:,:3].T, color=dirColorGRot, s=10)


ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color=dirColorRPre, s=10)
ax.scatter(*cembRRot[:,:3].T, color=dirColorRRot, s=10)


#compute centroids
centGPre, centGRot = get_centroids(cembGPre, cembGRot, cposGPre[:,0], cposGRot[:,0], 
                                                cdirGPre, cdirGRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midGPre = np.median(centGPre, axis=0).reshape(-1,1)
midGRot = np.median(centGRot, axis=0).reshape(-1,1)
normVector =  midGPre - midGRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centGPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centGPre-midGPre.T, centGRot-midGRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotGAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Green: {rotGAngle:2f} degrees")


#compute centroids
centRPre, centRRot = get_centroids(cembRPre, cembRRot, cposRPre[:,0], cposRRot[:,0], 
                                                cdirRPre, cdirRRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midRPre = np.median(centRPre, axis=0).reshape(-1,1)
midRRot = np.median(centRRot, axis=0).reshape(-1,1)
normVector =  midRPre - midRRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centRPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centRPre-midRPre.T, centRRot-midRRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotRAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Red: {rotRAngle:2f} degrees")

###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################       G22 REGISTERED       #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################

import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import copy
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


######################
#       PARAMS       #
######################
nNeigh = 120
dim = 3
velTh = 5

greenNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG22/Inscopix_data/ThyG22_lt_green_raw.csv'
greenNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG22/Inscopix_data/ThyG22_rot_green_raw.csv'
redNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG22/Inscopix_data/ThyG22_lt_red_raw.csv'
redNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG22/Inscopix_data/ThyG22_rot_red_raw.csv'

posNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG22/Inscopix_data/ThyG22_lt_position.mat'
posNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG22/Inscopix_data/ThyG22_rot_position.mat'


######################
#     LOAD SIGNAL    #
######################
signalGPre = pd.read_csv(greenNamePre).to_numpy()[1:,1:].astype(np.float64)
signalGRot = pd.read_csv(greenNameRot).to_numpy()[1:,1:].astype(np.float64)
signalRPre = pd.read_csv(redNamePre).to_numpy()[1:,1:].astype(np.float64)
signalRRot = pd.read_csv(redNameRot).to_numpy()[1:,1:].astype(np.float64)

######################
#      LOAD POS      #
######################
posPre = scipy.io.loadmat(posNamePre)['Position']
posPre = posPre[::2,:]/10
posRot = scipy.io.loadmat(posNameRot)['Position']
posRot = posRot[::2,:]/10

######################
#     DELETE NANs    #
######################
nanIdx = np.where(np.sum(np.isnan(signalGPre),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRPre),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posPre),axis=1)>0)[0]),axis=0)
signalGPre = np.delete(signalGPre,nanIdx, axis=0)
signalRPre = np.delete(signalRPre,nanIdx, axis=0)
posPre = np.delete(posPre,nanIdx, axis=0)

nanIdx = np.where(np.sum(np.isnan(signalGRot),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRRot),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posRot),axis=1)>0)[0]),axis=0)
signalGRot = np.delete(signalGRot,nanIdx, axis=0)
signalRRot = np.delete(signalRRot,nanIdx, axis=0)
posRot = np.delete(posRot,nanIdx, axis=0)

######################
#    MATCH LENGTH    #
######################
if signalGPre.shape[0]>signalRPre.shape[0]:
    signalGPre = signalGPre[:signalRPre.shape[0],:]
elif signalRPre.shape[0]>signalGPre.shape[0]:
    signalRPre = signalRPre[:signalGPre.shape[0],:]

if posPre.shape[0]>signalGPre.shape[0]:
    posPre = posPre[:signalGPre.shape[0],:]
else:
    signalGPre = signalGPre[:posPre.shape[0],:]
    signalRPre = signalRPre[:posPre.shape[0],:]

if signalGRot.shape[0]>signalRRot.shape[0]:
    signalGRot = signalGRot[:signalRRot.shape[0],:]
elif signalRRot.shape[0]>signalGRot.shape[0]:
    signalRRot = signalRRot[:signalGRot.shape[0],:]

if posRot.shape[0]>signalGRot.shape[0]:
    posRot = posRot[:signalGRot.shape[0],:]
else:
    signalGRot = signalGRot[:posRot.shape[0],:]
    signalRRot = signalRRot[:posRot.shape[0],:]


######################
#     COMPUTE DIR    #
######################
velPre = np.diff(posPre[:,0]).reshape(-1,1)*10
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
dirPre = np.zeros((velPre.shape))
dirPre[velPre>0] = 1
dirPre[velPre<0] = 2

velRot = np.diff(posRot[:,0]).reshape(-1,1)*10
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
dirRot = np.zeros((velRot.shape))
dirRot[velRot>0] = 1
dirRot[velRot<0] = 2

######################
#     COMPUTE VEL    #
######################
velPre = np.abs(np.diff(posPre[:,0]).reshape(-1,1)*10)
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)


velRot = np.abs(np.diff(posRot[:,0]).reshape(-1,1)*10)
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
velRot = gaussian_filter1d(velRot, sigma = 5, axis = 0)

######################
#  DELETE LOW SPEED  #
######################
lowSpeedIdxPre = np.where(velPre<velTh)[0]
signalGPre = np.delete(signalGPre,lowSpeedIdxPre, axis=0)
signalRPre = np.delete(signalRPre,lowSpeedIdxPre, axis=0)
posPre = np.delete(posPre,lowSpeedIdxPre, axis=0)
velPre = np.delete(velPre,lowSpeedIdxPre, axis=0)
dirPre = np.delete(dirPre,lowSpeedIdxPre, axis=0)


lowSpeedIdxRot = np.where(velRot<velTh)[0]
signalGRot = np.delete(signalGRot,lowSpeedIdxRot, axis=0)
signalRRot = np.delete(signalRRot,lowSpeedIdxRot, axis=0)
posRot = np.delete(posRot,lowSpeedIdxRot, axis=0)
velRot = np.delete(velRot,lowSpeedIdxRot, axis=0)
dirRot = np.delete(dirRot,lowSpeedIdxRot, axis=0)

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])
timeRot = np.arange(posRot.shape[0])

######################
#    CLEAN TRACES    #
######################
signalGPre = clean_traces(signalGPre)
signalRPre = clean_traces(signalRPre)
signalGRot = clean_traces(signalGRot)
signalRRot = clean_traces(signalRRot)

############################
#  ALL CELLS INDIVIDUAL    #
############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)


plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'All Cells - Individual {velTh}')

############################
#     REGISTER SIGNALS     #
############################
preGreenCells = [2,10,11,15,17,18,21,22,23,24,25,26,27,28,30,31,32,38,39,40,41,42,50,51,52,53,54,55,56,60,61,63,64,66,68,69,71,72,73,74,75,76,77,78,79,80,81,82,83,84,86,90,92,94,95,97,98,100,101,107,108,112,115,116,117,118,119,120,121,122,123,124,126,127,128,129,130,131,132,133,134,135,136,137,139,140,144,145,150,153,155,156,158,159,160,161,162,163,164,167,168,171,175,176,177,178,180,181,182,183,184,185,187,189,190,191,192,198,199,200,201,203,204,205,206,207,208,209,212,214,216,217,220,221,222,223,224,225,226,227,228,230,236,237,238,240,242,243,244,245,246,247,249,251,252,253,254,255,257,258,259,260,261,262,263,264,265,266,269,274,280,288,289,290,291,292,293,294,295,296,299,300,301,302,303,304,305,306,308,309,310,311,315,316,317,318,319,321,322,324,325,326,331,332,333,334,335,336,337]
preGreenCells = [x-1 for x in preGreenCells]
preRedCells = [1,2,3,4,5,6,7,9,10,16,17,22,23,25,26,27,28,30,33,37,41,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,65,68,70,71,77,79,80,81,82,83,84,86,87,89,91,92,96,99,100,102,103,105,106,107,109,110,111,112,113,114,117,120,122,124,129,130,131,133,135,136,139,140,142,144,146,147,148,151,152,153,154,155,156,160,161,163,165,166,170,172,173,175,178,179,180,182,183,188,189,190,191,196,197,198,200,201,202,203,208,209,210,213,214,215,216,217,219,220,221,222,223,224,225,226,228,229,234,236,239,243,244,245,246,247,248,250,251,252,253,254,255,256,258,259,260,261,263,266,267,269,270,271,272,273,276,279,282,283,285,286,287,288,289,290,291,293,297,298,301,303,306,307,308,311,313,314,315,316,319,324,331,332,333,334,335,336,341,343,344,349,351,352,354,355,356,357,358,359,360,362,364,365,367,368,369,371,372,374,375,378,383,385,386,387,388,389,390,391,392,393,394,395,396,397,398,400,401,403,405,408,421,422,423,424,427,428,430,432,433,437,439,440,441,442,443,445,447,449,450,451,452,453,454,455,456,465,466,469,470,471,476,479,480,481]
preRedCells = [x-1 for x in preRedCells]

rotGreenCells = [0,1,3,4,10,7,9,11,12,22,17,14,18,21,13,15,16,28,287,29,25,24,23,35,48,36,38,42,47,39,43,45,49,34,290,53,50,52,54,55,59,71,56,64,67,60,121,58,63,70,66,75,61,65,68,77,80,82,76,84,94,89,86,88,95,104,99,108,103,105,116,114,101,96,97,113,109,110,115,112,117,118,125,119,131,120,288,130,127,126,128,129,132,147,134,140,145,139,141,136,142,135,138,146,148,149,144,152,151,150,153,154,158,156,159,155,162,157,165,234,174,163,164,168,173,236,176,167,172,170,175,166,192,178,246,201,183,185,188,180,179,182,181,177,193,195,220,198,211,203,200,217,219,196,205,213,210,199,218,207,197,216,194,269,221,206,227,222,228,229,235,238,239,241,243,245,244,248,247,242,250,251,253,252,255,267,257,256,268,262,261,265,263,266,254,270,274,275,276,277,281,282,285,286,186,32,90,74,40]
rotGreenCells = [x-1 for x in rotGreenCells]
rotRedCells = [3,4,2,10,0,9,8,1,64,7,11,14,23,15,13,89,20,21,18,22,65,24,12,25,38,27,36,26,33,37,28,41,29,34,35,32,31,40,339,30,42,43,44,143,51,48,47,49,50,54,59,53,62,69,60,66,63,67,57,68,46,80,70,76,71,79,77,78,84,337,73,83,86,82,95,93,74,81,72,85,92,343,97,98,122,121,102,103,99,120,101,115,104,107,100,112,118,114,190,117,116,119,131,109,128,130,123,124,126,133,145,138,148,139,141,142,144,147,140,134,136,135,149,170,163,155,154,152,158,160,157,173,175,176,161,177,169,164,249,156,171,151,174,165,172,153,184,179,178,183,181,193,186,194,192,182,202,196,276,201,180,187,197,203,191,200,188,228,209,207,216,340,189,217,227,206,211,210,223,225,229,230,231,232,233,235,236,234,252,237,254,241,246,242,251,240,248,250,243,244,253,238,239,245,273,257,258,270,262,261,259,264,260,275,266,269,271,265,277,274,267,255,272,296,280,279,287,283,284,295,281,278,288,291,290,286,289,297,329,298,299,300,333,304,308,309,310,306,320,312,314,313,316,318,311,319,322,325,321,324,323,326,327,330,332,338,16,17,105,150,137,162]
rotRedCells = [x-1 for x in rotRedCells]

signalGPre = signalGPre[:,preGreenCells]
signalRPre = signalRPre[:, preRedCells]
signalGRot = signalGRot[:,rotGreenCells]
signalRRot = signalRRot[:, rotRedCells]

###############################
# REGISTERED CELLS INDIVIDUAL #
###############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)

plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'Reg Cells - Individual {velTh}')

#############################
# REGISTERED CELLS TOGETHER #
#############################
#%%all data
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]

#%%all data
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]


D = pairwise_distances(embGPre)
noiseIdx = filter_noisy_outliers(embGPre,D=D)
csignalGPre = signalGPre[~noiseIdx,:]
cembGPre = embGPre[~noiseIdx,:]
cposGPre = posPre[~noiseIdx,:]
cdirGPre = dirPre[~noiseIdx]


D = pairwise_distances(embRPre)
noiseIdx = filter_noisy_outliers(embRPre,D=D)
csignalRPre = signalRPre[~noiseIdx,:]
cembRPre = embRPre[~noiseIdx,:]
cposRPre = posPre[~noiseIdx,:]
cdirRPre = dirPre[~noiseIdx]

D = pairwise_distances(embGRot)
noiseIdx = filter_noisy_outliers(embGRot,D=D)
csignalGRot = signalGRot[~noiseIdx,:]
cembGRot = embGRot[~noiseIdx,:]
cposGRot = posRot[~noiseIdx,:]
cdirGRot = dirRot[~noiseIdx]

D = pairwise_distances(embRRot)
noiseIdx = filter_noisy_outliers(embRRot,D=D)
csignalRRot = signalRRot[~noiseIdx,:]
cembRRot = embRRot[~noiseIdx,:]
cposRRot = posRot[~noiseIdx,:]
cdirRRot = dirRot[~noiseIdx]


dirColorGPre = np.zeros((cdirGPre.shape[0],3))
for point in range(cdirGPre.shape[0]):
    if cdirGPre[point]==0:
        dirColorGPre[point] = [14/255,14/255,143/255]
    elif cdirGPre[point]==1:
        dirColorGPre[point] = [12/255,136/255,249/255]
    else:
        dirColorGPre[point] = [17/255,219/255,224/255]

dirColorRPre = np.zeros((cdirRPre.shape[0],3))
for point in range(cdirRPre.shape[0]):
    if cdirRPre[point]==0:
        dirColorRPre[point] = [14/255,14/255,143/255]
    elif cdirRPre[point]==1:
        dirColorRPre[point] = [12/255,136/255,249/255]
    else:
        dirColorRPre[point] = [17/255,219/255,224/255]

dirColorGRot = np.zeros((cdirGRot.shape[0],3))
for point in range(cdirGRot.shape[0]):
    if cdirGRot[point]==0:
        dirColorGRot[point] = [14/255,14/255,143/255]
    elif cdirGRot[point]==1:
        dirColorGRot[point] = [12/255,136/255,249/255]
    else:
        dirColorGRot[point] = [17/255,219/255,224/255]

dirColorRRot = np.zeros((cdirRRot.shape[0],3))
for point in range(cdirRRot.shape[0]):
    if cdirRRot[point]==0:
        dirColorRRot[point] = [14/255,14/255,143/255]
    elif cdirRRot[point]==1:
        dirColorRRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRRot[point] = [17/255,219/255,224/255]


plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.set_title('Green')
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color=dirColorGPre, s=10)
ax.scatter(*cembGRot[:,:3].T, color=dirColorGRot, s=10)


ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color=dirColorRPre, s=10)
ax.scatter(*cembRRot[:,:3].T, color=dirColorRRot, s=10)


#compute centroids
centGPre, centGRot = get_centroids(cembGPre, cembGRot, cposGPre[:,0], cposGRot[:,0], 
                                                cdirGPre, cdirGRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midGPre = np.median(centGPre, axis=0).reshape(-1,1)
midGRot = np.median(centGRot, axis=0).reshape(-1,1)
normVector =  midGPre - midGRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centGPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centGPre-midGPre.T, centGRot-midGRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotGAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Green: {rotGAngle:2f} degrees")


#compute centroids
centRPre, centRRot = get_centroids(cembRPre, cembRRot, cposRPre[:,0], cposRRot[:,0], 
                                                cdirRPre, cdirRRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midRPre = np.median(centRPre, axis=0).reshape(-1,1)
midRRot = np.median(centRRot, axis=0).reshape(-1,1)
normVector =  midRPre - midRRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centRPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centRPre-midRPre.T, centRRot-midRRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotRAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Red: {rotRAngle:2f} degrees")

###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################       G23 REGISTERED       #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################
import os
import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import copy
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

def clean_traces_together(ogSignalPre, ogSignalRot, sigma = 6, sig_up = 4, sig_down = 12, peak_th=0.1):
    ogSignal = np.concatenate((ogSignalPre, ogSignalRot), axis=0)
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
    return biSignal[:ogSignalPre.shape[0],:], biSignal[ogSignalPre.shape[0]:,:]

######################
#       PARAMS       #
######################

saveDir = '/home/julio/Documents/SP_project/LT_DualColor/emb_examples/'
nNeigh = 120
dim = 3
velTh = 5

greenNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG23/Inscopix_data/ThyG23_lt_green_raw.csv'
greenNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG23/Inscopix_data/ThyG23_rot_green_raw.csv'
redNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG23/Inscopix_data/ThyG23_lt_red_raw.csv'
redNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG23/Inscopix_data/ThyG23_rot_red_raw.csv'

posNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG23/Inscopix_data/ThyG23_lt_position.mat'
posNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG23/Inscopix_data/ThyG23_rot_position.mat'


######################
#     LOAD SIGNAL    #
######################
signalGPre = pd.read_csv(greenNamePre).to_numpy()[1:,1:].astype(np.float64)
signalGRot = pd.read_csv(greenNameRot).to_numpy()[1:,1:].astype(np.float64)
signalRPre = pd.read_csv(redNamePre).to_numpy()[1:,1:].astype(np.float64)
signalRRot = pd.read_csv(redNameRot).to_numpy()[1:,1:].astype(np.float64)

######################
#      LOAD POS      #
######################
posPre = scipy.io.loadmat(posNamePre)['Position']
posPre = posPre[::2,:]/10
posRot = scipy.io.loadmat(posNameRot)['Position']
posRot = posRot[::2,:]/10

######################
#     DELETE NANs    #
######################
nanIdx = np.where(np.sum(np.isnan(signalGPre),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRPre),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posPre),axis=1)>0)[0]),axis=0)
signalGPre = np.delete(signalGPre,nanIdx, axis=0)
signalRPre = np.delete(signalRPre,nanIdx, axis=0)
posPre = np.delete(posPre,nanIdx, axis=0)

nanIdx = np.where(np.sum(np.isnan(signalGRot),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRRot),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posRot),axis=1)>0)[0]),axis=0)
signalGRot = np.delete(signalGRot,nanIdx, axis=0)
signalRRot = np.delete(signalRRot,nanIdx, axis=0)
posRot = np.delete(posRot,nanIdx, axis=0)

######################
#    MATCH LENGTH    #
######################
if signalGPre.shape[0]>signalRPre.shape[0]:
    signalGPre = signalGPre[:signalRPre.shape[0],:]
elif signalRPre.shape[0]>signalGPre.shape[0]:
    signalRPre = signalRPre[:signalGPre.shape[0],:]

if posPre.shape[0]>signalGPre.shape[0]:
    posPre = posPre[:signalGPre.shape[0],:]
else:
    signalGPre = signalGPre[:posPre.shape[0],:]
    signalRPre = signalRPre[:posPre.shape[0],:]

if signalGRot.shape[0]>signalRRot.shape[0]:
    signalGRot = signalGRot[:signalRRot.shape[0],:]
elif signalRRot.shape[0]>signalGRot.shape[0]:
    signalRRot = signalRRot[:signalGRot.shape[0],:]

if posRot.shape[0]>signalGRot.shape[0]:
    posRot = posRot[:signalGRot.shape[0],:]
else:
    signalGRot = signalGRot[:posRot.shape[0],:]
    signalRRot = signalRRot[:posRot.shape[0],:]

######################
#     COMPUTE DIR    #
######################
velPre = np.diff(posPre[:,0]).reshape(-1,1)*10
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
dirPre = np.zeros((velPre.shape))
dirPre[velPre>0] = 1
dirPre[velPre<0] = 2

velRot = np.diff(posRot[:,0]).reshape(-1,1)*10
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
dirRot = np.zeros((velRot.shape))
dirRot[velRot>0] = 1
dirRot[velRot<0] = 2

######################
#     COMPUTE VEL    #
######################
velPre = np.abs(np.diff(posPre[:,0]).reshape(-1,1)*10)
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)

velRot = np.abs(np.diff(posRot[:,0]).reshape(-1,1)*10)
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
velRot = gaussian_filter1d(velRot, sigma = 5, axis = 0)

######################
#  DELETE LOW SPEED  #
######################
lowSpeedIdxPre = np.where(velPre<velTh)[0]
signalGPre = np.delete(signalGPre,lowSpeedIdxPre, axis=0)
signalRPre = np.delete(signalRPre,lowSpeedIdxPre, axis=0)
posPre = np.delete(posPre,lowSpeedIdxPre, axis=0)
velPre = np.delete(velPre,lowSpeedIdxPre, axis=0)
dirPre = np.delete(dirPre,lowSpeedIdxPre, axis=0)

lowSpeedIdxRot = np.where(velRot<velTh)[0]
signalGRot = np.delete(signalGRot,lowSpeedIdxRot, axis=0)
signalRRot = np.delete(signalRRot,lowSpeedIdxRot, axis=0)
posRot = np.delete(posRot,lowSpeedIdxRot, axis=0)
velRot = np.delete(velRot,lowSpeedIdxRot, axis=0)
dirRot = np.delete(dirRot,lowSpeedIdxRot, axis=0)

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])
timeRot = np.arange(posRot.shape[0])

######################
#    CLEAN TRACES    #
######################
signalGPre = clean_traces(signalGPre)
signalRPre = clean_traces(signalRPre)
signalGRot = clean_traces(signalGRot)
signalRRot = clean_traces(signalRRot)

############################
#  ALL CELLS INDIVIDUAL    #
############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)


plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'All Cells - Individual {velTh}')


############################
#     REGISTER SIGNALS     #
############################
preGreenCells = [4,5,6,7,8,9,10,11,14,16,17,18,21,25,26,27,28,30,32,33,34,35,36,41,42,44,45,46,47,48,49,51,52,53,54,56,57,58,60,61,62,63,64,65,66,67,70,71,72,73,75,76,77,78,79,80,81,83,84,86,87,88,89,90,93,95,96,97,98,104,105,107,108,110,112,113,114,115,119,120,122,123,124,125,127,129,130,131,134,136,137,138,139,140,141,144,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,172,173,176,177,178,180,182,184,185,186,188,189,191,193,194,195,196,198,199,201,202,207,208,209,210,211,212,218,219,220,222,223,224,225,226,227,229,232,236,237,238,240,241,242,243,245,247,248,249,250,251,252,253,254,255,256,257,258,259,260,262,263,264,267,268,269,270,272,273,274,278,280,281,282,283,285,286,287,288,291,292,293,294,295,296,300,301,302,306,307,309,310,314,315,316,318]
# preGreenCells = [x-1 for x in preGreenCells]
preRedCells = [1,3,4,8,9,10,11,12,13,14,15,16,21,22,23,24,27,28,30,31,37,43,45,46,47,48,50,51,52,53,54,57,59,61,64,65,66,67,68,69,70,71,72,73,75,76,77,78,79,82,83,91,92,93,94,95,98,99,101,102,103,104,106,108,110,116,122,123,124,126,129,130,131,132,133,137,138,139,140,141,145,146,147,148,149,150,152,153,154,156,157,159,160,162,164,165,166,167,168,169,170,172,173,174,178,179,180,181,182,192,193,194,196,197,199,201,202,205,207,208,210,214,215,218,229,230,231,232,233,234,235,236,237,241,242,244,245,252,254,255,256,257,258,260,261,263,264,265,267,268,270,271,272,273,276,277,278,279,280,281,282,283,284,286,288,289,293,294,299,300,305,306,307,308,310,311,313,314,315,316,318,319,320,321,323,328,330,331,333,334,342,343,347,349,351,354,355,357,359,361,362,363,364,365,368,369,370,371,372,376,377,378,380,381,383,384,385,386,387,389,395,396,398,399,400,403,404,406,407,409,410,411,412,413,414,415,416,417,418,419,422,423,425,426,427,428,432,435,438,440,441,443,446,447,448,449,451,452,453,456,458,459,461,462]
# preRedCells = [x-1 for x in preRedCells]

rotGreenCells = [12,3,4,5,14,20,9,13,19,16,15,17,18,21,23,24,32,25,37,26,36,27,0,28,35,34,39,40,64,38,46,52,51,43,60,44,53,48,58,63,67,41,62,45,66,59,65,47,42,68,82,78,72,80,77,73,81,74,76,75,83,85,86,88,333,91,92,94,93,95,97,96,22,100,99,102,101,106,107,112,134,113,160,119,120,127,129,125,135,131,130,128,114,117,136,159,157,70,146,148,142,151,158,137,144,332,145,155,138,140,161,164,166,163,165,168,167,172,179,178,181,192,180,190,188,185,191,182,193,184,208,196,214,205,206,201,210,202,204,199,213,197,195,118,227,224,221,215,219,233,249,245,242,238,234,240,231,259,253,258,256,254,260,251,257,252,250,267,274,176,263,268,266,270,264,265,269,272,271,276,275,283,285,281,279,220,288,292,289,230,297,299,247,301,303,304,309,306,305,307,311,308,310,316,313,320,321,319,330,323,200,8,116,132]
# rotGreenCells = [x-1 for x in rotGreenCells]
rotRedCells = [1,2,0,24,23,6,9,8,7,22,12,11,10,14,17,32,26,27,29,28,34,37,39,51,42,52,45,41,47,57,395,43,3,40,59,74,66,85,63,80,84,13,76,67,81,79,61,77,73,82,64,70,87,62,83,71,69,58,89,105,96,95,99,98,94,104,90,397,109,116,114,112,111,117,118,121,120,122,123,124,128,126,36,145,138,130,136,133,134,135,140,144,139,147,129,173,149,68,153,152,171,158,174,154,164,168,156,160,155,169,172,165,178,182,180,103,177,194,193,200,186,196,191,197,199,190,179,198,202,214,203,207,206,204,113,208,215,219,220,221,226,240,230,224,237,238,233,231,141,241,146,236,235,229,228,225,244,246,243,264,258,256,249,253,254,266,259,263,247,270,275,272,273,278,280,281,267,284,289,288,299,297,293,291,295,303,201,287,305,306,314,310,313,315,317,318,320,321,319,323,325,322,327,326,227,345,332,328,329,331,336,252,337,349,339,338,348,340,347,346,350,290,352,357,355,363,356,282,351,371,359,364,375,361,362,366,378,368,370,369,373,365,367,381,374,380,382,386,389,391,383,392,390,151,5,170,279,248,344,4,38,250,101,161]
# rotRedCells = [x-1 for x in rotRedCells]

signalGPre = signalGPre[:,preGreenCells]
signalRPre = signalRPre[:, preRedCells]
signalGRot = signalGRot[:,rotGreenCells]
signalRRot = signalRRot[:, rotRedCells]

# signalGPre, signalGRot = clean_traces_together(signalGPre,signalGRot)
# signalRPre, signalRRot = clean_traces_together(signalRPre,signalRRot)
###############################
# REGISTERED CELLS INDIVIDUAL #
###############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)

plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'Reg Cells - Individual {velTh}')

#############################
# REGISTERED CELLS TOGETHER #
#############################
#%%all data
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]

#%%all data
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]

D = pairwise_distances(embGPre)
gPreNoiseIdx = filter_noisy_outliers(embGPre,D=D)
csignalGPre = signalGPre[~gPreNoiseIdx,:]
cembGPre = embGPre[~gPreNoiseIdx,:]
cposGPre = posPre[~gPreNoiseIdx,:]
cdirGPre = dirPre[~gPreNoiseIdx]
ctimeGPre = timePre[~gPreNoiseIdx]

D = pairwise_distances(embRPre)
rPreNoiseIdx = filter_noisy_outliers(embRPre,D=D)
csignalRPre = signalRPre[~rPreNoiseIdx,:]
cembRPre = embRPre[~rPreNoiseIdx,:]
cposRPre = posPre[~rPreNoiseIdx,:]
cdirRPre = dirPre[~rPreNoiseIdx]
ctimeRPre = timePre[~rPreNoiseIdx]

D = pairwise_distances(embGRot)
gRotNoiseIdx = filter_noisy_outliers(embGRot,D=D)
csignalGRot = signalGRot[~gRotNoiseIdx,:]
cembGRot = embGRot[~gRotNoiseIdx,:]
cposGRot = posRot[~gRotNoiseIdx,:]
cdirGRot = dirRot[~gRotNoiseIdx]
ctimeGRot = timeRot[~gRotNoiseIdx]

D = pairwise_distances(embRRot)
rRotNoiseIdx = filter_noisy_outliers(embRRot,D=D)
csignalRRot = signalRRot[~rRotNoiseIdx,:]
cembRRot = embRRot[~rRotNoiseIdx,:]
cposRRot = posRot[~rRotNoiseIdx,:]
cdirRRot = dirRot[~rRotNoiseIdx]
ctimeRRot = timeRot[~rRotNoiseIdx]

dirColorGPre = np.zeros((cdirGPre.shape[0],3))
for point in range(cdirGPre.shape[0]):
    if cdirGPre[point]==0:
        dirColorGPre[point] = [14/255,14/255,143/255]
    elif cdirGPre[point]==1:
        dirColorGPre[point] = [12/255,136/255,249/255]
    else:
        dirColorGPre[point] = [17/255,219/255,224/255]

dirColorRPre = np.zeros((cdirRPre.shape[0],3))
for point in range(cdirRPre.shape[0]):
    if cdirRPre[point]==0:
        dirColorRPre[point] = [14/255,14/255,143/255]
    elif cdirRPre[point]==1:
        dirColorRPre[point] = [12/255,136/255,249/255]
    else:
        dirColorRPre[point] = [17/255,219/255,224/255]

dirColorGRot = np.zeros((cdirGRot.shape[0],3))
for point in range(cdirGRot.shape[0]):
    if cdirGRot[point]==0:
        dirColorGRot[point] = [14/255,14/255,143/255]
    elif cdirGRot[point]==1:
        dirColorGRot[point] = [12/255,136/255,249/255]
    else:
        dirColorGRot[point] = [17/255,219/255,224/255]

dirColorRRot = np.zeros((cdirRRot.shape[0],3))
for point in range(cdirRRot.shape[0]):
    if cdirRRot[point]==0:
        dirColorRRot[point] = [14/255,14/255,143/255]
    elif cdirRRot[point]==1:
        dirColorRRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRRot[point] = [17/255,219/255,224/255]



viewAngleGreen = [22, -45]
viewAngleRed = [20,15]


fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembGPre[0,:3].T, c = cposGPre[0,0], cmap = 'inferno',s = 10)
ax.view_init(viewAngleGreen[0], viewAngleGreen[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_empty.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_empty.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color = 'b',s = 10)
ax.scatter(*cembGRot[:,:3].T, color = 'r',s = 10)
ax.view_init(viewAngleGreen[0], viewAngleGreen[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_preRot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_preRot.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], cmap = 'inferno',s = 10)
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], cmap = 'inferno',s = 10)
ax.view_init(viewAngleGreen[0], viewAngleGreen[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_pos.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color = dirColorGPre, s = 10)
ax.scatter(*cembGRot[:,:3].T, color = dirColorGRot, s = 10)
ax.view_init(viewAngleGreen[0], viewAngleGreen[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_dir.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_dir.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = ctimeGPre, cmap = 'YlGn_r',s = 10)
ax.scatter(*cembGRot[:,:3].T, c = ctimeGRot, cmap = 'YlGn_r',s = 10)
ax.view_init(viewAngleGreen[0], viewAngleGreen[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_time.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Green_umapEmb_time.png'), dpi = 400,bbox_inches="tight")




fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembRPre[0,:3].T, c = cposRPre[0,0], cmap = 'inferno',s = 10)
ax.view_init(viewAngleRed[0], viewAngleRed[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_empty.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_empty.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color = 'b',s = 10)
ax.scatter(*cembRRot[:,:3].T, color = 'r',s = 10)
ax.view_init(viewAngleRed[0], viewAngleRed[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_preRot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_preRot.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], cmap = 'inferno',s = 10)
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], cmap = 'inferno',s = 10)
ax.view_init(viewAngleRed[0], viewAngleRed[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_pos.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color = dirColorRPre, s = 10)
ax.scatter(*cembRRot[:,:3].T, color = dirColorRRot, s = 10)
ax.view_init(viewAngleRed[0], viewAngleRed[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_dir.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_dir.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=((6,6)))
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = ctimeRPre, cmap = 'YlGn_r',s = 10)
ax.scatter(*cembRRot[:,:3].T, c = ctimeRRot, cmap = 'YlGn_r',s = 10)
ax.view_init(viewAngleRed[0], viewAngleRed[1])
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_time.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,f'G23_Red_umapEmb_time.png'), dpi = 400,bbox_inches="tight")




plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.view_init(25,-30)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title('Green')
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')
ax.view_init(25,-30)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color=dirColorGPre, s=10)
ax.scatter(*cembGRot[:,:3].T, color=dirColorGRot, s=10)
ax.view_init(25,-30)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.view_init(30,-20)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title('Red')
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
ax.view_init(30,-20)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color=dirColorRPre, s=10)
ax.scatter(*cembRRot[:,:3].T, color=dirColorRRot, s=10)
ax.view_init(30,-20)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.suptitle(f'G23 Reg Cells - Together {velTh}')
plt.tight_layout()

plt.savefig(os.path.join(dataDir,'G23_manifold_plot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'G23_manifold_plot.png'), dpi = 400,bbox_inches="tight")
#compute centroids
centGPre, centGRot = get_centroids(cembGPre, cembGRot, cposGPre[:,0], cposGRot[:,0], 
                                                cdirGPre, cdirGRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midGPre = np.median(centGPre, axis=0).reshape(-1,1)
midGRot = np.median(centGRot, axis=0).reshape(-1,1)
normVector =  midGPre - midGRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centGPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centGPre-midGPre.T, centGRot-midGRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotGAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Green: {rotGAngle:2f} degrees")


#compute centroids
centRPre, centRRot = get_centroids(cembRPre, cembRRot, cposRPre[:,0], cposRRot[:,0], 
                                                cdirRPre, cdirRRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midRPre = np.median(centRPre, axis=0).reshape(-1,1)
midRRot = np.median(centRRot, axis=0).reshape(-1,1)
normVector =  midRPre - midRRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centRPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centRPre-midRPre.T, centRRot-midRRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotRAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Red: {rotRAngle:2f} degrees")

mouseDict = {
    'greenNamePre': greenNamePre,
    'redNamePre': redNamePre,
    'greenNameRot': greenNameRot,
    'redNameRot': redNameRot,
    'posNamePre': posNamePre,
    'posNameRot': posNameRot,
    'signalGPre':signalGPre,
    'signalGRot': signalGRot,
    'signalRPre': signalRPre,
    'signalRRot': signalRRot,
    'posPre': posPre,
    'posRot': posRot,
    'velPre': velPre,
    'velRot': velRot,
    'dirPre': dirPre,
    'dirRot': dirRot,
    'preGreenCells': preGreenCells,
    'preRedCells': preRedCells,
    'rotGreenCells': rotGreenCells,
    'rotRedCells': rotRedCells,
    'embGPre': embGPre,
    'embGRot': embGRot,
    'embRPre': embRPre,
    'embRRot': embRRot,

    'gPreNoiseIdx': gPreNoiseIdx,
    'cembGPre': cembGPre,
    'cposGPre': cposGPre,
    'cdirGPre': cdirGPre,

    'rPreNoiseIdx': rPreNoiseIdx,
    'cembRPre': cembRPre,
    'cposRPre': cposRPre,
    'cdirRPre': cdirRPre,

    'gRotNoiseIdx': gRotNoiseIdx,
    'cembGRot': cembGRot,
    'cposGRot': cposGRot,
    'cdirGRot': cdirGRot,

    'rRotNoiseIdx': rRotNoiseIdx,
    'cembRRot': cembRRot,
    'cposRRot': cposRRot,
    'cdirRRot': cdirRRot,

    'params': {
        'nNeigh': nNeigh,
        'velTh': velTh,
        'dim': dim
    }

}
with open(os.path.join(dataDir, "ThyG23_data_dict.pkl"), "wb") as file:
    pickle.dump(mouseDict, file, protocol=pickle.HIGHEST_PROTOCOL)
###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################      G21 CONCATENATED      #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################


greenName = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/concatenated/Inscopix_data/ThyG21_ltm_green_raw.csv'
redName = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/concatenated/Inscopix_data/ThyG21_ltm_red_raw.csv'

posNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_lt_position.mat'
posNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_position.mat'


######################
#     LOAD SIGNAL    #
######################
signalG = pd.read_csv(greenName).to_numpy()[1:,1:].astype(np.float64)
signalR = pd.read_csv(redName).to_numpy()[1:,1:].astype(np.float64)

signalGPre = signalG[:12035,:]
signalGRot = signalG[12035:,:]
signalRPre = signalR[12036:,:]
signalRRot = signalR[12036:,:]


######################
#      LOAD POS      #
######################
posPre = scipy.io.loadmat(posNamePre)['Position']
posPre = posPre[::2,:]/10
posRot = scipy.io.loadmat(posNameRot)['Position']
posRot = posRot[::2,:]/10

######################
#     DELETE NANs    #
######################
nanIdx = np.where(np.sum(np.isnan(signalGPre),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRPre),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posPre),axis=1)>0)[0]),axis=0)
signalGPre = np.delete(signalGPre,nanIdx, axis=0)
signalRPre = np.delete(signalRPre,nanIdx, axis=0)
posPre = np.delete(posPre,nanIdx, axis=0)

nanIdx = np.where(np.sum(np.isnan(signalGRot),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRRot),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posRot),axis=1)>0)[0]),axis=0)
signalGRot = np.delete(signalGRot,nanIdx, axis=0)
signalRRot = np.delete(signalRRot,nanIdx, axis=0)
posRot = np.delete(posRot,nanIdx, axis=0)

######################
#    MATCH LENGTH    #
######################
if signalGPre.shape[0]>signalRPre.shape[0]:
    signalGPre = signalGPre[:signalRPre.shape[0],:]
elif signalRPre.shape[0]>signalGPre.shape[0]:
    signalRPre = signalRPre[:signalGPre.shape[0],:]

if posPre.shape[0]>signalGPre.shape[0]:
    posPre = posPre[:signalGPre.shape[0],:]
else:
    signalGPre = signalGPre[:posPre.shape[0],:]
    signalRPre = signalRPre[:posPre.shape[0],:]

if signalGRot.shape[0]>signalRRot.shape[0]:
    signalGRot = signalGRot[:signalRRot.shape[0],:]
elif signalRRot.shape[0]>signalGRot.shape[0]:
    signalRRot = signalRRot[:signalGRot.shape[0],:]

if posRot.shape[0]>signalGRot.shape[0]:
    posRot = posRot[:signalGRot.shape[0],:]
else:
    signalGRot = signalGRot[:posRot.shape[0],:]
    signalRRot = signalRRot[:posRot.shape[0],:]


######################
#     COMPUTE DIR    #
######################
velPre = np.diff(posPre[:,0]).reshape(-1,1)*10
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
dirPre = np.zeros((velPre.shape))
dirPre[velPre>0] = 1
dirPre[velPre<0] = 2

velRot = np.diff(posRot[:,0]).reshape(-1,1)*10
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
dirRot = np.zeros((velRot.shape))
dirRot[velRot>0] = 1
dirRot[velRot<0] = 2

######################
#     COMPUTE VEL    #
######################
velPre = np.abs(np.diff(posPre[:,0]).reshape(-1,1)*10)
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)


velRot = np.abs(np.diff(posRot[:,0]).reshape(-1,1)*10)
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
velRot = gaussian_filter1d(velRot, sigma = 5, axis = 0)

######################
#  DELETE LOW SPEED  #
######################
lowSpeedIdxPre = np.where(velPre<2.5)[0]
signalGPre = np.delete(signalGPre,lowSpeedIdxPre, axis=0)
signalRPre = np.delete(signalRPre,lowSpeedIdxPre, axis=0)
posPre = np.delete(posPre,lowSpeedIdxPre, axis=0)
velPre = np.delete(velPre,lowSpeedIdxPre, axis=0)
dirPre = np.delete(dirPre,lowSpeedIdxPre, axis=0)


lowSpeedIdxRot = np.where(velRot<2.5)[0]
signalGRot = np.delete(signalGRot,lowSpeedIdxRot, axis=0)
signalRRot = np.delete(signalRRot,lowSpeedIdxRot, axis=0)
posRot = np.delete(posRot,lowSpeedIdxRot, axis=0)
velRot = np.delete(velRot,lowSpeedIdxRot, axis=0)
dirRot = np.delete(dirRot,lowSpeedIdxRot, axis=0)

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])
timeRot = np.arange(posRot.shape[0])

######################
#    CLEAN TRACES    #
######################
signalGPre = clean_traces(signalGPre)
signalRPre = clean_traces(signalRPre)
signalGRot = clean_traces(signalGRot)
signalRRot = clean_traces(signalRRot)


#############################
# REGISTERED CELLS TOGETHER #
#############################
#%%all data
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]

#%%all data
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]


D = pairwise_distances(embGPre)
noiseIdx = filter_noisy_outliers(embGPre,D=D)
cembGPre = embGPre[~noiseIdx,:]
cposGPre = posPre[~noiseIdx,:]
cdirGPre = dirPre[~noiseIdx]

D = pairwise_distances(embRPre)
noiseIdx = filter_noisy_outliers(embRPre,D=D)
cembRPre = embRPre[~noiseIdx,:]
cposRPre = posPre[~noiseIdx,:]
cdirRPre = dirPre[~noiseIdx]

D = pairwise_distances(embGRot)
noiseIdx = filter_noisy_outliers(embGRot,D=D)
cembGRot = embGRot[~noiseIdx,:]
cposGRot = posRot[~noiseIdx,:]
cdirGRot = dirRot[~noiseIdx]

D = pairwise_distances(embRRot)
noiseIdx = filter_noisy_outliers(embRRot,D=D)
cembRRot = embRRot[~noiseIdx,:]
cposRRot = posRot[~noiseIdx,:]
cdirRRot = dirRot[~noiseIdx]

dirColorGPre = np.zeros((cdirGPre.shape[0],3))
for point in range(cdirGPre.shape[0]):
    if cdirGPre[point]==0:
        dirColorGPre[point] = [14/255,14/255,143/255]
    elif cdirGPre[point]==1:
        dirColorGPre[point] = [12/255,136/255,249/255]
    else:
        dirColorGPre[point] = [17/255,219/255,224/255]

dirColorRPre = np.zeros((cdirRPre.shape[0],3))
for point in range(cdirRPre.shape[0]):
    if cdirRPre[point]==0:
        dirColorRPre[point] = [14/255,14/255,143/255]
    elif cdirRPre[point]==1:
        dirColorRPre[point] = [12/255,136/255,249/255]
    else:
        dirColorRPre[point] = [17/255,219/255,224/255]

dirColorGRot = np.zeros((cdirGRot.shape[0],3))
for point in range(cdirGRot.shape[0]):
    if cdirGRot[point]==0:
        dirColorGRot[point] = [14/255,14/255,143/255]
    elif cdirGRot[point]==1:
        dirColorGRot[point] = [12/255,136/255,249/255]
    else:
        dirColorGRot[point] = [17/255,219/255,224/255]

dirColorRRot = np.zeros((cdirRRot.shape[0],3))
for point in range(cdirRRot.shape[0]):
    if cdirRRot[point]==0:
        dirColorRRot[point] = [14/255,14/255,143/255]
    elif cdirRRot[point]==1:
        dirColorRRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRRot[point] = [17/255,219/255,224/255]


plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.set_title('Green')
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color=dirColorGPre, s=10)
ax.scatter(*cembGRot[:,:3].T, color=dirColorGRot, s=10)
ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color=dirColorRPre, s=10)
ax.scatter(*cembRRot[:,:3].T, color=dirColorRRot, s=10)


#compute centroids
centGPre, centGRot = get_centroids(cembGPre, cembGRot, cposGPre[:,0], cposGRot[:,0], 
                                                cdirGPre, cdirGRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midGPre = np.median(centGPre, axis=0).reshape(-1,1)
midGRot = np.median(centGRot, axis=0).reshape(-1,1)
normVector =  midGPre - midGRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centGPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centGPre-midGPre.T, centGRot-midGRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotGAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Green: {rotGAngle:2f} degrees")


#compute centroids
centRPre, centRRot = get_centroids(cembRPre, cembRRot, cposRPre[:,0], cposRRot[:,0], 
                                                cdirRPre, cdirRRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midRPre = np.median(centRPre, axis=0).reshape(-1,1)
midRRot = np.median(centRRot, axis=0).reshape(-1,1)
normVector =  midRPre - midRRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centRPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centRPre-midRPre.T, centRRot-midRRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotRAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Red: {rotRAngle:2f} degrees")



###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

import scipy, umap, copy, os, pickle
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from neural_manifold import place_cells as pc

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


######################
#       PARAMS       #
######################
nNeigh = 120
dim = 3
velTh = 0

signalName = '/home/julio/Documents/SP_project/LT_DualColor/data/CalbGRIN1/Inscopix_data/CalbGrin1_raw.csv'
posNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/CalbGRIN1/Inscopix_data/spit/CalbGRIN1_lt_position.mat'
posNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/CalbGRIN1/Inscopix_data/spit/CalbGRIN1_rot_position.mat'

######################
#     LOAD SIGNAL    #
######################
signalBoth = pd.read_csv(signalName).to_numpy()[1:,1:].astype(np.float64)
signalPre = signalBoth[:24179,:]
######################
#      LOAD POS      #
######################
posPre = scipy.io.loadmat(posNamePre)['Position']/10
######################
#     DELETE NANs    #
######################
nanIdx = np.where(np.sum(np.isnan(signalPre),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posPre),axis=1)>0)[0]),axis=0)
signalPre = np.delete(signalPre,nanIdx, axis=0)
posPre = np.delete(posPre,nanIdx, axis=0)

######################
#    MATCH LENGTH    #
######################
if posPre.shape[0]>signalPre.shape[0]:
    posPre = posPre[:signalPre.shape[0],:]
else:
    signalPre = signalPre[:posPre.shape[0],:]

######################
#     COMPUTE DIR    #
######################
velPre = np.diff(posPre[:,0]).reshape(-1,1)*10
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)
dirPre = np.zeros((velPre.shape))
dirPre[velPre>0.3] = 1
dirPre[velPre<-0.3] = 2

######################
#     COMPUTE VEL    #
######################
velPre = np.abs(np.diff(posPre[:,0]).reshape(-1,1)*10)
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)
velPre = velPre - np.percentile(velPre,5)
velPre[velPre<0] = 0
######################
#  DELETE LOW SPEED  #
######################
lowSpeedIdxPre = np.where(velPre<velTh)[0]
signalPre = np.delete(signalPre,lowSpeedIdxPre, axis=0)
posPre = np.delete(posPre,lowSpeedIdxPre, axis=0)
velPre = np.delete(velPre,lowSpeedIdxPre, axis=0)
dirPre = np.delete(dirPre,lowSpeedIdxPre, axis=0)

######################
#  DELETE NON-DIR  #
######################
noDirIdx = np.where(dirPre==0)[0]
signalPre = np.delete(signalPre,noDirIdx, axis=0)
posPre = np.delete(posPre,noDirIdx, axis=0)
velPre = np.delete(velPre,noDirIdx, axis=0)
dirPre = np.delete(dirPre,noDirIdx, axis=0)

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])

######################
#    CLEAN TRACES    #
######################
signalPre = clean_traces(signalPre)

############################
#  ALL CELLS INDIVIDUAL    #
############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalPre)
embPre = model.transform(signalPre)

dirColorPre = np.zeros((dirPre.shape[0],3))
for point in range(dirPre.shape[0]):
    if dirPre[point]==0:
        dirColorPre[point] = [14/255,14/255,143/255]
    elif dirPre[point]==1:
        dirColorPre[point] = [12/255,136/255,249/255]
    else:
        dirColorPre[point] = [17/255,219/255,224/255]

plt.figure()
ax = plt.subplot(1,3,1, projection = '3d')
ax.scatter(*embPre[:,:3].T, color = dirColorPre, s=10)
ax = plt.subplot(1,3,2, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(1,3,3, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Pre')

############################
#        PLACE CELLS       #
############################
saveDir = '/home/julio/Documents/SP_project/LT_DualColor/place_cells/'

params = {
    'sF': 20,
    'bin_width': 5,
    'std_pos': 0,
    'std_pdf': 5,
    'method': 'spatial_info',
    'num_shuffles': 1000,
    'min_shift': 10,
    'th_metric': 99,
    'ignore_edges':5
    }

toKeep = np.logical_and(dirPre[:,0]>0,dirPre[:,0]<=2)
cposPre = posPre[toKeep,:] 
cvelPre = velPre[toKeep] 
csignalPre = signalPre[toKeep,:] 
cdirPre = dirPre[toKeep,:] 

mousePC = pc.get_place_cells(cposPre, csignalPre, vel_signal = cvelPre, dim = 1,
                      direction_signal = cdirPre, mouse = 'CalbGRIN1', save_dir = saveDir, **params)


print('\tNum place cells:')
num_cells = signalPre.shape[1]
num_place_cells = np.sum(mousePC['place_cells_dir'][:,0]*(mousePC['place_cells_dir'][:,1]==0))
print(f'\t\t Only left cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
num_place_cells = np.sum(mousePC['place_cells_dir'][:,1]*(mousePC['place_cells_dir'][:,0]==0))
print(f'\t\t Only right cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
num_place_cells = np.sum(mousePC['place_cells_dir'][:,0]*mousePC['place_cells_dir'][:,1])
print(f'\t\t Both dir cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
with open(os.path.join(saveDir,'CalbGRIN1_pc_dict.pkl'), 'wb') as f:
    pickle.dump(mousePC, f)


############################
#     GET TAGGED CELLS     #
############################
#deepCells = [16,19,24,27,58,59,61,68,69,74,81,88,93,107,109,118,119,120,121,130,135,137,141,146,148,149,150,169,170,174,177,180,184,191,194,195,198,204,234,239,241,243,245,266,271,272,273,274,287,294,303,305,306,309,317,316,336,349]
deepCells = [9,29,42,48,49,52,67,77,78,80,87,103,107,109,110,112,127,136,141,146,148,155,158,162,168,200,202,211,214,217,219,225,234,238,250,261,265,274,276,283,286,296,312,316,322,324,326,329,330,359,362]

deepCells = [x-1 for x in deepCells]
supCells = [x for x in range(signalPre.shape[1]) if x not in deepCells]

#load place cells
neuPDF = mousePC['neu_pdf']
normNeuPDF = np.zeros(neuPDF.shape)
nCells = neuPDF.shape[1]
for d in range(neuPDF.shape[2]):
    for c in range(nCells):
        normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.max(neuPDF[:,c,d])
deepPDF = normNeuPDF[:,deepCells,:]
supPDF = normNeuPDF[:, supCells, :]

meanDeepPDF = np.nanmean(deepPDF, axis=1)
meanSupPDF = np.nanmean(supPDF, axis=1)


deepOrderLeft =  np.argsort(np.argmax(deepPDF[:,:,0], axis=0))
deepOrderRight =  np.argsort(np.argmax(deepPDF[:,:,1], axis=0))
supOrderLeft =  np.argsort(np.argmax(supPDF[:,:,0], axis=0))
supOrderRight =  np.argsort(np.argmax(supPDF[:,:,1], axis=0))

mapAxis = mousePC['mapAxis']
deepManifoldSignal = np.zeros((embPre.shape[0]))*np.nan
supManifoldSignal = np.zeros((embPre.shape[0]))*np.nan
for p in range(embPre.shape[0]):
    try:
        x = np.where(mapAxis[0]<=posPre[p,0])[0][-1]
    except: 
        x = 0
    dire = int(dirPre[p,0])
    if dire==0:
        deepManifoldSignal[p] = np.nan
        supManifoldSignal[p] = np.nan
    else:
        deepManifoldSignal[p] = meanDeepPDF[x,dire-1]
        supManifoldSignal[p] = meanSupPDF[x,dire-1]

nDeepCells = len(deepCells)
nSupCells = len(supCells)


fig = plt.figure(figsize=(8,12))
ax = plt.subplot(2,2,1)
ax.matshow(deepPDF[:,deepOrderLeft,0].T, aspect = 'auto')
histSignalLeft = nDeepCells - 0.5*nDeepCells*(meanDeepPDF[:,0]/np.max(meanDeepPDF[:,0]))
ax.plot(histSignalLeft, color = 'white', linewidth = 5)
ax.set_title('izq Deep')
ax.set_ylabel('cell number')
ax.set_xlabel('pos-x')

ax = plt.subplot(2,2,2)
ax.matshow(deepPDF[:,deepOrderRight,1].T, aspect = 'auto')
histSignalRight = nDeepCells - 0.5*nDeepCells*(meanDeepPDF[:,1]/np.max(meanDeepPDF[:,1]))
ax.plot(histSignalRight, color = 'white', linewidth = 3)
ax.set_title('dcha Deep')
ax.set_ylabel('cell number')
ax.set_xlabel('pos-x')

ax = plt.subplot(2,2,3, projection = '3d')
b = ax.scatter(*embPre[:,:3].T, c = posPre[:,0],s = 30, cmap = 'inferno')
cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax = plt.subplot(2,2,4, projection = '3d')
b = ax.scatter(*embPre[:,:3].T, c = deepManifoldSignal,s = 30, vmin=0.3, vmax=0.6)
cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])




fig = plt.figure(figsize=(8,12))
ax = plt.subplot(2,2,1)
ax.matshow(supPDF[:,supOrderLeft,0].T, aspect = 'auto')
histSignalLeft = nSupCells - 0.5*nSupCells*(meanSupPDF[:,0]/np.max(meanSupPDF[:,0]))
ax.plot(histSignalLeft, color = 'white', linewidth = 5)
ax.set_title('izq Sup')
ax.set_ylabel('cell number')
ax.set_xlabel('pos-x')

ax = plt.subplot(2,2,2)
ax.matshow(supPDF[:,supOrderRight,1].T, aspect = 'auto')
histSignalRight = nSupCells - 0.5*nSupCells*(meanSupPDF[:,1]/np.max(meanSupPDF[:,1]))
ax.plot(histSignalRight, color = 'white', linewidth = 3)
ax.set_title('dcha Sup')
ax.set_ylabel('cell number')
ax.set_xlabel('pos-x')

ax = plt.subplot(2,2,3, projection = '3d')
b = ax.scatter(*embPre[:,:3].T, c = posPre[:,0],s = 30, cmap = 'inferno')
cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax = plt.subplot(2,2,4, projection = '3d')
b = ax.scatter(*embPre[:,:3].T, c = supManifoldSignal,s = 30, vmin=0.3, vmax=0.6)
cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])




plt.tight_layout()
fig.suptitle(mouse)
plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=(8,12))
ax = plt.subplot(2,2,1)
ax.matshow(normNeuPDF[:,orderLeft,0].T, aspect = 'auto')
histSignalLeft = nCells - 0.5*nCells*(meanNormNeuPDF[:,0]/np.max(meanNormNeuPDF[:,0]))
ax.plot(histSignalLeft, color = 'white', linewidth = 5)
ax.set_title('izq')
ax.set_ylabel('cell number')
ax.set_xlabel('pos-x')

ax = plt.subplot(2,2,2)
ax.matshow(normNeuPDF[:,orderRight,1].T, aspect = 'auto')
histSignalRight = nCells - 0.5*nCells*(meanNormNeuPDF[:,1]/np.max(meanNormNeuPDF[:,1]))
ax.plot(histSignalRight, color = 'white', linewidth = 3)
ax.set_title('dcha')
ax.set_ylabel('cell number')
ax.set_xlabel('pos-x')

histSignal = np.nanmean(np.squeeze(normNeuPDF.reshape(normNeuPDF.shape[0],-1,1)),axis=1)
ax = plt.subplot(2,1,2)
ax.plot(mapAxis[0][:,0], histSignal, linewidth=2)
ax.set_xlabel('pos-x')
ax.set_ylabel('histogram')
plt.tight_layout()
fig.suptitle(mouse)
plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells_histogram.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells_histogram.png'), dpi = 400,bbox_inches="tight")



###############################
# REGISTERED CELLS INDIVIDUAL #
###############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)

plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'Reg Cells - Individual {velTh}')

#############################
# REGISTERED CELLS TOGETHER #
#############################
#%%all data
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]

#%%all data
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]


D = pairwise_distances(embGPre)
noiseIdx = filter_noisy_outliers(embGPre,D=D)
csignalGPre = signalGPre[~noiseIdx,:]
cembGPre = embGPre[~noiseIdx,:]
cposGPre = posPre[~noiseIdx,:]
cdirGPre = dirPre[~noiseIdx]


D = pairwise_distances(embRPre)
noiseIdx = filter_noisy_outliers(embRPre,D=D)
csignalRPre = signalRPre[~noiseIdx,:]
cembRPre = embRPre[~noiseIdx,:]
cposRPre = posPre[~noiseIdx,:]
cdirRPre = dirPre[~noiseIdx]

D = pairwise_distances(embGRot)
noiseIdx = filter_noisy_outliers(embGRot,D=D)
csignalGRot = signalGRot[~noiseIdx,:]
cembGRot = embGRot[~noiseIdx,:]
cposGRot = posRot[~noiseIdx,:]
cdirGRot = dirRot[~noiseIdx]

D = pairwise_distances(embRRot)
noiseIdx = filter_noisy_outliers(embRRot,D=D)
csignalRRot = signalRRot[~noiseIdx,:]
cembRRot = embRRot[~noiseIdx,:]
cposRRot = posRot[~noiseIdx,:]
cdirRRot = dirRot[~noiseIdx]


dirColorGPre = np.zeros((cdirGPre.shape[0],3))
for point in range(cdirGPre.shape[0]):
    if cdirGPre[point]==0:
        dirColorGPre[point] = [14/255,14/255,143/255]
    elif cdirGPre[point]==1:
        dirColorGPre[point] = [12/255,136/255,249/255]
    else:
        dirColorGPre[point] = [17/255,219/255,224/255]

dirColorRPre = np.zeros((cdirRPre.shape[0],3))
for point in range(cdirRPre.shape[0]):
    if cdirRPre[point]==0:
        dirColorRPre[point] = [14/255,14/255,143/255]
    elif cdirRPre[point]==1:
        dirColorRPre[point] = [12/255,136/255,249/255]
    else:
        dirColorRPre[point] = [17/255,219/255,224/255]

dirColorGRot = np.zeros((cdirGRot.shape[0],3))
for point in range(cdirGRot.shape[0]):
    if cdirGRot[point]==0:
        dirColorGRot[point] = [14/255,14/255,143/255]
    elif cdirGRot[point]==1:
        dirColorGRot[point] = [12/255,136/255,249/255]
    else:
        dirColorGRot[point] = [17/255,219/255,224/255]

dirColorRRot = np.zeros((cdirRRot.shape[0],3))
for point in range(cdirRRot.shape[0]):
    if cdirRRot[point]==0:
        dirColorRRot[point] = [14/255,14/255,143/255]
    elif cdirRRot[point]==1:
        dirColorRRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRRot[point] = [17/255,219/255,224/255]


plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.set_title('Green')
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color=dirColorGPre, s=10)
ax.scatter(*cembGRot[:,:3].T, color=dirColorGRot, s=10)


ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color=dirColorRPre, s=10)
ax.scatter(*cembRRot[:,:3].T, color=dirColorRRot, s=10)


#compute centroids
centGPre, centGRot = get_centroids(cembGPre, cembGRot, cposGPre[:,0], cposGRot[:,0], 
                                                cdirGPre, cdirGRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midGPre = np.median(centGPre, axis=0).reshape(-1,1)
midGRot = np.median(centGRot, axis=0).reshape(-1,1)
normVector =  midGPre - midGRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centGPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centGPre-midGPre.T, centGRot-midGRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotGAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Green: {rotGAngle:2f} degrees")


#compute centroids
centRPre, centRRot = get_centroids(cembRPre, cembRRot, cposRPre[:,0], cposRRot[:,0], 
                                                cdirRPre, cdirRRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midRPre = np.median(centRPre, axis=0).reshape(-1,1)
midRRot = np.median(centRRot, axis=0).reshape(-1,1)
normVector =  midRPre - midRRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centRPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centRPre-midRPre.T, centRRot-midRRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotRAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Red: {rotRAngle:2f} degrees")


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import copy
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


######################
#       PARAMS       #
######################
nNeigh = 120
dim = 3
velTh = 0

signalName = '/home/julio/Documents/SP_project/LT_DualColor/data/CalbGRIN1/Inscopix_data/CalbGrin1_raw.csv'
posNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/CalbGRIN1/Inscopix_data/spit/CalbGRIN1_lt_position.mat'
posNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/CalbGRIN1/Inscopix_data/spit/CalbGRIN1_rot_position.mat'

######################
#     LOAD SIGNAL    #
######################
signalBoth = pd.read_csv(signalName).to_numpy()[1:,1:].astype(np.float64)

signalPre = signalBoth[:24179,:]
signalRot = signalBoth[24179:,:]

######################
#      LOAD POS      #
######################
posPre = scipy.io.loadmat(posNamePre)['Position']/10
posRot = scipy.io.loadmat(posNameRot)['Position']/10

######################
#     DELETE NANs    #
######################
nanIdx = np.where(np.sum(np.isnan(signalPre),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posPre),axis=1)>0)[0]),axis=0)
signalPre = np.delete(signalPre,nanIdx, axis=0)
posPre = np.delete(posPre,nanIdx, axis=0)

nanIdx = np.where(np.sum(np.isnan(signalRot),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posRot),axis=1)>0)[0]),axis=0)
signalRot = np.delete(signalRot,nanIdx, axis=0)
posRot = np.delete(posRot,nanIdx, axis=0)

######################
#    MATCH LENGTH    #
######################
if posPre.shape[0]>signalPre.shape[0]:
    posPre = posPre[:signalPre.shape[0],:]
else:
    signalPre = signalPre[:posPre.shape[0],:]


if posRot.shape[0]>signalRot.shape[0]:
    posRot = posRot[:signalRot.shape[0],:]
else:
    signalRot = signalRot[:posRot.shape[0],:]

######################
#     COMPUTE DIR    #
######################
velPre = np.diff(posPre[:,0]).reshape(-1,1)*10
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
dirPre = np.zeros((velPre.shape))
dirPre[velPre>0] = 1
dirPre[velPre<0] = 2

velRot = np.diff(posRot[:,0]).reshape(-1,1)*10
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
dirRot = np.zeros((velRot.shape))
dirRot[velRot>0] = 1
dirRot[velRot<0] = 2

######################
#     COMPUTE VEL    #
######################
velPre = np.abs(np.diff(posPre[:,0]).reshape(-1,1)*10)
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)


velRot = np.abs(np.diff(posRot[:,0]).reshape(-1,1)*10)
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
velRot = gaussian_filter1d(velRot, sigma = 5, axis = 0)

######################
#  DELETE LOW SPEED  #
######################
lowSpeedIdxPre = np.where(velPre<velTh)[0]
signalPre = np.delete(signalPre,lowSpeedIdxPre, axis=0)
posPre = np.delete(posPre,lowSpeedIdxPre, axis=0)
velPre = np.delete(velPre,lowSpeedIdxPre, axis=0)
dirPre = np.delete(dirPre,lowSpeedIdxPre, axis=0)


lowSpeedIdxRot = np.where(velRot<velTh)[0]
signalRot = np.delete(signalRot,lowSpeedIdxRot, axis=0)
posRot = np.delete(posRot,lowSpeedIdxRot, axis=0)
velRot = np.delete(velRot,lowSpeedIdxRot, axis=0)
dirRot = np.delete(dirRot,lowSpeedIdxRot, axis=0)

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])
timeRot = np.arange(posRot.shape[0])

######################
#    CLEAN TRACES    #
######################
signalGPre = clean_traces(signalPre)
signalGRot = clean_traces(signalRot)

############################
#  ALL CELLS INDIVIDUAL    #
############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalPre)
embPre = model.transform(signalPre)
#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRot)
embRot = model.transform(signalRot)

dirColorPre = np.zeros((dirPre.shape[0],3))
for point in range(dirPre.shape[0]):
    if dirPre[point]==0:
        dirColorPre[point] = [14/255,14/255,143/255]
    elif dirPre[point]==1:
        dirColorPre[point] = [12/255,136/255,249/255]
    else:
        dirColorPre[point] = [17/255,219/255,224/255]

dirColorRot = np.zeros((dirRot.shape[0],3))
for point in range(dirRot.shape[0]):
    if dirRot[point]==0:
        dirColorRot[point] = [14/255,14/255,143/255]
    elif dirRot[point]==1:
        dirColorRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRot[point] = [17/255,219/255,224/255]


plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*embPre[:,:3].T, color = dirColorPre, s=10)
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Pre')
ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*embRot[:,:3].T, color = dirColorRot, s=10)
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Rot')
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*embRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')


############################
#     GET TAGGED CELLS     #
############################
preGreenCells = [1,2,3,6,7,12,13,15,16,17,19,20,21,22,23,25,26,27,29,30,32,33,34,35,37,39,41,43,44,46,47,48,50,51,52,55,66,67,68,69,73,75,76,77,78,79,80,82,84,86,87,89,91,92,93,94,97,98,99,100,101,102,103,106,108,110,111,112,114,115,117,119,120,122,123,124,125,126,127,128,129,130,131,133,134,136,137,138,139,141,145,146,147,148,149,151,152,153,155,156,157,158,165,166,167,170,171,172,173,174,175,176,177,178,179,180,182,183,185,189,190,191,192,193,194,195,196,197,200,202,206,208,209,210,211,214,216,218,219,224,225,226,227,228, 229,231,232,234,235,236,239,240,248,249,251,254,258,259,261,262,266,267,273,277,278,279,282,284,285,288,290,293]
preGreenCells = [x-1 for x in preGreenCells]
preRedCells = [2,3,4,6,7,10,12,13,18,22,23,24,25,27,28,29,30,31,32,34,36,37,40,41,42,43,45,46,47,49,50,51,53,54,55,56,57,58,61,62,63,64,65,67,68,69,70,72,74,75,79,81,82,83,84,85,86,87,88,89,94,95,97,98,99,100,101,102,103,105,107,109,112,113,114,117,118,119,120,121,122,124,125,126,127,128,129,130,131,133,135,137,138,139,140,141,142,143,144,145,146,149,152,153,154,155,157,158,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,182,184,185,186,187,188,191,192,196,197,198,199,200,201,203,204,206,207,208,210,211,212,214,215,216,217,218,219,220,221,225,226,227,232,233,234,235,236,237,238,239,240,241,242,244,245,251,252,257,258,259,260,261,262,263,264,266,269,270,271,275,276,277,278,279,280,281,282,284,285,286,287,290,291,292,293,294,295,296,297,299,302,303,304,305,306,307,308,309,310,312,313,314,315,316,319,320,321,324,325,326,329,330,331,332,334,336,337,343,344,346,347,348,350,351,353,354,355,357,361,364,366,367,368,369,375,376,378,379,381,382,383,384,387,389,390,393,395,402,405,406,407,409,410,412,413,415,416,418,422,426,427,432,433,434,435,436,437,438,439,440,441,442,443,445,447,449,450,454,455,457,460,461,462,463,464,465,466,468,471,475,484,486]
preRedCells = [x-1 for x in preRedCells]

rotRedCells = [6,7,0,1,2,3,4,8,9,10,17,24,16,28,12,11,15,13,18,22,23,29,32,21,27,30,5,34,36,37,40,39,38,41,42,48,43,20,51,47,158,46,44,57,56,54,170,63,65,64,59,60,61,55,58,62,53,68,71,69,73,72,207,86,67,89,74,77,83,85,82,80,76,88,91,75,94,96,95,92,93,97,125,115,101,100,90,99,103,236,111,113,109,116,105,106,104,108,121,114,122,119,107,120,117,123,126,130,128,133,259,159,146,127,135,141,150,154,138,151,152,134,147,144,131,137,140,160,149,153,142,129,156,136,161,157,145,165,163,164,162,277,167,174,173,172,186,176,168,171,182,181,184,179,169,178,180,177,188,187,183,192,185,166,190,367,301,195,200,198,197,196,199,307,201,202,204,205,219,209,206,208,214,212,215,217,216,210,220,228,242,221,247,223,224,244,230,249,315,226,229,234,245,232,235,241,240,227,238,237,246,368,231,233,252,251,263,250,276,274,270,264,254,256,269,258,261,273,260,266,255,262,268,265,222,267,275,253,279,281,283,282,280,278,291,290,284,288,293,287,300,285,286,298,294,299,303,304,311,302,305,309,363,306,364,310,314,313,329,322,318,317,323,327,319,326,320,328,324,321,312,316,351,330,350,331,332,344,337,343,339,335,333,338,345,342,346,349,336,355,356,360,361,352,357,362,354,353,365,132,257]
rotRedCells=[x-1 for x in rotRedCells]
rotGreenCells = [2,0,4,45,6,1,3,5,8,18,13,11,15,16,10,19,12,20,23,22,24,21,26,27,30,25,37,101,32,36,39,33,35,34,38,40,43,123,42,47,46,50,54,56,61,66,59,73,63,60,138,64,221,71,58,70,57,77,74,87,79,85,80,225,78,76,89,148,113,88,86,83,82,75,103,96,90,97,99,100,94,92,98,95,106,104,110,220,105,112,108,93,191,120,179,115,116,114,186,118,224,119,223,121,124,133,122,128,127,129,126,134,135,143,137,144,139,159,140,136,153,146,149,147,151,152,154,158,160,157,150,166,174,164,162,168,169,172,167,165,171,178,163,161,183,180,189,182,181,188,184,187,185,192,190,193,195,194,201,199,206,210,205,200,208,212,209,213,216,218,68,203]
rotGreenCells = [x-1 for x in rotGreenCells]

signalGPre = signalGPre[:,preGreenCells]
signalRPre = signalRPre[:, preRedCells]
signalGRot = signalGRot[:,rotGreenCells]
signalRRot = signalRRot[:, rotRedCells]

###############################
# REGISTERED CELLS INDIVIDUAL #
###############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)

plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'Reg Cells - Individual {velTh}')

#############################
# REGISTERED CELLS TOGETHER #
#############################
#%%all data
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]

#%%all data
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]


D = pairwise_distances(embGPre)
noiseIdx = filter_noisy_outliers(embGPre,D=D)
csignalGPre = signalGPre[~noiseIdx,:]
cembGPre = embGPre[~noiseIdx,:]
cposGPre = posPre[~noiseIdx,:]
cdirGPre = dirPre[~noiseIdx]


D = pairwise_distances(embRPre)
noiseIdx = filter_noisy_outliers(embRPre,D=D)
csignalRPre = signalRPre[~noiseIdx,:]
cembRPre = embRPre[~noiseIdx,:]
cposRPre = posPre[~noiseIdx,:]
cdirRPre = dirPre[~noiseIdx]

D = pairwise_distances(embGRot)
noiseIdx = filter_noisy_outliers(embGRot,D=D)
csignalGRot = signalGRot[~noiseIdx,:]
cembGRot = embGRot[~noiseIdx,:]
cposGRot = posRot[~noiseIdx,:]
cdirGRot = dirRot[~noiseIdx]

D = pairwise_distances(embRRot)
noiseIdx = filter_noisy_outliers(embRRot,D=D)
csignalRRot = signalRRot[~noiseIdx,:]
cembRRot = embRRot[~noiseIdx,:]
cposRRot = posRot[~noiseIdx,:]
cdirRRot = dirRot[~noiseIdx]


dirColorGPre = np.zeros((cdirGPre.shape[0],3))
for point in range(cdirGPre.shape[0]):
    if cdirGPre[point]==0:
        dirColorGPre[point] = [14/255,14/255,143/255]
    elif cdirGPre[point]==1:
        dirColorGPre[point] = [12/255,136/255,249/255]
    else:
        dirColorGPre[point] = [17/255,219/255,224/255]

dirColorRPre = np.zeros((cdirRPre.shape[0],3))
for point in range(cdirRPre.shape[0]):
    if cdirRPre[point]==0:
        dirColorRPre[point] = [14/255,14/255,143/255]
    elif cdirRPre[point]==1:
        dirColorRPre[point] = [12/255,136/255,249/255]
    else:
        dirColorRPre[point] = [17/255,219/255,224/255]

dirColorGRot = np.zeros((cdirGRot.shape[0],3))
for point in range(cdirGRot.shape[0]):
    if cdirGRot[point]==0:
        dirColorGRot[point] = [14/255,14/255,143/255]
    elif cdirGRot[point]==1:
        dirColorGRot[point] = [12/255,136/255,249/255]
    else:
        dirColorGRot[point] = [17/255,219/255,224/255]

dirColorRRot = np.zeros((cdirRRot.shape[0],3))
for point in range(cdirRRot.shape[0]):
    if cdirRRot[point]==0:
        dirColorRRot[point] = [14/255,14/255,143/255]
    elif cdirRRot[point]==1:
        dirColorRRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRRot[point] = [17/255,219/255,224/255]


plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.set_title('Green')
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color=dirColorGPre, s=10)
ax.scatter(*cembGRot[:,:3].T, color=dirColorGRot, s=10)


ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color=dirColorRPre, s=10)
ax.scatter(*cembRRot[:,:3].T, color=dirColorRRot, s=10)


#compute centroids
centGPre, centGRot = get_centroids(cembGPre, cembGRot, cposGPre[:,0], cposGRot[:,0], 
                                                cdirGPre, cdirGRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midGPre = np.median(centGPre, axis=0).reshape(-1,1)
midGRot = np.median(centGRot, axis=0).reshape(-1,1)
normVector =  midGPre - midGRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centGPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centGPre-midGPre.T, centGRot-midGRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotGAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Green: {rotGAngle:2f} degrees")


#compute centroids
centRPre, centRRot = get_centroids(cembRPre, cembRRot, cposRPre[:,0], cposRRot[:,0], 
                                                cdirRPre, cdirRRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midRPre = np.median(centRPre, axis=0).reshape(-1,1)
midRRot = np.median(centRRot, axis=0).reshape(-1,1)
normVector =  midRPre - midRRot
normVector = normVector/np.linalg.norm(normVector)
k = np.dot(np.median(centRPre, axis=0), normVector)

angles = np.linspace(-np.pi,np.pi,100)
error = find_rotation(centRPre-midRPre.T, centRRot-midRRot.T, normVector)
normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
rotRAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
print(f"Red: {rotRAngle:2f} degrees")