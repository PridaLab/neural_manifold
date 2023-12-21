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
import neural_manifold.general_utils as gu


def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

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

def get_neuronal_fields(trial_data, ref_field=None):
    """
    Identify time-varying fields in the dataset
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    neuronal_fields : list of str
        list of fieldnames that store time-varying signals
    """
    if ref_field is None:
        # look for a spikes field
        ref_field = [col for col in trial_data.columns.values
                     if col.endswith("spikes") or col.endswith("rates")][0]

    # identify candidates based on the first trial
    first_trial = trial_data.iloc[0]
    T = first_trial[ref_field].shape[1]
    neuronal_fields = []
    for col in first_trial.index:
        try:
            if first_trial[col].shape[1] == T:
                neuronal_fields.append(col)
        except:
            pass

    # but check the rest of the trials, too
    ref_lengths = np.array([arr.shape[1] for arr in trial_data[ref_field]])
    for col in neuronal_fields:
        col_lengths = np.array([arr.shape[1] for arr in trial_data[col]])
        assert np.all(col_lengths == ref_lengths), f"not all lengths in {col} match the reference {ref_field}"

    return neuronal_fields

def add_dir_mat_field(pd_struct):
    out_pd = copy.deepcopy(pd_struct)
    if 'dir_mat' not in out_pd.columns:
        out_pd["dir_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            ('L' == out_pd["dir"][idx])+ 2*('R' == out_pd["dir"][idx])+
                            4*('F' in out_pd["dir"][idx]) for idx in out_pd.index]
    return out_pd

def preprocess_traces(pd_struct_p, pd_struct_r, signal_field, save_signal_field, sigma = 5,sig_up = 4, sig_down = 12, peak_th=0.1):
    out_pd_p = copy.deepcopy(pd_struct_p)
    out_pd_r = copy.deepcopy(pd_struct_r)

    out_pd_p["index_mat"] = [np.zeros((out_pd_p[signal_field][idx].shape[0],1))+out_pd_p["trial_id"][idx] 
                                  for idx in range(out_pd_p.shape[0])]                     
    index_mat_p = np.concatenate(out_pd_p["index_mat"].values, axis=0)

    out_pd_r["index_mat"] = [np.zeros((out_pd_r[signal_field][idx].shape[0],1))+out_pd_r["trial_id"][idx] 
                                  for idx in range(out_pd_r.shape[0])]
    index_mat_r = np.concatenate(out_pd_r["index_mat"].values, axis=0)

    signal_p_og = copy.deepcopy(np.concatenate(pd_struct_p[signal_field].values, axis=0))
    lowpass_p = uniform_filter1d(signal_p_og, size = 4000, axis = 0)
    signal_p = gaussian_filter1d(signal_p_og, sigma = sigma, axis = 0)

    signal_r_og = copy.deepcopy(np.concatenate(pd_struct_r[signal_field].values, axis=0))
    lowpass_r = uniform_filter1d(signal_r_og, size = 4000, axis = 0)
    signal_r = gaussian_filter1d(signal_r_og, sigma = sigma, axis = 0)

    for nn in range(signal_p.shape[1]):
        base_p = np.histogram(signal_p_og[:,nn], 100)
        base_p = base_p[1][np.argmax(base_p[0])]
        base_p = base_p + lowpass_p[:,nn] - np.min(lowpass_p[:,nn]) 

        base_r = np.histogram(signal_r_og[:,nn], 100)
        base_r = base_r[1][np.argmax(base_r[0])]   
        base_r = base_r + lowpass_r[:,nn] - np.min(lowpass_r[:,nn])   

        concat_signal = np.concatenate((signal_p[:,nn]-base_p, signal_r[:,nn]-base_r))

        concat_signal = concat_signal/np.max(concat_signal,axis = 0)
        concat_signal[concat_signal<0] = 0
        signal_p[:,nn] = concat_signal[:signal_p.shape[0]]
        signal_r[:,nn] = concat_signal[signal_p.shape[0]:]

    bi_signal_p = np.zeros(signal_p.shape)
    bi_signal_r = np.zeros(signal_r.shape)

    gaus = lambda x,sig,amp,vo: amp*np.exp(-(((x)**2)/(2*sig**2)))+vo;

    x = np.arange(-5*sig_down, 5*sig_down,1);
    gaus_up = gaus(x,sig_up, 1, 0); 
    gaus_up[5*sig_down+1:] = 0
    gaus_down = gaus(x,sig_down, 1, 0); 
    gaus_down[:5*sig_down+1] = 0
    gaus_final = gaus_down + gaus_up;

    for nn in range(signal_p.shape[1]):
        peaks_p,_ =find_peaks(signal_p[:,nn],height=peak_th)
        bi_signal_p[peaks_p, nn] = signal_p[peaks_p, nn]
        if gaus_final.shape[0]<signal_p.shape[0]:
            bi_signal_p[:, nn] = np.convolve(bi_signal_p[:, nn],gaus_final, 'same')

        peaks_r,_ =find_peaks(signal_r[:,nn],height=peak_th)
        bi_signal_r[peaks_r, nn] = signal_r[peaks_r, nn]
        if gaus_final.shape[0]<signal_r.shape[0]:
            bi_signal_r[:, nn] = np.convolve(bi_signal_r[:, nn],gaus_final, 'same')


    out_pd_p[save_signal_field] = [bi_signal_p[index_mat_p[:,0]==out_pd_p["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_p.shape[0])]
    out_pd_r[save_signal_field] = [bi_signal_r[index_mat_r[:,0]==out_pd_r["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_r.shape[0])]
    return out_pd_p, out_pd_r


velTh = 2
sigma = 6
sigUp = 4
sigDown = 12
nNeigh = 120
dim = 3


miceList = ['ThyCalbRCaMP2']
baseLoadDir = '/home/julio/Documents/SP_project/LT_DualColor/data/'
baseSaveDir = '/home/julio/Documents/SP_project/LT_DualColor/processed_data/'
signalField = ['green_raw_traces', 'red_raw_traces']

mouse = miceList[0]


loadDir = os.path.join(baseLoadDir, mouse)
saveDir = os.path.join(baseSaveDir, mouse)
#__________________________________________________________________________
#|                                                                        |#
#|                               LOAD DATA                                |#
#|________________________________________________________________________|#
animal = gu.load_files(loadDir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")
fnames = list(animal.keys())
fnamePre = [fname for fname in fnames if 'lt' in fname][0]
fnameRot = [fname for fname in fnames if 'rot' in fname][0]
animalPre= copy.deepcopy(animal[fnamePre])
animalRot= copy.deepcopy(animal[fnameRot])

#__________________________________________________________________________
#|                                                                        |#
#|                          PREPROCESS TRACES                             |#
#|________________________________________________________________________|#
animalPre = add_dir_mat_field(animalPre)
animalRot = add_dir_mat_field(animalRot)

animalPre = gu.select_trials(animalPre,"dir == ['L','R','N']")
animalRot = gu.select_trials(animalRot,"dir == ['L','R','N']")
animalPre, _ = gu.keep_only_moving(animalPre, velTh)
animalRot, _ = gu.keep_only_moving(animalRot, velTh)
for color in ['green', 'red']:
    animalPre, animalRot = preprocess_traces(animalPre, animalRot, color+'_raw_traces', color+'_clean_traces', sigma=sigma, sig_up = sigUp, sig_down = sigDown)


signalGPre = copy.deepcopy(np.concatenate(animalPre['green_clean_traces'].values, axis=0))
signalRPre = copy.deepcopy(np.concatenate(animalPre['red_clean_traces'].values, axis=0))
posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
velPre = copy.deepcopy(np.concatenate(animalPre['vel'].values, axis=0))
dirPre =copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))

signalGRot = copy.deepcopy(np.concatenate(animalRot['green_clean_traces'].values, axis=0))
signalRRot = copy.deepcopy(np.concatenate(animalRot['red_clean_traces'].values, axis=0))
posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
velRot = copy.deepcopy(np.concatenate(animalRot['vel'].values, axis=0))
dirRot =copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])
timeRot = np.arange(posRot.shape[0])
# ######################
# #    CLEAN TRACES    #
# ######################
# signalGPre = clean_traces(signalGPre)
# signalRPre = clean_traces(signalRPre)
# signalGRot = clean_traces(signalGRot)
# signalRRot = clean_traces(signalRRot)


##########################
# PROYECT CELLS TOGETHER #
##########################
#all data green
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]


#all data red
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]

#all data both colors
signalBPre = np.hstack((signalGPre, signalRPre))
signalBRot = np.hstack((signalGRot, signalRRot))
index = np.vstack((np.zeros((signalBPre.shape[0],1)),np.ones((signalBRot.shape[0],1))))
concatSignalB = np.vstack((signalBPre, signalBRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalB)
embBoth = model.transform(concatSignalB)
embBPre = embBoth[index[:,0]==0,:]
embBRot = embBoth[index[:,0]==1,:]

#clean outliers
D = pairwise_distances(embGPre)
noiseIdxGPre = filter_noisy_outliers(embGPre,D=D)
csignalGPre = signalGPre[~noiseIdxGPre,:]
cembGPre = embGPre[~noiseIdxGPre,:]
cposGPre = posPre[~noiseIdxGPre,:]
cdirGPre = dirPre[~noiseIdxGPre]

D = pairwise_distances(embRPre)
noiseIdxRPre = filter_noisy_outliers(embRPre,D=D)
csignalRPre = signalRPre[~noiseIdxRPre,:]
cembRPre = embRPre[~noiseIdxRPre,:]
cposRPre = posPre[~noiseIdxRPre,:]
cdirRPre = dirPre[~noiseIdxRPre]

D = pairwise_distances(embBPre)
noiseIdxBPre = filter_noisy_outliers(embBPre,D=D)
csignalBPre = signalRPre[~noiseIdxRPre,:]
cembBPre = embBPre[~noiseIdxBPre,:]
cposBPre = posPre[~noiseIdxBPre,:]
cdirBPre = dirPre[~noiseIdxBPre]

D = pairwise_distances(embGRot)
noiseIdxGRot = filter_noisy_outliers(embGRot,D=D)
csignalGRot = signalGRot[~noiseIdxGRot,:]
cembGRot = embGRot[~noiseIdxGRot,:]
cposGRot = posRot[~noiseIdxGRot,:]
cdirGRot = dirRot[~noiseIdxGRot]

D = pairwise_distances(embRRot)
noiseIdxRRot = filter_noisy_outliers(embRRot,D=D)
csignalRRot = signalRRot[~noiseIdxRRot,:]
cembRRot = embRRot[~noiseIdxRRot,:]
cposRRot = posRot[~noiseIdxRRot,:]
cdirRRot = dirRot[~noiseIdxRRot]    

D = pairwise_distances(embBRot)
noiseIdxBRot = filter_noisy_outliers(embBRot,D=D)
csignalBRot = signalRRot[~noiseIdxRRot,:]
cembBRot = embBRot[~noiseIdxBRot,:]
cposBRot = posRot[~noiseIdxBRot,:]
cdirBRot = dirRot[~noiseIdxBRot]


#PLOT
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

dirColorBPre = np.zeros((cdirBPre.shape[0],3))
for point in range(cdirBPre.shape[0]):
    if cdirBPre[point]==0:
        dirColorBPre[point] = [14/255,14/255,143/255]
    elif cdirBPre[point]==1:
        dirColorBPre[point] = [12/255,136/255,249/255]
    else:
        dirColorBPre[point] = [17/255,219/255,224/255]

dirColorBRot = np.zeros((cdirBRot.shape[0],3))
for point in range(cdirBRot.shape[0]):
    if cdirBRot[point]==0:
        dirColorBRot[point] = [14/255,14/255,143/255]
    elif cdirBRot[point]==1:
        dirColorBRot[point] = [12/255,136/255,249/255]
    else:
        dirColorBRot[point] = [17/255,219/255,224/255]


plt.figure()
ax = plt.subplot(3,3,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.set_title('Green')
ax = plt.subplot(3,3,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')
ax = plt.subplot(3,3,3, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color=dirColorGPre, s=10)
ax.scatter(*cembGRot[:,:3].T, color=dirColorGRot, s=10)

ax = plt.subplot(3,3,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(3,3,5, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')
ax = plt.subplot(3,3,6, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color=dirColorRPre, s=10)
ax.scatter(*cembRRot[:,:3].T, color=dirColorRRot, s=10)

ax = plt.subplot(3,3,7, projection = '3d')
ax.scatter(*cembBPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembBRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(3,3,8, projection = '3d')
ax.scatter(*cembBPre[:,:3].T, c = cposBPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembBRot[:,:3].T, c = cposBRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')
ax = plt.subplot(3,3,9, projection = '3d')
ax.scatter(*cembBPre[:,:3].T, color=dirColorBPre, s=10)
ax.scatter(*cembBRot[:,:3].T, color=dirColorBRot, s=10)
plt.suptitle(f"{mouse}")


# plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb.svg'), dpi = 400,bbox_inches="tight")
# plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb.png'), dpi = 400,bbox_inches="tight")



#compute centroids
centGPre, centGRot = get_centroids(cembGPre, cembGRot, cposGPre[:,0], cposGRot[:,0], 
                                                cdirGPre, cdirGRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midGPre = np.median(centGPre, axis=0).reshape(-1,1)
midGRot = np.median(centGRot, axis=0).reshape(-1,1)
normVectorG =  midGPre - midGRot
normVectorG = normVectorG/np.linalg.norm(normVectorG)
k = np.dot(np.median(centGPre, axis=0), normVectorG)

anglesG = np.linspace(-np.pi,np.pi,100)
errorG = find_rotation(centGPre-midGPre.T, centGRot-midGRot.T, normVectorG)
normErrorG = (np.array(errorG)-np.min(errorG))/(np.max(errorG)-np.min(errorG))
rotGAngle = np.abs(anglesG[np.argmin(normErrorG)])*180/np.pi
print(f"\tGreen: {rotGAngle:2f} degrees")


#compute centroids
centRPre, centRRot = get_centroids(cembRPre, cembRRot, cposRPre[:,0], cposRRot[:,0], 
                                                 cdirRPre, cdirRRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midRPre = np.median(centRPre, axis=0).reshape(-1,1)
midRRot = np.median(centRRot, axis=0).reshape(-1,1)
normVectorR =  midRPre - midRRot
normVectorR = normVectorR/np.linalg.norm(normVectorR)
k = np.dot(np.median(centRPre, axis=0), normVectorR)

anglesR = np.linspace(-np.pi,np.pi,100)
errorR = find_rotation(centRPre-midRPre.T, centRRot-midRRot.T, normVectorR)
normErrorR = (np.array(errorR)-np.min(errorR))/(np.max(errorR)-np.min(errorR))
rotRAngle = np.abs(anglesR[np.argmin(normErrorR)])*180/np.pi
print(f"\tRed: {rotRAngle:2f} degrees")


#compute centroids
centBPre, centBRot = get_centroids(cembBPre, cembBRot, cposBPre[:,0], cposBRot[:,0], 
                                                 cdirBPre, cdirBRot, ndims = 3, nCentroids=40)   
#find axis of rotatio                                                
midBPre = np.median(centBPre, axis=0).reshape(-1,1)
midBRot = np.median(centBRot, axis=0).reshape(-1,1)
normVectorB =  midBPre - midBRot
normVectorB = normVectorB/np.linalg.norm(normVectorB)
k = np.dot(np.median(centBPre, axis=0), normVectorB)

anglesB = np.linspace(-np.pi,np.pi,100)
errorB = find_rotation(centBPre-midBPre.T, centBRot-midBRot.T, normVectorB)
normErrorB = (np.array(errorB)-np.min(errorB))/(np.max(errorB)-np.min(errorB))
rotBAngle = np.abs(anglesB[np.argmin(normErrorR)])*180/np.pi
print(f"\tBoth: {rotBAngle:2f} degrees")


plt.figure()
plt.plot(anglesG, normErrorG, 'r')
plt.plot(anglesR, normErrorR, 'g')



indexMatPre = np.concatenate(animalPre['index_mat'].values, axis=0)
animalPre['umap_green'] = [embGPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                                for idx in animalPre.index]
animalPre['umap_red'] = [embRPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                                for idx in animalPre.index]
animalPre['umap_both'] = [embBPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                                for idx in animalPre.index]
animalPre['both_clean_traces'] = [signalBPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                                for idx in animalPre.index]


indexMatRot = np.concatenate(animalRot['index_mat'].values, axis=0)
animalRot['umap_green'] = [embGRot[indexMatRot[:,0]==animalRot["trial_id"][idx] ,:] 
                                                for idx in animalRot.index]
animalRot['umap_red'] = [embRRot[indexMatRot[:,0]==animalRot["trial_id"][idx] ,:] 
                                                for idx in animalRot.index]
animalRot['umap_both'] = [embBRot[indexMatRot[:,0]==animalRot["trial_id"][idx] ,:] 
                                                for idx in animalRot.index]
animalRot['both_clean_traces'] = [signalBRot[indexMatRot[:,0]==animalRot["trial_id"][idx] ,:] 
                                                for idx in animalRot.index]



mouseSaveDir = os.path.join(baseSaveDir, mouse)
save_df = open(os.path.join(mouseSaveDir, mouse+ "_df_pre_dict.pkl"), "wb")
pickle.dump(animalPre, save_df)
save_df.close()


save_df = open(os.path.join(mouseSaveDir, mouse+ "_df_rot_dict.pkl"), "wb")
pickle.dump(animalRot, save_df)
save_df.close()

aligment = {
    'centGPre': centGPre,
    'centGRot': centGRot,
    'midGPre': midGPre,
    'midGRot': midGRot,
    'normGVector': normVectorG,
    'anglesG': anglesG,
    'errorG': errorG,
    'normErrorG': normErrorR,
    'rotGAngle': rotGAngle,

    'centRPre': centRPre,
    'centRRot': centRRot,
    'midRPre': midRPre,
    'midRRot': midRRot,
    'normRVector': normVectorR,
    'anglesR': anglesR,
    'errorR': errorR,
    'normErrorR': normErrorR,
    'rotRAngle': rotRAngle,

    'centBPre': centBPre,
    'centBRot': centBRot,
    'midBPre': midBPre,
    'midBRot': midBRot,
    'normBVector': normVectorB,
    'anglesB': anglesB,
    'errorB': errorB,
    'normErrorB': normErrorB,
    'rotBAngle': rotBAngle,

    'noiseIdxGPre': noiseIdxGPre,
    'csignalGPre': csignalGPre,
    'cembGPre': cembGPre,
    'cposGPre': cposGPre,
    'cdirGPre': cdirGPre,

    'noiseIdxRPre': noiseIdxRPre,
    'csignalRPre': csignalRPre,
    'cembRPre': cembRPre,
    'cposRPre': cposRPre,
    'cdirRPre': cdirRPre,

    'noiseIdxBPre': noiseIdxBPre,
    'csignalBPre': csignalBPre,
    'cembBPre': cembBPre,
    'cposBPre': cposBPre,
    'cdirBPre': cdirBPre,

    'noiseIdxGRot': noiseIdxGRot,
    'csignalGRot': csignalGRot,
    'cembGRot': cembGRot,
    'cposGRot': cposGRot,
    'cdirGRot': cdirGRot,

    'noiseIdxRRot': noiseIdxRRot,
    'csignalRRot': csignalRRot,
    'cembRRot': cembRRot,
    'cposRRot': cposRRot,
    'cdirRRot': cdirRRot,

    'noiseIdxBRot': noiseIdxBRot,
    'csignalBRot': csignalBRot,
    'cembBRot': cembBRot,
    'cposBRot': cposBRot,
    'cdirBRot': cdirBRot,

}

save_df = open(os.path.join(mouseSaveDir, mouse+ "_alignment_dict.pkl"), "wb")
pickle.dump(aligment, save_df)
save_df.close()