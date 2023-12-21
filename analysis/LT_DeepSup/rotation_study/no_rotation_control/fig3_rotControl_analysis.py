import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neural_manifold.general_utils as gu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from neural_manifold import dimensionality_reduction as dim_red
import seaborn as sns
from neural_manifold.dimensionality_reduction import validation as dim_validation 
import umap
from kneed import KneeLocator
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from structure_index import compute_structure_index, draw_graph
from neural_manifold import decoders as dec
from datetime import datetime
from neural_manifold import place_cells as pc
from scipy import stats
import skdim
import time
from scipy.stats import pearsonr

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

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

def preprocess_traces(pd_struct_p, pd_struct_r, signalField, sigma = 5,sigUp = 4, sigDown = 12, peak_th=0.1):
    out_pd_p = copy.deepcopy(pd_struct_p)
    out_pd_r = copy.deepcopy(pd_struct_r)

    out_pd_p["index_mat"] = [np.zeros((out_pd_p[signalField][idx].shape[0],1))+out_pd_p["trial_id"][idx] 
                                  for idx in range(out_pd_p.shape[0])]                     
    index_mat_p = np.concatenate(out_pd_p["index_mat"].values, axis=0)

    out_pd_r["index_mat"] = [np.zeros((out_pd_r[signalField][idx].shape[0],1))+out_pd_r["trial_id"][idx] 
                                  for idx in range(out_pd_r.shape[0])]
    index_mat_r = np.concatenate(out_pd_r["index_mat"].values, axis=0)

    signalPre_og = copy.deepcopy(np.concatenate(pd_struct_p[signalField].values, axis=0))
    lowpass_p = uniform_filter1d(signalPre_og, size = 4000, axis = 0)
    signalPre = gaussian_filter1d(signalPre_og, sigma = sigma, axis = 0)

    signalRot_og = copy.deepcopy(np.concatenate(pd_struct_r[signalField].values, axis=0))
    lowpass_r = uniform_filter1d(signalRot_og, size = 4000, axis = 0)
    signalRot = gaussian_filter1d(signalRot_og, sigma = sigma, axis = 0)

    for nn in range(signalPre.shape[1]):
        base_p = np.histogram(signalPre_og[:,nn], 100)
        base_p = base_p[1][np.argmax(base_p[0])]
        base_p = base_p + lowpass_p[:,nn] - np.min(lowpass_p[:,nn]) 

        base_r = np.histogram(signalRot_og[:,nn], 100)
        base_r = base_r[1][np.argmax(base_r[0])]   
        base_r = base_r + lowpass_r[:,nn] - np.min(lowpass_r[:,nn])   

        concat_signal = np.concatenate((signalPre[:,nn]-base_p, signalRot[:,nn]-base_r))

        concat_signal = concat_signal/np.max(concat_signal,axis = 0)
        concat_signal[concat_signal<0] = 0
        signalPre[:,nn] = concat_signal[:signalPre.shape[0]]
        signalRot[:,nn] = concat_signal[signalPre.shape[0]:]

    bi_signalPre = np.zeros(signalPre.shape)
    bi_signalRot = np.zeros(signalRot.shape)

    gaus = lambda x,sig,amp,vo: amp*np.exp(-(((x)**2)/(2*sig**2)))+vo;

    x = np.arange(-5*sigDown, 5*sigDown,1);
    gaus_up = gaus(x,sigUp, 1, 0); 
    gaus_up[5*sigDown+1:] = 0
    gaus_down = gaus(x,sigDown, 1, 0); 
    gaus_down[:5*sigDown+1] = 0
    gaus_final = gaus_down + gaus_up;

    for nn in range(signalPre.shape[1]):
        peaks_p,_ =find_peaks(signalPre[:,nn],height=peak_th)
        bi_signalPre[peaks_p, nn] = signalPre[peaks_p, nn]
        if gaus_final.shape[0]<signalPre.shape[0]:
            bi_signalPre[:, nn] = np.convolve(bi_signalPre[:, nn],gaus_final, 'same')

        peaks_r,_ =find_peaks(signalRot[:,nn],height=peak_th)
        bi_signalRot[peaks_r, nn] = signalRot[peaks_r, nn]
        if gaus_final.shape[0]<signalRot.shape[0]:
            bi_signalRot[:, nn] = np.convolve(bi_signalRot[:, nn],gaus_final, 'same')


    out_pd_p['clean_traces'] = [bi_signalPre[index_mat_p[:,0]==out_pd_p["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_p.shape[0])]
    out_pd_r['clean_traces'] = [bi_signalRot[index_mat_r[:,0]==out_pd_r["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_r.shape[0])]
    return out_pd_p, out_pd_r

def remove_cells(pd_struct, cells_to_keep):
    out_pd = copy.deepcopy(pd_struct)

    out_pd["index_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            out_pd["trial_id"][idx]
                            for idx in out_pd.index]

    index_mat = np.concatenate(out_pd["index_mat"].values, axis=0)
    neuro_fields = get_neuronal_fields(out_pd, 'raw_traces')
    for field in neuro_fields:
        signal = np.concatenate(out_pd[field].values, axis=0)
        out_pd[field] = [signal[index_mat[:,0]==out_pd["trial_id"][idx]][:, cells_to_keep] 
                                                for idx in range(out_pd.shape[0])]
    return out_pd

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx

def compute_entanglement(points):
    distance_b = pairwise_distances(points)
    model_iso = Isomap(n_neighbors = 10, n_components = 1)
    emb = model_iso.fit_transform(points)
    distance_a = model_iso.dist_matrix_
    entanglement_boundary = np.max(distance_a[1:,0])/np.min(distance_b[1:,0])
    entanglement = np.max((distance_a[1:,0]/distance_b[1:,0]))
    return (entanglement-1)/entanglement_boundary

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

#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#

miceList = ['ChZ4', 'GC2']
baseloadDir = '/home/julio/Documents/SP_project/Fig3_rotControl/data/'
basesaveDir = '/home/julio/Documents/SP_project/Fig3_rotControl/processed_data/'
signalField = 'raw_traces'
velTh = 6
sigma = 6
sigUp = 4
sigDown = 12
nNeigh = 120
dim = 3

for mouse in miceList:
    print(f"Working on mouse: {mouse}")
    loadDir = os.path.join(baseloadDir, mouse)
    saveDir = os.path.join(basesaveDir, mouse)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    saveDirFig = os.path.join(saveDir, 'figures')
    if not os.path.exists(saveDirFig):
        os.mkdir(saveDirFig)
    now = datetime.now()
    params = {
        "date": now.strftime("%d/%m/%Y %H:%M:%S"),
        "mouse": mouse,
        "loadDir": loadDir,
        "saveDir": saveDir,
        "signalField": signalField,
        "velTh": velTh,
        "sigma": sigma,
        "sigUp": sigUp,
        "sigDown": sigDown
    }
    #__________________________________________________________________________
    #|                                                                        |#
    #|                               LOAD DATA                                |#
    #|________________________________________________________________________|#
    animal = gu.load_files(loadDir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")
    fnames = list(animal.keys())

    fnamePre = [fname for fname in fnames if 'ltpre' in fname][0]
    fnamePost = [fname for fname in fnames if 'ltpost' in fname][0]

    animalPre= copy.deepcopy(animal[fnamePre])
    animalPost= copy.deepcopy(animal[fnamePost])

    #__________________________________________________________________________
    #|                                                                        |#
    #|                          PREPROCESS TRACES                             |#
    #|________________________________________________________________________|#
    animalPre = add_dir_mat_field(animalPre)
    animalPost = add_dir_mat_field(animalPost)

    animalPre = gu.select_trials(animalPre,"dir == ['L','R','N']")
    animalPost = gu.select_trials(animalPost,"dir == ['L','R','N']")
    animalPre, animalPre_still = gu.keep_only_moving(animalPre, velTh)
    animalPost, animalPost_still = gu.keep_only_moving(animalPost, velTh)

    animalPre, animalPost = preprocess_traces(animalPre, animalPost, signalField, sigma=sigma, sigUp = sigUp, sigDown = sigDown)
    animalPre_still, animalPost_still = preprocess_traces(animalPre_still, animalPost_still, signalField, sigma=sigma, sigUp = sigUp, sigDown = sigDown)


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                PLOT UMAP                               |#
    #|________________________________________________________________________|#
    signalPre = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis=0))
    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    signalPost = copy.deepcopy(np.concatenate(animalPost['clean_traces'].values, axis=0))
    posPost = copy.deepcopy(np.concatenate(animalPost['pos'].values, axis=0))
    #%%all data
    index = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalPost.shape[0],1))+1))
    concat_signal = np.vstack((signalPre, signalPost))
    model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
    # model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
    model.fit(concat_signal)
    concat_emb = model.transform(concat_signal)
    embPre = concat_emb[index[:,0]==0,:]
    embPost = concat_emb[index[:,0]==1,:]

    #%%
    fig = plt.figure()
    ax = plt.subplot(1,2,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*embPost[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax.set_title('All')
    ax = plt.subplot(1,2,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*embPost[:,:3].T, c = posPost[:,0], s= 30, cmap = 'magma')
    plt.suptitle(f"{mouse}: clean_traces - vel: {velTh} - nn: {nNeigh} - dim: {dim}")
    plt.savefig(os.path.join(saveDirFig,mouse+'_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                  SAVE                                  |#
    #|________________________________________________________________________|#
    with open(os.path.join(saveDir, mouse+"_params.pkl"), "wb") as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

    animal_dict = {
        fnamePre: animalPre,
        fnamePost: animalPost
    }
    with open(os.path.join(saveDir, mouse+"_df_dict.pkl"), "wb") as file:
        pickle.dump(animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    animal_still_dict = {
        fnamePre: animalPre_still,
        fnamePost: animalPost_still
    }
    with open(os.path.join(saveDir, mouse+"_df_still_dict.pkl"), "wb") as file:
        pickle.dump(animal_still_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                              SAVE DIM RED                              |#
#|________________________________________________________________________|#

miceList = ['ChZ4', 'GC2']
params = {
    'dim':3,
    'nNeigh': 120,
    'minDist': 0.1,
    'signalName': 'clean_traces',
}
dataDir =  '/home/julio/Documents/SP_project/Fig3_rotControl/processed_data/'

for mouse in miceList:
    dim_red_object = dict()
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'ltpre' in fname][0]
    fnamePost = [fname for fname in fnames if 'ltpost' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalPost= copy.deepcopy(animal[fnamePost])

    signalPre = np.concatenate(animalPre[params['signalName']].values, axis = 0)
    posPre = np.concatenate(animalPre['pos'].values, axis = 0)
    dirMatPre = np.concatenate(animalPre['dir_mat'].values, axis=0)
    indexMatPre = np.concatenate(animalPre['index_mat'].values, axis=0)

    signalPost = np.concatenate(animalPost[params['signalName']].values, axis = 0)
    posPost = np.concatenate(animalPost['pos'].values, axis = 0)
    dirMatPost = np.concatenate(animalPost['dir_mat'].values, axis=0)
    indexMatPost = np.concatenate(animalPost['index_mat'].values, axis=0)

    indexPrePost = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalPost.shape[0],1))+1))
    signalBoth = np.vstack((signalPre, signalPost))

    #umap
    print("\tFitting umap model...", sep= '', end = '')
    modelUmap = umap.UMAP(n_neighbors =params['nNeigh'], n_components =params['dim'], min_dist=params['minDist'])
    modelUmap.fit(signalBoth)
    embBoth = modelUmap.transform(signalBoth)
    embPre = embBoth[indexPrePost[:,0]==0,:]
    embPost = embBoth[indexPrePost[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*embPost[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*embPost[:,:3].T, c = posPost[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = dirMatPre, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    ax.scatter(*embPost[:,:3].T, c = dirMatPost, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_umap.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animalPre['umap'] = [embPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                   for idx in animalPre.index]
    animalPost['umap'] = [embPost[indexMatPost[:,0]==animalPost["trial_id"][idx] ,:] 
                                   for idx in animalPost.index]
    dim_red_object['umap'] = copy.deepcopy(modelUmap)
    print("\b\b\b: Done")

    #isomap
    print("\tFitting isomap model...", sep= '', end = '')
    modelIsomap = Isomap(n_neighbors =params['nNeigh'], n_components = signalBoth.shape[1])
    modelIsomap.fit(signalBoth)
    embBoth = modelIsomap.transform(signalBoth)
    embPre = embBoth[indexPrePost[:,0]==0,:]
    embPost = embBoth[indexPrePost[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*embPost[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*embPost[:,:3].T, c = posPost[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = dirMatPre, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    ax.scatter(*embPost[:,:3].T, c = dirMatPost, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_isomap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_isomap.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animalPre['isomap'] = [embPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                   for idx in animalPre.index]
    animalPost['isomap'] = [embPost[indexMatPost[:,0]==animalPost["trial_id"][idx] ,:] 
                                   for idx in animalPost.index]
    dim_red_object['isomap'] = copy.deepcopy(modelIsomap)
    print("\b\b\b: Done")

    #pca
    print("\tFitting PCA model...", sep= '', end = '')
    modelPCA = PCA(signalBoth.shape[1])
    modelPCA.fit(signalBoth)
    embBoth = modelPCA.transform(signalBoth)
    embPre = embBoth[indexPrePost[:,0]==0,:]
    embPost = embBoth[indexPrePost[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*embPost[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*embPost[:,:3].T, c = posPost[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = dirMatPre, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    ax.scatter(*embPost[:,:3].T, c = dirMatPost, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_PCA.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_PCA.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animalPre['pca'] = [embPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                   for idx in animalPre.index]
    animalPost['pca'] = [embPost[indexMatPost[:,0]==animalPost["trial_id"][idx] ,:] 
                                   for idx in animalPost.index]
    dim_red_object['pca'] = copy.deepcopy(modelPCA)
    print("\b\b\b: Done")

    newAnimalDict = {
        fnamePre: animalPre,
        fnamePost: animalPost
    }
    with open(os.path.join(filePath,fileName), "wb") as file:
        pickle.dump(newAnimalDict, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(filePath, mouse+"_umap_object.pkl"), "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE ROTATION                            |#
#|________________________________________________________________________|#

miceList = ['ChZ4', 'GC2']
dataDir =  '/home/julio/Documents/SP_project/Fig3_rotControl/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig3_rotControl/rotation/'

rot_error_dict = dict()
for mouse in miceList:
    rot_error_dict[mouse] = dict()
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'ltpre' in fname][0]
    fnameRot = [fname for fname in fnames if 'ltpost' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    posPre = np.concatenate(animalPre['pos'].values, axis = 0)
    dirMatPre = np.concatenate(animalPre['dir_mat'].values, axis=0)
    posRot = np.concatenate(animalRot['pos'].values, axis = 0)
    dirMatRot = np.concatenate(animalRot['dir_mat'].values, axis=0)

    for embName in ['pca','isomap','umap']:
        embPre = np.concatenate(animalPre[embName].values, axis = 0)[:,:3]
        embRot = np.concatenate(animalRot[embName].values, axis = 0)[:,:3]

        DPre = pairwise_distances(embPre)
        noiseIdxPre = filter_noisy_outliers(embPre,DPre)
        max_dist = np.nanmax(DPre)
        cembPre = embPre[~noiseIdxPre,:]
        cposPre = posPre[~noiseIdxPre,:]
        cdirMatPre = dirMatPre[~noiseIdxPre]

        DRot = pairwise_distances(embRot)
        noiseIdxRot = filter_noisy_outliers(embRot,DRot)
        max_dist = np.nanmax(DRot)
        cembRot = embRot[~noiseIdxRot,:]
        cposRot = posRot[~noiseIdxRot,:]
        cdirMatRot = dirMatRot[~noiseIdxRot]

        #compute centroids
        centPre, centRot = get_centroids(cembPre, cembRot, cposPre[:,0], cposRot[:,0], 
                                                        cdirMatPre, cdirMatRot, ndims = 3, nCentroids=40)   
        #find axis of rotatio                                                
        midPre = np.median(cembPre, axis=0).reshape(-1,1)
        midRot = np.median(cembRot, axis=0).reshape(-1,1)
        normVector =  midPre - midRot
        normVector = normVector/np.linalg.norm(normVector)
        k = np.dot(np.median(cembPre, axis=0), normVector)

        angles = np.linspace(-np.pi,np.pi,100)
        error = find_rotation(centPre-midPre.T, centRot-midRot.T, normVector)
        normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
        rotAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
        print(f"\t{embName}: {rotAngle:2f} degrees")

        rot_error_dict[mouse][embName] = {
            'embPre': embPre,
            'embRot': embRot,
            'posPre': posPre,
            'posRot': posRot,
            'dirMatPre': dirMatPre,
            'dirMatRot': dirMatRot,
            'noiseIdxPre': noiseIdxPre,
            'noiseIdxRot': noiseIdxRot,
            'centPre': centPre,
            'centRot': centRot,
            'midPre': midPre,
            'midRot': midRot,
            'normVector': normVector,
            'angles': angles,
            'error': error,
            'normError': normError,
            'rotAngle': rotAngle
        }

        with open(os.path.join(saveDir, f"rot_error_dict.pkl"), "wb") as file:
            pickle.dump(rot_error_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                     MEASURE REMMAPING DISTANCE                         |#
#|________________________________________________________________________|#

miceList = ['ChZ4', 'GC2']
dataDir =  '/home/julio/Documents/SP_project/Fig3_rotControl/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig3_rotControl/rotation/'

remapDist_dict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'ltpre' in fname][0]
    fnameRot = [fname for fname in fnames if 'ltpost' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])
    remapDist_dict[mouse] = dict()
    for emb in ['umap', 'isomap', 'pca']:
        posPre = np.concatenate(animalPre['pos'].values, axis = 0)
        dirMatPre = np.concatenate(animalPre['dir_mat'].values, axis=0)
        posRot = np.concatenate(animalRot['pos'].values, axis = 0)
        dirMatRot = np.concatenate(animalRot['dir_mat'].values, axis=0)

        embPre = np.concatenate(animalPre[emb].values, axis = 0)[:,:3]
        embRot = np.concatenate(animalRot[emb].values, axis = 0)[:,:3]

        DPre = pairwise_distances(embPre)
        noiseIdxPre = filter_noisy_outliers(embPre,DPre)
        max_dist = np.nanmax(DPre)
        cembPre = embPre[~noiseIdxPre,:]
        cposPre = posPre[~noiseIdxPre,:]
        cdirMatPre = dirMatPre[~noiseIdxPre]

        DRot = pairwise_distances(embRot)
        noiseIdxRot = filter_noisy_outliers(embRot,DRot)
        max_dist = np.nanmax(DRot)
        cembRot = embRot[~noiseIdxRot,:]
        cposRot = posRot[~noiseIdxRot,:]
        cdirMatRot = dirMatRot[~noiseIdxRot]

        #compute centroids
        centPre, centRot = get_centroids(cembPre, cembRot, cposPre[:,0], cposRot[:,0], 
                                                        cdirMatPre, cdirMatRot, ndims = 3, nCentroids=40)  


        interDist = np.mean(pairwise_distances(centPre, centRot))
        intraPre = np.percentile(pairwise_distances(centPre),95)
        intraRot = np.percentile(pairwise_distances(centRot),95)

        remapDist = interDist/np.max((intraPre, intraRot))
        print(f"Remmap Dist: {remapDist:.4f}")

        remapDist_dict[mouse][emb] = {
            'embPre': embPre,
            'embRot': embRot,
            'posPre': posPre,
            'posRot': posRot,
            'dirMatPre': dirMatPre,
            'dirMatRot': dirMatRot,
            'noiseIdxPre': noiseIdxPre,
            'noiseIdxRot': noiseIdxRot,
            'centPre': centPre,
            'centRot': centRot,
            'interDist': interDist,
            'intraPre': intraPre,
            'intraRot': intraRot,
            'remapDist': remapDist
        }

    with open(os.path.join(saveDir,'remap_distance_dict.pkl'), 'wb') as f:
        pickle.dump(remapDist_dict, f)


#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT ROTATION                               |#
#|________________________________________________________________________|#

dataDirControl = '/home/julio/Documents/SP_project/Fig3_rotControl/rotation/'
rotErrorDictControl = load_pickle(dataDirControl, 'rot_error_dict.pkl')
miceList = list(rotErrorDictControl.keys())

dataDirRotation = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'
rotErrorDictRotation = load_pickle(dataDirRotation, 'rot_error_dict.pkl')

#PLOT LINES
for embName in ['pca', 'isomap', 'umap']:
    rotErrorControl = np.zeros((100, len(miceList)))
    rotErrorRotation = np.zeros((100, len(miceList)))
    angleDeg = rotErrorDictControl[miceList[0]][embName]['angles']

    for idx, mouse in enumerate(miceList):
        rotErrorControl[:,idx] = rotErrorDictControl[mouse][embName]['normError']
        rotErrorRotation[:,idx] = rotErrorDictRotation[mouse][embName]['normError']

    plt.figure()
    ax = plt.subplot(111)
    m = np.mean(rotErrorControl,axis=1)
    sd = np.std(rotErrorControl,axis=1)
    ax.plot(angleDeg, m, color = '#32E653',label = 'Control')
    ax.fill_between(angleDeg, m-sd, m+sd, color = '#32E653', alpha = 0.3)
    m = np.mean(rotErrorRotation,axis=1)
    sd = np.std(rotErrorRotation,axis=1)
    ax.plot(angleDeg, m, color = '#E632C5', label = 'Rotation')
    ax.fill_between(angleDeg, m-sd, m+sd, color = '#E632C5', alpha = 0.3)
    ax.set_xlabel('Angle of rotation (º)')
    ax.set_ylabel('Aligment Error')
    ax.set_title(embName)
    ax.legend()
    plt.savefig(os.path.join(dataDirControl,f'control_{embName}_rotation_error.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(dataDirControl,f'control_{embName}_rotation_error.svg'), dpi = 400,bbox_inches="tight",transparent=True)

#PLOT BOXPLOTS
rotAngleList = list()
conditionList = list()
embList = list()
for mouse in miceList:
    for embName in['pca', 'isomap', 'umap']:
        rotAngleList.append(rotErrorDictControl[mouse][embName]['rotAngle'])
        embList.append(embName)
        conditionList.append('control')

        rotAngleList.append(rotErrorDictRotation[mouse][embName]['rotAngle'])
        embList.append(embName)
        conditionList.append('rotation')

anglePD = pd.DataFrame(data={'angle': rotAngleList,
                            'emb': embList,
                            'condition': conditionList})

palette= ["#32e653", "#E632C5"]
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='emb', y='angle', hue='condition', data=anglePD, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='emb', y='angle', hue = 'condition', data=anglePD, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylabel('Angle Rotation')
plt.tight_layout()

plt.savefig(os.path.join(dataDirControl,f'control_rotation_boxplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(dataDirControl,f'control_rotation_boxplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)


#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT DISTANCE                               |#
#|________________________________________________________________________|#

miceList = ['ChZ4', 'GC2']

dataDirControl = '/home/julio/Documents/SP_project/Fig3_rotControl/rotation/'
dataDirRotation = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'

remapDistControl = load_pickle(dataDirControl, 'remap_distance_dict.pkl')
remapDistRotation = load_pickle(dataDirRotation, 'remap_distance_dict.pkl')

fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, emb in enumerate(['umap','isomap','pca']):
    remapDist = list()
    conditionList = list()
    mouseList = list()
    for mouse in miceList:
        remapDist.append(remapDistControl[mouse][emb]['remapDist'])
        remapDist.append(remapDistRotation[mouse][emb]['remapDist'])
        conditionList.append('control')
        conditionList.append('rotation')
        mouseList.append(mouse)
        mouseList.append(mouse)


    # print(emb, ':', stats.ttest_ind(remapDist[[0,2]], remapDist[[1,3]], equal_var=True))
    pd_dist = pd.DataFrame(data={'condition': conditionList,
                                     'dist': remapDist,
                                     'mouse': mouseList})


    palette= ["#1EE153", "#E11EAC"]
    b = sns.barplot(x='condition', y='dist', data=pd_dist,
                palette = palette, linewidth = 1, width= .5, ax = ax[idx])
    sns.lineplot(x = 'condition', y= 'dist', data = pd_dist, units = 'mouse',
                     ax = ax[idx], estimator = None, color = ".7", markers = True)
    ax[idx].set_title(emb)
    ax[idx].set_ylim([0,3.2])
plt.tight_layout()

plt.savefig(os.path.join(dataDirControl,'remapDist.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDirControl,'remapDist.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT EMBEDDING                              |#
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

dataDir = '/home/julio/Documents/SP_project/Fig3_rotControl/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig3_rotControl/rotation/emb_example/'

view_init_dict = {
    'GC2': [25,-175],
    'ChZ4': [20, 135]
}

for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())

    fnamePre = [fname for fname in fnames if 'pre' in fname][0]
    fnameRot = [fname for fname in fnames if 'post' in fname][0]

    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
    embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

    DPre= pairwise_distances(embPre)
    noiseIdx = filter_noisy_outliers(embPre,DPre)
    embPre = embPre[~noiseIdx,:]
    posPre = posPre[~noiseIdx,:]
    dirMatPre = dirMatPre[~noiseIdx]

    DRot= pairwise_distances(embRot)
    noiseIdx = filter_noisy_outliers(embRot,DRot)
    embRot = embRot[~noiseIdx,:]
    posRot = posRot[~noiseIdx,:]
    dirMatRot = dirMatRot[~noiseIdx]

    dirColorPre = np.zeros((dirMatPre.shape[0],3))
    for point in range(dirMatPre.shape[0]):
        if dirMatPre[point]==0:
            dirColorPre[point] = [14/255,14/255,143/255]
        elif dirMatPre[point]==1:
            dirColorPre[point] = [12/255,136/255,249/255]
        else:
            dirColorPre[point] = [17/255,219/255,224/255]

    dirColorRot = np.zeros((dirMatRot.shape[0],3))
    for point in range(dirMatRot.shape[0]):
        if dirMatRot[point]==0:
            dirColorRot[point] = [14/255,14/255,143/255]
        elif dirMatRot[point]==1:
            dirColorRot[point] = [12/255,136/255,249/255]
        else:
            dirColorRot[point] = [17/255,219/255,224/255]

    if mouse in list(view_init_dict.keys()):
        view_init_values = view_init_dict[mouse]
    else:
        view_init_values = None

    fig = plt.figure(figsize=((15,5)))
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color = '#00fe42ff',s = 20)
    ax.scatter(*embRot[:,:3].T, color = 'b',s = 20)
    personalize_ax(ax,view_init_values)

    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color = dirColorPre, s = 20)
    ax.scatter(*embRot[:,:3].T, color = dirColorRot, s = 20)
    personalize_ax(ax,view_init_values)

    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap = 'inferno',s = 20)
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap = 'inferno',s = 20)
    personalize_ax(ax,view_init_values)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,f'{mouse}_control_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_control_emb.png'), dpi = 400,bbox_inches="tight")