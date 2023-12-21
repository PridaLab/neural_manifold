import sys, os, copy, pickle, timeit, random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from neural_manifold import general_utils as gu
from neural_manifold import decoders as dec
from neural_manifold import place_cells as pc
from neural_manifold import dimensionality_reduction as dim_red

from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import seaborn as sns
from kneed import KneeLocator

import umap
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

from sklearn.metrics import pairwise_distances
from structure_index import compute_structure_index, draw_graph
from statistics import mode

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

def get_centroids(input_A, label_A, dir_A = None, ndims = 2, nCentroids = 20):
    input_A = input_A[:,:ndims]
    if label_A.ndim>1:
        label_A = label_A[:,0]
    #compute label max and min to divide into centroids
    labelLimits = np.array([(np.percentile(label_A,5), np.percentile(label_A,95))]).T[:,0] 
    #find centroid size
    centSize = (labelLimits[1] - labelLimits[0]) / (nCentroids)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    centEdges = np.column_stack((np.linspace(labelLimits[0],labelLimits[0]+centSize*(nCentroids),nCentroids),
                                np.linspace(labelLimits[0],labelLimits[0]+centSize*(nCentroids),nCentroids)+centSize))

    if isinstance(dir_A, type(None)) :
        centLabel_A = np.zeros((nCentroids,ndims))
        ncentLabel_A = np.zeros((nCentroids,))
        for c in range(nCentroids):
            points_A = input_A[np.logical_and(label_A >= centEdges[c,0], label_A<centEdges[c,1]),:]
            centLabel_A[c,:] = np.median(points_A, axis=0)
            ncentLabel_A[c] = points_A.shape[0]
    else:
        input_A_left = copy.deepcopy(input_A[dir_A[:,0]==1,:])
        label_A_left = copy.deepcopy(label_A[dir_A[:,0]==1])
        input_A_right = copy.deepcopy(input_A[dir_A[:,0]==2,:])
        label_A_right = copy.deepcopy(label_A[dir_A[:,0]==2])
        
        centLabel_A = np.zeros((2*nCentroids,ndims))
        ncentLabel_A = np.zeros((2*nCentroids,))
        for c in range(nCentroids):
            points_A_left = input_A_left[np.logical_and(label_A_left >= centEdges[c,0], label_A_left<centEdges[c,1]),:]
            centLabel_A[2*c,:] = np.median(points_A_left, axis=0)
            ncentLabel_A[2*c] = points_A_left.shape[0]
            points_A_right = input_A_right[np.logical_and(label_A_right >= centEdges[c,0], label_A_right<centEdges[c,1]),:]
            centLabel_A[2*c+1,:] = np.median(points_A_right, axis=0)
            ncentLabel_A[2*c+1] = points_A_right.shape[0]

    del_cent_nan = np.all(np.isnan(centLabel_A), axis= 1)
    del_cent_num = (ncentLabel_A<20)
    del_cent = del_cent_nan + del_cent_num
    centLabel_A = np.delete(centLabel_A, del_cent, 0)
    return centLabel_A

def _create_save_folders(saveDir, mouse):
    #try creating general folder
    try:
        os.mkdir(saveDir)
    except:
        pass
    #add new folder with mouse name + current date-time
    saveDir = os.path.join(saveDir, mouse)
    #create this new folder
    try:
        os.mkdir(saveDir)
    except:
        pass
    return saveDir

def add_dir_mat_field(pdMouse):
    pdOut = copy.deepcopy(pdMouse)
    if 'dir_mat' not in pdOut.columns:
        pdOut["dir_mat"] = [np.zeros((pdOut["pos"][idx].shape[0],1)).astype(int)+
                            ('L' == pdOut["dir"][idx])+ 2*('R' == pdOut["dir"][idx])+
                            4*('F' in pdOut["dir"][idx]) for idx in pdOut.index]
    return pdOut

def add_inner_trial_time_field(pdMouse):
    pdOut = copy.deepcopy(pdMouse)
    pdOut["inner_trial_time"] = [np.arange(pdOut["pos"][idx].shape[0]).reshape(-1,1).astype(int)
                        for idx in pdOut.index]
    return pdOut

def preprocess_traces(pdMouse, signal_field, sigma = 5, sig_up = 4, sig_down = 12, peak_th=0.1):
    pdOut = copy.deepcopy(pdMouse)

    pdOut["index_mat"] = [np.zeros((pdOut[signal_field][idx].shape[0],1))+pdOut["trial_id"][idx] 
                                  for idx in range(pdOut.shape[0])]                     
    indexMat = np.concatenate(pdOut["index_mat"].values, axis=0)

    ogSignal = copy.deepcopy(np.concatenate(pdMouse[signal_field].values, axis=0))
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

    pdOut['clean_traces'] = [biSignal[indexMat[:,0]==pdOut["trial_id"][idx] ,:] 
                                                                for idx in range(pdOut.shape[0])]

    return pdOut

#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'

#%% PARAMS
sigma = 6
upSig = 4
downSig = 12
signalField = 'raw_traces'
peakTh = 0.1
velTh = 6
verbose = True

for mouse in miceList:
    #initialize time
    globalTime = timeit.default_timer()
    #create save folder data by adding time suffix
    mouseDataDir = os.path.join(dataDir, mouse)
    mouseSaveDir = _create_save_folders(saveDir, mouse)
    #check if verbose has to be saved into txt file
    if verbose:
        f = open(os.path.join(mouseSaveDir,mouse + '_logFile.txt'), 'w')
        original = sys.stdout
        sys.stdout = gu.Tee(sys.stdout, f)
    #%% 1.LOAD DATA
    localTime = timeit.default_timer()
    print('\n### 1. LOAD DATA ###')
    print('1 Searching & loading data in directory:\n', mouseDataDir)
    dfMouse = gu.load_files(mouseDataDir, '*_PyalData_struct*.mat', verbose=verbose)
    fileNames = list(dfMouse.keys())
    sessionPre = [fname for fname in fileNames if 'lt' in fname][0]
    print(f"\tSession selected: {sessionPre}")
    if verbose:
        gu.print_time_verbose(localTime, globalTime)
    #%% KEEP ONLY LAST SESSION
    pdMouse = copy.deepcopy(dfMouse[sessionPre])
    print('\n### 2. PROCESS DATA ###')
    print(f'Working on session: {sessionPre}')
    #%% 2. PROCESS DATA
    pdMouse = add_dir_mat_field(pdMouse)
    pdMouse = add_inner_trial_time_field(pdMouse)

    #2.1 keep only moving epochs
    print(f'2.1 Dividing into moving/still data w/ velTh= {velTh:.2f}.')
    if velTh>0:
        og_dur = np.concatenate(pdMouse["pos"].values, axis=0).shape[0]
        pdMouse, still_pdMouse = gu.keep_only_moving(pdMouse, velTh)
        move_dur = np.concatenate(pdMouse["pos"].values, axis=0).shape[0]
        print(f"\t{sessionPre}: Og={og_dur} ({og_dur/20}s) Move= {move_dur} ({move_dur/20}s)")
    else:
        print('2.1 Keeping all data (not limited to moving periods).')
        still_pdMouse = dict()
    #2.2 compute clean traces
    print(f"2.2 Computing clean-traces from {signalField} with sigma = {sigma}," +
        f" sigma_up = {upSig}, sigma_down = {downSig}", sep='')
    pdMouse = preprocess_traces(pdMouse, signalField, sigma = sigma, sig_up = upSig,
                            sig_down = downSig, peak_th = peakTh)
    if velTh>0:
        still_pdMouse = preprocess_traces(still_pdMouse, signalField, sigma = sigma, sig_up = upSig,
                                sig_down = downSig, peak_th = peakTh)
        save_still = open(os.path.join(mouseSaveDir, mouse+ "_still_df_dict.pkl"), "wb")
        pickle.dump(still_pdMouse, save_still)
        save_still.close()

    save_df = open(os.path.join(mouseSaveDir, mouse+ "_df_dict.pkl"), "wb")
    pickle.dump(pdMouse, save_df)
    save_df.close()
    params = {
        'sigma': sigma,
        'upSig': upSig,
        'downSig': downSig,
        'signalField': signalField,
        'peakTh': peakTh,
        'dataDir': mouseDataDir,
        'saveDir': mouseSaveDir,
        'mouse': mouse
    }
    save_params = open(os.path.join(mouseSaveDir, mouse+ "_params.pkl"), "wb")
    pickle.dump(params, save_params)
    save_params.close()
    # create list of strings
    paramsList = [ f'{key} : {params[key]}' for key in params]
    # write string one by one adding newline
    saveParamsFile = open(os.path.join(mouseSaveDir, mouse+ "_params.txt"), "w")
    with saveParamsFile as saveFile:
        [saveFile.write("%s\n" %st) for st in paramsList]
    saveParamsFile.close()
    sys.stdout = original
    f.close()

#__________________________________________________________________________
#|                                                                        |#
#|                               INNER DIM                                |#
#|________________________________________________________________________|#
import skdim
import time
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

params = {
    'signalName': 'clean_traces',
    'nNeigh': 30,
    'verbose': True
}
saveDir =  '/home/julio/Documents/SP_project/Fig2/dimensionality/inner_dim'
try:
    os.mkdir(saveDir)
except:
    pass
dataDir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'

idDict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    #compute abids dim
    abidsDim = np.nanmean(dim_red.compute_abids(signal, params['nNeigh']))
    print(f"\tABIDS: {abidsDim:.2f}", end='', flush=True)
    time.sleep(.2)

    corrIntDim = skdim.id.CorrInt(k1 = 5, k2 = params['nNeigh']).fit_transform(signal)
    print(f" | CorrInt: {corrIntDim:.2f}", end='', flush=True)
    time.sleep(.2)

    dancoDim = skdim.id.DANCo().fit_transform(signal)
    print(f" | DANCo: {dancoDim:.2f}", end='', flush=True)
    time.sleep(.2)

    essDim = skdim.id.ESS().fit_transform(signal,n_neighbors = params['nNeigh'])
    print(f" | ESS: {essDim:.2f}", end='', flush=True)
    time.sleep(.2)

    fishersDim = skdim.id.FisherS(conditional_number=5).fit_transform(signal)
    print(f" | FisherS: {fishersDim:.2f}", end='', flush=True)
    time.sleep(.2)

    knnDim = skdim.id.KNN(k=params['nNeigh']).fit_transform(signal)
    print(f" | KNN: {knnDim:.2f}", end='', flush=True)
    time.sleep(.2)

    lPCADim = skdim.id.lPCA(ver='broken_stick').fit_transform(signal)
    print(f" | lPCA: {lPCADim:.2f}", end='', flush=True)
    time.sleep(.2)

    madaDim = skdim.id.MADA().fit_transform(signal)
    print(f" | MADA: {madaDim:.2f}", end='', flush=True)
    time.sleep(.2)

    mindDim = skdim.id.MiND_ML(k=params['nNeigh']).fit_transform(signal)
    print(f" | MiND_ML: {mindDim:.2f}", end='', flush=True)
    time.sleep(.2)

    mleDim = skdim.id.MLE(K=params['nNeigh']).fit_transform(signal)
    print(f" | MLE: {mleDim:.2f}", end='', flush=True)
    time.sleep(.2)

    momDim = skdim.id.MOM().fit_transform(signal,n_neighbors = params['nNeigh'])
    print(f" | MOM: {momDim:.2f}", end='', flush=True)
    time.sleep(.2)

    tleDim = skdim.id.TLE().fit_transform(signal,n_neighbors = params['nNeigh'])
    print(f" | TLE: {tleDim:.2f}")
    time.sleep(.2)

    #save results
    idDict[mouse] = {
        'abidsDim': abidsDim,
        'corrIntDim': corrIntDim,
        'dancoDim': dancoDim,
        'essDim': essDim,
        'fishersDim': fishersDim,
        'knnDim': knnDim,
        'lPCADim': lPCADim,
        'madaDim': madaDim,
        'mindDim': mindDim,
        'mleDim': mleDim,
        'momDim': momDim,
        'tleDim': tleDim,
        'params': params
    }

    saveFile = open(os.path.join(saveDir, 'inner_dim_dict.pkl'), "wb")
    pickle.dump(idDict, saveFile)
    saveFile.close()

saveFile = open(os.path.join(saveDir, 'inner_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# create list of strings
paramsList = [ f'{key} : {params[key]}' for key in params]
# write string one by one adding newline
saveParamsFile = open(os.path.join(saveDir, "inner_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                                UMAP DIM                                |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

params = {
    'maxDim':10,
    'nNeigh': 120,
    'minDist': 0.1,
    'nnDim': 30,
    'signalName': 'clean_traces',
}

saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
umap_dim_dict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    print("Computing rank indices og space...", end = '', sep = '')
    rankIdx = dim_red.validation.compute_rank_indices(signal)
    print("\b\b\b: Done")
    trustNum = np.zeros((params['maxDim'],))
    contNum = np.zeros((params['maxDim'],))
    for dim in range(params['maxDim']):
        emb_space = np.arange(dim+1)
        print(f"Dim: {dim+1} ({dim+1}/{params['maxDim']})")
        model = umap.UMAP(n_neighbors = params['nNeigh'], n_components =dim+1, min_dist=params['minDist'])
        print("\tFitting model...", sep= '', end = '')
        emb = model.fit_transform(signal)
        print("\b\b\b: Done")
        #1. Compute trustworthiness
        print("\tComputing trustworthiness...", sep= '', end = '')
        temp = dim_red.validation.trustworthiness_vector(signal, emb, params['nnDim'], indices_source = rankIdx)
        trustNum[dim] = temp[-1]
        print(f"\b\b\b: {trustNum[dim]:.4f}")
        #2. Compute continuity
        print("\tComputing continuity...", sep= '', end = '')
        temp = dim_red.validation.continuity_vector(signal, emb ,params['nnDim'])
        contNum[dim] = temp[-1]
        print(f"\b\b\b: {contNum[dim]:.4f}")

    dimSpace = np.linspace(1,params['maxDim'], params['maxDim']).astype(int)   

    kl = KneeLocator(dimSpace, trustNum, curve = "concave", direction = "increasing")
    if kl.knee:
        trustDim = kl.knee
        print('Trust final dimension: %d - Final error of %.4f' %(trustDim, 1-trustNum[dim-1]))
    else:
        trustDim = np.nan
    kl = KneeLocator(dimSpace, contNum, curve = "concave", direction = "increasing")
    if kl.knee:
        contDim = kl.knee
        print('Cont final dimension: %d - Final error of %.4f' %(contDim, 1-contNum[dim-1]))
    else:
        contDim = np.nan
    hmeanDim = (2*trustDim*contDim)/(trustDim+contDim)
    print('Hmean final dimension: %d' %(hmeanDim))

    umap_dim_dict[mouse] = {
        'trustNum': trustNum,
        'contNum': contNum,
        'trustDim': trustDim,
        'contDim':contDim,
        'hmeanDim': hmeanDim,
        'params': params
    }
    saveFile = open(os.path.join(saveDir, 'umap_dim_dict.pkl'), "wb")
    pickle.dump(umap_dim_dict, saveFile)
    saveFile.close()

saveFile = open(os.path.join(saveDir, 'umap_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# write string one by one adding newline
paramsList = [ f'{key} : {params[key]}' for key in params]
saveParamsFile = open(os.path.join(saveDir, "umap_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveParamsFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                               ISOMAP DIM                               |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'maxDim':10,
    'nNeigh': 120,
    'signalName': 'clean_traces',
}
saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
isomap_dim_dict = dict()

#define kernel function for rec error
K = lambda D: -0.5*((np.eye(D.shape[0])-(1/D.shape[0])).dot(np.square(D))).dot(np.eye(D.shape[0])-(1/D.shape[0]))

for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    #initialize isomap object
    isoModel = Isomap(n_neighbors = params['nNeigh'], n_components = params['maxDim'])
    #fit and project data
    print("\tFitting model...", sep= '', end = '')
    emb = isoModel.fit_transform(signal)
    print("\b\b\b: Done")
    resVar = np.zeros((params['maxDim'], ))
    recError = np.zeros((params['maxDim'], ))
    nSamples = signal.shape[0]
    #compute Isomap kernel for input data once before going into the loop
    signalKD = K(isoModel.dist_matrix_)
    for dim in range(1, params['maxDim']+1):
        print(f"Dim {dim+1}/{params['maxDim']}")
        embD = pairwise_distances(emb[:,:dim], metric = 'euclidean') 
        #compute residual variance
        resVar[dim-1] = 1 - pearsonr(np.matrix.flatten(isoModel.dist_matrix_),
                                            np.matrix.flatten(embD.astype('float32')))[0]**2
        #compute residual error
        embKD = K(embD)
        recError[dim-1] = np.linalg.norm((signalKD-embKD)/nSamples, 'fro')

    #find knee in residual variance
    dimSpace = np.linspace(1,params['maxDim'], params['maxDim']).astype(int)

    kl = KneeLocator(dimSpace, resVar, curve = "convex", direction = "decreasing")                                                      
    if kl.knee:
        resVarDim = kl.knee
        print('Final dimension: %d - Final residual variance: %.4f' %(resVarDim, resVar[resVarDim-1]))
    else:
        resVarDim = np.nan
        print('Could estimate final dimension (knee not found). Returning nan.')

    #find knee in reconstruction error      
    kl = KneeLocator(dimSpace, recError, curve = "convex", direction = "decreasing")                                                      
    if kl.knee:
        recErrorDim = kl.knee
        print('Final dimension: %d - Final reconstruction error: %.4f' %(recErrorDim, recError[recErrorDim-1]))
    else:
        recErrorDim = np.nan
        print('Could estimate final dimension (knee not found). Returning nan.')
    isomap_dim_dict[mouse] = {
        'resVar': resVar,
        'resVarDim': resVarDim,
        'recError': recError,
        'recErrorDim': recErrorDim,
        'params': params
    }
    saveFile = open(os.path.join(saveDir, 'isomap_dim_dict.pkl'), "wb")
    pickle.dump(isomap_dim_dict, saveFile)
    saveFile.close()

saveFile = open(os.path.join(saveDir, 'isomap_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# write string one by one adding newline
paramsList = [ f'{key} : {params[key]}' for key in params]
saveParamsFile = open(os.path.join(saveDir, "isomap_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                                 PCA DIM                                |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'signalName': 'clean_traces',
}
saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
pca_dim_dict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    #initialize isomap object
    modelPCA = PCA(signal.shape[1])
    #fit data
    modelPCA = modelPCA.fit(signal)
    #dim 80% variance explained
    var80Dim = np.where(np.cumsum(modelPCA.explained_variance_ratio_)>=0.80)[0][0]
    print('80%% Final dimension: %d' %(var80Dim))

    #dim drop in variance explained by knee
    dimSpace = np.linspace(1,signal.shape[1], signal.shape[1]).astype(int)        
    kl = KneeLocator(dimSpace, modelPCA.explained_variance_ratio_, curve = "convex", direction = "decreasing")                                                      
    if kl.knee:
        dim = kl.knee
        print('Knee Final dimension: %d - Final variance: %.4f' %(dim, np.cumsum(modelPCA.explained_variance_ratio_)[dim]))
    else:
        dim = np.nan
        print('Could estimate final dimension (knee not found). Returning nan.')

    pca_dim_dict[mouse] = {
        'explained_variance_ratio_': modelPCA.explained_variance_ratio_,
        'var80Dim': var80Dim,
        'kneeDim': dim,
        'params': params
    }
    saveFile = open(os.path.join(saveDir, 'pca_dim_dict.pkl'), "wb")
    pickle.dump(pca_dim_dict, saveFile)
    saveFile.close()

saveFile = open(os.path.join(saveDir, 'pca_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# write string one by one adding newline
paramsList = [ f'{key} : {params[key]}' for key in params]
saveParamsFile = open(os.path.join(saveDir, "pca_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                              SAVE DIM RED                              |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'dim':3,
    'nNeigh': 120,
    'minDist': 0.1,
    'signalName': 'clean_traces',
}
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    pos = np.concatenate(pdMouse['pos'].values, axis=0)
    dirMat = np.concatenate(pdMouse['dir_mat'].values, axis=0)
    indexMat = np.concatenate(pdMouse['index_mat'].values, axis=0)

    print("\tFitting umap model...", sep= '', end = '')
    modelUmap = umap.UMAP(n_neighbors =params['nNeigh'], n_components =params['dim'], min_dist=params['minDist'])
    modelUmap.fit(signal)
    embUmap = modelUmap.transform(signal)
    print("\b\b\b: Done")
    pdMouse['umap'] = [embUmap[indexMat[:,0]==pdMouse["trial_id"][idx] ,:] 
                                                    for idx in pdMouse.index]
    dim_red_object['umap'] = copy.deepcopy(modelUmap)

    print("\tFitting isomap model...", sep= '', end = '')
    modelIsomap = Isomap(n_neighbors =params['nNeigh'], n_components = signal.shape[1])
    modelIsomap.fit(signal)
    embIsomap = modelIsomap.transform(signal)
    print("\b\b\b: Done")
    pdMouse['isomap'] = [embIsomap[indexMat[:,0]==pdMouse["trial_id"][idx] ,:] 
                                                    for idx in pdMouse.index]
    dim_red_object['isomap'] = copy.deepcopy(modelIsomap)

    print("\tFitting pca model...", sep= '', end = '')
    modelPCA = PCA(signal.shape[1])
    modelPCA.fit(signal)
    embPCA = modelPCA.transform(signal)
    print("\b\b\b: Done")
    pdMouse['pca'] = [embPCA[indexMat[:,0]==pdMouse["trial_id"][idx] ,:] 
                                                    for idx in pdMouse.index]
    dim_red_object['pca'] = copy.deepcopy(modelPCA)
    #%%
    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(2,3,1, projection = '3d')
    b = ax.scatter(*embUmap[:,:3].T, c = dirMat, cmap = 'Accent',s = 10, vmin= 0, vmax = 8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Umap')
    ax = plt.subplot(2,3,4, projection = '3d')
    b = ax.scatter(*embUmap[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)

    ax = plt.subplot(2,3,2, projection = '3d')
    b = ax.scatter(*embIsomap[:,:3].T, c = dirMat, cmap = 'Accent',s = 10, vmin= 0, vmax = 8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Isomap')
    ax = plt.subplot(2,3,5, projection = '3d')
    b = ax.scatter(*embIsomap[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)

    ax = plt.subplot(2,3,3, projection = '3d')
    b = ax.scatter(*embPCA[:,:3].T, c = dirMat, cmap = 'Accent',s = 10, vmin= 0, vmax = 8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('PCA')
    ax = plt.subplot(2,3,6, projection = '3d')
    b = ax.scatter(*embPCA[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    plt.suptitle(f"{mouse}")
    plt.savefig(os.path.join(filePath,f'{mouse}_dim_red.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(filePath, fileName), "wb") as file:
        pickle.dump(pdMouse, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(filePath, mouse+"_dim_red_object.pkl"), "wb") as file:
        pickle.dump(dim_red_object, file, protocol=pickle.HIGHEST_PROTOCOL)

#__________________________________________________________________________
#|                                                                        |#
#|                           COMPUTE SI NN ABS                            |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
saveDir = '/home/julio/Documents/SP_project/Fig2/SI/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
sIDict = dict()
featParamsDict = {
    'pos': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    'dir': {'discrete_label':True, 'num_shuffles':100, 'verbose':False},
    'vel': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    '(pos,dir)': {'discrete_label':[False, True], 'num_shuffles':1, 'verbose':False},
    'globalTime': {'discrete_label':False, 'n_bins':20, 'num_shuffles':1, 'verbose':False},
    'trial': {'discrete_label':False, 'n_bins':20, 'num_shuffles':1, 'verbose':False},
    'trialTime': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False}
}
nnList = [5,15,25,50,100,250]
for mouse in miceList:
    print(f"\nWorking on mouse {mouse}: ")
    sIDict[mouse] = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #keep only right left trials
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")
    #get data
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
    dirMat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pdMouse['vel'].values, axis=0))
    vecFeat = np.concatenate((pos[:,0].reshape(-1,1),dirMat),axis=1)
    time = np.arange(pos.shape[0])
    trial = copy.deepcopy(np.concatenate(pdMouse['index_mat'].values, axis=0))
    trialTime = copy.deepcopy(np.concatenate(pdMouse['inner_trial_time'].values, axis=0))

    for signalName in ['clean_traces', 'umap', 'isomap', 'pca']:
        signal = copy.deepcopy(np.concatenate(pdMouse[signalName].values, axis=0))
        if signalName == 'pca' or signalName =='isomap':
            signal = signal[:,:3]
        D = pairwise_distances(signal)
        noiseIdx = filter_noisy_outliers(signal,D=D)
        csignal = signal[~noiseIdx,:]

        featDict = {
            'pos': pos[~noiseIdx,0],
            'dir': dirMat[~noiseIdx],
            'vel': vel[~noiseIdx],
            '(pos,dir)': vecFeat[~noiseIdx,:],
            'globalTime': time[~noiseIdx],
            'trial': trial[~noiseIdx],
            'trialTime': trialTime[~noiseIdx]
        }
        sIDict[mouse][signalName] = dict()
        sIDict[mouse][signalName]['featDict'] = featDict
        sIDict[mouse][signalName]['featParamsDict'] = featParamsDict
        sIDict[mouse][signalName]['signalDict'] = {
                                                'signal': signal,
                                                'csignal': csignal,
                                                'noiseIdx': noiseIdx
                                                }
        sIDict[mouse][signalName]['results'] = dict()
        for featName in list(featDict.keys()):
            sI = np.zeros((len(nnList),))*np.nan
            binLabel = list()
            overlapMat = list()
            ssI = np.zeros((len(nnList),featParamsDict[featName]['num_shuffles']))

            for nnIdx in range(len(nnList)):
                print(f'\t{signalName} - {featName}: {nnList[nnIdx]} ({nnIdx+1}/{len(nnList)})', end='\r', flush=True)
                nn = nnList[nnIdx]
                sI[nnIdx], tbinLabel, toverlapMat, ssI[nnIdx] = compute_structure_index(csignal, featDict[featName], 
                                            n_neighbors=nn, **featParamsDict[featName])
                binLabel.append(tbinLabel)
                overlapMat.append(toverlapMat)
            print(f'\t{signalName} - {featName}: {np.nanmean(sI)}')
            sIDict[mouse][signalName]['results'][featName] = {
                'sI': sI,
                'binLabel': binLabel,
                'overlapMat': overlapMat,
                'ssI': ssI,
                'nnList': nnList,
                'params': featParamsDict[featName]
            }

        with open(os.path.join(saveDir,'sI_abs_dict.pkl'), 'wb') as f:
            pickle.dump(sIDict, f)

#__________________________________________________________________________
#|                                                                        |#
#|                          COMPUTE SI NN PERC                            |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7']#, 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
saveDir = '/home/julio/Documents/SP_project/Fig2/SI/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
sIDict = dict()
featParamsDict = {
    'pos': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    'dir': {'discrete_label':True, 'num_shuffles':1, 'verbose':False},
    'vel': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    '(pos,dir)': {'discrete_label':[False, True], 'num_shuffles':1, 'verbose':False},
    'globalTime': {'discrete_label':False, 'n_bins':20, 'num_shuffles':1, 'verbose':False},
    'trial': {'discrete_label':False, 'n_bins':20, 'num_shuffles':1, 'verbose':False},
    'trialTime': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False}
}
nnPercList = [0.001, 0.002, 0.005,0.01, 0.02, 0.05]
for mouse in miceList:

    print(f"\nWorking on mouse {mouse}: ")
    sIDict[mouse] = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #keep only right left trials
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")
    #get data
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
    dirMat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pdMouse['vel'].values, axis=0))
    vecFeat = np.concatenate((pos[:,0].reshape(-1,1),dirMat),axis=1)
    time = np.arange(pos.shape[0])
    trial = copy.deepcopy(np.concatenate(pdMouse['index_mat'].values, axis=0))
    trialTime = copy.deepcopy(np.concatenate(pdMouse['inner_trial_time'].values, axis=0))

    for signalName in ['clean_traces', 'umap','isomap', 'pca']:
        signal = copy.deepcopy(np.concatenate(pdMouse[signalName].values, axis=0))
        if signalName == 'pca' or signalName =='isomap':
            signal = signal[:,:3]
        D = pairwise_distances(signal)
        noiseIdx = filter_noisy_outliers(signal,D=D)
        csignal = signal[~noiseIdx,:]

        featDict = {
            'pos': pos[~noiseIdx,0],
            'dir': dirMat[~noiseIdx],
            'vel': vel[~noiseIdx],
            '(pos,dir)': vecFeat[~noiseIdx,:],
            'globalTime': time[~noiseIdx],
            'trial': trial[~noiseIdx],
            'trialTime': trialTime[~noiseIdx]
        }

        nnList = [np.max([np.round(nnPercList[idx]*csignal.shape[0]).astype(int),3]) for idx in range(len(nnPercList))]
        sIDict[mouse][signalName] = dict()
        sIDict[mouse][signalName]['featDict'] = featDict
        sIDict[mouse][signalName]['featParamsDict'] = featParamsDict
        sIDict[mouse][signalName]['signalDict'] = {
                                                'signal': signal,
                                                'csignal': csignal,
                                                'noiseIdx': noiseIdx
                                                }
        sIDict[mouse][signalName]['results'] = dict()
        for featName in list(featDict.keys()):
            sI = np.zeros((len(nnList),))*np.nan
            binLabel = list()
            overlapMat = list()
            ssI = np.zeros((len(nnList),featParamsDict[featName]['num_shuffles']))

            for nnIdx in range(len(nnList)):
                print(f'\t{signalName} - {featName}: {nnList[nnIdx]} ({nnIdx+1}/{len(nnList)})', end='\r', flush=True)
                nn = nnList[nnIdx]
                sI[nnIdx], tbinLabel, toverlapMat, ssI[nnIdx] = compute_structure_index(csignal, featDict[featName], 
                                            n_neighbors=nn, **featParamsDict[featName])
                binLabel.append(tbinLabel)
                overlapMat.append(toverlapMat)
            print(f'\t{signalName} - {featName}: {np.nanmean(sI)}')
            sIDict[mouse][signalName]['results'][featName] = {
                'sI': sI,
                'binLabel': binLabel,
                'overlapMat': overlapMat,
                'ssI': ssI,
                'nnList': nnList,
                'nnPercList': nnPercList,
                'params': featParamsDict[featName]
            }

        with open(os.path.join(saveDir,'sI_perc_dict.pkl'), 'wb') as f:
            pickle.dump(sIDict, f)

#__________________________________________________________________________
#|                                                                        |#
#|                      COMPUTE ELLIPSE ECCENTRICITY                      |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/eccentricity/'
ellipseDict = dict()

for mouse in miceList:
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    pdMouse = load_pickle(filePath,fileName)

    pos = np.concatenate(pdMouse['pos'].values, axis = 0)
    dirMat = np.concatenate(pdMouse['dir_mat'].values, axis=0)
    emb = np.concatenate(pdMouse['umap'].values, axis = 0)[:,:3]

    D = pairwise_distances(emb)
    noiseIdx = filter_noisy_outliers(emb,D)
    cemb = emb[~noiseIdx,:]
    cpos = pos[~noiseIdx,:]
    cdirMat = dirMat[~noiseIdx]

    #compute centroids
    cent = get_centroids(cemb, cpos[:,0],cdirMat, ndims = 3, nCentroids=40)  
    modelPCA = PCA(2)
    modelPCA.fit(cent)
    cent2D = modelPCA.transform(cent)
    cent2D = cent2D - np.tile(np.mean(cent2D,axis=0), (cent2D.shape[0],1))

    # cent2D = get_centroids(cemb2D, cpos[:,0],cdirMat, ndims = 2, nCentroids=40)  
    # cent2D = cent2D - np.tile(np.mean(cent2D,axis=0), (cent2D.shape[0],1))

    ########################
    #         PRE          #
    ########################
    # Formulate and solve the least squares problem ||Ax - b ||^2
    X = cent2D[:,0].reshape(-1,1)
    Y = cent2D[:,1].reshape(-1,1)
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(cent2D[:,0])
    x = np.linalg.lstsq(A, b, rcond=-1)[0].squeeze()

    xLim = [np.min(cent2D[:,0]), np.max(cent2D[:,0])]
    yLim = [np.min(cent2D[:,1]), np.max(cent2D[:,1])]
    x_coord = np.linspace(xLim[0]-np.abs(xLim[0]*0.1),xLim[1]+np.abs(xLim[1]*0.1),10000)
    y_coord = np.linspace(yLim[0]-np.abs(yLim[0]*0.1),yLim[1]+np.abs(yLim[1]*0.1),10000)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0]*X_coord**2 + x[1]*X_coord*Y_coord + x[2]*Y_coord**2 + x[3]*X_coord + x[4]*Y_coord

    flatX = X_coord.reshape(-1,1)
    flatY = Y_coord.reshape(-1,1)
    flatZ = Z_coord.reshape(-1,1)
    idxValid = np.abs(flatZ-1)
    idxValid = idxValid<np.percentile(idxValid,0.01)
    xValid = flatX[idxValid]
    yValid = flatY[idxValid]

    x0 = (x[1]*x[4] - 2*x[2]*x[3])/(4*x[0]*x[2] - x[1]**2)
    y0 = (x[1]*x[3] - 2*x[0]*x[4])/(4*x[0]*x[2] - x[1]**2)
    center = [x0, y0]

    #assign to each point a pos and direction value according to closest k-neighbors
    k = 50
    cemb2D = modelPCA.transform(cemb)
    featEllipse = np.zeros((xValid.shape[0],2))
    for idx in range(xValid.shape[0]):
        point = [xValid[idx], yValid[idx]]
        d = ((cemb2D - point)**2).sum(axis=1)
        nIdx = d.argsort()[:k]  
        featEllipse[idx,:] = [np.mean(cpos[nIdx,0]), mode(cdirMat[nIdx].T.tolist()[0])]

    #Find axis of features gradient
    nSteps = 200
    angleArray = np.linspace(0, np.pi, nSteps)
    corrDir = np.zeros((nSteps,2))*np.nan
    ellipsePoints = np.concatenate((xValid.reshape(-1,1), yValid.reshape(-1,1)), axis=1)
    centeredPoints = ellipsePoints - center
    for idx, angle in enumerate(angleArray):
        unitDir = [np.cos(angle), np.sin(angle)]
        ellipseProjected = np.sum(centeredPoints*unitDir, axis=1)
        corrDir[idx,0] = np.abs(np.corrcoef(ellipseProjected,featEllipse[:,0])[0,1])
        corrDir[idx,1] = np.abs(np.corrcoef(ellipseProjected,featEllipse[:,1])[0,1])
    #find perendicular directions that maximize corrDir
    ortAngleIdx = np.argmin(np.abs(angleArray-np.pi/2))

    mixedCorrelation = 0.3*corrDir[:,0] + 0.2*(1 - corrDir[:,1]) + \
                        0.2*(1-np.roll(corrDir[:, 0],-ortAngleIdx)) + 0.3*np.roll(corrDir[:, 1],-ortAngleIdx)

    # mixedCorrelation = 0.7*corrDir[:,0] + 0.3*np.roll(corrDir[:, 1],-ortAngleIdx)
    bestAngleIdx = np.argmax(mixedCorrelation)
    #Find length of axis
    posAngle = angleArray[bestAngleIdx]
    posUnitDir = [np.cos(posAngle), np.sin(posAngle)]
    posAxisProjection = np.sum(centeredPoints*posUnitDir, axis=1)
    posLength = np.percentile(posAxisProjection,95) - np.percentile(posAxisProjection,5)

    dirAngle = angleArray[bestAngleIdx]+np.pi/2
    if dirAngle >= np.pi: dirAngle -= np.pi
    dirUnitDir = [np.cos(dirAngle), np.sin(dirAngle)]
    dirAxisProjection = np.sum(centeredPoints*dirUnitDir, axis=1)
    dirLength = np.percentile(dirAxisProjection,95) - np.percentile(dirAxisProjection,5)
    eccentricity = posLength/dirLength

    #Plot computations
    plotLims = [np.min([xLim[0], yLim[0]]), np.max([xLim[1], yLim[1]])]
    plotLims = [0.15*x + x for x in plotLims]
    embPlotLims = [np.min(cemb2D, axis=(0,1)), np.max(cemb2D, axis=(0,1))]
    embPlotLims = [0.15*x + x for x in embPlotLims]
    xposLine = np.array(range(np.floor(plotLims[0]).astype(int), np.ceil(plotLims[1]).astype(int)))
    ellipDirColor = np.zeros((featEllipse.shape[0],3))
    for point in range(featEllipse.shape[0]):
        if featEllipse[point,1]==0:
            ellipDirColor[point] = [14/255,14/255,143/255]
        elif featEllipse[point,1]==1:
            ellipDirColor[point] = [12/255,136/255,249/255]
        else:
            ellipDirColor[point] = [17/255,219/255,224/255]

    embDirColor = np.zeros((cdirMat.shape[0],3))
    for point in range(cdirMat.shape[0]):
        if cdirMat[point]==0:
            embDirColor[point] = [14/255,14/255,143/255]
        elif cdirMat[point]==1:
            embDirColor[point] = [12/255,136/255,249/255]
        else:
            embDirColor[point] = [17/255,219/255,224/255]

    fig = plt.figure(figsize=(15,6))

    ax = plt.subplot2grid(shape=(2, 5), loc=(0,0), rowspan=1, colspan = 1, projection = '3d')
    b = ax.scatter(*cemb[:,:3].T, c = cpos[:,0], cmap = 'inferno',s = 10)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot2grid(shape=(2, 5), loc=(1,0), rowspan=1, colspan = 1, projection = '3d')
    b = ax.scatter(*cemb[:,:3].T, color=embDirColor,s = 10)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot2grid(shape=(2, 5), loc=(0,1), rowspan=2, colspan = 2)
    ax.scatter(*cemb2D[:,:3].T, c = cpos[:,0], s=10, cmap = 'magma')
    ax.scatter(xValid, yValid, color ='m', s=10)
    ax.scatter(*cent2D[:,:2].T, color ='b', s=20)
    ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    ax.scatter(center[0], center[1], color = 'm', s=30)
    ax.set_xlim(embPlotLims)
    ax.set_ylim(embPlotLims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"2D embedding and ellipse")

    ax = plt.subplot2grid(shape=(2, 5), loc=(0,3), rowspan=1, colspan = 1)
    ax.scatter(*cent2D[:,:2].T, color ='b', s=20)
    ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    ax.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, c = featEllipse[:,0], s=10, cmap = 'magma')
    yposLine= posUnitDir[1]/posUnitDir[0]*xposLine 
    ax.plot(xposLine,yposLine, color = 'r', linestyle='--')
    ax.set_xlim(plotLims)
    ax.set_ylim(plotLims)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Length pos: {posLength:2f}")

    ax = plt.subplot2grid(shape=(2, 5), loc=(1,3), rowspan=1, colspan = 1)
    ax.scatter(*cent2D[:,:2].T, color ='b', s=20)
    ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    ax.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, s=10, color=ellipDirColor)
    yposLine= dirUnitDir[1]/dirUnitDir[0]*xposLine 
    ax.plot(xposLine,yposLine, color = 'r', linestyle='--')
    ax.set_xlim(plotLims)
    ax.set_ylim(plotLims)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Length dir: {dirLength:2f}")

    ax = plt.subplot2grid(shape=(2, 5), loc=(0,4), rowspan=2, colspan = 1)
    ax.plot(corrDir[:,0], angleArray, color = 'b', label = 'position')
    ax.plot(corrDir[:,1], angleArray, color = 'orange',  label = 'direction')
    ax.plot(mixedCorrelation, angleArray, color = 'k',  label = 'mixed')
    ax.plot([0,1], [posAngle,posAngle], color = 'b', linestyle= '--')
    ax.plot([0,1], [dirAngle,dirAngle], color = 'orange', linestyle= '--')
    ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_ylabel('Angle of projection')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(f"{mixedCorrelation[bestAngleIdx]:2f}")
    ax.legend()
    plt.suptitle(f"{mouse} - Eccentricity: {eccentricity:4f}")
    plt.tight_layout()

    retval = -1
    def onclick(event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata))
        # Record the x location of the user's click in the global variable and close the figure
        global retval
        retval = event.ydata
        plt.close()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # Bring up the figure (and wait)
    plt.show()

    #Find length of axis
    bestAngleIdx = np.argmin(np.abs(angleArray - retval))    
    #Find length of axis
    posAngle = angleArray[bestAngleIdx]
    posUnitDir = [np.cos(posAngle), np.sin(posAngle)]
    posAxisProjection = np.sum(centeredPoints*posUnitDir, axis=1)
    posLength = np.percentile(posAxisProjection,95) - np.percentile(posAxisProjection,5)

    dirAngle = angleArray[bestAngleIdx]+np.pi/2
    if dirAngle >= np.pi: dirAngle -= np.pi
    dirUnitDir = [np.cos(dirAngle), np.sin(dirAngle)]
    dirAxisProjection = np.sum(centeredPoints*dirUnitDir, axis=1)
    dirLength = np.percentile(dirAxisProjection,95) - np.percentile(dirAxisProjection,5)


    #Compute eccentricity
    eccentricity = posLength/dirLength
    print(posAngle, eccentricity)

    ellipseDict[mouse] = {
        'pos':pos,
        'dirMat': dirMat,
        'emb': emb,
        'D': D,
        'noiseIdx': noiseIdx,
        'cpos': cpos,
        'cdirMat': cdirMat,
        'cemb': cemb,
        'cent': cent,
        'cent2D': cent2D,
        'cemb2D': cemb2D,
        'ellipseCoeff': x,
        'xLim': xLim,
        'yLim': yLim,
        'X_coord': X_coord,
        'Y_coord': Y_coord,
        'Z_coord': Z_coord,
        'idxValid':idxValid,
        'xValid': xValid,
        'yValid': yValid,
        'center': center,
        'angleArray': angleArray,
        'corrDir':corrDir,
        'ellipsePoints': ellipsePoints,
        'centeredPoints':centeredPoints,
        'ortAngleIdx':ortAngleIdx,
        'mixedCorrelation': mixedCorrelation,
        'bestAngleIdx': bestAngleIdx,
        'posAngle': posAngle,
        'posUnitDir': posUnitDir,
        'posAxisProjection': posAxisProjection,
        'posLength': posLength,
        'dirAngle': dirAngle,
        'dirUnitDir': dirUnitDir,
        'dirAxisProjection': dirAxisProjection,
        'dirLength': dirLength,
        'eccentricity': eccentricity
    }

    with open(os.path.join(saveDir,'ellipse_fit_dict.pkl'), 'wb') as f:
        pickle.dump(ellipseDict, f)


    fig = plt.figure(figsize=(15,6))

    ax = plt.subplot2grid(shape=(2, 5), loc=(0,0), rowspan=1, colspan = 1, projection = '3d')
    b = ax.scatter(*cemb[:,:3].T, c = cpos[:,0], cmap = 'inferno',s = 10)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot2grid(shape=(2, 5), loc=(1,0), rowspan=1, colspan = 1, projection = '3d')
    b = ax.scatter(*cemb[:,:3].T, color=embDirColor,s = 10)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot2grid(shape=(2, 5), loc=(0,1), rowspan=2, colspan = 2)
    ax.scatter(*cemb2D[:,:3].T, c = cpos[:,0], s=10, cmap = 'magma')
    ax.scatter(xValid, yValid, color ='m', s=10)
    ax.scatter(*cent2D[:,:2].T, color ='b', s=20)
    ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    ax.scatter(center[0], center[1], color = 'm', s=30)
    ax.set_xlim(embPlotLims)
    ax.set_ylim(embPlotLims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"2D embedding and ellipse")

    ax = plt.subplot2grid(shape=(2, 5), loc=(0,3), rowspan=1, colspan = 1)
    ax.scatter(*cent2D[:,:2].T, color ='b', s=20)
    ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    ax.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, c = featEllipse[:,0], s=10, cmap = 'magma')
    yposLine= posUnitDir[1]/posUnitDir[0]*xposLine 
    ax.plot(xposLine,yposLine, color = 'r', linestyle='--')
    ax.set_xlim(plotLims)
    ax.set_ylim(plotLims)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Length pos: {posLength:2f}")

    ax = plt.subplot2grid(shape=(2, 5), loc=(1,3), rowspan=1, colspan = 1)
    ax.scatter(*cent2D[:,:2].T, color ='b', s=20)
    ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    ax.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, s=10, color=ellipDirColor)
    yposLine= dirUnitDir[1]/dirUnitDir[0]*xposLine 
    ax.plot(xposLine,yposLine, color = 'r', linestyle='--')
    ax.set_xlim(plotLims)
    ax.set_ylim(plotLims)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Length dir: {dirLength:2f}")

    ax = plt.subplot2grid(shape=(2, 5), loc=(0,4), rowspan=2, colspan = 1)
    ax.plot(corrDir[:,0], angleArray, color = 'b', label = 'position')
    ax.plot(corrDir[:,1], angleArray, color = 'orange',  label = 'direction')
    ax.plot(mixedCorrelation, angleArray, color = 'k',  label = 'mixed')

    ax.plot([0,1], [posAngle,posAngle], color = 'b', linestyle= '--')
    ax.plot([0,1], [dirAngle,dirAngle], color = 'orange', linestyle= '--')
    ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_ylabel('Angle of projection')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(f"{mixedCorrelation[bestAngleIdx]:2f}")
    ax.legend()
    plt.suptitle(f"{mouse} - Eccentricity: {eccentricity:4f}")
    plt.tight_layout()

    plt.savefig(os.path.join(saveDir,f'{mouse}_ellipse_fit.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDir,f'{mouse}_ellipse_fit.svg'), dpi = 400,bbox_inches="tight",transparent=True)

    plt.close(fig)


#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/decoders/'

params = {
    'x_base_signal': 'clean_traces',
    'y_signal_list': ['posx', 'posy','vel', 'index_mat', 'dir_mat'],
    'verbose': True,
    'trial_signal': 'index_mat',
    'nn': 120,
    'min_dist':0.1,
    'n_splits': 10,
    'n_dims': 3,
    'emb_list': ['pca', 'isomap', 'umap']
    }

dec_R2s = dict()
dec_pred = dict()
for mouse in miceList:
    print(f'\n{mouse}: ')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = gu.load_files(filePath,'*'+fileName,verbose=True,struct_type="pickle")
    dec_R2s[mouse], dec_pred[mouse] = dec.decoders_1D(pd_object = copy.deepcopy(pdMouse), **params)

    with open(os.path.join(saveDir,'dec_R2s_dict.pkl'), 'wb') as f:
        pickle.dump(dec_R2s, f)
    with open(os.path.join(saveDir,'dec_pred_dict.pkl'), 'wb') as f:
        pickle.dump(dec_pred, f)
    with open(os.path.join(saveDir,'dec_params.pkl'), 'wb') as f:
        pickle.dump(params, f)



#__________________________________________________________________________
#|                                                                        |#
#|                              PLACE CELLS                               |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'

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

for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    pdMouse = load_pickle(filePath,fileName)

    neuSignal = np.concatenate(pdMouse['clean_traces'].values, axis=0)
    posSignal = np.concatenate(pdMouse['pos'].values, axis=0)
    velSignal = np.concatenate(pdMouse['vel'].values, axis=0)
    dirSignal = np.concatenate(pdMouse['dir_mat'].values, axis=0)

    to_keep = np.logical_and(dirSignal[:,0]>0,dirSignal[:,0]<=2)
    posSignal = posSignal[to_keep,:] 
    velSignal = velSignal[to_keep] 
    neuSignal = neuSignal[to_keep,:] 
    dirSignal = dirSignal[to_keep,:] 

    mousePC = pc.get_place_cells(posSignal, neuSignal, vel_signal = velSignal, dim = 1,
                          direction_signal = dirSignal, mouse = mouse, save_dir = saveDir, **params)

    print('\tNum place cells:')
    num_cells = neuSignal.shape[1]
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,0]*(mousePC['place_cells_dir'][:,1]==0))
    print(f'\t\t Only left cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,1]*(mousePC['place_cells_dir'][:,0]==0))
    print(f'\t\t Only right cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,0]*mousePC['place_cells_dir'][:,1])
    print(f'\t\t Both dir cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    with open(os.path.join(saveDir,mouse+'_pc_dict.pkl'), 'wb') as f:
        pickle.dump(mousePC, f)

#__________________________________________________________________________
#|                                                                        |#
#|                           MANIFOLD ACTIVITY                            |#
#|________________________________________________________________________|#

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']

dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2/manifold_cells'

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

for mouse in miceList:
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    signal = copy.deepcopy(np.concatenate(pdMouse['clean_traces'].values, axis = 0))
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis = 0))
    emb = copy.deepcopy(np.concatenate(pdMouse['umap'].values, axis = 0))

    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')
    neuPDF = pcDict['neu_pdf']
    normNeuPDF = np.zeros(neuPDF.shape)
    nCells = neuPDF.shape[1]
    for d in range(neuPDF.shape[2]):
        for c in range(nCells):
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.max(neuPDF[:,c,d])
    orderLeft =  np.argsort(np.argmax(normNeuPDF[:,:,0], axis=0))
    orderRight =  np.argsort(np.argmax(normNeuPDF[:,:,1], axis=0))

    meanNormNeuPDF = np.nanmean(normNeuPDF, axis=1)
    mapAxis = pcDict['mapAxis']
    manifoldSignal = np.zeros((emb.shape[0]))*np.nan
    for p in range(emb.shape[0]):
        try:
            x = np.where(mapAxis[0]<=pos[p,0])[0][-1]
        except: 
            x = 0
        dire = direction[p]
        if dire==0:
            manifoldSignal[p] = np.nan
        else:
            manifoldSignal[p] = meanNormNeuPDF[x,dire-1]

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

    ax = plt.subplot(2,2,3, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 30, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax = plt.subplot(2,2,4, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 30)
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


