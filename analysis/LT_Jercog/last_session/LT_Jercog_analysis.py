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
from scipy.stats import pearsonr
import random

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
    sum(noiseIdx)
    return noiseIdx

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

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
dataDir = '/home/julio/Documents/SP_project/LT_Jercog/data/'
saveDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
#%% PARAMS
sigma = 6
upSig = 4
downSig = 12
signalField = 'rawProb'
peakTh = 0.1
velTh = 3
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
    sessionName = fileNames[0]
    for fileName in fileNames[1:]:
        if int(fileName[6:14])>int(sessionName[6:14]):
            sessionName = fileName
    print(f"\tSession selected: {sessionName}")
    if verbose:
        gu.print_time_verbose(localTime, globalTime)

    #%% KEEP ONLY LAST SESSION
    pdMouse = copy.deepcopy(dfMouse[sessionName])
    print('\n### 2. PROCESS DATA ###')
    print(f'Working on session: {sessionName}')
    #%% 2. PROCESS DATA
    pdMouse = add_dir_mat_field(pdMouse)
    pdMouse = add_inner_trial_time_field(pdMouse)
    
    #2.1 keep only moving epochs
    print(f'2.1 Dividing into moving/still data w/ velTh= {velTh:.2f}.')
    if velTh>0:
        og_dur = np.concatenate(pdMouse["pos"].values, axis=0).shape[0]
        pdMouse, still_pdMouse = gu.keep_only_moving(pdMouse, velTh)
        move_dur = np.concatenate(pdMouse["pos"].values, axis=0).shape[0]
        print(f"\t{sessionName}: Og={og_dur} ({og_dur/20}s) Move= {move_dur} ({move_dur/20}s)")
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

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
params = {
    'signalName': 'clean_traces',
    'nNeigh': 30,
    'verbose': True
}

saveDir =  '/home/julio/Documents/SP_project/Fig1/dimensionality/inner_dim'
try:
    os.mkdir(saveDir)
except:
    pass
dataDir =  '/home/julio/Documents/SP_project/Fig1/processed_data/'

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

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']

params = {
    'maxDim':10,
    'nNeigh': 120,
    'minDist': 0.1,
    'nnDim': 30,
    'signalName': 'clean_traces',
}

saveDir = '/home/julio/Documents/SP_project/Fig1/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
umap_dim_dict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    print("Computing rank indices og space...", end = '', sep = '')
    rankIdx = dim_validation.compute_rank_indices(signal)
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
        temp = dim_validation.trustworthiness_vector(signal, emb, params['nnDim'], indices_source = rankIdx)
        trustNum[dim] = temp[-1]
        print(f"\b\b\b: {trustNum[dim]:.4f}")
        #2. Compute continuity
        print("\tComputing continuity...", sep= '', end = '')
        temp = dim_validation.continuity_vector(signal, emb ,params['nnDim'])
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

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
params = {
    'maxDim':10,
    'nNeigh': 120,
    'signalName': 'clean_traces',
}
saveDir = '/home/julio/Documents/SP_project/Fig1/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
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
miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
params = {
    'signalName': 'clean_traces',
}
saveDir = '/home/julio/Documents/SP_project/Fig1/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
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
miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
params = {
    'dim':3,
    'nNeigh': 120,
    'minDist': 0.1,
    'signalName': 'clean_traces',
}
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
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
#|                               COMPUTE SI                               |#
#|________________________________________________________________________|#

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
saveDir = '/home/julio/Documents/SP_project/Fig1/SI/'
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
sI_dict = dict()
neighList = [25,50,75,100,200]
numShuffles = 1
for mouse in miceList:
    print(f"\nWorking on mouse {mouse}: ")
    sI_dict[mouse] = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #keep only right left trials
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")
    #get data
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pdMouse['vel'].values, axis=0))
    vectorial_feature = np.concatenate((pos[:,0].reshape(-1,1),dir_mat),axis=1)
    trial = copy.deepcopy(np.concatenate(pdMouse['index_mat'].values, axis=0))
    time = copy.deepcopy(np.concatenate(pdMouse['inner_trial_time'].values, axis=0))

    for signalName in ['clean_traces', 'umap', 'isomap', 'pca']:
        print(f'\t{signalName}: ')
        signal = copy.deepcopy(np.concatenate(pdMouse[signalName].values, axis=0))
        if signalName == 'pca' or signalName =='isomap':
            signal = signal[:,:3]
        D = pairwise_distances(signal)
        noiseIdx = filter_noisy_outliers(signal,D=D)
        csignal = signal[~noiseIdx,:]
        cpos = pos[~noiseIdx,:]
        cdir_mat = dir_mat[~noiseIdx]
        cvel = vel[~noiseIdx]
        cvectorial_feature = vectorial_feature[~noiseIdx,:]
        ctime = time[~noiseIdx]
        ctrial = trial[~noiseIdx]

        sI_dict[mouse][signalName] = dict()
        print('\t\tpos')

        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], overlapMat[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, cpos[:,0], 
                                                        n_neighbors=nVal, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['pos'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\tdir')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], overlapMat[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, cdir_mat, 
                                                        n_neighbors=nVal, discrete_label=True, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['dir'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\t(pos,dir)')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], overlapMat[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, cvectorial_feature, 
                                                        n_neighbors=nVal, discrete_label=[False, True], num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['(pos_dir)'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\tvel')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], overlapMat[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, cvel, 
                                                        n_neighbors=nVal, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['vel'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\ttime')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], overlapMat[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, ctime, 
                                                        n_neighbors=nVal, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['inner_time'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\ttrial')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], overlapMat[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, ctrial, 
                                                        n_neighbors=nVal, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['time'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }
        with open(os.path.join(saveDir,'sI_clean_dict.pkl'), 'wb') as f:
            pickle.dump(sI_dict, f)

sI_dict = dict()
for mouse in miceList:
    print(f"\nWorking on mouse {mouse}: ")
    sI_dict[mouse] = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #keep only right left trials
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")
    #get data
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pdMouse['vel'].values, axis=0))
    vectorial_feature = np.concatenate((pos[:,0].reshape(-1,1),dir_mat),axis=1)
    trial = copy.deepcopy(np.concatenate(pdMouse['index_mat'].values, axis=0))
    time = copy.deepcopy(np.concatenate(pdMouse['inner_trial_time'].values, axis=0))

    for signalName in ['clean_traces', 'umap', 'isomap', 'pca']:
        print(f'\t{signalName}: ')
        signal = copy.deepcopy(np.concatenate(pdMouse[signalName].values, axis=0))
        if signalName == 'pca' or signalName =='isomap':
            signal = signal[:,:3]

        noiseIdx = np.isnan(pos[:,0])
        csignal = signal[~noiseIdx,:]
        cpos = pos[~noiseIdx,:]
        cdir_mat = dir_mat[~noiseIdx]
        cvel = vel[~noiseIdx]
        cvectorial_feature = vectorial_feature[~noiseIdx,:]
        ctime = time[~noiseIdx]
        ctrial = trial[~noiseIdx]

        sI_dict[mouse][signalName] = dict()
        print('\t\tpos')

        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], binLabel[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, cpos[:,0], 
                                                        n_neighbors=nVal, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['pos'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\tdir')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], binLabel[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, cdir_mat, 
                                                        n_neighbors=nVal, discrete_label=True, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['dir'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\t(pos,dir)')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], binLabel[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, cvectorial_feature, 
                                                        n_neighbors=nVal, discrete_label=[False, True], num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['(pos_dir)'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\tvel')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], binLabel[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, cvel, 
                                                        n_neighbors=nVal, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['vel'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\ttime')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], binLabel[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, ctime, 
                                                        n_neighbors=nVal, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['time'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }

        print('\t\ttrial')
        sI = np.zeros((len(neighList),))*np.nan
        ssI = np.zeros((len(neighList),numShuffles))*np.nan
        binLabel = [np.nan]*len(neighList)
        overlapMat = [np.nan]*len(neighList)
        for nIdx, nVal in enumerate(neighList):
            sI[nIdx], binLabel[nIdx], overlapMat[nIdx], ssI[nIdx,:] = compute_structure_index(csignal, ctrial, 
                                                        n_neighbors=nVal, num_shuffles=numShuffles, verbose=True)
        sI_dict[mouse][signalName]['trial'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI,
            'neighList': neighList
        }
        with open(os.path.join(saveDir,'sI_dict.pkl'), 'wb') as f:
            pickle.dump(sI_dict, f)

#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig1/decoders/'

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
#|                        DECODERS ACROSS ANIMALS                         |#
#|________________________________________________________________________|#
from neural_manifold.decoders.decoder_classes import DECODERS  #decoders classes
from sklearn.metrics import median_absolute_error


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


def get_point_registration(p1, p2, verbose=True):
    #from https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
    if p1.shape[0]>p1.shape[1]:
        p1 = p1.transpose()
        p2 = p2.transpose()
    #Calculate centroids
    p1_c = np.nanmedian(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.nanmedian(p2, axis = 1).reshape((-1,1))
    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c
    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())
    #Calculate singular value decomposition (SVD)
    try:
        U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt
        #Calculate rotation matrix
        R = np.matmul(V_t.transpose(),U.transpose())
        if not np.allclose(np.linalg.det(R), 1.0) and verbose:
            print("Rotation matrix of N-point registration not 1, see paper Arun et al.")
        #Calculate translation matrix
        T = p2_c - np.matmul(R,p1_c)
    except:
        R = np.zeros((p1.shape[0], p1.shape[0]))*np.nan
        T = np.zeros((p1.shape[0], 1))
    return T,R

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig1/decoders/'

params = {
    'x_base_signal': 'clean_traces',
    'y_signal_list': ['posx', 'posy','vel', 'index_mat', 'dir_mat'],
    'verbose': True,
    'trial_signal': 'index_mat',
    'n_splits': 10,
    'n_dims': 3,
    'emb_list': ['pca', 'isomap', 'umap'],
    'decoder_list': ["wf", "wc", "xgb", "svr"]
    }

cross_dec_R2s = dict()
cross_dec_Rmat = dict()
for mouse_A in miceList:
    fileName =  mouse_A+'_df_dict.pkl'
    filePath_A = os.path.join(dataDir, mouse_A)
    pdMouse_A = gu.load_files(filePath_A,'*'+mouse_A+'_df_dict.pkl',verbose=True,struct_type="pickle")

    y_signal_list_A = list()
    for y_signal in params['y_signal_list']:
        y_signal_list_A.append(gu.pd_to_array_translator(pdMouse_A, y_signal).reshape(-1,1))

    x_signal_list_A = list()
    for emb_signal in params['emb_list']:
        x_signal_list_A.append(gu.pd_to_array_translator(pdMouse_A, emb_signal)[:,:3])

    #train all decoders for all signals and all y_labels
    print(f"Training decoders for mouse: {mouse_A}")
    decoder_objects = dict()
    for x_idx, x_signal in enumerate(params['emb_list']):
        for y_idx, y_signal in enumerate(params['y_signal_list']):
            for dec_name in params['decoder_list']:
                model_decoder = copy.deepcopy(DECODERS[dec_name]())
                model_decoder.fit(x_signal_list_A[x_idx], y_signal_list_A[y_idx])
                decoder_objects[f"({x_signal},{y_signal},{dec_name})"] = copy.deepcopy(model_decoder)

    for mouse_B in miceList:
        if mouse_B == mouse_A: continue;
        print(f"\n*******{mouse_A}vs{mouse_B}*******")
        filePath_B = os.path.join(dataDir, mouse_B)
        pdMouse_B = gu.load_files(filePath_B,'*'+mouse_B+'_df_dict.pkl',verbose=True,struct_type="pickle")
        y_signal_list_B = list()
        for y_signal in params['y_signal_list']:
            y_signal_list_B.append(gu.pd_to_array_translator(pdMouse_B, y_signal).reshape(-1,1))

        x_signal_list_B = list()        
        for emb_signal in params['emb_list']:
            x_signal_list_B.append(gu.pd_to_array_translator(pdMouse_B, emb_signal)[:,:3])

        R2s = dict()
        Rot_sol = dict()
        for emb_signal in params['emb_list']:
            R2s[emb_signal] = dict()
            for dec_name in params['decoder_list']:
                R2s[emb_signal][dec_name] = np.zeros((len(y_signal_list_A),2))

        #pre alignment
        for x_idx, x_signal in enumerate(params['emb_list']):
            for y_idx, y_signal in enumerate(params['y_signal_list']):
                for dec_name in params['decoder_list']:
                    model_decoder = copy.deepcopy(decoder_objects[f"({x_signal},{y_signal},{dec_name})"])
                    pred_B = model_decoder.predict(x_signal_list_B[x_idx])

                    test_error = median_absolute_error(y_signal_list_B[y_idx][:,0], pred_B[:,0])
                    #store results
                    R2s[x_signal][dec_name][y_idx,0] = test_error

        #after alignment
        for x_idx, x_signal in enumerate(params['emb_list']):
            cent_A, cent_B = get_centroids(x_signal_list_A[x_idx], x_signal_list_B[x_idx], 
                                                y_signal_list_A[0], y_signal_list_B[0],
                                                y_signal_list_A[-1], y_signal_list_B[-1], 
                                                ndims = 3, nCentroids = 40)
            T,R = get_point_registration(cent_B, cent_A)

            Rot_sol[x_signal] = {
                'R': R,
                'T': T,
                'emb_A': x_signal_list_A[x_idx],
                'emb_B': x_signal_list_B[x_idx],
                'label_A': y_signal_list_A[0],
                'label_B': y_signal_list_B[0]
            }

            emb_BA_rot = np.transpose(T + np.matmul(R, x_signal_list_B[x_idx].T))
            for y_idx, y_signal in enumerate(params['y_signal_list']):
                for dec_name in params['decoder_list']:
                    model_decoder = copy.deepcopy(decoder_objects[f"({x_signal},{y_signal},{dec_name})"])
                    pred_B = model_decoder.predict(emb_BA_rot)
                    test_error = median_absolute_error(y_signal_list_B[y_idx][:,0], pred_B[:,0])
                    #store results
                    R2s[x_signal][dec_name][y_idx,1] = test_error

        cross_dec_R2s[f"({mouse_A},{mouse_B})"] = copy.deepcopy(R2s)
        cross_dec_Rmat[f"({mouse_A},{mouse_B})"] = copy.deepcopy(Rot_sol)

        with open(os.path.join(saveDir,'cross_dec_R2s_dict.pkl'), 'wb') as f:
            pickle.dump(cross_dec_R2s, f)
        with open(os.path.join(saveDir,'cross_dec_Rmat_dict.pkl'), 'wb') as f:
            pickle.dump(cross_dec_Rmat, f)
        with open(os.path.join(saveDir,'cross_dec_params.pkl'), 'wb') as f:
            pickle.dump(params, f)

#__________________________________________________________________________
#|                                                                        |#
#|                           DECODERS VS NOISE                            |#
#|________________________________________________________________________|#

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig1/decoders/'

params = {
    'x_base_signal': 'clean_traces',
    'y_signal_list': ['posx', 'posy','vel', 'index_mat', 'dir_mat'],
    'verbose': True,
    'trial_signal': 'index_mat',
    'nn': 120,
    'min_dist':0.1,
    'n_splits': 3,
    'n_dims': 3,
    'emb_list': ['pca', 'isomap', 'umap'],
    'noise_list': [0, 0.01, 0.05, 0.1, 1],
    }

dec_R2s = dict()
dec_pred = dict()
SNR_vals = dict()
for mouse in miceList:
    print(f'\n{mouse}: ')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = gu.load_files(filePath,'*'+fileName,verbose=True,struct_type="pickle")
    dec_R2s[mouse], dec_pred[mouse], SNR_vals[mouse] = dec.decoders_noise_1D(pd_object = copy.deepcopy(pdMouse), **params)

    with open(os.path.join(saveDir,'noise_dec_R2s_dict.pkl'), 'wb') as f:
        pickle.dump(dec_R2s, f)
    with open(os.path.join(saveDir,'noise_dec_pred_dict.pkl'), 'wb') as f:
        pickle.dump(dec_pred, f)
    with open(os.path.join(saveDir,'noise_dec_params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    with open(os.path.join(saveDir,'noise_dec_SNR.pkl'), 'wb') as f:
        pickle.dump(SNR_vals, f)

#__________________________________________________________________________
#|                                                                        |#
#|                               PLACE CELLS                              |#
#|________________________________________________________________________|#
from neural_manifold import place_cells as pc

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
dataDir = '/media/julio/DATOS/spatial_navigation/paper/Fig1/data'
params = {
    'sF': 20,
    'bin_width': 2.5,
    'std_pos': 0,
    'std_pdf': 2.5,
    'method': 'spatial_info',
    'num_shuffles': 1000,
    'min_shift': 5,
    'fluo_th': 0.3,
    'th_metric': 99,
    }


for mouse in miceList:
    print(f'\n{mouse}: ')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    pdMouse = gu.load_files(filePath,'*'+fileName,verbose=True,struct_type="pickle")

    neu_signal = np.concatenate(pdMouse['clean_traces'].values, axis=0)
    pos_signal = np.concatenate(pdMouse['pos'].values, axis=0)
    vel_signal = np.concatenate(pdMouse['vel'].values, axis=0)
    direction_signal = np.concatenate(pdMouse['dir_mat'].values, axis=0)

    to_keep = np.logical_and(direction_signal[:,0]>0,direction_signal[:,0]<=2)
    pos_signal = pos_signal[to_keep,:] 
    vel_signal = vel_signal[to_keep] 
    neu_signal = neu_signal[to_keep,:] 
    direction_signal = direction_signal[to_keep,:] 

    mouse_pc = pc.get_place_cells(pos_signal, neu_signal, vel_signal = vel_signal, dim = 1,
                          direction_signal = direction_signal, mouse = mouse, saveDir = filePath, **params)

    save_name =  mouse+'_pc_dict.pkl'
    with open(os.path.join(filePath,save_name), 'wb') as f:
        pickle.dump(mouse_pc, f)

#__________________________________________________________________________
#|                                                                        |#
#|                              NUMBER CELLS                              |#
#|________________________________________________________________________|#
saveDir = '/home/julio/Documents/SP_project/Fig1/nCells/'
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
numCellList = np.unique(np.logspace(np.log10(5), np.log10(200),10,dtype=int))
iters = 5
paramsABIDS = {
    'nNeigh': 30,
}
paramsUMAP = {
    'maxDim':10,
    'nNeigh': 120,
    'minDist': 0.1,
    'nnDim': 30,
}
for mouse in miceList:
    print(f"\nWorking on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")

    signal = copy.deepcopy(np.concatenate(pdMouse['clean_traces'].values, axis=0))
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pdMouse['vel'].values, axis=0))
    vectorial_feature = np.concatenate((pos[:,0].reshape(-1,1),dir_mat),axis=1)
    pickedCellsArray = np.zeros((len(numCellList),iters,signal.shape[1]))*np.nan
    abidsDimArray = np.zeros((len(numCellList),iters))*np.nan
    trustNumArray = np.zeros((len(numCellList),iters, paramsUMAP['maxDim']))*np.nan
    contNumArray = np.zeros((len(numCellList),iters, paramsUMAP['maxDim']))*np.nan
    trustDimArray = np.zeros((len(numCellList),iters))*np.nan
    contDimArray = np.zeros((len(numCellList),iters))*np.nan
    sIArray = np.zeros((len(numCellList),iters,4))*np.nan
    for itCell, nCell in enumerate(numCellList):
        if nCell>signal.shape[1]: continue;
        print(f"Checking number of cells: {nCell} ({itCell+1}/10):")
        print("\tIteration X/X", sep= '', end = '')
        pre_del = '\b\b\b'
        for itSplit in range(iters):
            print(pre_del, f"{itSplit+1}/{iters}", sep = '', end = '')
            pre_del = (len(str(itSplit+1))+len(str(iters))+1)*'\b'
            pickedCells = random.sample(list(np.arange(signal.shape[1]).astype(int)), nCell)
            pickedCellsArray[itCell, itSplit, pickedCells] = True 
            itSignal = copy.deepcopy(signal[:, pickedCells])
            #inner dim
            abids = dim_red.compute_abids(signal, paramsABIDS['nNeigh'])
            abidsDimArray[itCell, itSplit] = np.nanmean(abids)
            #umap dim
            rankIdx = dim_validation.compute_rank_indices(itSignal)
            for dim in range(paramsUMAP['maxDim']):
                model = umap.UMAP(n_neighbors = paramsUMAP['nNeigh'], n_components =dim+1, min_dist=paramsUMAP['minDist'])
                emb = model.fit_transform(itSignal)
                trustNumArray[itCell, itSplit, dim] = dim_validation.trustworthiness_vector(itSignal, emb, paramsUMAP['nnDim'], indices_source = rankIdx)[-1]
                #2. Compute continuity
                contNumArray[itCell, itSplit, dim] = dim_validation.continuity_vector(itSignal, emb ,paramsUMAP['nnDim'])[-1]
            dimSpace = np.linspace(1,paramsUMAP['maxDim'], paramsUMAP['maxDim']).astype(int)   
            kl = KneeLocator(dimSpace, trustNum, curve = "concave", direction = "increasing")
            if kl.knee:
                trustDimArray[itCell, itSplit] = kl.knee
            else:
                trustDimArray[itCell, itSplit] = np.nan
            kl = KneeLocator(dimSpace, contNum, curve = "concave", direction = "increasing")
            if kl.knee:
                contDimArray[itCell, itSplit] = kl.knee
            else:
                contDimArray[itCell, itSplit] = np.nan

            #SI
            D = pairwise_distances(itSignal)
            noiseIdx = filter_noisy_outliers(itSignal,D=D)
            csignal = itSignal[~noiseIdx,:]
            cpos = pos[~noiseIdx,:]
            cdir_mat = dir_mat[~noiseIdx]
            cvel = vel[~noiseIdx]
            cvectorial_feature = vectorial_feature[~noiseIdx,:]
            sIArray[itCell, itSplit,0], _, _, _ = compute_structure_index(csignal, cpos[:,0],n_neighbors=20, num_shuffles=0, verbose=False)
            sIArray[itCell, itSplit,0], _, _, _ = compute_structure_index(csignal, cdir_mat, n_neighbors=20, discrete_label=True, num_shuffles=0, verbose=False)
            sIArray[itCell, itSplit,0], _, _, _ = compute_structure_index(csignal, cvectorial_feature, n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)
            sIArray[itCell, itSplit,0], _, _, _ = compute_structure_index(csignal, cvel, n_neighbors=20, num_shuffles=0, verbose=False)
    nCellDict = {
        'pickedCellsArray': pickedCellsArray,
        'abidsDimArray': abidsDimArray,
        'trustNumArray': trustNumArray,
        'contNumArray': contNumArray,
        'trustDimArray': trustDimArray,
        'contDimArray': contDimArray,
        'sIArray': sIArray,
        'paramsUMAP': paramsUMAP,
        'paramsABIDS': paramsABIDS
    }
    with open(os.path.join(saveDir,mouse+'nCell_dict.pkl'), 'wb') as f:
            pickle.dump(nCellDict, f)