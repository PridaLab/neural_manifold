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

#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#

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


miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
baseloadDir = '/home/julio/Documents/SP_project/LT_DeepSup/data/'
basesaveDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
signalField = 'raw_traces'
velTh = 5
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
    animalPre, animalPre_still = gu.keep_only_moving(animalPre, velTh)
    animalRot, animalRot_still = gu.keep_only_moving(animalRot, velTh)

    animalPre, animalRot = preprocess_traces(animalPre, animalRot, signalField, sigma=sigma, sigUp = sigUp, sigDown = sigDown)
    animalPre_still, animalRot_still = preprocess_traces(animalPre_still, animalRot_still, signalField, sigma=sigma, sigUp = sigUp, sigDown = sigDown)


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                PLOT UMAP                               |#
    #|________________________________________________________________________|#
    signalPre = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis=0))
    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    signalRot = copy.deepcopy(np.concatenate(animalRot['clean_traces'].values, axis=0))
    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    #%%all data
    index = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalRot.shape[0],1))+1))
    concat_signal = np.vstack((signalPre, signalRot))
    model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
    # model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
    model.fit(concat_signal)
    concat_emb = model.transform(concat_signal)
    embPre = concat_emb[index[:,0]==0,:]
    embRot = concat_emb[index[:,0]==1,:]

    #%%
    fig = plt.figure()
    ax = plt.subplot(1,2,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax.set_title('All')
    ax = plt.subplot(1,2,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s= 30, cmap = 'magma')
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
        fnameRot: animalRot
    }
    with open(os.path.join(saveDir, mouse+"_df_dict.pkl"), "wb") as file:
        pickle.dump(animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    animal_still_dict = {
        fnamePre: animalPre_still,
        fnameRot: animalRot_still
    }
    with open(os.path.join(saveDir, mouse+"_df_still_dict.pkl"), "wb") as file:
        pickle.dump(animal_still_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

#__________________________________________________________________________
#|                                                                        |#
#|                               INNER DIM                                |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'signalName': 'clean_traces',
    'nNeigh': 30,
    'verbose': True
}

saveDir =  '/home/julio/Documents/SP_project/LT_DeepSup/dimensionality/inner_dim'
try:
    os.mkdir(saveDir)
except:
    pass
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

idDict = dict()
for mouse in miceList:
    idDict[mouse] = dict()
    print(f"Working on mouse {mouse}: ", sep='', end='')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())

    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]

    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    signalPre = np.concatenate(animalPre[params['signalName']].values, axis = 0)
    signalRot = np.concatenate(animalRot[params['signalName']].values, axis = 0)
    signalBoth = np.concatenate((signalPre, signalRot), axis = 0)

    for case, signal in [('pre', signalPre), ('rot', signalRot), ('both', signalBoth)]:
        print(f"{case}")
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
        idDict[mouse][case] = {
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

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

params = {
    'maxDim':10,
    'nNeigh': 120,
    'minDist': 0.1,
    'nnDim': 30,
    'signalName': 'clean_traces',
}

saveDir =  '/home/julio/Documents/SP_project/LT_DeepSup/dimensionality/inner_dim'
try:
    os.mkdir(saveDir)
except:
    pass
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

umap_dim_dict = dict()
for mouse in miceList:
    umap_dim_dict[mouse] = dict()
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    signalPre = np.concatenate(animalPre[params['signalName']].values, axis = 0)
    signalRot = np.concatenate(animalRot[params['signalName']].values, axis = 0)
    signalBoth = np.concatenate((signalPre, signalRot), axis = 0)
    signalDict = {
        'pre': signalPre,
        'rot': signalRot,
        'both': signalBoth
    }
    for case, signal in signalDict.items():
        print(f"---Case: {case}---")
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

        umap_dim_dict[mouse][case] = {
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

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'maxDim':10,
    'nNeigh': 120,
    'signalName': 'clean_traces',
}

saveDir =  '/home/julio/Documents/SP_project/LT_DeepSup/dimensionality/inner_dim'
try:
    os.mkdir(saveDir)
except:
    pass
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

#define kernel function for rec error
K = lambda D: -0.5*((np.eye(D.shape[0])-(1/D.shape[0])).dot(np.square(D))).dot(np.eye(D.shape[0])-(1/D.shape[0]))

isomap_dim_dict = dict()
for mouse in miceList:
    isomap_dim_dict[mouse] = dict()
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    signalPre = np.concatenate(animalPre[params['signalName']].values, axis = 0)
    signalRot = np.concatenate(animalRot[params['signalName']].values, axis = 0)
    signalBoth = np.concatenate((signalPre, signalRot), axis = 0)
    signalDict = {
        'pre': signalPre,
        'rot': signalRot,
        'both': signalBoth
    }

    for case, signal in signalDict.items():
        print(f"---Case: {case}---")
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
        isomap_dim_dict[mouse][case] = {
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
    [saveParamsFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                                 PCA DIM                                |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

params = {
    'signalName': 'clean_traces',
}
saveDir =  '/home/julio/Documents/SP_project/LT_DeepSup/dimensionality/inner_dim'
try:
    os.mkdir(saveDir)
except:
    pass
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

pca_dim_dict = dict()
for mouse in miceList:
    pca_dim_dict[mouse] = dict()
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    signalPre = np.concatenate(animalPre[params['signalName']].values, axis = 0)
    signalRot = np.concatenate(animalRot[params['signalName']].values, axis = 0)
    signalBoth = np.concatenate((signalPre, signalRot), axis = 0)
    signalDict = {
        'pre': signalPre,
        'rot': signalRot,
        'both': signalBoth
    }
    for case, signal in signalDict.items():
        print(f"---Case: {case}---")
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

        pca_dim_dict[mouse][case] = {
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
    [saveParamsFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()


#__________________________________________________________________________
#|                                                                        |#
#|                              SAVE DIM RED                              |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'dim':3,
    'nNeigh': 120,
    'minDist': 0.1,
    'signalName': 'clean_traces',
}
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

for mouse in miceList:
    dim_red_object = dict()
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    signalPre = np.concatenate(animalPre[params['signalName']].values, axis = 0)
    posPre = np.concatenate(animalPre['pos'].values, axis = 0)
    dirMatPre = np.concatenate(animalPre['dir_mat'].values, axis=0)
    indexMatPre = np.concatenate(animalPre['index_mat'].values, axis=0)

    signalRot = np.concatenate(animalRot[params['signalName']].values, axis = 0)
    posRot = np.concatenate(animalRot['pos'].values, axis = 0)
    dirMatRot = np.concatenate(animalRot['dir_mat'].values, axis=0)
    indexMatRot = np.concatenate(animalRot['index_mat'].values, axis=0)

    indexPreRot = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalRot.shape[0],1))+1))
    signalBoth = np.vstack((signalPre, signalRot))


    #umap
    print("\tFitting umap model...", sep= '', end = '')
    modelUmap = umap.UMAP(n_neighbors =params['nNeigh'], n_components =params['dim'], min_dist=params['minDist'])
    modelUmap.fit(signalBoth)
    embBoth = modelUmap.transform(signalBoth)
    embPre = embBoth[indexPreRot[:,0]==0,:]
    embRot = embBoth[indexPreRot[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = dirMatPre, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    ax.scatter(*embRot[:,:3].T, c = dirMatRot, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_umap.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animalPre['umap'] = [embPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                   for idx in animalPre.index]
    animalRot['umap'] = [embRot[indexMatRot[:,0]==animalRot["trial_id"][idx] ,:] 
                                   for idx in animalRot.index]
    dim_red_object['umap'] = copy.deepcopy(modelUmap)
    print("\b\b\b: Done")

    #isomap
    print("\tFitting isomap model...", sep= '', end = '')
    modelIsomap = Isomap(n_neighbors =params['nNeigh'], n_components = signalBoth.shape[1])
    modelIsomap.fit(signalBoth)
    embBoth = modelIsomap.transform(signalBoth)
    embPre = embBoth[indexPreRot[:,0]==0,:]
    embRot = embBoth[indexPreRot[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = dirMatPre, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    ax.scatter(*embRot[:,:3].T, c = dirMatRot, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_isomap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_isomap.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animalPre['isomap'] = [embPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                   for idx in animalPre.index]
    animalRot['isomap'] = [embRot[indexMatRot[:,0]==animalRot["trial_id"][idx] ,:] 
                                   for idx in animalRot.index]
    dim_red_object['isomap'] = copy.deepcopy(modelIsomap)
    print("\b\b\b: Done")

    #pca
    print("\tFitting PCA model...", sep= '', end = '')
    modelPCA = PCA(signalBoth.shape[1])
    modelPCA.fit(signalBoth)
    embBoth = modelPCA.transform(signalBoth)
    embPre = embBoth[indexPreRot[:,0]==0,:]
    embRot = embBoth[indexPreRot[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = dirMatPre, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    ax.scatter(*embRot[:,:3].T, c = dirMatRot, cmap = 'Accent',s = 30, vmin= 0, vmax = 8)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_PCA.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDirFig,mouse+'_saved_PCA.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animalPre['pca'] = [embPre[indexMatPre[:,0]==animalPre["trial_id"][idx] ,:] 
                                   for idx in animalPre.index]
    animalRot['pca'] = [embRot[indexMatRot[:,0]==animalRot["trial_id"][idx] ,:] 
                                   for idx in animalRot.index]
    dim_red_object['pca'] = copy.deepcopy(modelPCA)
    print("\b\b\b: Done")

    newAnimalDict = {
        fnamePre: animalPre,
        fnameRot: animalRot
    }
    with open(os.path.join(filePath,fileName), "wb") as file:
        pickle.dump(newAnimalDict, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(filePath, mouse+"_umap_object.pkl"), "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE ROTATION                            |#
#|________________________________________________________________________|#

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
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

dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
rot_error_dict = dict()
for mouse in miceList:
    rot_error_dict[mouse] = dict()
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
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
#|                               COMPUTE SI                               |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/SI/'

sI_dict = dict()
numShuffles = 1
for mouse in miceList:
    print(f"\nWorking on mouse {mouse}: ")
    sI_dict[mouse] = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis = 0))
    dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
    velPre = copy.deepcopy(np.concatenate(animalPre['vel'].values, axis=0))
    trialPre = copy.deepcopy(np.concatenate(animalPre['index_mat'].values, axis=0))
    vecFeaturePre = np.concatenate((posPre[:,0].reshape(-1,1),dirMatPre),axis=1)
    timePre = np.arange(posPre.shape[0])

    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis = 0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    velRot = copy.deepcopy(np.concatenate(animalRot['vel'].values, axis=0))
    trialRot = copy.deepcopy(np.concatenate(animalRot['index_mat'].values, axis=0))
    vecFeatureRot = np.concatenate((posRot[:,0].reshape(-1,1),dirMatRot),axis=1)
    timeRot = np.arange(posRot.shape[0])

    #get data
    posBoth = np.vstack((posPre, posRot))
    dirMatBoth = np.concatenate((dirMatPre, dirMatRot),axis=0)
    velBoth = np.concatenate((velPre, velRot),axis=0)
    indexPreRot = np.vstack((np.zeros((posPre.shape[0],1)),np.zeros((posRot.shape[0],1))+1))
    vecFeature = np.concatenate((posBoth[:,0].reshape(-1,1),dirMatBoth),axis=1)
    trialBoth = np.concatenate((trialPre, trialRot+trialPre[-1]+1), axis=0)
    timeBoth = np.arange(posBoth.shape[0])

    for signalName in ['clean_traces', 'umap', 'isomap', 'pca']:
        print(f'\t{signalName}: ')
        signalPre = copy.deepcopy(np.concatenate(animalPre[signalName].values, axis = 0))
        signalRot = copy.deepcopy(np.concatenate(animalRot[signalName].values, axis = 0))
        if signalName == 'pca' or signalName =='isomap':
            signalPre = signalPre[:,:3]
            signalRot = signalRot[:,:3]

        D = pairwise_distances(signalPre)
        noiseIdx = filter_noisy_outliers(signalPre,D=D)
        csignalPre = signalPre[~noiseIdx,:]
        cposPre = posPre[~noiseIdx,:]
        cdirMatPre = dirMatPre[~noiseIdx]
        cvelPre = velPre[~noiseIdx]
        cvecFeaturePre = vecFeaturePre[~noiseIdx,:]
        ctrialPre = trialPre[~noiseIdx,:]
        ctimePre = timePre[~noiseIdx]

        D = pairwise_distances(signalRot)
        noiseIdx = filter_noisy_outliers(signalRot,D=D)
        csignalRot = signalRot[~noiseIdx,:]
        cposRot = posRot[~noiseIdx,:]
        cdirMatRot = dirMatRot[~noiseIdx]
        cvelRot = velRot[~noiseIdx]
        cvecFeatureRot = vecFeatureRot[~noiseIdx,:]
        ctrialRot = trialRot[~noiseIdx,:]
        ctimeRot = timeRot[~noiseIdx]

        signalBoth = np.vstack((signalPre, signalRot))
        D = pairwise_distances(signalBoth)
        noiseIdx = filter_noisy_outliers(signalBoth,D=D)
        csignalBoth = signalBoth[~noiseIdx,:]
        cposBoth = posBoth[~noiseIdx,:]
        cdirMatBoth = dirMatBoth[~noiseIdx]
        cvelBoth = velBoth[~noiseIdx]
        cvecFeature = vecFeature[~noiseIdx,:]
        cindexPreRot = indexPreRot[~noiseIdx,:]
        ctrialBoth = trialBoth[~noiseIdx,:]
        ctimeBoth = timeBoth[~noiseIdx]

        sI_dict[mouse][signalName] = dict()
        sI_dict[mouse][signalName]['pre'] = dict()
        sI_dict[mouse][signalName]['rot'] = dict()
        sI_dict[mouse][signalName]['both'] = dict()
        #--------------------------------------------------------------------------------------------------
        print('\t\tpos')
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalPre, cposPre[:,0], 
                                                    n_neighbors=20, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['pre']['pos'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalRot, cposRot[:,0], 
                                                    n_neighbors=20, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['rot']['pos'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalBoth, cposBoth[:,0], 
                                                    n_neighbors=20, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['both']['pos'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        #--------------------------------------------------------------------------------------------------
        print('\t\tdir')
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalPre, cdirMatPre, 
                                                    n_neighbors=20, discrete_label=True, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['pre']['dir'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalRot, cdirMatRot, 
                                                    n_neighbors=20, discrete_label=True, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['rot']['dir'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalBoth, cdirMatBoth, 
                                                    n_neighbors=20, discrete_label=True, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['both']['dir'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        #--------------------------------------------------------------------------------------------------
        print('\t\t(pos,dir)')
        try:
            sI, binLabel, overlapMat, ssI = compute_structure_index(csignalPre, cvecFeaturePre, 
                                                    n_neighbors=20, discrete_label=[False, True], num_shuffles=numShuffles, verbose=False)
        except:
            sI = np.nan
            binLabel = np.nan
            overlapMat = np.nan
            ssI = np.nan
        sI_dict[mouse][signalName]['pre']['(pos_dir)'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        try:
            sI, binLabel, overlapMat, ssI = compute_structure_index(csignalRot, cvecFeatureRot, 
                                                        n_neighbors=20, discrete_label=[False, True], num_shuffles=numShuffles, verbose=False)
        except:
            sI = np.nan
            binLabel = np.nan
            overlapMat = np.nan
            ssI = np.nan
        sI_dict[mouse][signalName]['rot']['(pos_dir)'] = {
                'sI': sI,
                'binLabel': binLabel,
                'overlapMat': overlapMat,
                'ssI': ssI
            }

        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalBoth, cvecFeature, 
                                                    n_neighbors=20, discrete_label=[False, True], num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['both']['(pos_dir)'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        #--------------------------------------------------------------------------------------------------
        print('\t\tvel')
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalPre, cvelPre, 
                                                    n_neighbors=20, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['pre']['vel'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalRot, cvelRot, 
                                                    n_neighbors=20, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['rot']['vel'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalBoth, cvelBoth, 
                                                    n_neighbors=20, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['both']['vel'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }

        #--------------------------------------------------------------------------------------------------
        print('\t\tsession')
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalBoth, cindexPreRot, 
                                                    n_neighbors=20, discrete_label=True, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['both']['session'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }

        #--------------------------------------------------------------------------------------------------
        print('\t\ttrial')
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalPre, ctrialPre, 
                                                    n_neighbors=10, discrete_label=False, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['pre']['trial'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalRot, ctrialRot, 
                                                    n_neighbors=10, discrete_label=False, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['rot']['trial'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalBoth, ctrialBoth, 
                                                    n_neighbors=10, discrete_label=False, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['both']['trial'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }

                #--------------------------------------------------------------------------------------------------
        print('\t\ttime')
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalPre, ctimePre, 
                                                    n_neighbors=10, discrete_label=False, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['pre']['time'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalRot, ctimeRot, 
                                                    n_neighbors=10, discrete_label=False, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['rot']['time'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        sI, binLabel, overlapMat, ssI = compute_structure_index(csignalBoth, ctimeBoth, 
                                                    n_neighbors=10, discrete_label=False, num_shuffles=numShuffles, verbose=False)
        sI_dict[mouse][signalName]['both']['time'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }
        #--------------------------------------------------------------------------------------------------
        with open(os.path.join(saveDir,'sI_clean_dict.pkl'), 'wb') as f:
            pickle.dump(sI_dict, f)

#__________________________________________________________________________
#|                                                                        |#
#|                              PLACE CELLS                               |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/'

params = {
    'sF': 20,
    'bin_width': 2.5,
    'std_pos': 0,
    'std_pdf': 5,
    'method': 'spatial_info',
    'num_shuffles': 1000,
    'min_shift': 10,
    'th_metric': 99,
    }

for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]

    mousePC = dict()
    for fname in [fnamePre, fnameRot]:
        print(f'\t{fname}')
        pdSession = copy.deepcopy(animal[fname])

        neuSignal = np.concatenate(pdSession['clean_traces'].values, axis=0)
        posSignal = np.concatenate(pdSession['pos'].values, axis=0)
        velSignal = np.concatenate(pdSession['vel'].values, axis=0)
        dirSignal = np.concatenate(pdSession['dir_mat'].values, axis=0)

        to_keep = np.logical_and(dirSignal[:,0]>0,dirSignal[:,0]<=2)
        posSignal = posSignal[to_keep,:] 
        velSignal = velSignal[to_keep] 
        neuSignal = neuSignal[to_keep,:] 
        dirSignal = dirSignal[to_keep,:] 

        mousePC[fname] = pc.get_place_cells(posSignal, neuSignal, vel_signal = velSignal, dim = 1,
                              direction_signal = dirSignal, mouse = mouse, save_dir = saveDir, **params)

        print('\tNum place cells:')
        num_cells = neuSignal.shape[1]
        num_place_cells = np.sum(mousePC[fname]['place_cells_dir'][:,0]*(mousePC[fname]['place_cells_dir'][:,1]==0))
        print(f'\t\t Only left cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
        num_place_cells = np.sum(mousePC[fname]['place_cells_dir'][:,1]*(mousePC[fname]['place_cells_dir'][:,0]==0))
        print(f'\t\t Only right cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
        num_place_cells = np.sum(mousePC[fname]['place_cells_dir'][:,0]*mousePC[fname]['place_cells_dir'][:,1])
        print(f'\t\t Both dir cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')

        with open(os.path.join(saveDir,mouse+'_pc_dict.pkl'), 'wb') as f:
            pickle.dump(mousePC, f)

#__________________________________________________________________________
#|                                                                        |#
#|                     MEASURE REMMAPING DISTANCE                         |#
#|________________________________________________________________________|#

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

dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

remapDist_dict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
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
#|                      COMPUTE ELLIPSE ECCENTRICITY                      |#
#|________________________________________________________________________|#


def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
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

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/ellipse/'
ellipseDict = dict()

for mouse in miceList:
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    ellipseDict[mouse] = dict()

    posPre = np.concatenate(animalPre['pos'].values, axis = 0)
    dirMatPre = np.concatenate(animalPre['dir_mat'].values, axis=0)
    posRot = np.concatenate(animalRot['pos'].values, axis = 0)
    dirMatRot = np.concatenate(animalRot['dir_mat'].values, axis=0)

    embPre = np.concatenate(animalPre['umap'].values, axis = 0)[:,:3]
    embRot = np.concatenate(animalRot['umap'].values, axis = 0)[:,:3]

    DPre = pairwise_distances(embPre)
    noiseIdxPre = filter_noisy_outliers(embPre,DPre)
    cembPre = embPre[~noiseIdxPre,:]
    cposPre = posPre[~noiseIdxPre,:]
    cdirMatPre = dirMatPre[~noiseIdxPre]

    DRot = pairwise_distances(embRot)
    noiseIdxRot = filter_noisy_outliers(embRot,DRot)
    cembRot = embRot[~noiseIdxRot,:]
    cposRot = posRot[~noiseIdxRot,:]
    cdirMatRot = dirMatRot[~noiseIdxRot]

    #compute centroids
    centPre, centRot = get_centroids(cembPre, cembRot, cposPre[:,0], cposRot[:,0], 
                                                    cdirMatPre, cdirMatRot, ndims = 3, nCentroids=40)  

    modelPCA = PCA(2)
    modelPCA.fit(centPre)
    centPre2D = modelPCA.transform(centPre)
    centPre2D = centPre2D - np.tile(np.mean(centPre2D,axis=0), (centPre2D.shape[0],1))

    modelPCA = PCA(2)
    modelPCA.fit(centPre)
    centRot2D = modelPCA.transform(centRot)
    centRot2D = centRot2D - np.tile(np.mean(centRot2D,axis=0), (centRot2D.shape[0],1))

    plt.figure()
    ax = plt.subplot(2,2,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')

    ax = plt.subplot(2,2,2, projection = '3d')
    ax.scatter(*centPre[:,:3].T, color ='b', s=10)
    ax.scatter(*centRot[:,:3].T, color = 'r', s=10)

    ########################
    #         PRE          #
    ########################
    # Formulate and solve the least squares problem ||Ax - b ||^2
    X = centPre2D[:,0].reshape(-1,1)
    Y = centPre2D[:,1].reshape(-1,1)
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(centPre2D[:,0])
    x = np.linalg.lstsq(A, b)[0].squeeze()

    xLim = [np.min(centPre2D[:,0]), np.max(centPre2D[:,0])]
    yLim = [np.min(centPre2D[:,1]), np.max(centPre2D[:,1])]
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

    #Compute Excentricity
    distEllipse = np.sqrt((xValid-center[0])**2 + (yValid-center[1])**2)
    pointLong = [xValid[np.argmax(distEllipse)], yValid[np.argmax(distEllipse)]]
    pointShort = [xValid[np.argmin(distEllipse)], yValid[np.argmin(distEllipse)]]
    longAxis = np.max(distEllipse)
    shortAxis = np.min(distEllipse)

    #Plot
    ax = plt.subplot(2,2,3)
    ax.scatter(*centPre2D[:,:2].T, color ='b', s=10)
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    plt.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, color ='m', s=10)
    ax.plot([center[0],pointLong[0]], [center[1], pointLong[1]], color = 'c')
    ax.plot([center[0],pointShort[0]], [center[1], pointShort[1]], color = 'c')
    ax.set_xlim(np.min(centPre2D[:,0])-0.2, np.max(centPre2D[:,0])+0.2)
    ax.set_ylim(np.min(centPre2D[:,1])-0.2, np.max(centPre2D[:,1])+0.2)
    ax.set_xlim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_ylim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{longAxis/shortAxis:4f}")

    ellipseDict[mouse][fnamePre] = {
        'pos':posPre,
        'dirMat': dirMatPre,
        'emb': embPre,
        'D': DPre,
        'noiseIdx': noiseIdxPre,
        'cpos': cposPre,
        'cdirMat': cdirMatPre,
        'cemb': cembPre,
        'cent': centPre,
        'cent2D': centPre2D,
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
        'distEllipse': distEllipse,
        'pointLong': pointLong,
        'pointShort': pointShort,
        'longAxis': longAxis,
        'shortAxis': shortAxis,
        'eccentricity': longAxis/shortAxis
    }

    ########################
    #         ROT          #
    ########################
    # Formulate and solve the least squares problem ||Ax - b ||^2
    X = centRot2D[:,0].reshape(-1,1)
    Y = centRot2D[:,1].reshape(-1,1)
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(centRot2D[:,0])
    x = np.linalg.lstsq(A, b)[0].squeeze()

    xLim = [np.min(centRot2D[:,0]), np.max(centRot2D[:,0])]
    yLim = [np.min(centRot2D[:,1]), np.max(centRot2D[:,1])]
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

    #Compute Excentricity
    distEllipse = np.sqrt((xValid-center[0])**2 + (yValid-center[1])**2)
    pointLong = [xValid[np.argmax(distEllipse)], yValid[np.argmax(distEllipse)]]
    pointShort = [xValid[np.argmin(distEllipse)], yValid[np.argmin(distEllipse)]]
    longAxis = np.max(distEllipse)
    shortAxis = np.min(distEllipse)

    #Plot
    ax = plt.subplot(2,2,4)
    ax.scatter(*centRot2D[:,:2].T, color ='b', s=10)
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    plt.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, color ='m', s=10)
    ax.plot([center[0],pointLong[0]], [center[1], pointLong[1]], color = 'c')
    ax.plot([center[0],pointShort[0]], [center[1], pointShort[1]], color = 'c')
    ax.set_xlim(np.min(centRot2D[:,0])-0.2, np.max(centRot2D[:,0])+0.2)
    ax.set_ylim(np.min(centRot2D[:,1])-0.2, np.max(centRot2D[:,1])+0.2)

    ax.set_xlim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_ylim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{longAxis/shortAxis:4f}")
    plt.suptitle(mouse)


    ellipseDict[mouse][fnameRot] = {
        'pos':posRot,
        'dirMat': dirMatRot,
        'emb': embRot,
        'D': DRot,
        'noiseIdx': noiseIdxRot,
        'cpos': cposRot,
        'cdirMat': cdirMatRot,
        'cemb': cembRot,
        'cent': centRot,
        'cent2D': centRot2D,
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
        'distEllipse': distEllipse,
        'pointLong': pointLong,
        'pointShort': pointShort,
        'longAxis': longAxis,
        'shortAxis': shortAxis,
        'eccentricity': longAxis/shortAxis
    }

    with open(os.path.join(saveDir,'ellipse_fit_dict.pkl'), 'wb') as f:
        pickle.dump(ellipseDict, f)

    plt.savefig(os.path.join(saveDir,f'{mouse}_ellipse_fit.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDir,f'{mouse}_ellipse_fit.svg'), dpi = 400,bbox_inches="tight",transparent=True)


#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT PLACE CELLS                           |#
#|________________________________________________________________________|#
import scipy.signal as scs
from scipy.ndimage import convolve1d
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

#Adapted from PyalData package (19/10/21) (added variable win_length)
def _norm_gauss_window(bin_size, std, num_std = 5):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_size (float): binning size of the array we want to smooth in same time 
                units as the std
    
    std (float): standard deviation of the window use hw_to_std to calculate 
                std based from half-width (same time units as bin_size)
                
    num_std (int): size of the window to convolve in #of stds

    Returns
    -------
    win (1D np.array): Gaussian kernel with length: num_bins*std/bin_length
                mass normalized to 1
    """
    win_len = int(num_std*std/bin_size)
    if win_len%2==0:
        win_len = win_len+1
    win = scs.gaussian(win_len, std/bin_size)      
    return win / np.sum(win)

#Copied from PyalData package (19/10/21)
def _hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))

#Copied from PyalData package (19/10/21)
def _smooth_data(mat, bin_size=None, std=None, hw=None, win=None, axis=0):
    """
    Smooth a 1D array or every column of a 2D array

    Parameters
    ----------
    mat : 1D or 2D np.array
        vector or matrix whose columns to smooth
        e.g. recorded spikes in a time x neuron array
    bin_size : float
        length of the timesteps in seconds
    std : float (optional)
        standard deviation of the smoothing window
    hw : float (optional)
        half-width of the smoothing window
    win : 1D array-like (optional)
        smoothing window to convolve with

    Returns
    -------
    np.array of the same size as mat
    """
    #assert mat.ndim == 1 or mat.ndim == 2, "mat has to be a 1D or 2D array"
    assert  sum([arg is not None for arg in [win, hw, std]]) == 1, "only give win, hw, or std"
    if win is None:
        assert bin_size is not None, "specify bin_size if not supplying window"
        if std is None:
            std = _hw_to_std(hw)
        win = _norm_gauss_window(bin_size, std)
    return convolve1d(mat, win, axis=axis, output=np.float32, mode='reflect')

def _get_pdf(pos, neu_signal, mapAxis, dim, dir_mat = None):
    """Obtain probability density function of 'pos' along each bin defined by 
    the limits mapAxis as well as the mean signal of 'neu_signal' on those bins.
    If dir_mat is provided, the above is computed for each possible direction.

    Parameters
    ----------
    pos: numpy array (T, space_dims)
        Numpy array with the position
        
    neu_signal: numpy array (T, nn)
        Numpy array containing the neural signal (spikes, rates, or traces).

    mapAxis: list
        List with as many entries as spatial dimensions. Each entry contains 
        the lower limits of the bins one wants to divide that spatial dimension 
        into. 
            e.g.
            If we want to divide our 2D space, then mapAxis will contain 2 
            entries. The first entry will have the lower limits of the x-bins 
            and the second entry those of the y-bins.
    
    dim: integer
        Integer indicating the number of spatial dimenions of the bins. That
        is, 
            dim = len(mapAxis) <= pos.shape[1]
            
    Optional parameters:
    --------------------
    dir_mat: numpy array (T,1)
        Numpy array containing the labels of the direction the animal was 
        moving on each timestamp. For each label, a separate pdf will be 
        computed.
        
    Returns
    -------
    pos_pdf: numpy array (space_dims_bins, 1, dir_labels)
        Numpy array containing the probability the animal is at each bin for
        each of the directions if dir_mat was indicated.
    
    neu_pdf: numpy array (space_dims_bins, nn, dir_labels)
        Numpy array containing the mean neural activity of each neuron at each 
        bin for each of the directions if dir_mat was indicated.
    """
    if isinstance(dir_mat, type(None)):
        dir_mat = np.zeros((pos.shape[0],1))
        
    val_dirs = np.array(np.unique(dir_mat))
    num_dirs = len(val_dirs)
                   
    subAxis_length =  [len(mapsubAxis) for mapsubAxis in mapAxis] #nbins for each axis
    access_neu = np.linspace(0,neu_signal.shape[1]-1,neu_signal.shape[1]).astype(int) #used to later access neu_pdf easily

    neu_pdf = np.zeros(subAxis_length+[neu_signal.shape[1]]+[num_dirs])
    pos_pdf = np.zeros(subAxis_length+[1]+[num_dirs])
    
    for sample in range(pos.shape[0]):
        pos_entry_idx = list()
        for d in range(dim):
            temp_entry = np.where(pos[sample,d]>=mapAxis[d])
            if np.any(temp_entry):
                temp_entry = temp_entry[0][-1]
            else:
                temp_entry = 0
            pos_entry_idx.append(temp_entry)
        
        dir_idx = np.where(dir_mat[sample]==val_dirs)[0][0]
        pos_idxs = tuple(pos_entry_idx + [0]+[dir_idx])
        neu_idxs = tuple(pos_entry_idx + [access_neu]+[dir_idx])
        
        pos_pdf[pos_idxs] += 1
        neu_pdf[neu_idxs] += (1/pos_pdf[pos_idxs])* \
                                   (neu_signal[sample,:]-neu_pdf[neu_idxs]) #online average
    


    pos_pdf = pos_pdf/np.sum(pos_pdf,axis=tuple(range(pos_pdf.ndim - 1))) #normalize probability density function
    
    return pos_pdf, neu_pdf

def _get_edges(pos, limit, dim):
    """Obtain which points of 'pos' are inside the limits of the border defined
    by limit (%).
    
    Parameters
    ----------
    pos: numpy array
        Numpy array with the position
        
    limit: numerical (%)
        Limit (%) used to decided which points are kepts and which ones are 
        discarded. (i.e. if limit=10 and pos=[0,100], only the points inside the
                    range [10,90] will be kepts).
    
    dim: integer
        Dimensionality of the division.    
                
    Returns
    -------
    signal: numpy array
        Concatenated dataframe column into numpy array along axis=0.
    """  
    assert pos.shape[1]>=dim, f"pos has less dimensions ({pos.shape[1]}) " + \
                                f"than the indicated in dim ({dim})"
    norm_limit = limit/100
    minpos = np.min(pos, axis=0)
    maxpos = np.max(pos,axis=0)
    
    trackLength = maxpos - minpos
    trackLimits = np.vstack((minpos + norm_limit*trackLength,
                             maxpos - norm_limit*trackLength))
    
    points_inside_lim = list()
    for d in range(dim):
        points_inside_lim.append(np.vstack((pos[:,d]<trackLimits[0,d],
                                       pos[:,d]>trackLimits[1,d])).T)
        
    points_inside_lim = np.concatenate(points_inside_lim, axis=1)
    points_inside_lim = ~np.any(points_inside_lim ,axis=1)
    
    return points_inside_lim

def get_neu_pdf(pos, neuSignal, vel = None, dim=2, **kwargs):
    #smooth pos signal if applicable

    if kwargs['std_pos']>0:
        pos = _smooth_data(pos, std = kwargs['std_pos'], bin_size = 1/kwargs['sF'])
    if pos.ndim == 1:
        pos = pos.reshape(-1,1)

    #Compute velocity if not provided
    if isinstance(vel, type(None)):
        vel = np.linalg.norm(np.diff(pos_signal, axis= 0), axis=1)*kwargs['sF']
        vel = np.hstack((vel[0], vel))
    vel = vel.reshape(-1)

    #Compute dirSignal if not provided
    if 'dirSignal' in kwargs:
        dirSignal = kwargs["dirSignal"]
    else:
        dirSignal = np.zeros((pos.shape[0],1))

    #Discard edges
    if kwargs['ignoreEdges']>0:
        posBoolean = _get_edges(pos, kwargs['ignoreEdges'], 1)
        pos = pos[posBoolean]
        neuSignal = neuSignal[posBoolean]
        dirSignal = dirSignal[posBoolean]
        vel = vel[posBoolean]

    #compute moving epochs
    moveEpochs = vel>=kwargs['velTh'] 
    #keep only moving epochs
    pos = pos[moveEpochs]
    neuSignal = neuSignal[moveEpochs]
    dirSignal = dirSignal[moveEpochs]
    vel = vel[moveEpochs]
        
    #Create grid along each dimensions
    minPos = np.percentile(pos,1, axis=0) #(pos.shape[1],)
    maxPos = np.percentile(pos,99, axis = 0) #(pos.shape[1],)
    obsLength = maxPos - minPos #(pos.shape[1],)
    if 'binWidth' in kwargs:
        binWidth = kwargs['binWidth']
        if isinstance(binWidth, list):
            binWidth = np.array(binWidth)
        else:
            binWidth = np.repeat(binWidth, dim, axis=0) #(pos.shape[1],)
        nbins = np.ceil(obsLength[:dim]/binWidth).astype(int) #(pos.shape[1],)
        kwargs['nbins'] = nbins
    elif 'binNum' in kwargs:
        nbins = kwargs['binNum']
        if isinstance(nbins, list):
            nbins = np.array(nbins)
        else:
            nbins = np.repeat(nbins, dim, axis=0) #(pos.shape[1],)
        binWidth = np.round(obsLength[:dim]/nbins,4) #(pos.shape[1],)
        kwargs['binWidth'] = binWidth
    mapAxis = list()
    for d in range(dim):
        mapAxis.append(np.linspace(minPos[d], maxPos[d], nbins[d]+1)[:-1].reshape(-1,1)); #(nbins[d],)

    #Compute probability density function
    posPDF, neuPDF = _get_pdf(pos, neuSignal, mapAxis, dim, dirSignal)
    for d in range(dim):
        posPDF = _smooth_data(posPDF, std=kwargs['stdPDF'], bin_size=binWidth[d], axis=d)
        neuPDF = _smooth_data(neuPDF, std=kwargs['stdPDF'], bin_size=binWidth[d], axis=d)
    return posPDF, neuPDF, mapAxis


supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/manifold_cells/pre/'
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

params = {
    'std_pos': 0.5,
    'sF': 20,
    'ignoreEdges': 0,
    'velTh': 0,
    'binWidth': 2.5,
    'stdPDF': 5
}


pcManifoldDict = dict()
for mouse in miceList:
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])

    emb = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis = 0))
    signal = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis = 0))
    pos = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis = 0))
    vel = copy.deepcopy(np.concatenate(animalPre['vel'].values, axis = 0))

    posPDF, neuPDF, mapAxis = get_neu_pdf(pos,signal, vel = vel, dim= 1, **params)
    neuPDF_norm = np.zeros((neuPDF.shape[0],neuPDF.shape[1]))
    for c in range(neuPDF.shape[1]):
        neuPDF_norm[:,c] = neuPDF[:,c,0]/np.max(neuPDF[:,c,0])

    mneuPDF = np.nanmean(neuPDF_norm, axis=1)

    manifoldSignal = np.zeros((emb.shape[0]))
    for p in range(emb.shape[0]):
        try:
            x = np.where(mapAxis[0]<=pos[p,0])[0][-1]
        except: 
            x = 0
        manifoldSignal[p] = mneuPDF[x]
    manifoldSignal = gaussian_filter1d(manifoldSignal, sigma = 3, axis = 0)
    pcManifoldDict[mouse] = {
        'emb': emb,
        'pos': pos,
        'signal': signal,
        'vel': vel,
        'mapAxis': mapAxis,
        'posPDF': posPDF,
        'neuPDF': neuPDF,
        'mneuPDF': mneuPDF,
        'manifoldSignal': manifoldSignal,
        'fname': fnamePre,
        'params': params
    }

    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(1,2,1, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 30, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    # ax.view_init(130,110,90)
    ax.view_init(-65,-85,90)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax = plt.subplot(1,2,2, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 30)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    # ax.view_init(130,110,90)
    ax.view_init(-65,-85,90)

    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,mouse+'_PDF_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_PDF_emb.png'), dpi = 400,bbox_inches="tight")

    with open(os.path.join(saveDir, "manifold_pc_dict.pkl"), "wb") as file:
        pickle.dump(pcManifoldDict, file, protocol=pickle.HIGHEST_PROTOCOL)

#__________________________________________________________________________
#|                                                                        |#
#|                             remove ROT Cells                           |#
#|________________________________________________________________________|#

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
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

def compute_entanglement(points):
    distance_b = pairwise_distances(points)
    model_iso = Isomap(n_neighbors = 10, n_components = 1)
    emb = model_iso.fit_transform(points)
    distance_a = model_iso.dist_matrix_
    entanglement_boundary = np.max(distance_a[1:,0])/np.min(distance_b[1:,0])
    entanglement = np.max((distance_a[1:,0]/distance_b[1:,0]))
    return (entanglement-1)/entanglement_boundary

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

dim= 3
nNeigh = 120
minDist = 0.1
numIters = 100

deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
cellTypeDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'

for mouse in deepMice:
    print(f'Working on mouse: {mouse}')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    signalPre = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis=0))
    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
    vfeatPre = np.concatenate((posPre[:,0].reshape(-1,1),dirMatPre),axis=1)

    signalRot = copy.deepcopy(np.concatenate(animalRot['clean_traces'].values, axis=0))
    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    vfeatRot = np.concatenate((posRot[:,0].reshape(-1,1),dirMatRot),axis=1)

    #%%all data
    index = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalRot.shape[0],1))+1))
    signalBoth = np.vstack((signalPre, signalRot))

    cellTypePath = os.path.join(cellTypeDir, mouse)
    cellTypeName = mouse+'_cellType.npy'
    cellType = np.load(os.path.join(cellTypePath,cellTypeName))
    rotCells = np.where(np.logical_and(cellType<4,cellType>0))[0]
    rotError = np.zeros((100, numIters, rotCells.shape[0]))
    rotAngle = np.zeros((numIters, rotCells.shape[0]))
    SIVal = np.zeros((2,numIters,rotCells.shape[0]))
    remapDist = np.zeros((numIters, rotCells.shape[0]))
    interDist = np.zeros((numIters, rotCells.shape[0]))
    intraPre = np.zeros((numIters, rotCells.shape[0]))
    intraRot = np.zeros((numIters, rotCells.shape[0]))
    entang = np.zeros((2,numIters, rotCells.shape[0]))
    embPreSave = dict()
    embRotSave = dict()
    newOrderList = list()
    for it in range(numIters):
        new_order = np.random.permutation(signalBoth.shape[1])
        newOrderList.append(new_order)
        itSignalBoth = copy.deepcopy(signalBoth[:, new_order])
        scellType = cellType[new_order]
        shufRotCells = np.where(np.logical_and(scellType<4,scellType>0))[0]
        for idx in range(shufRotCells.shape[0]):
            idxSignalBoth = copy.deepcopy(itSignalBoth)
            idxSignalBoth = np.delete(idxSignalBoth, shufRotCells[:idx+1], axis = 1)

            model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=minDist)
            model.fit(idxSignalBoth)
            concat_emb = model.transform(idxSignalBoth)
            embPre = concat_emb[index[:,0]==0,:]
            embRot = concat_emb[index[:,0]==1,:]

            DPre = pairwise_distances(embPre)
            noiseIdxPre = filter_noisy_outliers(embPre,DPre)
            cembPre = embPre[~noiseIdxPre,:]
            cposPre = posPre[~noiseIdxPre,:]
            cvfeatPre = vfeatPre[~noiseIdxPre,:]
            cdirMatPre = dirMatPre[~noiseIdxPre]

            DRot = pairwise_distances(embRot)
            noiseIdxRot = filter_noisy_outliers(embRot,DRot)
            # noiseIdxRot[dirMatRot==0] = True
            cembRot = embRot[~noiseIdxRot,:]
            cposRot = posRot[~noiseIdxRot,:]
            cvfeatRot = vfeatRot[~noiseIdxRot,:]
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
            rotError[:,it, idx] = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
            rotAngle[it, idx] = angles[np.argmin(error)]


            interDist[it, idx] = np.mean(pairwise_distances(centPre, centRot))
            intraPre[it, idx] = np.percentile(pairwise_distances(centPre),95)
            intraRot[it, idx] = np.percentile(pairwise_distances(centRot),95)

            remapDist[it, idx] = interDist[it, idx]/np.max((intraPre[it, idx], intraRot[it, idx]))

            entang[0,it,idx] = compute_entanglement(centPre)
            entang[1,it,idx] = compute_entanglement(centRot)
            #compute SI of emb
            try:
                SIVal[0,it, idx],_,_,_ = compute_structure_index(cembPre, cvfeatPre, 
                                                            n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)
            except:
                SIVal[0,it, idx] = np.nan
            try:
                SIVal[1,it, idx],_,_,_ = compute_structure_index(cembRot, cvfeatRot, 
                                                            n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)
            except:
                SIVal[1,it, idx] = np.nan
            print(f"Iter: {it+1}/{numIters} | Idx: {idx+1}/{shufRotCells.shape[0]} | Rot: {(rotAngle[it,idx]*180/np.pi):.2f} | ",
                f"SI_p: {SIVal[0,it,idx]:.2f} | SI_r: {SIVal[1,it,idx]:.2f} | Dist: {remapDist[it,idx]:.2f} | Entang: {np.mean(entang[:, it,idx]):.4f}")

            embPreSave[it,idx] = embPre
            embRotSave[it,idx] = embRot

        remove_rotCells_dict = {
            'rotAngle': rotAngle,
            'rotError': rotError,
            'remapDist': remapDist,
            'interDist': interDist,
            'intraPre': intraPre,
            'intraRot': intraRot,
            'SIVal': SIVal,
            'newOrderList': newOrderList,
            'embPreSave': embPreSave,
            'embRotSave': embRotSave,
            'entang': entang
        }
        with open(os.path.join(cellTypePath,mouse+'_remRotCells.pkl'), 'wb') as f:
            pickle.dump(remove_rotCells_dict, f)

#__________________________________________________________________________
#|                                                                        |#
#|                           remove Remap Cells                           |#
#|________________________________________________________________________|#
dim= 3
nNeigh = 120
minDist = 0.1
numIters = 100
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
cellTypeDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    signalPre = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis=0))
    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
    vfeatPre = np.concatenate((posPre[:,0].reshape(-1,1),dirMatPre),axis=1)

    signalRot = copy.deepcopy(np.concatenate(animalRot['clean_traces'].values, axis=0))
    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    vfeatRot = np.concatenate((posRot[:,0].reshape(-1,1),dirMatRot),axis=1)

    #%%all data
    index = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalRot.shape[0],1))+1))
    signalBoth = np.vstack((signalPre, signalRot))

    cellTypePath = os.path.join(cellTypeDir, mouse)
    cellTypeName = mouse+'_cellType.npy'
    cellType = np.load(os.path.join(cellTypePath,cellTypeName))
    remapCells = np.where(cellType==4)[0]
    rotError = np.zeros((100, numIters, remapCells.shape[0]))
    rotAngle = np.zeros((numIters, remapCells.shape[0]))
    SIVal = np.zeros((2,numIters,remapCells.shape[0]))
    remapDist = np.zeros((numIters, remapCells.shape[0]))
    interDist = np.zeros((numIters, remapCells.shape[0]))
    intraPre = np.zeros((numIters, remapCells.shape[0]))
    intraRot = np.zeros((numIters, remapCells.shape[0]))
    entang = np.zeros((2,numIters, remapCells.shape[0]))
    embPreSave = dict()
    embRotSave = dict()
    newOrderList = list()
    for it in range(numIters):
        new_order = np.random.permutation(signalBoth.shape[1])
        newOrderList.append(new_order)
        itSignalBoth = copy.deepcopy(signalBoth[:, new_order])
        scellType = cellType[new_order]
        shufRemmapCells = np.where(scellType==4)[0]
        for idx in range(shufRemmapCells.shape[0]):
            idxSignalBoth = copy.deepcopy(itSignalBoth)
            idxSignalBoth = np.delete(idxSignalBoth, shufRemmapCells[:idx+1], axis = 1)

            model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=minDist)
            model.fit(idxSignalBoth)
            concat_emb = model.transform(idxSignalBoth)
            embPre = concat_emb[index[:,0]==0,:]
            embRot = concat_emb[index[:,0]==1,:]

            DPre = pairwise_distances(embPre)
            noiseIdxPre = filter_noisy_outliers(embPre,DPre)
            cembPre = embPre[~noiseIdxPre,:]
            cposPre = posPre[~noiseIdxPre,:]
            cvfeatPre = vfeatPre[~noiseIdxPre,:]
            cdirMatPre = dirMatPre[~noiseIdxPre]

            DRot = pairwise_distances(embRot)
            noiseIdxRot = filter_noisy_outliers(embRot,DRot)
            # noiseIdxRot[dirMatRot==0] = True
            cembRot = embRot[~noiseIdxRot,:]
            cposRot = posRot[~noiseIdxRot,:]
            cvfeatRot = vfeatRot[~noiseIdxRot,:]
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
            rotError[:,it, idx] = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
            rotAngle[it, idx] = angles[np.argmin(error)]


            interDist[it, idx] = np.mean(pairwise_distances(centPre, centRot))
            intraPre[it, idx] = np.percentile(pairwise_distances(centPre),95)
            intraRot[it, idx] = np.percentile(pairwise_distances(centRot),95)

            remapDist[it, idx] = interDist[it, idx]/np.max((intraPre[it, idx], intraRot[it, idx]))

            entang[0,it,idx] = compute_entanglement(centPre)
            entang[1,it,idx] = compute_entanglement(centRot)
            #compute SI of emb
            try:
                SIVal[0,it, idx],_,_,_ = compute_structure_index(cembPre, cvfeatPre, 
                                                            n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)
            except:
                SIVal[0,it, idx] = np.nan
            try:
                SIVal[1,it, idx],_,_,_ = compute_structure_index(cembRot, cvfeatRot, 
                                                            n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)
            except:
                SIVal[1,it, idx] = np.nan
            print(f"Iter: {it+1}/{numIters} | Idx: {idx+1}/{shufRemmapCells.shape[0]} | Rot: {(rotAngle[it,idx]*180/np.pi):.2f} | ",
                f"SI_p: {SIVal[0,it,idx]:.2f} | SI_r: {SIVal[1,it,idx]:.2f} | Dist: {remapDist[it,idx]:.2f} | Entang: {np.mean(entang[:, it,idx]):.4f}")

            embPreSave[it,idx] = embPre
            embRotSave[it,idx] = embRot

        remove_remapCells_dict = {
            'rotAngle': rotAngle,
            'rotError': rotError,
            'remapDist': remapDist,
            'interDist': interDist,
            'intraPre': intraPre,
            'intraRot': intraRot,
            'SIVal': SIVal,
            'newOrderList': newOrderList,
            'embPreSave': embPreSave,
            'embRotSave': embRotSave,
            'entang': entang
        }
        with open(os.path.join(cellTypePath,mouse+'_remRemapCells.pkl'), 'wb') as f:
            pickle.dump(remove_remapCells_dict, f)

#__________________________________________________________________________
#|                                                                        |#
#|                        remove Allocentric Cells                        |#
#|________________________________________________________________________|#
dim= 3
nNeigh = 120
minDist = 0.1
numIters = 100

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
cellTypeDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'

for mouse in supMice:
    print(f'Working on mouse: {mouse}')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    signalPre = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis=0))
    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
    vfeatPre = np.concatenate((posPre[:,0].reshape(-1,1),dirMatPre),axis=1)

    signalRot = copy.deepcopy(np.concatenate(animalRot['clean_traces'].values, axis=0))
    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    vfeatRot = np.concatenate((posRot[:,0].reshape(-1,1),dirMatRot),axis=1)

    #%%all data
    index = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalRot.shape[0],1))+1))
    signalBoth = np.vstack((signalPre, signalRot))

    cellTypePath = os.path.join(cellTypeDir, mouse)
    cellTypeName = mouse+'_cellType.npy'
    cellType = np.load(os.path.join(cellTypePath,cellTypeName))
    AlloCells = np.where(cellType==0)[0]

    rotError = np.zeros((100, numIters, AlloCells.shape[0]))
    rotAngle = np.zeros((numIters, AlloCells.shape[0]))
    SIVal = np.zeros((2,numIters,AlloCells.shape[0]))
    remapDist = np.zeros((numIters, AlloCells.shape[0]))
    interDist = np.zeros((numIters, AlloCells.shape[0]))
    intraPre = np.zeros((numIters, AlloCells.shape[0]))
    intraRot = np.zeros((numIters, AlloCells.shape[0]))
    entang = np.zeros((2,numIters, AlloCells.shape[0]))
    embPreSave = dict()
    embRotSave = dict()
    newOrderList = list()
    for it in range(numIters):
        new_order = np.random.permutation(signalBoth.shape[1])
        newOrderList.append(new_order)
        itSignalBoth = copy.deepcopy(signalBoth[:, new_order])
        scellType = cellType[new_order]
        shufAlloCells = np.where(scellType==0)[0]
        for idx in range(shufAlloCells.shape[0]):
            idxSignalBoth = copy.deepcopy(itSignalBoth)
            idxSignalBoth = np.delete(idxSignalBoth, shufAlloCells[:idx+1], axis = 1)

            model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=minDist)
            model.fit(idxSignalBoth)
            concat_emb = model.transform(idxSignalBoth)
            embPre = concat_emb[index[:,0]==0,:]
            embRot = concat_emb[index[:,0]==1,:]

            DPre = pairwise_distances(embPre)
            noiseIdxPre = filter_noisy_outliers(embPre,DPre)
            cembPre = embPre[~noiseIdxPre,:]
            cposPre = posPre[~noiseIdxPre,:]
            cvfeatPre = vfeatPre[~noiseIdxPre,:]
            cdirMatPre = dirMatPre[~noiseIdxPre]

            DRot = pairwise_distances(embRot)
            noiseIdxRot = filter_noisy_outliers(embRot,DRot)
            # noiseIdxRot[dirMatRot==0] = True
            cembRot = embRot[~noiseIdxRot,:]
            cposRot = posRot[~noiseIdxRot,:]
            cvfeatRot = vfeatRot[~noiseIdxRot,:]
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
            rotError[:,it, idx] = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
            rotAngle[it, idx] = angles[np.argmin(error)]


            interDist[it, idx] = np.mean(pairwise_distances(centPre, centRot))
            intraPre[it, idx] = np.percentile(pairwise_distances(centPre),95)
            intraRot[it, idx] = np.percentile(pairwise_distances(centRot),95)

            remapDist[it, idx] = interDist[it, idx]/np.max((intraPre[it, idx], intraRot[it, idx]))

            entang[0,it,idx] = compute_entanglement(centPre)
            entang[1,it,idx] = compute_entanglement(centRot)
            #compute SI of emb
            try:
                SIVal[0,it, idx],_,_,_ = compute_structure_index(cembPre, cvfeatPre, 
                                                            n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)
            except:
                SIVal[0,it, idx] = np.nan
            try:
                SIVal[1,it, idx],_,_,_ = compute_structure_index(cembRot, cvfeatRot, 
                                                            n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)
            except:
                SIVal[1,it, idx] = np.nan
            print(f"Iter: {it+1}/{numIters} | Idx: {idx+1}/{shufAlloCells.shape[0]} | Rot: {(rotAngle[it,idx]*180/np.pi):.2f} | ",
                f"SI_p: {SIVal[0,it,idx]:.2f} | SI_r: {SIVal[1,it,idx]:.2f} | Dist: {remapDist[it,idx]:.2f} | Entang: {np.mean(entang[:, it,idx]):.4f}")

            embPreSave[it,idx] = embPre
            embRotSave[it,idx] = embRot

        remove_AlloCells_dict = {
            'rotAngle': rotAngle,
            'rotError': rotError,
            'remapDist': remapDist,
            'interDist': interDist,
            'intraPre': intraPre,
            'intraRot': intraRot,
            'SIVal': SIVal,
            'newOrderList': newOrderList,
            'embPreSave': embPreSave,
            'embRotSave': embRotSave,
            'entang': entang
        }
        with open(os.path.join(cellTypePath,mouse+'_remAlloCells.pkl'), 'wb') as f:
            pickle.dump(remove_AlloCells_dict, f)

