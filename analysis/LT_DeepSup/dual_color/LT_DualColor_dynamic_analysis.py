import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neural_manifold.general_utils as gu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from neural_manifold import dimensionality_reduction as dim_red
import seaborn as sns
# from neural_manifold.dimensionality_reduction import validation as dim_validation 
import umap
# import math
# from kneed import KneeLocator
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from structure_index import compute_structure_index, draw_graph
from neural_manifold import decoders as dec
from datetime import datetime
from neural_manifold import place_cells as pc

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

def remove_cells(pd_struct, color, cells_to_keep):
    out_pd = copy.deepcopy(pd_struct)

    out_pd["index_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            out_pd["trial_id"][idx]
                            for idx in out_pd.index]

    index_mat = np.concatenate(out_pd["index_mat"].values, axis=0)
    neuro_fields = get_neuronal_fields(out_pd, color+'_raw_traces')
    for field in neuro_fields:
        signal = np.concatenate(out_pd[field].values, axis=0)
        out_pd[field] = [signal[index_mat[:,0]==out_pd["trial_id"][idx]][:, cells_to_keep] 
                                                for idx in range(out_pd.shape[0])]
    return out_pd


miceList = ['ThyCalbRCaMP2']
baseLoadDir = '/home/julio/Documents/SP_project/LT_DualColor/data/'
baseSaveDir = '/home/julio/Documents/SP_project/LT_DualColor/processed_data/'
signalField = ['green_raw_traces', 'red_raw_traces']

velTh = 1
sigma = 6
sigUp = 4
sigDown = 12
nNeigh = 120
dim = 3

for mouse in miceList:
    print(f"Working on mouse: {mouse}")
    loadDir = os.path.join(baseLoadDir, mouse)
    saveDir = os.path.join(baseSaveDir, mouse)
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
    animalPre, animalPreStill = gu.keep_only_moving(animalPre, velTh)
    animalRot, animalRotStill = gu.keep_only_moving(animalRot, velTh)
    for color in ['green', 'red']:
        animalPre, animalRot = preprocess_traces(animalPre, animalRot, color+'_raw_traces', color+'_clean_traces', sigma=sigma, sig_up = sigUp, sig_down = sigDown)
        animalPreStill, animalRotStill = preprocess_traces(animalPreStill, animalRotStill, color+'_raw_traces', color+'_clean_traces', sigma=sigma, sig_up = sigUp, sig_down = sigDown)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                               REMOVE CELLS                             |#
    #|________________________________________________________________________|#

    gsignalPre = copy.deepcopy(np.concatenate(animalPre['green_clean_traces'].values, axis=0))
    rsignalPre = copy.deepcopy(np.concatenate(animalPre['red_clean_traces'].values, axis=0))
    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    gsignalRot = copy.deepcopy(np.concatenate(animalRot['green_clean_traces'].values, axis=0))
    rsignalRot = copy.deepcopy(np.concatenate(animalRot['red_clean_traces'].values, axis=0))
    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))

    #%%all data
    indexGreen = np.vstack((np.zeros((gsignalPre.shape[0],1)),np.zeros((gsignalRot.shape[0],1))+1))
    gsignalBoth = np.vstack((gsignalPre, gsignalRot))
    model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=0.1)
    model.fit(gsignalBoth)
    gembBoth = model.transform(gsignalBoth)
    gembPre = gembBoth[indexGreen[:,0]==0,:]
    gembRot = gembBoth[indexGreen[:,0]==1,:]

    #%%all data
    indexRed = np.vstack((np.zeros((rsignalPre.shape[0],1)),np.zeros((rsignalRot.shape[0],1))+1))
    rsignalBoth = np.vstack((rsignalPre, rsignalRot))
    model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=0.1)
    model.fit(rsignalBoth)
    rembBoth = model.transform(rsignalBoth)
    rembPre = rembBoth[indexRed[:,0]==0,:]
    rembRot = rembBoth[indexRed[:,0]==1,:]

    # model.fit(gsignalPre)
    # gembPre = model.transform(gsignalPre)
    # model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=0.1)
    # model.fit(gsignalRot)
    # gembRot = model.transform(gsignalRot)

    # model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=0.1)
    # model.fit(rsignalPre)
    # rembPre = model.transform(rsignalPre)
    # model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=0.1)
    # model.fit(rsignalRot)
    # rembRot = model.transform(rsignalRot)

    #%%
    plt.figure()
    ax = plt.subplot(2,2,1, projection = '3d')
    ax.scatter(*gembPre[:,:3].T, color='b', s=10)
    ax.scatter(*gembRot[:,:3].T, color= 'r', s=10)
    ax.set_title('Green')
    ax = plt.subplot(2,2,2, projection = '3d')
    plotEmb = np.concatenate((gembPre, gembRot),axis=0)[:,:3]
    plotFeat = np.concatenate((posPre, posRot), axis=0)[:,0]
    ax.scatter(*plotEmb.T, c = plotFeat, s=10, cmap = 'magma')

    ax = plt.subplot(2,2,3, projection = '3d')
    ax.scatter(*rembPre[:,:3].T, color='b', s=10)
    ax.scatter(*rembRot[:,:3].T, color= 'r', s=10)
    ax.set_title('Red')
    ax = plt.subplot(2,2,4, projection = '3d')
    plotEmb = np.concatenate((rembPre, rembRot),axis=0)[:,:3]
    ax.scatter(*plotEmb.T, c = plotFeat, s= 5, cmap = 'magma')
    plt.tight_layout()
    plt.suptitle(f"{mouse}: clean_traces - vel: {velTh} - nn: {nNeigh} - dim: {dim}")
    plt.savefig(os.path.join(saveDirFig,mouse+'_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)


    for color in ['green', 'red']:
        signalPre = copy.deepcopy(np.concatenate(animalPre[color+'_clean_traces'].values, axis=0))
        signalRot = copy.deepcopy(np.concatenate(animalRot[color+'_clean_traces'].values, axis=0))

        doCells = input(f"{color} signal: {signalPre.shape[1]} cells. Do you want to review them?: ([Y]/N)")
        keepCells = np.zeros((signalPre.shape[1],), dtype=bool)*True

        if not any(doCells) or 'Y' in doCells:
            signalPreOg = copy.deepcopy(np.concatenate(animalPre[color+'_raw_traces'].values, axis=0))
            signalRotOg = copy.deepcopy(np.concatenate(animalRot[color+'_raw_traces'].values, axis=0))
            lowpassRot = uniform_filter1d(signalRotOg, size = 2000, axis = 0)
            lowpassPre = uniform_filter1d(signalPreOg, size = 2000, axis = 0)

            for n in range(signalPre.shape[1]):
                ylims = [np.min([np.min(signalPreOg[:,n]), np.min(signalRotOg[:,n])]),
                        1.1*np.max([np.max(signalPreOg[:,n]), np.max(signalRotOg[:,n])]) ]

                f = plt.figure()
                ax = plt.subplot(2,2,1)
                ax.plot(signalPreOg[:,n])
                base = np.histogram(signalPreOg[:,n], 100)
                basePre = base[1][np.argmax(base[0])]
                basePre = basePre + lowpassPre[:,n] - np.min(lowpassPre[:,n])   
                ax.plot(basePre, color = 'r')
                ax.set_ylim(ylims)

                ax = plt.subplot(2,2,2)
                ax.plot(signalRotOg[:,n])
                base = np.histogram(signalRotOg[:,n], 100)
                baseRot = base[1][np.argmax(base[0])]
                baseRot = baseRot + lowpassRot[:,n] - np.min(lowpassRot[:,n])   
                ax.plot(baseRot, color = 'r')
                ax.set_ylim(ylims)

                ax = plt.subplot(2,2,3)
                ax.plot(signalPre[:,n])
                ax.set_ylim([-0.05, 1.5])

                ax = plt.subplot(2,2,4)
                ax.plot(signalRot[:,n])
                ax.set_ylim([-0.05, 1.5])
                plt.suptitle(f"{n}/{signalPre.shape[1]}")
                a = input()
                keepCells[n] = not any(a)
                plt.close(f)

            print(f"Removing {np.sum(~keepCells)} cells ({100*np.sum(~keepCells)/keepCells.shape[0]:.2f} %)")
            index = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalRot.shape[0],1))+1))
            signalBoth = np.vstack((signalPre[:, keepCells], signalRot[:, keepCells]))
            model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
            model.fit(signalBoth)
            embBoth = model.transform(signalBoth)
            embPreClean = embBoth[index[:,0]==0,:]
            embRotClean = embBoth[index[:,0]==1,:]

            signalBoth = np.vstack((signalPre, signalRot))
            model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
            model.fit(signalBoth)
            embBoth = model.transform(signalBoth)
            embPre = embBoth[index[:,0]==0,:]
            embRot = embBoth[index[:,0]==1,:]

            plt.figure()
            ax = plt.subplot(2,2,1, projection = '3d')
            ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
            ax.scatter(*embRot[:,:3].T, color = 'r', s= 30, cmap = 'magma')
            ax.set_title('All')
            ax = plt.subplot(2,2,2, projection = '3d')
            ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
            ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s= 30, cmap = 'magma')
            ax = plt.subplot(2,2,3, projection = '3d')
            ax.scatter(*embPreClean[:,:3].T, color ='b', s= 30, cmap = 'magma')
            ax.scatter(*embRotClean[:,:3].T, color = 'r', s= 30, cmap = 'magma')
            ax.set_title('Clean')
            ax = plt.subplot(2,2,4, projection = '3d')
            ax.scatter(*embPreClean[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
            ax.scatter(*embRotClean[:,:3].T, c = posRot[:,0], s= 30, cmap = 'magma')
            plt.tight_layout()
            plt.suptitle(f"{mouse}: {color}-traces | vel: {velTh}")
            doRemoveCells = input(f"Do you want to remove cells?: ([Y]/N)")
            if not any(doRemoveCells) or 'Y' in doRemoveCells:
                animalPre = remove_cells(animalPre, color, keepCells)
                animalRot = remove_cells(animalRot, color, keepCells)

                animalPreStill = remove_cells(animalPreStill, color, keepCells)
                animalRotStill = remove_cells(animalRotStill, color, keepCells)
            else:
                keepCells = np.ones((signalPre.shape[1],), dtype=bool)
        params[color+"KeepCells"] = keepCells

    with open(os.path.join(saveDir, mouse+"_params.pkl"), "wb") as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
    animalDict = {
        fnamePre: animalPre,
        fnameRot: animalRot
    }
    with open(os.path.join(saveDir, mouse+"_df_dict.pkl"), "wb") as file:
        pickle.dump(animalDict, file, protocol=pickle.HIGHEST_PROTOCOL)

    animalStillDict = {
        fnamePre: animalPreStill,
        fnameRot: animalRotStill
    }
    with open(os.path.join(saveDir, mouse+"_df_still_dict.pkl"), "wb") as file:
        pickle.dump(animalStillDict, file, protocol=pickle.HIGHEST_PROTOCOL)



nNeigh = 120
#%%all data
# index = np.vstack((np.zeros((gsignal_p.shape[0],1)),np.zeros((gsignalRot.shape[0],1))+1))
# concat_signal = np.vstack((gsignal_p, gsignalRot))
# model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
# model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
# model.fit(concat_signal)
# concat_emb = model.transform(concat_signal)
# gembPre = concat_emb[index[:,0]==0,:]
# gembRot = concat_emb[index[:,0]==1,:]

gsignal_p = copy.deepcopy(np.concatenate(animalPre['green_clean_traces'].values, axis=0))
rsignal_p = copy.deepcopy(np.concatenate(animalPre['red_clean_traces'].values, axis=0))
posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
gsignalRot = copy.deepcopy(np.concatenate(animalRot['green_clean_traces'].values, axis=0))
rsignalRot = copy.deepcopy(np.concatenate(animalRot['red_clean_traces'].values, axis=0))
posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))

model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(gsignal_p)
gembPre = model.transform(gsignal_p)
gembRot = model.transform(gsignalRot)


#%%all data
# index = np.vstack((np.zeros((rsignal_p.shape[0],1)),np.zeros((rsignalRot.shape[0],1))+1))
# concat_signal = np.vstack((rsignal_p, rsignalRot))
# model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
# model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
# model.fit(concat_signal)
# concat_emb = model.transform(rsignal_p)
# rembPre = concat_emb[index[:,0]==0,:]
# rembRot = concat_emb[index[:,0]==1,:]
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(rsignal_p)
rembPre = model.transform(rsignal_p)
rembRot = model.transform(rsignalRot)

#%%
plt.figure()
ax = plt.subplot(1,2,1, projection = '3d')
ax.scatter(*gembPre[:,:3].T, color ='b', s= 10)
ax.scatter(*gembRot[:,:3].T, color = 'r', s= 10)
ax.set_title('Green')
ax = plt.subplot(1,2,2, projection = '3d')
ax.scatter(*gembPre[:,:3].T, c = posPre[:,0], s= 10, cmap = 'magma')
ax.scatter(*gembRot[:,:3].T, c = posRot[:,0], s= 10, cmap = 'magma')

ax = plt.subplot(2,2,3, projection = '3d')
ax.scatter(*rembPre[:,:3].T, color ='b', s= 10)
ax.scatter(*rembRot[:,:3].T, color = 'r', s= 10)
ax.set_title('Red')
ax = plt.subplot(2,2,4, projection = '3d')
ax.scatter(*rembPre[:,:3].T, c = posPre[:,0], s= 10, cmap = 'magma')
ax.scatter(*rembRot[:,:3].T, c = posRot[:,0], s= 10, cmap = 'magma')
plt.tight_layout()



plt.figure()
ax = plt.subplot(1,2,1, projection = '3d')
ax.scatter(*gembPre[:,:3].T, color ='b', s= 10)
ax.set_title('Green')
ax = plt.subplot(1,2,2, projection = '3d')
ax.scatter(*gembPre[:,:3].T, c = posPre[:,0], s= 10, cmap = 'magma')
