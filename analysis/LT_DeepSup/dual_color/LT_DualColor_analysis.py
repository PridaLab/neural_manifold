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

def preprocess_traces(pd_struct_p, pd_struct_r, signal_field, sigma = 5,sig_up = 4, sig_down = 12, peak_th=0.1):
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


    out_pd_p['clean_traces'] = [bi_signal_p[index_mat_p[:,0]==out_pd_p["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_p.shape[0])]
    out_pd_r['clean_traces'] = [bi_signal_r[index_mat_r[:,0]==out_pd_r["trial_id"][idx] ,:] 
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


miceList = ['CalbGRIN1', 'CalbGRIN2']

base_load_dir = '/home/julio/Documents/SP_project/LT_DualColor/data/'
base_save_dir = '/home/julio/Documents/SP_project/LT_DualColor/processed_data/'
signal_field = 'raw_traces'
vel_th = 6
sigma = 6
sig_up = 4
sig_down = 12
nn_val = 120
dim = 3

for mouse in miceList:
    print(f"Working on mouse: {mouse}")
    load_dir = os.path.join(base_load_dir, mouse)
    save_dir = os.path.join(base_save_dir, mouse)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir_fig = os.path.join(save_dir, 'figures')
    if not os.path.exists(save_dir_fig):
        os.mkdir(save_dir_fig)
    now = datetime.now()
    params = {
        "date": now.strftime("%d/%m/%Y %H:%M:%S"),
        "mouse": mouse,
        "load_dir": load_dir,
        "save_dir": save_dir,
        "signal_field": signal_field,
        "vel_th": vel_th,
        "sigma": sigma,
        "sig_up": sig_up,
        "sig_down": sig_down
    }
    #__________________________________________________________________________
    #|                                                                        |#
    #|                               LOAD DATA                                |#
    #|________________________________________________________________________|#
    animal = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")
    fnames = list(animal.keys())

    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]

    animal_p= copy.deepcopy(animal[fnamePre])
    animal_r= copy.deepcopy(animal[fnameRot])

    #__________________________________________________________________________
    #|                                                                        |#
    #|                          PREPROCESS TRACES                             |#
    #|________________________________________________________________________|#
    animal_p = add_dir_mat_field(animal_p)
    animal_r = add_dir_mat_field(animal_r)

    animal_p = gu.select_trials(animal_p,"dir == ['L','R','N']")
    animal_r = gu.select_trials(animal_r,"dir == ['L','R','N']")
    animal_p, animal_p_still = gu.keep_only_moving(animal_p, vel_th)
    animal_r, animal_r_still = gu.keep_only_moving(animal_r, vel_th)

    animal_p, animal_r = preprocess_traces(animal_p, animal_r, signal_field, sigma=sigma, sig_up = sig_up, sig_down = sig_down)
    animal_p_still, animal_r_still = preprocess_traces(animal_p_still, animal_r_still, signal_field, sigma=sigma, sig_up = sig_up, sig_down = sig_down)


    #__________________________________________________________________________
    #|                                                                        |#
    #|                               REMOVE CELLS                             |#
    #|________________________________________________________________________|#

    signal_p = copy.deepcopy(np.concatenate(animal_p['clean_traces'].values, axis=0))
    pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
    signal_r = copy.deepcopy(np.concatenate(animal_r['clean_traces'].values, axis=0))
    pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
    #%%all data
    index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
    concat_signal = np.vstack((signal_p, signal_r))
    model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
    # model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
    model.fit(concat_signal)
    concat_emb = model.transform(concat_signal)
    emb_p = concat_emb[index[:,0]==0,:]
    emb_r = concat_emb[index[:,0]==1,:]

    #%%
    plt.figure()
    ax = plt.subplot(1,2,1, projection = '3d')
    ax.scatter(*emb_p[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*emb_r[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax.set_title('All')
    ax = plt.subplot(1,2,2, projection = '3d')
    ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
    ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
    plt.suptitle(f"{mouse}: clean_traces - vel: {vel_th} - nn: {nn_val} - dim: {dim}")
    plt.savefig(os.path.join(save_dir_fig,mouse+'_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)

    do_cells = input(f"{signal_p.shape[1]} cells. Do you want to review them?: ([Y]/N)")
    cells_to_keep = np.zeros((signal_p.shape[1],), dtype=bool)*True


    if not any(do_cells) or 'Y' in do_cells:

        signal_p_og = copy.deepcopy(np.concatenate(animal_p[signal_field].values, axis=0))
        signal_r_og = copy.deepcopy(np.concatenate(animal_r[signal_field].values, axis=0))
        lowpass_r = uniform_filter1d(signal_r_og, size = 4000, axis = 0)
        lowpass_p = uniform_filter1d(signal_p_og, size = 4000, axis = 0)
        signal_p = copy.deepcopy(np.concatenate(animal_p['clean_traces'].values, axis=0))
        signal_r = copy.deepcopy(np.concatenate(animal_r['clean_traces'].values, axis=0))
        for n in range(signal_p.shape[1]):
            ylims = [np.min([np.min(signal_p_og[:,n]), np.min(signal_r_og[:,n])]),
                    1.1*np.max([np.max(signal_p_og[:,n]), np.max(signal_r_og[:,n])]) ]

            f = plt.figure()
            ax = plt.subplot(2,2,1)
            ax.plot(signal_p_og[:,n])
            base = np.histogram(signal_p_og[:,n], 100)
            base_p = base[1][np.argmax(base[0])]
            base_p = base_p + lowpass_p[:,n] - np.min(lowpass_p[:,n])   
            ax.plot(base_p, color = 'r')
            ax.set_ylim(ylims)

            ax = plt.subplot(2,2,2)
            ax.plot(signal_r_og[:,n])
            base = np.histogram(signal_r_og[:,n], 100)
            base_r = base[1][np.argmax(base[0])]
            base_r = base_r + lowpass_r[:,n] - np.min(lowpass_r[:,n])   
            ax.plot(base_r, color = 'r')
            ax.set_ylim(ylims)

            ax = plt.subplot(2,2,3)
            ax.plot(signal_p[:,n])
            ax.set_ylim([-0.05, 1.5])

            ax = plt.subplot(2,2,4)
            ax.plot(signal_r[:,n])
            ax.set_ylim([-0.05, 1.5])
            plt.suptitle(f"{n}/{signal_p.shape[1]}")
            a = input()
            cells_to_keep[n] = not any(a)
            plt.close(f)

        print(f"Removing {np.sum(~cells_to_keep)} cells ({100*np.sum(~cells_to_keep)/cells_to_keep.shape[0]:.2f} %)")

        concat_signal = np.vstack((signal_p[:, cells_to_keep], signal_r[:, cells_to_keep]))
        model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
        model.fit(concat_signal)
        concat_emb = model.transform(concat_signal)
        emb_p_clean = concat_emb[index[:,0]==0,:]
        emb_r_clean = concat_emb[index[:,0]==1,:]


        plt.figure()
        ax = plt.subplot(2,2,1, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, color ='b', s= 30, cmap = 'magma')
        ax.scatter(*emb_r[:,:3].T, color = 'r', s= 30, cmap = 'magma')
        ax.set_title('All')
        ax = plt.subplot(2,2,2, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
        ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
        ax = plt.subplot(2,2,3, projection = '3d')
        ax.scatter(*emb_p_clean[:,:3].T, color ='b', s= 30, cmap = 'magma')
        ax.scatter(*emb_r_clean[:,:3].T, color = 'r', s= 30, cmap = 'magma')
        ax.set_title('Clean')
        ax = plt.subplot(2,2,4, projection = '3d')
        ax.scatter(*emb_p_clean[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
        ax.scatter(*emb_r_clean[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
        plt.tight_layout()
        plt.suptitle(f"{mouse}: conv_traces vel: {vel_th}")
        do_remove_cells = input(f"Do you want to remove cells?: ([Y]/N)")
        if not any(do_remove_cells) or 'Y' in do_remove_cells:
            animal_p = remove_cells(animal_p, cells_to_keep)
            animal_r = remove_cells(animal_r, cells_to_keep)

            animal_p_still = remove_cells(animal_p_still, cells_to_keep)
            animal_r_still = remove_cells(animal_r_still, cells_to_keep)
        else:
            cells_to_keep = np.zeros((signal_p_og.shape[1],), dtype=bool)*True


    params["cells_to_keep"] = cells_to_keep
    with open(os.path.join(save_dir, mouse+"_params.pkl"), "wb") as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
    animal_dict = {
        fnamePre: animal_p,
        fnameRot: animal_r
    }
    with open(os.path.join(save_dir, mouse+"_df_dict.pkl"), "wb") as file:
        pickle.dump(animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    animal_still_dict = {
        fnamePre: animal_p_still,
        fnameRot: animal_r_still
    }
    with open(os.path.join(save_dir, mouse+"_df_still_dict.pkl"), "wb") as file:
        pickle.dump(animal_still_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#