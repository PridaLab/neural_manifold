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
import skdim

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data


import time
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


#__________________________________________________________________________
#|                                                                        |#
#|                                  START                                 |#
#|________________________________________________________________________|#

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig2/ncells/'

min_cells = 10
max_cells = 300
n_steps = 8
n_splits = 10
max_dim = 10


for mouse in mice_list:
    tic()
    print(f"Working on mouse {mouse}: ",  end = '')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)

    results = {'params': {
        'min_cells' : min_cells,
        'max_cells' : max_cells,
        'n_steps' : n_steps,
        'n_splits' : max_dim
    }}

    umap_embds = {}

    #signal
    signal = np.concatenate(pd_mouse['clean_traces'].values, axis = 0)
    pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pd_mouse['vel'].values, axis=0))
    trial = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))

    num_cells = signal.shape[1]
    cell_list = np.unique(np.logspace(np.log10(min_cells), np.log10(max_cells),n_steps,dtype=int))
    results['params']["og_num_cells"] = cell_list
    cell_list = cell_list[cell_list<=num_cells]
    results['params']["cell_list"] = cell_list
    cells_picked_list = np.zeros((n_steps,n_splits,num_cells)).astype(bool)


    inner_dim = {
        'abids': np.zeros((n_steps, n_splits))*np.nan,
        # 'mom': np.zeros((n_steps, n_splits))*np.nan,
        # 'tle': np.zeros((n_steps, n_splits))*np.nan,
        'params': {
            'signalName': 'clean_traces',
            'nNeigh': 30,
            'verbose': False
        }
    }

    # umap_dim = {
    #     'trustNum': np.zeros((n_steps, n_splits,max_dim))*np.nan,
    #     'contNum': np.zeros((n_steps, n_splits,max_dim))*np.nan,
    #     'trustDim': np.zeros((n_steps, n_splits))*np.nan,
    #     'contDim': np.zeros((n_steps, n_splits))*np.nan,
    #     'hmeanDim': np.zeros((n_steps, n_splits))*np.nan,
    #     'params': {
    #         'maxDim':10,
    #         'nNeigh': 120,
    #         'minDist': 0.1,
    #         'nnDim': 30,
    #         'signalName': 'clean_traces'
    #     }
    # }

    si_temp = {
        'pos': {
            'sI': np.zeros((n_steps, n_splits))*np.nan,
            'binLabel': {},
            'overlapMat': {},
            },
        'dir_mat': {
            'sI': np.zeros((n_steps, n_splits))*np.nan,
            'binLabel': {},
            'overlapMat': {},
        },
        # 'time': {
        #     'sI': np.zeros((n_steps, n_splits))*np.nan,
        #     'binLabel': {},
        #     'overlapMat': {},
        # },
        'vel': {
            'sI': np.zeros((n_steps, n_splits))*np.nan,
            'binLabel': {},
            'overlapMat': {},
        }
    }

    SI_dict = {
        'clean_traces': copy.deepcopy(si_temp),
        'umap': copy.deepcopy(si_temp),
        'params': {
            'nNeigh_perc': 0.005,
            'numShuffles': 0
        }
    }
    SI_dict['params']['nNeigh'] = np.round(SI_dict['params']['nNeigh_perc']*signal.shape[0]).astype(int)
    dec_params = {
        'x_base_signal': 'clean_traces',
        'y_signal_list': ['posx', 'vel'],
        'verbose': False,
        'trial_signal': 'index_mat',
        'nn': 120,
        'min_dist':0.1,
        'n_splits': 5,
        'n_dims': 3,
        'emb_list': ['umap'],
        'decoder_list': ['xgb']
    }   

    dec_dict = {
        'clean_traces': np.zeros((n_steps, n_splits, dec_params['n_splits'], len(dec_params['y_signal_list']),2))*np.nan,
        'umap': np.zeros((n_steps, n_splits, dec_params['n_splits'], len(dec_params['y_signal_list']),2))*np.nan,
        'params': dec_params
    }

    for ncell_idx, ncell_val in enumerate(cell_list):
        print(f"\nChecking number of cells {ncell_val} ({ncell_idx+1}/{n_steps}):")
        print("\tIteration X/X", sep= '', end = '')
        pre_del = '\b\b\b'
        for split_idx in range(n_splits):
            print(pre_del, f"{split_idx+1}/{n_splits}", sep = '', end = '')
            pre_del = (len(str(split_idx+1))+len(str(n_splits))+1)*"\b"

            cells_picked = random.sample(list(np.arange(num_cells).astype(int)), ncell_val)
            cells_picked_list[ncell_idx, split_idx, cells_picked] = True 
            it_signal = copy.deepcopy(signal[:, cells_picked])

            #compute inner dim
            inner_dim['abids'][ncell_idx, split_idx] = np.nanmean(dim_red.compute_abids(it_signal, inner_dim['params']['nNeigh'],verbose=False))
            # inner_dim['mom'][ncell_idx, split_idx] = skdim.id.MOM().fit_transform(it_signal,n_neighbors = inner_dim['params']['nNeigh'])
            # try:
            #     inner_dim['tle'][ncell_idx, split_idx] = skdim.id.TLE().fit_transform(it_signal,n_neighbors = inner_dim['params']['nNeigh'])
            # except:
            #     inner_dim['tle'][ncell_idx, split_idx] = np.nan

            #compute umap dim
            # rank_idxs = dim_validation.compute_rank_indices(it_signal)

            # for dim in range(umap_dim['params']['maxDim']):
            #     emb_space = np.arange(dim+1)
            #     model = umap.UMAP(n_neighbors = umap_dim['params']['nNeigh'], n_components =dim+1, min_dist=umap_dim['params']['minDist'])
            #     emb = model.fit_transform(it_signal)
            #     if dim==2:
            #         it_umap = copy.deepcopy(emb)
            #     #1. Compute trustworthiness
            #     temp = dim_validation.trustworthiness_vector(it_signal, emb, umap_dim['params']['nnDim'], indices_source = rank_idxs)
            #     umap_dim['trustNum'][ncell_idx,split_idx][dim] = temp[-1]

            #     #2. Compute continuity
            #     temp = dim_validation.continuity_vector(it_signal, emb ,umap_dim['params']['nnDim'])
            #     umap_dim['contNum'][ncell_idx,split_idx][dim] = temp[-1]

            # dimSpace = np.linspace(1,umap_dim['params']['maxDim'], umap_dim['params']['maxDim']).astype(int)   
            model = umap.UMAP(n_neighbors = 120, n_components =3, min_dist=0.1)
            it_umap = model.fit_transform(it_signal)

            # kl = KneeLocator(dimSpace, umap_dim['trustNum'][ncell_idx,split_idx], curve = "concave", direction = "increasing")
            # if kl.knee:
            #     umap_dim['trustDim'][ncell_idx, split_idx] = kl.knee
            # else:
            #     umap_dim['trustDim'][ncell_idx, split_idx] = np.nan
            # kl = KneeLocator(dimSpace, umap_dim['contNum'][ncell_idx,split_idx], curve = "concave", direction = "increasing")
            # if kl.knee:
            #     umap_dim['contDim'][ncell_idx, split_idx] = kl.knee
            # else:
            #     umap_dim['contDim'][ncell_idx, split_idx] = np.nan
            # umap_dim['hmeanDim'][ncell_idx, split_idx] = (2*umap_dim['trustDim'][ncell_idx, split_idx]*umap_dim['contDim'][ncell_idx, split_idx])/(umap_dim['trustDim'][ncell_idx, split_idx]+umap_dim['contDim'][ncell_idx, split_idx])


            #compute SI
            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, pos[:,0], n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['pos']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['clean_traces']['pos']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['clean_traces']['pos']['overlapMat'][ncell_idx, split_idx] = toverlapMat


            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, dir_mat, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['dir_mat']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['clean_traces']['dir_mat']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['clean_traces']['dir_mat']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            # tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, trial, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            # SI_dict['clean_traces']['time']['sI'][ncell_idx, split_idx] = tSI
            # SI_dict['clean_traces']['time']['binLabel'][ncell_idx, split_idx] = tbinLabel
            # SI_dict['clean_traces']['time']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, vel, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['vel']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['clean_traces']['vel']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['clean_traces']['vel']['overlapMat'][ncell_idx, split_idx] = toverlapMat


            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, pos[:,0], n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['pos']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['umap']['pos']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['umap']['pos']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, dir_mat, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['dir_mat']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['umap']['dir_mat']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['umap']['dir_mat']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            # tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, trial, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            # SI_dict['umap']['time']['sI'][ncell_idx, split_idx] = tSI
            # SI_dict['umap']['time']['binLabel'][ncell_idx, split_idx] = tbinLabel
            # SI_dict['umap']['time']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, vel, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['vel']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['umap']['vel']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['umap']['vel']['overlapMat'][ncell_idx, split_idx] = toverlapMat


            #compute decoders
            new_pd_mouse = copy.deepcopy(pd_mouse)
            for idx in range(pd_mouse.shape[0]):
                new_pd_mouse['clean_traces'][idx] = pd_mouse['clean_traces'][idx][:, cells_picked]

            dec_R2s, _ = dec.decoders_1D(pd_object = copy.deepcopy(new_pd_mouse), **dec_params)
            dec_dict['clean_traces'][ncell_idx, split_idx,:, :,:] = dec_R2s['base_signal']['xgb']
            dec_dict['umap'][ncell_idx, split_idx,:, :,:] = dec_R2s['umap']['xgb']

            umap_embds[ncell_idx, split_idx] = it_umap


        results['inner_dim'] = copy.deepcopy(inner_dim)
        # results['umap_dim'] = copy.deepcopy(umap_dim)
        results['SI_dict'] = copy.deepcopy(SI_dict)
        results['dec_dict'] = copy.deepcopy(dec_dict)
        results['umap_embds'] = copy.deepcopy(umap_embds)
        results['params']['cells_picked_list'] = copy.deepcopy(cells_picked_list)
        save_file = open(os.path.join(save_dir, f'{mouse}_ncells_results_dict.pkl'), "wb")
        pickle.dump(results, save_file)
        save_file.close()
    print('\n')
    toc()


mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig2/ncells/'


for mouse in mice_list:
    tic()
    print(f"Working on mouse {mouse}: ",  end = '')


    ncells_dict = load_pickle(save_dir, f'{mouse}_ncells_results_dict.pkl')
    cell_list = ncells_dict['params']['cell_list']
    cells_picked_list = ncells_dict['params']['cells_picked_list'] 
    umap_embds = ncells_dict['umap_embds']

    n_steps = ncells_dict['params']['n_steps']
    n_splits = ncells_dict['params']['n_splits']

    ncells_dict['dec_dict']['params']['y_signal_list'].append('dir_mat')


    dec_params = {
        'verbose': False,
        'nn': 120,
        'min_dist':0.1,
        'n_splits': 5,
        'n_dims': 3,
        'emb_list': ['umap'],
        'decoder_list': ['svc'], 
        'metric': 'f1_score'
    }   
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)


    dec_dict = {
        'clean_traces': np.zeros((n_steps, n_splits, dec_params['n_splits'], 3,2))*np.nan,
        'umap': np.zeros((n_steps, n_splits, dec_params['n_splits'], 3,2))*np.nan,
    }

    for a in list(dec_dict.keys()):
        dec_dict[a][:,:,:,:2,:] = copy.deepcopy(ncells_dict['dec_dict'][a])

    for ncell_idx, ncell_val in enumerate(cell_list):
        print(f"\nChecking number of cells {ncell_val} ({ncell_idx+1}/{n_steps}):")
        print("\tIteration X/X", sep= '', end = '')
        pre_del = '\b\b\b'
        for split_idx in range(n_splits):
            print(pre_del, f"{split_idx+1}/{n_splits}", sep = '', end = '')
            pre_del = (len(str(split_idx+1))+len(str(n_splits))+1)*"\b"

            cells_picked = cells_picked_list[ncell_idx, split_idx]

            signal = np.concatenate(pd_mouse['clean_traces'].values, axis = 0)[:, cells_picked]
            dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
            trial = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))


            to_keep = dir_mat[:,0]!=0
            signal = signal[to_keep,:] 
            trial = trial[to_keep,:] 
            dir_mat = dir_mat[to_keep,:].reshape(-1,1)


            dec_R2s, _ = dec.decoders_1D(x_base_signal=signal, y_signal_list = [dir_mat], trial_signal = trial, **dec_params)


            dec_dict['clean_traces'][ncell_idx, split_idx,:, 2,:] = dec_R2s['base_signal']['svc'][:,0,:]
            dec_dict['umap'][ncell_idx, split_idx,:, 2,:] = dec_R2s['umap']['svc'][:,0,:]



    for a in list(dec_dict.keys()):
        ncells_dict['dec_dict'][a] = copy.deepcopy(dec_dict[a])

    save_file = open(os.path.join(save_dir, f'{mouse}_ncells_results_dict.pkl'), "wb")
    pickle.dump(ncells_dict, save_file)
    save_file.close()
    print('\n')
    toc()


#__________________________________________________________________________
#|                                                                        |#
#|                                 PLOT                                   |#
#|________________________________________________________________________|#


sup_mice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deep_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ7', 'GC7','ChZ8', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir = '/home/julio/Documents/SP_project/Fig2/ncells/'

palette_deepsup = {'deep': "#cc9900ff", 'sup': "#9900ffff"}
base_cell_list = [ 10,  16,  26,  42,  69, 113, 184, 300]


#DEC UMAP

plt.figure(figsize = (13,6))
for idx, label in enumerate(['pos','vel', 'dir_mat']):
    ax = plt.subplot(1,3,idx+1)
    deep_vals = []
    sup_vals = []
    for mouse in mice_list:
        if label == 'vel' and mouse == 'CZ4': continue;
        ncells_dict = load_pickle(data_dir, f'{mouse}_ncells_results_dict.pkl')
        cell_list = ncells_dict['params']['cell_list']
        val = np.nanmean(ncells_dict['dec_dict']['umap'][:,:,:,idx,0],axis=(1,2))
        if mouse in deep_mice:
            color = palette_deepsup['deep']
            deep_vals.append(val)
        elif mouse in sup_mice:
            color = palette_deepsup['sup']
            sup_vals.append(val)
        ax.plot(cell_list, val[:len(cell_list)], color = color, alpha = 0.3)

    deep_vals = np.stack(deep_vals,axis=1)
    sup_vals = np.stack(sup_vals,axis=1)
    ax.plot(base_cell_list, np.nanmean(deep_vals, axis=1), label = 'deep', linewidth = 3, color = palette_deepsup['deep'])
    ax.plot(base_cell_list, np.nanmean(sup_vals, axis=1), label = 'sup', linewidth = 3, color = palette_deepsup['sup'])
    ax.set_xlabel('Num cells')
    if 'pos' in label:
        ax.set_ylabel('posx (cm)')
        ax.set_ylim([0, 18])
    elif 'vel' in label:
        ax.set_ylabel('vel (cm/s)')
        ax.set_ylim([0, 18])
    ax.legend()

plt.suptitle('Umap')
plt.savefig(os.path.join(data_dir,'nCells_decoders_umap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_decoders_umap.png'), dpi = 400,bbox_inches="tight")


#DEC TRACES
plt.figure(figsize = (13,6))
for idx, label in enumerate(['pos','vel', 'dir_mat']):
    ax = plt.subplot(1,3,idx+1)
    deep_vals = []
    sup_vals = []
    for mouse in mice_list:
        if label == 'vel' and mouse == 'CZ4': continue;
        if label =='dir_mat' and mouse == 'GC5_nvista': continue;
        if label =='dir_mat' and mouse == 'ChZ4': continue;
        ncells_dict = load_pickle(data_dir, f'{mouse}_ncells_results_dict.pkl')
        cell_list = ncells_dict['params']['cell_list']
        val = np.nanmean(ncells_dict['dec_dict']['clean_traces'][:,:,:,idx,0],axis=(1,2))
        if mouse in deep_mice:
            color = palette_deepsup['deep']
            deep_vals.append(val)
        elif mouse in sup_mice:
            color = palette_deepsup['sup']
            sup_vals.append(val)
        ax.plot(cell_list, val[:len(cell_list)], color = color, alpha = 0.3)

    deep_vals = np.stack(deep_vals,axis=1)
    sup_vals = np.stack(sup_vals,axis=1)
    ax.plot(base_cell_list, np.nanmean(deep_vals, axis=1), label = 'deep', linewidth = 3, color = palette_deepsup['deep'])
    ax.plot(base_cell_list, np.nanmean(sup_vals, axis=1), label = 'sup', linewidth = 3, color = palette_deepsup['sup'])
    ax.set_xlabel('Num cells')
    if 'pos' in label:
        ax.set_ylabel('posx (cm)')
        ax.set_ylim([0, 18])
    elif 'vel' in label:
        ax.set_ylabel('vel (cm/s)')
        ax.set_ylim([0, 18])
    elif 'dir_mat' in label:
        ax.set_ylabel('dir (F1_score)')
        ax.set_ylim([0.495, 1.005])
    ax.legend()


plt.suptitle('Traces')
plt.savefig(os.path.join(data_dir,'nCells_decoders_traces.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_decoders_traces.png'), dpi = 400,bbox_inches="tight")


#DEC TRACES BOXPLOT
plt.figure(figsize = (13,6))
for idx, label in enumerate(['pos','vel']):
    ax = plt.subplot(1,3,idx+1)
    mouse_name = []
    dec_vals = []
    layer_list = []
    for mouse in mice_list:
        if label == 'vel' and mouse == 'CZ4': continue;
        mouse_name.append(mouse)
        if mouse in deep_mice:
            layer_list.append('deep')
        elif mouse in sup_mice:
            layer_list.append('sup')

        ncells_dict = load_pickle(data_dir, f'{mouse}_ncells_results_dict.pkl')
        cell_list = ncells_dict['params']['cell_list']
        val = np.nanmean(ncells_dict['dec_dict']['clean_traces'][:,:,:,idx,0],axis=(1,2))
        dec_vals.append(val[len(cell_list)-1])

    sns.barplot(x = layer_list, y= dec_vals, ax=ax)

    if 'pos' in label:
        ax.set_ylabel('posx (cm)')
        ax.set_ylim([0, 8])
    elif 'vel' in label:
        ax.set_ylabel('vel (cm/s)')
        ax.set_ylim([0, 8])
    ax.legend()

ax = plt.subplot(1,3,3)
ax.set_xlabel('Num cells')
if 'pos' in label:
    ax.set_ylabel('posx (cm)')
    ax.set_ylim([0, 18])
elif 'vel' in label:
    ax.set_ylabel('vel (cm/s)')
    ax.set_ylim([0, 18])

plt.suptitle('Traces')
plt.savefig(os.path.join(data_dir,'nCells_decoders_traces_boxplot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_decoders_traces_boxplot.png'), dpi = 400,bbox_inches="tight")

#SI UMAP
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ7', 'GC7','ChZ8', 'CZ3','CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']

plt.figure(figsize = (13,6))
for idx, label in enumerate(['pos','vel', 'dir_mat']):
    ax = plt.subplot(1,3,idx+1)
    deep_vals = []
    sup_vals = []
    for mouse in mice_list:
        if label == 'vel' and mouse == 'CZ4': continue;
        ncells_dict = load_pickle(data_dir, f'{mouse}_ncells_results_dict.pkl')
        cell_list = ncells_dict['params']['cell_list']
        val = np.nanmean(ncells_dict['SI_dict']['umap'][label]['sI'][:,:],axis=1)
        if mouse in deep_mice:
            color = palette_deepsup['deep']
            deep_vals.append(val)
        elif mouse in sup_mice:
            color = palette_deepsup['sup']
            sup_vals.append(val)
        ax.plot(cell_list, val[:len(cell_list)], color = color, alpha = 0.3)

    deep_vals = np.stack(deep_vals,axis=1)
    sup_vals = np.stack(sup_vals,axis=1)
    ax.plot(base_cell_list, np.nanmean(deep_vals, axis=1), label = 'deep', linewidth = 3, color = palette_deepsup['deep'])
    ax.plot(base_cell_list, np.nanmean(sup_vals, axis=1), label = 'sup', linewidth = 3, color = palette_deepsup['sup'])
    ax.set_xlabel('Num cells')

    ax.set_ylabel(f'SI {label}')
    ax.set_ylim([-0.05, 1.05])
    ax.legend()

plt.suptitle('Umap')
plt.savefig(os.path.join(data_dir,'nCells_SI_umap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_SI_umap.png'), dpi = 400,bbox_inches="tight")

#SI TRACES
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ7', 'GC7','ChZ8', 'CZ3','CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']

plt.figure(figsize = (13,6))
for idx, label in enumerate(['pos','vel', 'dir_mat']):
    ax = plt.subplot(1,3,idx+1)
    deep_vals = []
    sup_vals = []
    for mouse in mice_list:
        if label == 'vel' and mouse == 'CZ4': continue;
        ncells_dict = load_pickle(data_dir, f'{mouse}_ncells_results_dict.pkl')
        cell_list = ncells_dict['params']['cell_list']
        val = np.nanmean(ncells_dict['SI_dict']['clean_traces'][label]['sI'][:,:],axis=1)
        if mouse in deep_mice:
            color = palette_deepsup['deep']
            deep_vals.append(val)
        elif mouse in sup_mice:
            color = palette_deepsup['sup']
            sup_vals.append(val)
        ax.plot(cell_list, val[:len(cell_list)], color = color, alpha = 0.3)

    deep_vals = np.stack(deep_vals,axis=1)
    sup_vals = np.stack(sup_vals,axis=1)
    ax.plot(base_cell_list, np.nanmean(deep_vals, axis=1), label = 'deep', linewidth = 3, color = palette_deepsup['deep'])
    ax.plot(base_cell_list, np.nanmean(sup_vals, axis=1), label = 'sup', linewidth = 3, color = palette_deepsup['sup'])
    ax.set_xlabel('Num cells')

    ax.set_ylabel(f'SI {label}')
    ax.set_ylim([-0.05, 1.05])
    ax.legend()

plt.suptitle('Traces')
plt.savefig(os.path.join(data_dir,'nCells_SI_traces.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_SI_traces.png'), dpi = 400,bbox_inches="tight")



    inner_dim_abids.append(np.nanmean(ncells_dict['inner_dim']['abids'],axis=1))

#INNER DIM
plt.figure(figsize = (13,6))

ax = plt.subplot(1,1,1)
deep_vals = []
sup_vals = []
for mouse in mice_list:
    ncells_dict = load_pickle(data_dir, f'{mouse}_ncells_results_dict.pkl')
    cell_list = ncells_dict['params']['cell_list']
    val = np.nanmean(ncells_dict['inner_dim']['abids'],axis=1)
    if mouse in deep_mice:
        color = palette_deepsup['deep']
        deep_vals.append(val)
    elif mouse in sup_mice:
        color = palette_deepsup['sup']
        sup_vals.append(val)
    ax.plot(cell_list, val[:len(cell_list)], color = color, alpha = 0.3)

deep_vals = np.stack(deep_vals,axis=1)
sup_vals = np.stack(sup_vals,axis=1)
ax.plot(base_cell_list, np.nanmean(deep_vals, axis=1), label = 'deep', linewidth = 3, color = palette_deepsup['deep'])
ax.plot(base_cell_list, np.nanmean(sup_vals, axis=1), label = 'sup', linewidth = 3, color = palette_deepsup['sup'])
ax.set_xlabel('Num cells')

ax.set_ylabel('ABID dim')
ax.set_ylim([0, 4])
ax.legend()

plt.suptitle('Traces')
plt.savefig(os.path.join(data_dir,'nCells_inner_dim_ABID.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_inner_dim_ABID.png'), dpi = 400,bbox_inches="tight")




#__________________________________________________________________________
#|                                                                        |#
#|                                 UMAP EMB                               |#
#|________________________________________________________________________|#
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
data_dir_ncells = '/home/julio/Documents/SP_project/Fig2/ncells/'
save_dir = '/home/julio/Documents/SP_project/Fig2/ncells/emb_examples/'


for mouse in mice_list:
    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
    pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
    trial = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))
    time = np.arange(pos.shape[0])
    dir_color = np.zeros((dir_mat.shape[0],3))

    ncells_dict = load_pickle(data_dir_ncells, f'{mouse}_ncells_results_dict.pkl')

    for point in range(dir_mat.shape[0]):
        if dir_mat[point]==0:
            dir_color[point] = [14/255,14/255,143/255]
        elif dir_mat[point]==1:
            dir_color[point] = [12/255,136/255,249/255]
        else:
            dir_color[point] = [17/255,219/255,224/255]
    for ncell_idx in range(8):
        for split_idx in range(10):
            umap_emb = copy.deepcopy(ncells_dict['umap_embds'][(ncell_idx,split_idx)])

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1, projection = '3d')
            b = ax.scatter(*umap_emb[:,:3].T, color = dir_color,s = 20)
            # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
            ax.view_init(45, -45)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_zlabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_aspect('equal')
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,:2].T, color = dir_color,s = 20)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir_xy.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir_xy.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,(0,2)].T, color = dir_color,s = 20)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir_xz.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir_xz.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1, projection = '3d')
            b = ax.scatter(*umap_emb[:,:3].T, c = pos[:,0],s = 20, cmap = 'inferno', vmin= 0, vmax = 120)
            # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
            ax.view_init(45, -45)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_zlabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_aspect('equal')

            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,:2].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos_xy.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos_xy.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,(0,2)].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos_xz.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos_xz.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1, projection = '3d')
            b = ax.scatter(*umap_emb[:,:3].T, c = time,s = 20, cmap = 'YlGn_r', vmax = 13000)
            ax.view_init(45, -45)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_zlabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_aspect('equal')

            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,:2].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time_xy.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time_xy.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,(0,2)].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time_xz.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time_xz.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            