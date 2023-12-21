import ripserplusplus as rpp_py
from persim import plot_diagrams
from matplotlib import gridspec
from numpy.random import choice
import numpy as np
from sklearn.metrics import pairwise_distances
import umap
import os, pickle, copy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from tqdm.auto import tqdm
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
import pandas as pd
import seaborn as sns
def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def save_pickle(path, name, data):
    if ('.pkl' not in name) and ('.pickle' not in name):
        name += '.pkl'
    save_pickle = open(os.path.join(path, name), "wb")
    pickle.dump(data, save_pickle)
    save_pickle.close()
    return True

def filter_noisy_outliers(data, D=None, dist_th = 5, noise_th = 25):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,dist_th), axis=1)
    noise_idx = np.where(nnDist < np.percentile(nnDist, noise_th))[0]
    signalIdx = np.where(nnDist >= np.percentile(nnDist, noise_th))[0]
    return noise_idx, signalIdx

def topological_denoising(data, num_samples=None, num_iters=100, inds=[],  sig=None, w=None, c=None, metric='euclidean'):
    n = np.float64(data.shape[0])
    d = data.shape[1]
    if len(inds)==0:
        inds = np.unique(np.floor(np.arange(0,n-1, n/num_samples)).astype(int))
    else:
        num_samples = len(inds)
    S = data[inds, :] 
    if not sig:
        sig = np.sqrt(np.var(S))
    if not c:
        c = 0.05*max(pdist(S, metric = metric)) 
    if not w:
        w = 0.3

    dF1 = np.zeros((len(inds), d), float)
    dF2 = np.zeros((len(inds), d), float)

    for i in range(num_samples):
        dF1[i, :] = np.dot((data - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], data, metric = metric), 2) / (2*sig)).T)[:, 0]
        dF2[i, :] = np.dot((S - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], S, metric = metric), 2) / (2*sig)).T)[:, 0]

    dF = 1/sig*(1/n * dF1 - (w / num_samples) * dF2)
    M = dF.max()
    for k in range(num_iters):
        S += c*dF/M
        for i in range(num_samples):
            dF1[i, :] = np.dot((data - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], data, metric = metric), 2) / (2*sig)).T)[:, 0]
            dF2[i, :] = np.dot((S - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], S, metric = metric), 2) / (2*sig)).T)[:, 0]
        dF = 1/sig*(1/n * dF1 - (w / num_samples) * dF2)
    data_denoised = S
    return data_denoised, inds

def plot_betti_bars(diagrams, max_dist=1e3, conf_interval = None):
    col_list = ['r', 'g', 'm', 'c']
    diagrams[0][~np.isfinite(diagrams[0])] = max_dist
    max_len = [np.max(h) for h in diagrams]
    max_len = np.max(max_len)
    # Plot the 30 longest barcodes only
    to_plot = []
    for curr_h in diagrams:
         bar_lens = curr_h[:,1] - curr_h[:,0]
         plot_h = curr_h[(-bar_lens).argsort()[:30]]
         to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(len(diagrams), 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            if conf_interval:
                ax.plot([interval[0], interval[0]+conf_interval[curr_betti]], [i,i], linewidth = 5, color =[.8,.8,.8])

            ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
                lw=1.5)

        ax.set_ylabel('H' + str(curr_betti))
        ax.set_xlim([-0.1, max_len+0.1])
        # ax.set_xticks([0, xlim])
        ax.set_ylim([-1, len(curr_bar)])
    return fig

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

#__________________________________________________________________________
#|                                                                        |#
#|                          COMPUTE INFORMATION                           |#
#|________________________________________________________________________|#
data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig2/mutual_info/'
mi_scores = {}
for mouse in miceList:
    mi_scores[mouse] = {
        'label_order': ['posx','posy','dir','vel','time']
    }
    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
    #signal
    signal = np.concatenate(pd_mouse['clean_traces'].values, axis = 0)
    pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pd_mouse['vel'].values, axis=0)).reshape(-1,1)
    trial = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))
    time = np.arange(pos.shape[0]).reshape(-1,1)
    x = np.concatenate((pos,dir_mat, vel, time),axis=1)
    mi_regression = np.zeros((x.shape[1], signal.shape[1]))*np.nan
    for cell in range(signal.shape[1]):
        mi_regression[:, cell] = mutual_info_regression(x, signal[:,cell], n_neighbors = 50, random_state = 16)
    mi_scores[mouse]['mi_scores'] = copy.deepcopy(mi_regression)
    saveFile = open(os.path.join(save_dir, 'mi_scores_dict.pkl'), "wb")
    pickle.dump(mi_scores, saveFile)
    saveFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                     COMPUTE BETTI IN OG WITH MI                        |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
mi_dir = '/home/julio/Documents/SP_project/Fig2/mutual_info/'
save_dir = '/home/julio/Documents/SP_project/Fig2/betti_numbers/og_mi_cells/'
mi_scores_dict = load_pickle(mi_dir,'mi_scores_dict.pkl')

dist_th = 5
noise_th = 20
num_samples = 500
num_iters = 150
dim = 20
num_neigh = 120
min_dist = 0.1
perc_cells = 20
perc_time = 0.1
fluo_th = 0.1
signal_name = 'clean_traces'
num_cells = 80

for mouse in miceList:
    try:
        os.mkdir(os.path.join(save_dir,mouse))
    except:
        pass
    #load data
    print(f"Working on mouse {mouse}: ")
    pd_mouse = load_pickle(os.path.join(data_dir, mouse),mouse+'_df_dict.pkl')

    #signal
    signal = np.concatenate(pd_mouse[signal_name].values, axis = 0)
    pos = np.concatenate(pd_mouse['pos'].values, axis=0)
    dir_mat = np.concatenate(pd_mouse['dir_mat'].values, axis=0)
    index_mat = np.concatenate(pd_mouse['index_mat'].values, axis=0)

    #select cells with highest mutual info
    label_order = mi_scores_dict[mouse]['label_order']
    mi_scores_idx = [x for x in range(len(label_order)) if (label_order[x]=='posx' or label_order[x]=='dir')]
    mi_scores = mi_scores_dict[mouse]['mi_scores'][mi_scores_idx,:]
    mi_scores = mi_scores/np.tile(np.max(mi_scores,axis=1).reshape(-1,1), (1,mi_scores.shape[1]))
    cell_order = np.argsort(np.sum(mi_scores,axis=0))[::-1]

    del_mi_cells = cell_order[num_cells:]
    signal = np.delete(signal, del_mi_cells,1)

    #delete cells that are almost inactive
    del_cells = np.mean(signal,axis=0)<np.percentile(np.mean(signal,axis=0), perc_cells)
    signal = np.delete(signal, np.where(del_cells)[0],1)
    #delete timestamps where almost all cells are inactive
    del_time = np.sum(signal>fluo_th, axis=1)<perc_time*signal.shape[1]
    signal = np.delete(signal, np.where(del_time)[0],0)
    pos = np.delete(pos, np.where(del_time)[0],0)
    dir_mat = np.delete(dir_mat, np.where(del_time)[0],0)
    index_mat = np.delete(index_mat, np.where(del_time)[0],0)
    #fit umap
    print("\tFitting umap model...", sep= '', end = '')
    model_umap = umap.UMAP(n_neighbors=num_neigh, n_components =dim, min_dist=min_dist)
    model_umap.fit(signal)
    emb_umap = model_umap.transform(signal)
    print("\b\b\b: Done")

    #filter outliers
    print("\tFiltering noisy outliers...", sep= '', end = '')
    D = pairwise_distances(signal)
    Demb = pairwise_distances(emb_umap)

    noise_idx, signalIdx = filter_noisy_outliers(signal, D=D, dist_th=dist_th, noise_th=noise_th)
    clean_signal = signal[signalIdx,:]
    clean_umap = emb_umap[signalIdx,:]
    clean_pos = pos[signalIdx,:]
    print("\b\b\b: Done")

    #topological denoising
    print("\tPerforming topological denoising...", sep= '', end = '')
    down_signal, down_idx = topological_denoising(clean_signal, num_samples=num_samples, num_iters=num_iters)
    down_umap, down_idx2 = topological_denoising(clean_umap, num_samples=num_samples, num_iters=num_iters)
    print("\b\b\b: Done")

    #plot embeddings
    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(2,3,1, projection = '3d')
    b = ax.scatter(*emb_umap[:,:3].T, c = dir_mat, cmap = 'Accent',s = 10, vmin= 0, vmax = 8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_title('Umap direction')

    ax = plt.subplot(2,3,4, projection = '3d')
    b = ax.scatter(*emb_umap[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_title('Umap pos')

    ax = plt.subplot(2,3,2, projection = '3d')
    b = ax.scatter(*emb_umap[:,:3].T, color = 'blue',s = 5)
    b = ax.scatter(*emb_umap[noise_idx,:3].T, color = 'red',s = 5)
    ax.legend(['Good', 'Outliers'])
    ax.set_title('Umap outliers')

    ax = plt.subplot(2,3,5, projection = '3d')
    b = ax.scatter(*clean_umap[:,:3].T, c = clean_pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_title('Umap pos w/o outliers')

    ax = plt.subplot(2,3,3, projection = '3d')
    b = ax.scatter(*clean_umap[down_idx,:3].T,c = clean_pos[down_idx,0], cmap = 'magma', s = 10)
    ax.set_title('Umap pos after topological denoising on signal')

    ax = plt.subplot(2,3,6, projection = '3d')
    b = ax.scatter(*down_umap[:,:3].T,c = clean_pos[down_idx2,0], cmap = 'magma', s = 10)
    ax.set_title('Umap pos after topological denoising on emb')
    fig.suptitle(f"{mouse}: dist_th={dist_th} | noise_th={noise_th} | num_samples={num_samples}")
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_cloudDenoising.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    #compute betti numbers
    print("\tComputing Betti Numbers Og...", sep= '', end = '')
    down_D = pairwise_distances(down_signal)
    a = rpp_py.run(f"--dim 2 --format point-cloud --threshold {int(np.ceil(np.nanmax(down_D)))}",down_signal)
    print("\b\b\b: Done")

    diagrams = list()
    diagrams.append(np.zeros((a[0].shape[0],2)))
    for b in range(diagrams[0].shape[0]):
        diagrams[0][b][0] = a[0][b][0]
        diagrams[0][b][1] = a[0][b][1]
    diagrams[0][-1,1] = np.nanmax(down_D)

    diagrams.append(np.zeros((a[1].shape[0],2)))
    for b in range(diagrams[1].shape[0]):
        diagrams[1][b][0] = a[1][b][0]
        diagrams[1][b][1] = a[1][b][1]

    diagrams.append(np.zeros((a[2].shape[0],2)))
    for b in range(diagrams[2].shape[0]):
        diagrams[2][b][0] = a[2][b][0]
        diagrams[2][b][1] = a[2][b][1]

    tril_D = down_D[np.tril_indices_from(down_D, k=-1)]
    num_edges = int(down_D.shape[0]*(down_D.shape[0]-1)/2)
    computed_distances = dict()
    dense_diagrams = copy.deepcopy(diagrams)
    for betti_num in range(len(diagrams)):
        for bar_num in range(diagrams[betti_num].shape[0]):
            st = diagrams[betti_num][bar_num][0]
            en = diagrams[betti_num][bar_num][1]

            if st in computed_distances.keys():
                dense_diagrams[betti_num][bar_num][0] = computed_distances[st]
            else:
                dense_start = sum(tril_D <= st)/num_edges
                dense_diagrams[betti_num][bar_num][0] = dense_start
                computed_distances[st] = dense_start

            if en in computed_distances.keys():
                dense_diagrams[betti_num][bar_num][1] = computed_distances[en]
            else:
                dense_end =  sum(tril_D <= en)/num_edges
                dense_diagrams[betti_num][bar_num][1] = dense_end
                computed_distances[en] = dense_end


    #plot birth/death scatter
    fig = plt.figure()
    plot_diagrams(diagrams, show=True)
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_diagrams_scatter_og.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_diagrams_scatter_og.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    #plot betti bars
    fig = plot_betti_bars(diagrams, max_dist=np.nanmax(down_D))
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_diagrams_bars_og.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_diagrams_bars_og.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plot_diagrams(dense_diagrams, show=True)
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_dense_diagrams_scatter_og.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_dense_diagrams_scatter_og.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    #plot betti bars
    fig = plot_betti_bars(dense_diagrams, max_dist=1)
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_dense_diagrams_bars_og.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_dense_diagrams_bars_og.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    #save
    params = {
        'dist_th':dist_th,
        'noise_th':noise_th,
        'num_samples':num_samples,
        'num_iters': num_iters,
        'dim': dim,
        'num_neigh': num_neigh,
        'min_dist': min_dist,
        'signal_name': signal_name,
        'perc_cells': perc_cells,
        'perc_time': perc_time,
        'fluo_th': fluo_th,
        'num_cells': num_cells
    }

    betti_dict = {
        'mouse': mouse,
        'data_dir': data_dir,
        'save_dir': save_dir,
        'pd_mouse': copy.deepcopy(pd_mouse),
        'D': D,
        'del_mi_cells': del_mi_cells,
        'del_cells': del_cells,
        'del_time': del_time,
        'signalIdx': signalIdx,
        'down_idx': down_idx,
        'down_signal': down_signal,
        'down_D': down_D,
        'diagrams': diagrams,
        'dense_diagrams': dense_diagrams,
        'params': copy.deepcopy(params)
    }
    save_pickle(os.path.join(save_dir, mouse), mouse+'_betti_dict_og.pkl', betti_dict)




data_dir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig2/betti_numbers/og_mi_cells/'

#shuffling og
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    betti_dict = load_pickle(os.path.join(save_dir,mouse),mouse+'_betti_dict_og.pkl')

    signal = betti_dict['down_signal']
    og_D = betti_dict['down_D']

    minShift = 1
    maxShift = signal.shape[0]-1
    shift_diagrams = list()
    dense_shift_diagrams = list()
    bar=tqdm(total=1000, desc='Computing Betti on Shuffling')
    for iter in range(1000):
        shift_signal = copy.deepcopy(signal)
        timeShift = np.random.randint(minShift, maxShift,signal.shape[1])
        for cell, shift in enumerate(timeShift):
            shift_signal[:-shift,cell] = copy.deepcopy(signal[shift:,cell])
            shift_signal[-shift:,cell] = copy.deepcopy(signal[:shift,cell])
        #compute betti numbers
        a = rpp_py.run(f"--dim 2 --format point-cloud --threshold {int(np.nanmax(og_D))}",shift_signal)
        #compute betti numbers
        shift_down_D = pairwise_distances(shift_signal)
        shift_tril_D = shift_down_D[np.tril_indices_from(shift_down_D, k=-1)]
        num_edges = int(shift_down_D.shape[0]*(shift_down_D.shape[0]-1)/2)
        a = rpp_py.run(f"--dim 2 --format point-cloud --threshold {int(np.ceil(np.nanmax(shift_down_D)))}",shift_signal)
        computedDistances = dict()
        if iter == 0:
            #betti 0
            shift_diagrams.append(np.zeros((a[0].shape[0],2)))
            dense_shift_diagrams.append(np.zeros((a[0].shape[0],2)))
            for b in range(shift_diagrams[0].shape[0]):
                shift_diagrams[0][b][0] = a[0][b][0]
                shift_diagrams[0][b][1] = a[0][b][1]
            #density
            bar_lens = shift_diagrams[0][:,1] - shift_diagrams[0][:,0]
            longest_bar = shift_diagrams[0][(-bar_lens).argsort()[0]]
            dense_shift_diagrams[0][b][0] = sum(shift_tril_D <= longest_bar[0])/num_edges
            dense_shift_diagrams[0][b][1] = sum(shift_tril_D <= longest_bar[1])/num_edges

            #betti 1
            shift_diagrams.append(np.zeros((a[1].shape[0],2)))
            dense_shift_diagrams.append(np.zeros((a[1].shape[0],2)))
            for b in range(shift_diagrams[1].shape[0]):
                shift_diagrams[1][b][0] = a[1][b][0]
                shift_diagrams[1][b][1] = a[1][b][1]
            #density
            bar_lens = shift_diagrams[1][:,1] - shift_diagrams[1][:,0]
            longest_bar = shift_diagrams[1][(-bar_lens).argsort()[0]]
            dense_shift_diagrams[1][b][0] = sum(shift_tril_D <= longest_bar[0])/num_edges
            dense_shift_diagrams[1][b][1] = sum(shift_tril_D <= longest_bar[1])/num_edges

            #betti 2
            shift_diagrams.append(np.zeros((a[2].shape[0],2)))
            dense_shift_diagrams.append(np.zeros((a[2].shape[0],2)))
            for b in range(shift_diagrams[2].shape[0]):
                shift_diagrams[2][b][0] = a[2][b][0]
                shift_diagrams[2][b][1] = a[2][b][1]
            #density
            bar_lens = shift_diagrams[2][:,1] - shift_diagrams[2][:,0]
            longest_bar = shift_diagrams[2][(-bar_lens).argsort()[0]]
            dense_shift_diagrams[2][b][0] = sum(shift_tril_D <= longest_bar[0])/num_edges
            dense_shift_diagrams[2][b][1] = sum(shift_tril_D <= longest_bar[1])/num_edges

        else:
            #betti 0 
            st = shift_diagrams[0].shape[0]
            shift_diagrams[0] = np.concatenate((shift_diagrams[0],np.zeros((a[0].shape[0],2))))
            dense_shift_diagrams[0] = np.concatenate((dense_shift_diagrams[0],np.zeros((a[0].shape[0],2))))
            for b in range(a[0].shape[0]):
                shift_diagrams[0][b+st][0] = a[0][b][0]
                shift_diagrams[0][b+st][1] = a[0][b][1]
            #density
            bar_lens = shift_diagrams[0][st:,1] - shift_diagrams[0][st:,0]
            longest_bar = shift_diagrams[0][st+(-bar_lens).argsort()[0]]
            dense_shift_diagrams[0][b+st][0] = sum(shift_tril_D <= longest_bar[0])/num_edges
            dense_shift_diagrams[0][b+st][1] = sum(shift_tril_D <= longest_bar[1])/num_edges

            #betti 1
            st = shift_diagrams[1].shape[0]
            shift_diagrams[1] = np.concatenate((shift_diagrams[1],np.zeros((a[1].shape[0],2))))
            dense_shift_diagrams[1] = np.concatenate((dense_shift_diagrams[1],np.zeros((a[1].shape[0],2))))
            for b in range(a[1].shape[0]):
                shift_diagrams[1][b+st][0] = a[1][b][0]
                shift_diagrams[1][b+st][1] = a[1][b][1]
            #density
            bar_lens = shift_diagrams[1][st:,1] - shift_diagrams[1][st:,0]
            longest_bar = shift_diagrams[1][st+(-bar_lens).argsort()[0]]
            dense_shift_diagrams[1][b+st][0] = sum(shift_tril_D <= longest_bar[0])/num_edges
            dense_shift_diagrams[1][b+st][1] = sum(shift_tril_D <= longest_bar[1])/num_edges


            #betti 2
            st = shift_diagrams[2].shape[0]
            shift_diagrams[2] = np.concatenate((shift_diagrams[2],np.zeros((a[2].shape[0],2))))
            dense_shift_diagrams[2] = np.concatenate((dense_shift_diagrams[2],np.zeros((a[2].shape[0],2))))
            for b in range(a[2].shape[0]):
                shift_diagrams[2][b+st][0] = a[2][b][0]
                shift_diagrams[2][b+st][1] = a[2][b][1]
            #density
            bar_lens = shift_diagrams[2][st:,1] - shift_diagrams[2][st:,0]
            longest_bar = shift_diagrams[0][st+(-bar_lens).argsort()[0]]
            dense_shift_diagrams[2][b+st][0] = sum(shift_tril_D <= longest_bar[0])/num_edges
            dense_shift_diagrams[2][b+st][1] = sum(shift_tril_D <= longest_bar[1])/num_edges
        bar.update(1)  
    bar.close()

    conf_interval = list()
    conf_interval.append(np.max(np.diff(shift_diagrams[0],axis=1)))
    conf_interval.append(np.max(np.diff(shift_diagrams[1],axis=1)))
    conf_interval.append(np.max(np.diff(shift_diagrams[2],axis=1)))
    betti_dict['conf_interval'] = conf_interval
    betti_dict['shift_diagrams'] = shift_diagrams


    dense_conf_interval = list()
    dense_conf_interval.append(np.percentile(np.diff(dense_shift_diagrams[0],axis=1), 99.8))
    dense_conf_interval.append(np.percentile(np.diff(dense_shift_diagrams[1],axis=1), 99.8))
    dense_conf_interval.append(np.percentile(np.diff(dense_shift_diagrams[2],axis=1), 99.8))  
    betti_dict['dense_conf_interval'] = dense_conf_interval
    betti_dict['dense_shift_diagrams'] = dense_shift_diagrams

    save_pickle(os.path.join(save_dir,mouse), mouse+'_betti_dict_og.pkl', betti_dict)

    fig = plot_betti_bars(betti_dict['diagrams'], conf_interval = betti_dict['conf_interval'])
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_diagrams_bars_conf_og.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_diagrams_bars_conf_og.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plot_betti_bars(betti_dict['dense_diagrams'], conf_interval = betti_dict['dense_conf_interval'])
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_dense_diagrams_bars_conf_og.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, mouse, f'{mouse}_dense_diagrams_bars_conf_og.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

#__________________________________________________________________________
#|                                                                        |#
#|                     PLOT BETTI NUMBERS LIFETIME                        |#
#|________________________________________________________________________|#


save_dir = '/home/julio/Documents/SP_project/Fig2/betti_numbers/og_mi_cells'
miceList = ['GC2','GC3','ChZ7', 'ChZ8', 'GC7','CZ4', 'CZ6', 'CZ8', 'CGrin1']
supMice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']

hList = list()
lifeTimeList = list()
mouseList = list()
layerList = list()
space = 'og'
for mouse in miceList:

    betti_dict = load_pickle(os.path.join(save_dir,mouse), mouse+'_betti_dict_'+space+'.pkl')
    try:
        dense_conf_interval1 = betti_dict['dense_conf_interval'][1]
        dense_conf_interval2 = betti_dict['dense_conf_interval'][2]
    except:
        dense_conf_interval1 = 0
        dense_conf_interval2 = 0

    h1Diagrams = np.array(betti_dict['dense_diagrams'][1])
    h1Length = np.sort(np.diff(h1Diagrams, axis=1)[:,0])
    second_length = np.max([dense_conf_interval1, h1Length[-2]])

    lifeTimeList.append((h1Length[-1]-second_length))
    hList.append('h1')

    h2Diagrams = np.array(betti_dict['dense_diagrams'][2])
    h2Length = np.sort(np.diff(h2Diagrams, axis=1)[:,0])
    second_length = np.max([dense_conf_interval2, h2Length[-2]])
    lifeTimeList.append((h2Length[-1]-second_length))
    hList.append('h2')

    if mouse in deepMice:
        layerList.append('deep')
        layerList.append('deep')
    elif mouse in supMice:
        layerList.append('sup')
        layerList.append('sup')

    mouseList.append(mouse)
    mouseList.append(mouse)

pd_betti = pd.DataFrame(data={'mouse': mouseList,
                     'denseLifeTime': lifeTimeList,
                     'layer': layerList,
                     'betti': hList})    

fig, ax = plt.subplots(1, 1, figsize=(6,10))
b = sns.boxplot(x='betti', y='denseLifeTime', data=pd_betti, hue='layer',
            linewidth = 1, width= .5, ax = ax)

sns.swarmplot(x='betti', y='denseLifeTime', data=pd_betti, hue='layer',
            palette='dark:gray', ax = ax)
ax.plot([-.25,.25], [0,0], linestyle='--', color='black')
ax.plot([.75,1.25], [0,0], linestyle='--', color='black')
# ax.set_ylim([-0.1, 0.6])

plt.savefig(os.path.join(save_dir,f'dense_lifetime_betti_{space}.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dense_lifetime_betti_{space}.png'), dpi = 400,bbox_inches="tight")




deepH1 = pd_betti.loc[(pd_betti['betti']=='h1')*(pd_betti['layer']=='deep')]['denseLifeTime']
deepH2 = pd_betti.loc[(pd_betti['betti']=='h2')*(pd_betti['layer']=='deep')]['denseLifeTime']

supH1 = pd_betti.loc[(pd_betti['betti']=='h1')*(pd_betti['layer']=='sup')]['denseLifeTime']
supH2 = pd_betti.loc[(pd_betti['betti']=='h2')*(pd_betti['layer']=='sup')]['denseLifeTime']

deepH1_norm = stats.shapiro(deepH1)
deepH2_norm = stats.shapiro(deepH2)
supH1_norm = stats.shapiro(supH1)
supH2_norm = stats.shapiro(supH2)

if deepH1_norm.pvalue<=0.05 or deepH2_norm.pvalue<=0.05:
    print('deepH1 vs deepH2:',stats.ks_2samp(deepH1, deepH2))
else:
    print('deepH1 vs deepH2:', stats.ttest_ind(deepH1, deepH2))

if deepH1_norm.pvalue<=0.05 or supH1_norm.pvalue<=0.05:
    print('deepH1 vs supH1:',stats.ks_2samp(deepH1, supH1))
else:
    print('deepH1 vs supH1:',stats.ttest_ind(deepH1, supH1))

if deepH1_norm.pvalue<=0.05 or supH2_norm.pvalue<=0.05:
    print('deepH1 vs supH2:',stats.ks_2samp(deepH1, supH2))
else:
    print('deepH1 vs supH2:', stats.ttest_ind(deepH1, supH2))

if supH1_norm.pvalue<=0.05 or deepH2_norm.pvalue<=0.05:
    print('supH1 vs deepH2',stats.ks_2samp(supH1, deepH2))
else:
    print('supH1 vs deepH2:', stats.ttest_ind(supH1, deepH2))

if supH1_norm.pvalue<=0.05 or supH2_norm.pvalue<=0.05:
    print('supH1 vs supH2',stats.ks_2samp(supH1, supH2))
else:
    print('supH1 vs supH2:', stats.ttest_ind(supH1, supH2))

if deepH2_norm.pvalue<=0.05 or supH2_norm.pvalue<=0.05:
    print('deepH2 vs supH2',stats.ks_2samp(deepH2, supH2))
else:
    print('deepH2 vs supH2:', stats.ttest_ind(deepH2, supH2))