import umap, math
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
from datetime import datetime
import os
from neural_manifold import dimensionality_reduction as dim_red

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def add_dir_mat_field(pd_struct):
    out_pd = copy.deepcopy(pd_struct)

    out_pd["dir_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                        ('L' == out_pd["dir"][idx])+ 2*('R' == out_pd["dir"][idx])
                        for idx in out_pd.index]
    return out_pd

#__________________________________________________________________________
#|                                                                        |#
#|                           COMPUTE INNER DIM                            |#
#|________________________________________________________________________|#

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'signal_name': 'clean_traces',
    'n_neigh': 40,
    'verbose': True
}
data_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/processed_data'
save_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/inner_dim'

for mouse in mice_list:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    animal_dict = load_pickle(file_path,file_name)
    fnames = list(animal_dict.keys())
    animal_p= copy.deepcopy(animal_dict[fnames[0]])
    animal_r= copy.deepcopy(animal_dict[fnames[1]])

    signal_p = copy.deepcopy(np.concatenate(animal_p[params['signal_name']].values, axis=0))
    signal_r = copy.deepcopy(np.concatenate(animal_r[params['signal_name']].values, axis=0))


    abids_p = dim_red.compute_abids(signal_p, params['n_neigh'])
    abids_p_dim = np.nanmean(abids_p)
    print(f'\t{mouse} pre: {abids_p_dim:.2f} dim')

    abids_r = dim_red.compute_abids(signal_r, params['n_neigh'])
    abids_r_dim = np.nanmean(abids_r)
    print(f'\t{mouse} rot: {abids_r_dim:.2f} dim')

    concat_signal = np.vstack((signal_p, signal_r))
    abids_both = dim_red.compute_abids(concat_signal, params['n_neigh'])
    abids_both_dim = np.nanmean(abids_both)
    print(f'\t{mouse} both: {abids_both_dim:.2f} dim')

    inner_dim = {
        'abids_p': abids_p,
        'abids_p_dim': abids_p_dim,

        'abids_r': abids_r,
        'abids_r_dim': abids_r_dim,

        'abids_both': abids_both,
        'abids_both_dim': abids_both_dim,

        'params': params
    }

    with open(os.path.join(save_dir, mouse+"_inner_dim.pkl"), "wb") as file:
        pickle.dump(inner_dim, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT INNER DIM                             |#
#|________________________________________________________________________|#
import pandas as pd
import seaborn as sns
deep_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
sup_list = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

deep_dict = dict()
deep_ID = list()
for mouse in deep_list:
    file_name = mouse + '_inner_dim.pkl'
    deep_dict[mouse] = load_pickle(save_dir,file_name)
    deep_ID.append(deep_dict[mouse]['abids_both_dim'])


sup_dict = dict()
sup_ID = list()
for mouse in sup_list:
    file_name = mouse + '_inner_dim.pkl'
    sup_dict[mouse] = load_pickle(save_dir,file_name)
    sup_ID.append(sup_dict[mouse]['abids_both_dim'])

nn = sup_dict[mouse]['params']['n_neigh']


type_list = ['Deep']*len(deep_list) + ['Sup']*len(sup_list)
ID_struct = pd.DataFrame(data= {'cell_type': type_list,
                                'abids': np.array(deep_ID+sup_ID).reshape(-1,1).T[0,:]})

#%% PLOTS
palette=['#62C370', '#C360B4', '#6083C3', '#C3A060']
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
b = sns.boxplot(x='cell_type', y='abids', data=ID_struct,
                palette = palette, linewidth = 1, width= .5, ax = ax)

sns.swarmplot(x='cell_type', y='abids', data=ID_struct, 
    palette = palette, edgecolor = 'gray', ax = ax)

b.set_xlabel(" ",fontsize=15)
b.set_ylabel(f"abids (nn={nn})",fontsize=15)
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)
b.set_ylim([-0.05, 4.1])
b.tick_params(labelsize=12)
b.set_yticks([0, 1, 2, 3, 4])
plt.savefig(os.path.join(save_dir,'DeepSup_ID.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'DeepSup_ID.svg'), dpi = 400,bbox_inches="tight",transparent=True)









def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx


from sklearn.metrics import pairwise_distances
from matplotlib import gridspec

data_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/processed_data'
signal_field = 'clean_traces'
mouse = 'TGrin1'
animal_dict = load_pickle(os.path.join(data_dir, mouse), mouse+'_df_dict.pkl')
fnames = list(animal_dict.keys())
animal_p= copy.deepcopy(animal_dict[fnames[0]])
animal_r= copy.deepcopy(animal_dict[fnames[1]])
animal_p = add_dir_mat_field(animal_p)
animal_r = add_dir_mat_field(animal_r)

nn_val = 120
dim = 4

signal_p = copy.deepcopy(np.concatenate(animal_p['clean_traces'].values, axis=0))
pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
dir_mat_p = copy.deepcopy(np.concatenate(animal_p['dir_mat'].values, axis=0))

signal_r = copy.deepcopy(np.concatenate(animal_r['clean_traces'].values, axis=0))
pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
dir_mat_r = copy.deepcopy(np.concatenate(animal_r['dir_mat'].values, axis=0))

#%%all data
index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
concat_signal = np.vstack((signal_p, signal_r))
model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
# model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
model.fit(concat_signal)
concat_emb = model.transform(concat_signal)
emb_p = concat_emb[index[:,0]==0,:]
emb_r = concat_emb[index[:,0]==1,:]



plt.figure()
ax = plt.subplot(1,2,1, projection = '3d')
ax.scatter(*emb_p[:,1:].T, color ='b', s= 30, cmap = 'magma')
ax.scatter(*emb_r[:,1:].T, color = 'r', s= 30, cmap = 'magma')
ax.set_title('All')
ax = plt.subplot(1,2,2, projection = '3d')
ax.scatter(*emb_p[:,1:].T, c = pos_p[:,0], s= 30, cmap = 'magma')
ax.scatter(*emb_r[:,1:].T, c = pos_r[:,0], s= 30, cmap = 'magma')


D_p = pairwise_distances(emb_p)
noiseIdx_p = filter_noisy_outliers(emb_p,D_p)
max_dist = np.nanmax(D_p)
clean_emb = emb_p[~noiseIdx_p,:]
clean_pos = pos_p[~noiseIdx_p,:]



#%%
plt.figure()
ax = plt.subplot(2,2,1, projection = '3d')
ax.scatter(*emb_p[:,:3].T, color = 'b', s= 30, cmap = 'magma')
ax.scatter(*emb_p[noiseIdx_p,:3].T, color = 'r', s= 30, cmap = 'magma')
ax = plt.subplot(2,2,2, projection = '3d')
ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')

ax = plt.subplot(2,2,3, projection = '3d')
ax.scatter(*clean_emb[:,:3].T, c = clean_pos[:,0], s= 30, cmap = 'magma')
plt.suptitle(f"{mouse}")

n_points = 1000
clean_dir_mat = dir_mat_p[~noiseIdx_p]
idx = np.random.choice(clean_emb.shape[0], n_points)
sclean_emb = clean_emb.copy()[idx,:]


plt.figure()
ax = plt.subplot(2,2,1, projection = '3d')
ax.scatter(*clean_emb[:,:3].T, c = clean_dir_mat, s= 30, cmap = 'magma')
ax = plt.subplot(2,2,2, projection = '3d')
ax.scatter(*clean_emb[:,:3].T, c = clean_pos[:,0], s= 30, cmap = 'magma')

ax = plt.subplot(2,2,3, projection = '3d')
ax.scatter(*clean_emb[idx,:3].T, c = clean_dir_mat[idx], s= 30, cmap = 'magma')
ax = plt.subplot(2,2,4, projection = '3d')
ax.scatter(*clean_emb[idx,:3].T, c = clean_pos[idx,0], s= 30, cmap = 'magma')

plt.suptitle(f"{mouse}")


barcodes = tda(sclean_emb, maxdim=1, coeff=2, thresh=max_dist)['dgms']
col_list = ['r', 'g', 'm', 'c']

h0, h1, h2 = barcodes[0], barcodes[1], barcodes[1]
# replace the infinity bar (-1) in H0 by a really large number
h0[~np.isfinite(h0)] = max_dist
# Plot the 30 longest barcodes only
to_plot = []
for curr_h in [h0, h1, h2]:
     bar_lens = curr_h[:,1] - curr_h[:,0]
     plot_h = curr_h[(-bar_lens).argsort()[:30]]
     to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 4)
for curr_betti, curr_bar in enumerate(to_plot):
    ax = fig.add_subplot(gs[curr_betti, :])
    for i, interval in enumerate(reversed(curr_bar)):
        ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
            lw=1.5)
    ax.set_ylabel('H' + str(curr_betti))
    ax.set_xlim([-0.5, np.max(np.vstack((h0,h1,h2)))+0.5])
    # ax.set_xticks([0, xlim])
    ax.set_ylim([-1, len(curr_bar)])
plt.suptitle(f"{fname}: {s}% ({n_points})")


D_p = pairwise_distances(signal_p)
noiseIdx_p = filter_noisy_outliers(signal_p,D_p)
max_dist = np.nanmax(D_p)
n_points = 2000
clean_emb = signal_p[~noiseIdx_p,:]
clean_pos = pos_p[~noiseIdx_p,:]
clean_dir_mat = dir_mat_p[~noiseIdx_p]
idx = np.random.choice(clean_emb.shape[0], n_points)
sclean_emb = clean_emb.copy()[idx,:]


#%%
plt.figure()
ax = plt.subplot(1,2,1, projection = '3d')
ax.scatter(*emb_p[:,:3].T, color ='b', s= 30, cmap = 'magma')
ax.scatter(*emb_r[:,:3].T, color = 'r', s= 30, cmap = 'magma')

ax = plt.subplot(1,2,2, projection = '3d')
ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
plt.suptitle(f"{mouse}")

#__________________________________________________________________________
#|                                                                        |#
#|                           COMPUTE UMAP                                 |#
#|________________________________________________________________________|#

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

base_load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data'
base_save_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/processed_data'
signal_field = 'raw_traces'
vel_th = 6
sigma = 6
sig_up = 4
sig_down = 12
nn_val = 120
dim = 3


nn_val = 120
dim = 3

signal_p = copy.deepcopy(np.concatenate(animal_p['clean_traces'].values, axis=0))
pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
index_mat_p = copy.deepcopy(np.concatenate(animal_p['index_mat'].values, axis=0))

signal_r = copy.deepcopy(np.concatenate(animal_r['clean_traces'].values, axis=0))
pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
index_mat_r = copy.deepcopy(np.concatenate(animal_p['index_mat'].values, axis=0))

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

ax = plt.subplot(1,2,2, projection = '3d')
ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
plt.suptitle(f"{mouse}: clean_traces - vel: {vel_th} - nn: {nn_val} - dim: {dim}")
plt.tight_layout()
plt.savefig(os.path.join(save_dir_fig,mouse+'_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir_fig,'umap.svg'), dpi = 400,bbox_inches="tight",transparent=True)

save_dir_fig
index_mat_p = np.concatenate(animal_p["index_mat"].values, axis=0)
animal_p['umap'] = [emb_p[index_mat_p[:,0]==animal_p["trial_id"][idx] ,:] 
                               for idx in animal_p.index]
index_mat_r = np.concatenate(animal_r["index_mat"].values, axis=0)
animal_r['umap'] = [emb_r[index_mat_r[:,0]==animal_r["trial_id"][idx] ,:] 
                               for idx in animal_r.index]

params["nn"] = nn_val
params["dim"] = dim
animal_dict = {
    fnames[0]: animal_p,
    fnames[1]: animal_r
}
with open(os.path.join(save_dir, mouse+"_df_dict.pkl"), "wb") as file:
    pickle.dump(animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(save_dir, mouse+"_params.pkl"), "wb") as file:
    pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(save_dir, mouse+"_umap_object.pkl"), "wb") as file:
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

