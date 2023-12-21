from neural_manifold import general_utils as gu

import sys, os, timeit
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA


mouse = sys.argv[1]
data_dir = sys.argv[2]
save_dir = sys.argv[3]
if len(sys.argv)>4:
    signal = sys.argv[4]
else:
    signal = 'revents_SNR3'

if len(sys.argv)>5:
    dim = sys.argv[5]
else:
    dim = 3

n_neigh = 50
min_dist = 0.1
#__________________________________________________________________________
#|                                                                        |#
#|                              1. LOAD DATA                              |#
#|________________________________________________________________________|#
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

figures_dir = os.path.join(save_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

f = open(os.path.join(save_dir,mouse + '_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

global_starttime = timeit.default_timer()

print(f"Working on mouse {mouse}:")
print(f"\tdata_dir: {data_dir}")
print(f"\tsave_dir: {save_dir}")
print(f"\tDate: {datetime.now():%Y-%m-%d %H:%M}")

#1. Load data
local_starttime = timeit.default_timer()
print('### 1. LOAD DATA ###')
print('1 Searching & loading data in directory:\n', data_dir)
mouse_pd = gu.load_files(data_dir, '*'+mouse+'_rates_dict*.pkl', verbose=True, 
                                                    struct_type = "pickle")

fnames = list(mouse_pd.keys())
gu.print_time_verbose(local_starttime, global_starttime)

#__________________________________________________________________________
#|                                                                        |#
#|                            2. CONCAT DATA                              |#
#|________________________________________________________________________|#
print('### 2. CONCAT DATA ###')
signal_pre = np.concatenate(mouse_pd[fnames[0]][signal].values, axis = 0)
signal_rot = np.concatenate(mouse_pd[fnames[1]][signal].values, axis = 0)
concat_signal = np.concatenate((signal_pre,signal_rot), axis = 0)

session_idx_pre = np.zeros((signal_pre.shape[0],1))
session_idx_rot = np.zeros((signal_rot.shape[0],1))+1
session_idx = np.concatenate((session_idx_pre,session_idx_rot), axis = 0)

if 'index_mat' not in mouse_pd[fnames[0]]:
    mouse_pd[fnames[0]]["index_mat"] = [np.zeros((mouse_pd[fnames[0]][signal][idx].shape[0],1))+mouse_pd[fnames[0]]["trial_id"][idx] 
                                  for idx in mouse_pd[fnames[0]].index]
index_mat_pre = np.concatenate(mouse_pd[fnames[0]]["index_mat"].values, axis=0)

if 'index_mat' not in mouse_pd[fnames[1]]:
    mouse_pd[fnames[1]]["index_mat"] = [np.zeros((mouse_pd[fnames[1]][signal][idx].shape[0],1))+mouse_pd[fnames[1]]["trial_id"][idx] 
                                  for idx in mouse_pd[fnames[1]].index]
index_mat_rot = np.concatenate(mouse_pd[fnames[1]]["index_mat"].values, axis=0)

if 'dir_mat' not in mouse_pd[fnames[0]].columns:
        mouse_pd[fnames[0]]["dir_mat"] = [np.zeros((mouse_pd[fnames[0]]["pos"][idx].shape[0],1)).astype(int)+
                                    ('L' == mouse_pd[fnames[0]]["dir"][idx])+ 2*('R' == mouse_pd[fnames[0]]["dir"][idx])
                                    for idx in mouse_pd[fnames[0]].index]

if 'dir_mat' not in mouse_pd[fnames[1]].columns:
        mouse_pd[fnames[1]]["dir_mat"] = [np.zeros((mouse_pd[fnames[1]]["pos"][idx].shape[0],1)).astype(int)+
                                    ('L' == mouse_pd[fnames[1]]["dir"][idx])+ 2*('R' == mouse_pd[fnames[1]]["dir"][idx])
                                    for idx in mouse_pd[fnames[1]].index]
#__________________________________________________________________________
#|                                                                        |#
#|                               3. UMAP                                  |#
#|________________________________________________________________________|#
print('### 3. UMAP ###')
umap_model = umap.UMAP(n_neighbors = n_neigh, n_components = dim, min_dist=min_dist)

umap_model.fit(concat_signal)
concat_emb = umap_model.transform(concat_signal)

emb_p = concat_emb[session_idx[:,0]==0,:]
emb_r = concat_emb[session_idx[:,0]==1,:]


mouse_pd[fnames[0]]['umap'] = [emb_p[index_mat_pre[:,0]==mouse_pd[fnames[0]]["trial_id"][idx] ,:] 
                                   for idx in mouse_pd[fnames[0]].index]

mouse_pd[fnames[1]]['umap'] = [emb_r[index_mat_rot[:,0]==mouse_pd[fnames[1]]["trial_id"][idx] ,:] 
                                   for idx in mouse_pd[fnames[1]].index]



#save umap umap_model
file_name = os.path.join(save_dir, mouse+ "_umap_model.pkl")
save_df = open(file_name, "wb")
pickle.dump(umap_model, save_df)
save_df.close()

#save mouse_pd
file_name = os.path.join(save_dir_step, mouse+ "_dict.pkl")
save_df = open(file_name, "wb")
pickle.dump(mouse_pd, save_df)
save_df.close()


#__________________________________________________________________________
#|                                                                        |#
#|                               4. ISOMAP                                |#
#|________________________________________________________________________|#
print('### 4. ISOMAP ###')
isomap_model = Isomap(n_neighbors = n_neigh, n_components = dim)

isomap_model.fit(concat_signal)
concat_emb = isomap_model.transform(concat_signal)

emb_p = concat_emb[session_idx[:,0]==0,:]
emb_r = concat_emb[session_idx[:,0]==1,:]


mouse_pd[fnames[0]]['isomap'] = [emb_p[index_mat_pre[:,0]==mouse_pd[fnames[0]]["trial_id"][idx] ,:] 
                                   for idx in mouse_pd[fnames[0]].index]

mouse_pd[fnames[1]]['isomap'] = [emb_r[index_mat_rot[:,0]==mouse_pd[fnames[1]]["trial_id"][idx] ,:] 
                                   for idx in mouse_pd[fnames[1]].index]

#save umap isomap_model
file_name = os.path.join(save_dir, mouse+ "_isomap_model.pkl")
save_df = open(file_name, "wb")
pickle.dump(isomap_model, save_df)
save_df.close()

#save mouse_pd
file_name = os.path.join(save_dir_step, mouse+ "_dict.pkl")
save_df = open(file_name, "wb")
pickle.dump(mouse_pd, save_df)
save_df.close()


#__________________________________________________________________________
#|                                                                        |#
#|                                5. PCA                                  |#
#|________________________________________________________________________|#
print('### 5. PCA ###')

pca_model = PCA(concat_signal.shape[1])
pca_model.fit(concat_signal)
concat_emb = pca_model.transform(concat_signal)

emb_p = concat_emb[session_idx[:,0]==0,:]
emb_r = concat_emb[session_idx[:,0]==1,:]


mouse_pd[fnames[0]]['pca'] = [emb_p[index_mat_pre[:,0]==mouse_pd[fnames[0]]["trial_id"][idx] ,:] 
                                   for idx in mouse_pd[fnames[0]].index]

mouse_pd[fnames[1]]['pca'] = [emb_r[index_mat_rot[:,0]==mouse_pd[fnames[1]]["trial_id"][idx] ,:] 
                                   for idx in mouse_pd[fnames[1]].index]

#save umap pca_model
file_name = os.path.join(save_dir, mouse+ "_pca_model.pkl")
save_df = open(file_name, "wb")
pickle.dump(pca_model, save_df)
save_df.close()

#save mouse_pd
file_name = os.path.join(save_dir_step, mouse+ "_dict.pkl")
save_df = open(file_name, "wb")
pickle.dump(mouse_pd, save_df)
save_df.close()

#__________________________________________________________________________
#|                                                                        |#
#|                               6. PLOT                                  |#
#|________________________________________________________________________|#
#PLOT DATA
fig = plt.figure(figsize=(12,14))
for idx in range(2):
    ax = plt.subplot(4,3,1+6*idx, projection='3d')
    data = np.concatenate(mouse_pd[fnames[idx]]['pca'].values, axis = 0)
    label = np.concatenate(mouse_pd[fnames[idx]]['pos'].values, axis = 0)
    b = ax.scatter(*data[:,:3].T, c = label[:,0], cmap = 'inferno')
    fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('PC 1', labelpad = -8)
    ax.set_ylabel('PC 2', labelpad = -8)
    ax.set_zlabel('PC 3', labelpad = -8)

    ax = plt.subplot(4,3,2+6*idx, projection='3d')
    data = np.concatenate(mouse_pd[fnames[idx]]['isomap'].values, axis = 0)
    label = np.concatenate(mouse_pd[fnames[idx]]['pos'].values, axis = 0)
    b = ax.scatter(*data[:,:3].T, c = label[:,0], cmap = 'inferno')
    fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('IsoDim 1', labelpad = -8)
    ax.set_ylabel('IsoDim 2', labelpad = -8)
    ax.set_zlabel('IsoDim 3', labelpad = -8)

    ax = plt.subplot(4,3,3+6*idx, projection='3d')
    data = np.concatenate(mouse_pd[fnames[idx]]['umap'].values, axis = 0)
    label = np.concatenate(mouse_pd[fnames[idx]]['pos'].values, axis = 0)
    b = ax.scatter(*data[:,:3].T, c = label[:,0], cmap = 'inferno')
    fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('UDim 1', labelpad = -8)
    ax.set_ylabel('UDim 2', labelpad = -8)
    ax.set_zlabel('UDim 3', labelpad = -8)


    ax = plt.subplot(4,3,4+6*idx, projection='3d')
    data = np.concatenate(mouse_pd[fnames[idx]]['pca'].values, axis = 0)
    label = np.concatenate(mouse_pd[fnames[idx]]['dir_mat'].values, axis = 0)
    b = ax.scatter(*data[:,:3].T, c = label[:,0], cmap = 'inferno')
    fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('PC 1', labelpad = -8)
    ax.set_ylabel('PC 2', labelpad = -8)
    ax.set_zlabel('PC 3', labelpad = -8)

    ax = plt.subplot(4,3,5+6*idx, projection='3d')
    data = np.concatenate(mouse_pd[fnames[idx]]['isomap'].values, axis = 0)
    label = np.concatenate(mouse_pd[fnames[idx]]['dir_mat'].values, axis = 0)
    b = ax.scatter(*data[:,:3].T, c = label[:,0], cmap = 'inferno')
    fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('IsoDim 1', labelpad = -8)
    ax.set_ylabel('IsoDim 2', labelpad = -8)
    ax.set_zlabel('IsoDim 3', labelpad = -8)

    ax = plt.subplot(4,3,6+6*idx, projection='3d')
    data = np.concatenate(mouse_pd[fnames[idx]]['umap'].values, axis = 0)
    label = np.concatenate(mouse_pd[fnames[idx]]['dir_mat'].values, axis = 0)
    b = ax.scatter(*data[:,:3].T, c = label[:,0], cmap = 'inferno')
    fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('UDim 1', labelpad = -8)
    ax.set_ylabel('UDim 2', labelpad = -8)
    ax.set_zlabel('UDim 3', labelpad = -8)


plt.tight_layout()
plt.savefig(os.path.join(figures_dir, mouse + '_embs_plot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(figures_dir,mouse + '_embs_plot.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)
