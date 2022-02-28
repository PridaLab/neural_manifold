# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:39:29 2022

@author: JulioEI
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

from neural_manifold import general_utils as gu

from neural_manifold.pipelines.LT_manifold_pipeline import LT_manifold_pipeline
from neural_manifold.pipelines.LT_sI_pipeline import compute_sI
import neural_manifold.decoders as dec

save_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\JP_data\\LT_inscopix\\results'
###############################################################################
#                              MANIFOLD STUDY                                 #
###############################################################################
#%% PARAMETERS
params = {
    #general params
    'keep_only_moving': False,
    'max_session_length': 'adapt_to_min',
    'check_inner_dim': False,
    'compute_place_cells': False,
    #spike field info
    'spikes_field': "Inscopix_events_spikes",
    'rates_kernel_std': 0.4,
    'th_rates_freq': 0.02,
    #traces field info
    'traces_field': "Inscopix_traces",
    #isomap
    'compute_iso_resvar': True,
    #umap params
    'umap_dims': 'optimize_to_umap_trust',
    'check_dim_to_cells_umap': False,
    'neighbours_umap_rates':  0.01,
    'min_dist_umap_rates': 0.75,
    'neighbours_umap_traces': 0.01,
    'min_dist_umap_traces': 0.75,
    'apply_same_model': True
    }
# %% GC1
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\JP_data\\LT_inscopix\\GC1' 
mouse_GC1 = 'GC1'
results_dir_GC1, GC1_dict = LT_manifold_pipeline(data_dir, mouse_GC1, save_dir, **params);
plt.close('all')
# %% GC2
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\JP_data\\LT_inscopix\\GC2' 
mouse_GC2 = 'GC2'
results_dir_GC2, GC2_dict = LT_manifold_pipeline(data_dir, mouse_GC2, save_dir, **params);
plt.close('all')
# %% CZ3
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\JP_data\\LT_inscopix\\CZ3' 
mouse_CZ3 = 'CZ3'
results_dir_CZ3, CZ3_dict = LT_manifold_pipeline(data_dir, mouse_CZ3, save_dir, **params);
plt.close('all')
# %% CZ4
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\JP_data\\LT_inscopix\\CZ4' 
mouse_CZ4 = 'CZ4'
results_dir_CZ4, CZ4_dict = LT_manifold_pipeline(data_dir, mouse_CZ4, save_dir, **params);
plt.close('all')
# %% GC1
sI_GC1_dict = compute_sI(results_dir_GC1, mouse_GC1, ["Inscopix_events_rates","ML_pca", "ML_iso", "ML_umap", 
                                                           "Inscopix_traces", "Inscopix_traces_pca", "Inscopix_traces_iso",
                                                           "Inscopix_traces_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 3, comp_method ='all',load_old_dict = False, nBins = 20)
# %% GC2 
sI_GC2_dict = compute_sI(results_dir_GC2, mouse_GC2, ["Inscopix_events_rates","ML_pca", "ML_iso", "ML_umap", 
                                                           "Inscopix_traces", "Inscopix_traces_pca", "Inscopix_traces_iso",
                                                           "Inscopix_traces_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 3, comp_method ='all',load_old_dict = False, nBins = 20)
# %% CZ3
sI_CZ3_dict = compute_sI(results_dir_CZ3, mouse_CZ3, ["Inscopix_events_rates","ML_pca", "ML_iso", "ML_umap", 
                                                           "Inscopix_traces", "Inscopix_traces_pca", "Inscopix_traces_iso",
                                                           "Inscopix_traces_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 3, comp_method ='all',load_old_dict = False, nBins = 20)
# %% CZ4
sI_CZ4_dict = compute_sI(results_dir_CZ4, mouse_CZ4, ["Inscopix_events_rates","ML_pca", "ML_iso", "ML_umap", 
                                                           "Inscopix_traces", "Inscopix_traces_pca", "Inscopix_traces_iso",
                                                           "Inscopix_traces_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 3, comp_method ='all',load_old_dict = False, nBins = 20)
# %%
def get_sI(sI_dict):
    sI_struct = np.zeros((2,8,3))*np.nan
    new_dict = copy.deepcopy(sI_dict)
    file_idx = -1
    for file, it in new_dict.items():
        file_idx +=1
        field_idx = -1
        for field, it2 in it.items():
            field_idx +=1
            sI_struct[file_idx,field_idx, :] = it2["sI_all"]
    return sI_struct

sI_GC1_mat = get_sI(sI_GC1_dict)
sI_GC2_mat = get_sI(sI_GC2_dict)
sI_CZ3_mat = get_sI(sI_CZ3_dict)
sI_CZ4_mat = get_sI(sI_CZ4_dict)

sI_deep = np.stack((sI_GC1_mat, sI_GC2_mat), axis=3)
sI_sup = np.stack((sI_CZ3_mat, sI_CZ4_mat), axis=3)

fig, ax = plt.subplots(2,2,figsize=(8,8))
fields = ['Umap_events_rates', 'Umap_traces']
varList = ['x-pos', 'y-pos']
x_dims = 2
x_space = np.linspace(1, x_dims, x_dims).astype(int)
color_code = [ 'C4', 'C3', 'C0', 'gray']
for var_idx, var_label in enumerate(varList):
    for field_idx, field in enumerate(fields):
        if field:
            m = np.nanmean(sI_deep[:,3+4*field_idx,var_idx], axis=1)
            sd = np.nanstd(sI_deep[:,3+4*field_idx,var_idx], axis=1)
                        
            ax[var_idx, field_idx].plot(x_space, m, color = color_code[0], label = 'deep')
            ax[var_idx, field_idx].fill_between(x_space, m-sd, m+sd, color = color_code[0], alpha=0.25)
            
            m = np.nanmean(sI_sup[:,3+4*field_idx,var_idx], axis=1)
            sd = np.nanstd(sI_sup[:,3+4*field_idx,var_idx], axis=1)
                        
            ax[var_idx, field_idx].plot(x_space, m, color = color_code[1], label = 'sup')
            ax[var_idx, field_idx].fill_between(x_space, m-sd, m+sd, color = color_code[1], alpha=0.25)
        
        ax[var_idx, field_idx].set_ylabel('Structure Index for '+var_label,fontsize=14)
        ax[var_idx, field_idx].set_ylim([0,1])
        ax[var_idx, field_idx].set_xlim([x_space[0]-0.05, x_space[-1]+0.05])
        plt.setp(ax[var_idx, field_idx].spines.values(), linewidth=2)
        ax[var_idx, field_idx].spines['right'].set_visible(False)
        ax[var_idx, field_idx].spines['top'].set_visible(False)        
        ax[var_idx, field_idx].legend(fontsize=14)
        ax[var_idx, field_idx].set_title(field,fontsize=16)
        ax[var_idx, field_idx].set_xticks(x_space)
        ax[var_idx, field_idx].set_xticklabels(["Day1", "Day2-rotation"],rotation = 45, ha="right")
fig.tight_layout()
plt.show()

# %%
R2s_GC1_dict = gu.apply_to_dict(dec.decoders_1D, GC1_dict, field_signal = "Inscopix_traces", 
                                  emb_list = ["Inscopix_pca", "Inscopix_iso", "Inscopix_umap"], n_dims = 3,
                                  decoder_list = ["wf", "wc", "xgb", "svr"],
                                  input_label = ["posx", "posy","index_mat"], verbose = True, n_splits = 10)

save_file = open(os.path.join(results_dir_GC1, mouse_GC1+ "_traces_decoder_dict.pkl"), "wb")
pickle.dump(R2s_GC1_dict, save_file)
save_file.close

R2s_GC2_dict = gu.apply_to_dict(dec.decoders_1D, GC2_dict, field_signal = "Inscopix_traces", 
                                  emb_list = ["Inscopix_pca", "Inscopix_iso", "Inscopix_umap"], n_dims = 3,
                                  decoder_list = ["wf", "wc", "xgb", "svr"],
                                  input_label = ["posx", "posy","index_mat"], verbose = True, n_splits = 10)

save_file = open(os.path.join(results_dir_GC2, mouse_GC2+ "_traces_decoder_dict.pkl"), "wb")
pickle.dump(R2s_GC2_dict, save_file)
save_file.close
        
R2s_CZ3_dict = gu.apply_to_dict(dec.decoders_1D, CZ3_dict, field_signal = "Inscopix_traces", 
                                  emb_list = ["Inscopix_pca", "Inscopix_iso", "Inscopix_umap"], n_dims = 3,
                                  decoder_list = ["wf", "wc", "xgb", "svr"],
                                  input_label = ["posx", "posy","index_mat"], verbose = True, n_splits = 10)

save_file = open(os.path.join(results_dir_CZ3, mouse_CZ3+ "_traces_decoder_dict.pkl"), "wb")
pickle.dump(R2s_CZ3_dict, save_file)
save_file.close
        

R2s_CZ4_dict = gu.apply_to_dict(dec.decoders_1D, CZ4_dict, field_signal = "Inscopix_traces", 
                                  emb_list = ["Inscopix_pca", "Inscopix_iso", "Inscopix_umap"], n_dims = 3,
                                  decoder_list = ["wf", "wc", "xgb", "svr"],
                                  input_label = ["posx", "posy","index_mat"], verbose = True, n_splits = 10)

save_file = open(os.path.join(results_dir_CZ4, mouse_CZ4+ "_traces_decoder_dict.pkl"), "wb")
pickle.dump(R2s_CZ4_dict, save_file)
save_file.close
# %%
R2s_GC2_dict = gu.apply_to_dict(dec.decoders_delay_1D, GC2_dict, field_signal = "Inscopix_traces", 
                                  emb_list = ["traces_umap"], n_dims = 3,
                                  decoder_list = ["wf", "wc", "xgb", "svr"], input_label = ["posx", "posy","index_mat"],
                                  time_shift = np.arange(-10,10,2).astype(int), verbose = True, n_splits = 10)
save_file = open(os.path.join(results_dir_GC2, mouse_GC2+ "timeShift_decoder_dict.pkl"), "wb")
pickle.dump(R2s_GC2_dict, save_file)
save_file.close()  