# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:31:03 2022

@author: JulioEI
"""

import sys
path_to_module = "E:\\Users\\Usuario\\Documents\\GitHub\\hippocampal_manifolds\\Linear track"
sys.path.append(path_to_module) 
import LT_cI_module as ltcI
import LT_decoders_module as ltd

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

from neural_manifold.pipelines.LT_manifold_pipeline import LT_manifold_pipeline
from neural_manifold import general_utils as gu

save_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\results_pipeline\\move_data'
#%%
###############################################################################
#                              MANIFOLD STUDY                                 #
###############################################################################
#%% PARAMETERS
params = {
    #general params
    'keep_only_moving': True,
    'max_session_length': 'adapt_to_min',
    'check_inner_dim': True,
    'compute_place_cells': False,
    #spike field info
    'spikes_field': "ML_spikes",
    'rates_kernel_std': 0.4,
    'th_rates_freq': 0.05,
    #traces field info
    'traces_field': "deconvProb",
    #isomap
    'compute_iso_resvar': True,
    #umap params
    'umap_dims': 'optimize_to_umap_trust',
    'check_dim_to_cells_umap': False,
    'neighbours_umap_rates':  0.01,
    'min_dist_umap_rates': 0.75,
    'neighbours_umap_traces': 0.01,
    'min_dist_umap_traces': 0.75
    }
# %% M2019
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\2019' 
mouse_M2019 = 'M2019'
results_dir_M2019, M2019_dict = LT_manifold_pipeline(data_dir, mouse_M2019, save_dir, **params);
plt.close('all')
# %% M2021
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\2021' 
mouse_M2021 = 'M2021'
results_dir_M2021, M2021_dict = LT_manifold_pipeline(data_dir, mouse_M2021, save_dir, **params);
plt.close('all')
# %% M2022
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\2022' 
mouse_M2022 = 'M2022'
results_dir_M2022, M2022_dict = LT_manifold_pipeline(data_dir, mouse_M2022, save_dir, **params);
plt.close('all')
# %% M2023
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\2023' 
mouse_M2023 = 'M2023'
results_dir_M2023, M2023_dict = LT_manifold_pipeline(data_dir, mouse_M2023, save_dir, **params);
plt.close('all')
# %% M2024
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\2024' 
mouse_M2024 = 'M2024'
results_dir_M2024, M2024_dict = LT_manifold_pipeline(data_dir, mouse_M2024, save_dir, **params);
plt.close('all')
# %% M2025
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\2025' 
mouse_M2025 = 'M2025'
results_dir_M2025, M2025_dict = LT_manifold_pipeline(data_dir, mouse_M2025, save_dir, **params);
plt.close('all')
# %% M206
data_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\2026' 
mouse_M2026 = 'M2026'
results_dir_M2026, M2026_dict = LT_manifold_pipeline(data_dir, mouse_M2026, save_dir, **params);
plt.close('all')

#%%
###############################################################################
#                              STRUCTURE INDEX                                #
###############################################################################
#%% LOAD ALL DATA IF NEEDED (check path then)
if save_dir not in locals():
    verbose = True
    save_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\results_pipeline\\move_data'
    
    mouse_M2019 = 'M2019'
    results_dir_M2019 = os.path.join(save_dir, "M2019_170122_100510")
    M2019_dict = gu.load_files(results_dir_M2019, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    
    mouse_M2021 = 'M2021'
    results_dir_M2021 = os.path.join(save_dir, "M2021_170122_131036")
    M2021_dict = gu.load_files(results_dir_M2021, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    
    mouse_M2022 = 'M2022'
    results_dir_M2022 = os.path.join(save_dir, "M2022_170122_132357")
    M2022_dict = gu.load_files(results_dir_M2022, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
        
    mouse_M2023 = 'M2023'
    results_dir_M2023 = os.path.join(save_dir, "M2023_170122_134606")
    M2023_dict = gu.load_files(results_dir_M2023, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
        
    mouse_M2024 = 'M2024'
    results_dir_M2024 = os.path.join(save_dir, "M2024_170122_141956")
    M2024_dict = gu.load_files(results_dir_M2024, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
        
    mouse_M2025 = 'M2025'
    results_dir_M2025 = os.path.join(save_dir, "M2025_170122_143435")
    M2025_dict = gu.load_files(results_dir_M2025, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
        
    mouse_M2026 = 'M2026'
    results_dir_M2026 = os.path.join(save_dir, "M2026_170122_150559")
    M2026_dict = gu.load_files(results_dir_M2026, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    
# %% M2019
    #1. Pairwise Comparison
cI_M2019_dict = ltcI.compute_cI(results_dir_M2019, mouse_M2019, ["ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap',
                              comp_method ='pairwise', nBins = 20)
    #2. All Comparison
cI_M2019_dict = ltcI.compute_cI(results_dir_M2019, mouse_M2019, ["ML_rates","ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap', comp_method ='all',
                              load_old_dict = True, nBins = 20)

    #3. Triplet Comparison (find best dimensions to plot)
cI_M2019_dict = ltcI.compute_cI(results_dir_M2019, mouse_M2019, ["ML_umap"], ["posx","posy", "index_mat"], 
                              nRep = 1, n_dims = 'adapt', comp_method ='triplets',
                              load_old_dict = True)
# %% M2021
    #1. Pairwise Comparison
cI_M2021_dict = ltcI.compute_cI(results_dir_M2021, mouse_M2021, ["ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap',
                              comp_method ='pairwise', nBins = 20)
    #2. All Comparison
cI_M2021_dict = ltcI.compute_cI(results_dir_M2021, mouse_M2021, ["ML_rates","ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap', comp_method ='all',
                              load_old_dict = True, nBins = 20)

    #3. Triplet Comparison (find best dimensions to plot)
cI_M2021_dict = ltcI.compute_cI(results_dir_M2021, mouse_M2021, ["ML_umap"], ["posx","posy", "index_mat"], 
                              nRep = 1, n_dims = 'adapt', comp_method ='triplets',
                              load_old_dict = True)
# %% M2022
    #1. Pairwise Comparison
cI_M2022_dict = ltcI.compute_cI(results_dir_M2022, mouse_M2022, ["ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap',
                              comp_method ='pairwise', nBins = 20)
    #2. All Comparison
cI_M2022_dict = ltcI.compute_cI(results_dir_M2022, mouse_M2022, ["ML_rates","ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap', comp_method ='all',
                              load_old_dict = True, nBins = 20)

    #3. Triplet Comparison (find best dimensions to plot)
cI_M2022_dict = ltcI.compute_cI(results_dir_M2022, mouse_M2022, ["ML_umap"], ["posx","posy", "index_mat"], 
                              nRep = 1, n_dims = 'adapt', comp_method ='triplets',
                              load_old_dict = True)

# %% M2023
    #1. Pairwise Comparison
cI_M2023_dict = ltcI.compute_cI(results_dir_M2023, mouse_M2023, ["ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap',
                              comp_method ='pairwise', nBins = 20)
    #2. All Comparison
cI_M2023_dict = ltcI.compute_cI(results_dir_M2023, mouse_M2023, ["ML_rates","ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap', comp_method ='all',
                              load_old_dict = True, nBins = 20)

    #3. Triplet Comparison (find best dimensions to plot)
cI_M2023_dict = ltcI.compute_cI(results_dir_M2023, mouse_M2023, ["ML_umap"], ["posx","posy", "index_mat"], 
                              nRep = 1, n_dims = 'adapt', comp_method ='triplets',
                              load_old_dict = True)
# %% M2024
    #1. Pairwise Comparison
cI_M2024_dict = ltcI.compute_cI(results_dir_M2024, mouse_M2024, ["ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap',
                              comp_method ='pairwise', nBins = 20)
    #2. All Comparison
cI_M2024_dict = ltcI.compute_cI(results_dir_M2024, mouse_M2024, ["ML_rates","ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap', comp_method ='all',
                              load_old_dict = True, nBins = 20)

    #3. Triplet Comparison (find best dimensions to plot)
cI_M2024_dict = ltcI.compute_cI(results_dir_M2024, mouse_M2024, ["ML_umap"], ["posx","posy", "index_mat"], 
                              nRep = 1, n_dims = 'adapt', comp_method ='triplets',
                              load_old_dict = True)
# %% M2025
    #1. Pairwise Comparison
cI_M2025_dict = ltcI.compute_cI(results_dir_M2025, mouse_M2025, ["ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap',
                              comp_method ='pairwise', nBins = 20)
    #2. All Comparison
cI_M2025_dict = ltcI.compute_cI(results_dir_M2025, mouse_M2025, ["ML_rates","ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap', comp_method ='all',
                              load_old_dict = True, nBins = 20)

    #3. Triplet Comparison (find best dimensions to plot)
cI_M2025_dict = ltcI.compute_cI(results_dir_M2025, mouse_M2025, ["ML_umap"], ["posx","posy", "index_mat"], 
                              nRep = 1, n_dims = 'adapt', comp_method ='triplets',
                              load_old_dict = True)
# %% M2026
    #1. Pairwise Comparison
cI_M2026_dict = ltcI.compute_cI(results_dir_M2026, mouse_M2026, ["ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap',
                              comp_method ='pairwise', nBins = 20)
    #2. All Comparison
cI_M2026_dict = ltcI.compute_cI(results_dir_M2026, mouse_M2026, ["ML_rates","ML_pca", "ML_iso", "ML_umap"], 
                              ["posx","posy","index_mat"],nRep = 1,n_dims = 'adapt_to_umap', comp_method ='all',
                              load_old_dict = True, nBins = 20)

    #3. Triplet Comparison (find best dimensions to plot)
cI_M2026_dict = ltcI.compute_cI(results_dir_M2026, mouse_M2026, ["ML_umap"], ["posx","posy", "index_mat"], 
                              nRep = 1, n_dims = 'adapt', comp_method ='triplets',
                              load_old_dict = True)

#%%
###############################################################################
#                             PLOT STRUCTURE INDEX                            #
###############################################################################
def get_max_cI(cI_dict, skip = np.array([])):
    max_cI = np.zeros((4,4,3))*np.nan
    file_idx = -1
    new_dict = copy.deepcopy(cI_dict)
    for file, it in new_dict.items():
        if not np.any(skip==it):
            if '_LT_' not in file or '_LT_1' in file:
                file_div = 1
                file_idx +=1
            else:
                file_div +=1
            field_idx = -1
            for field, it2 in it.items():
                field_idx +=1
                if '_LT_' not in file or '_LT_1' in file:
                    if 'rates' not in field:
                        max_cI[file_idx, field_idx,:] = it2["cI_all"]
                    else:
                        max_cI[file_idx, field_idx,:] = it2["cI_all"]
                else:
                    if 'rates' not in field:
                        max_cI[file_idx, field_idx,:] += (1/file_div)* (it2["cI_all"]-max_cI[file_idx, field_idx,:]) #online average
                    else:
                        max_cI[file_idx, field_idx,:] += (1/file_div)* (it2["cI_all"]-max_cI[file_idx, field_idx,:]) #online average
    return max_cI

M2019_max_cI = get_max_cI(cI_M2019_dict)
M2021_max_cI = get_max_cI(cI_M2021_dict)
M2022_max_cI = get_max_cI(cI_M2022_dict)
M2023_max_cI = get_max_cI(cI_M2023_dict, skip = np.array([1]))
M2024_max_cI = get_max_cI(cI_M2024_dict, skip = np.array([1,2]))
M2025_max_cI = get_max_cI(cI_M2025_dict, skip = np.array([1]))
M2026_max_cI = get_max_cI(cI_M2026_dict, skip = np.array([1]))

fig, ax = plt.subplots(1,3,figsize=(10,5))
fields = ['', '', 'Umap', 'Rates']
varList = ['x-pos', 'y-pos', 'index_mat']
x_dims = 4
x_space = np.linspace(1, x_dims, x_dims).astype(int)
color_code = [ 'C4', 'C3', 'C0', 'gray']
for var_idx, var_label in enumerate(varList):
    for field_idx, field in enumerate(fields):
        if field:
            m = np.nanmean(np.vstack((M2019_max_cI[:,field_idx,var_idx],
                                   M2021_max_cI[:,field_idx,var_idx],
                                   M2022_max_cI[:,field_idx,var_idx],
                                   M2023_max_cI[:,field_idx,var_idx],
                                   M2024_max_cI[:,field_idx,var_idx],
                                   M2025_max_cI[:,field_idx,var_idx],
                                   M2026_max_cI[:,field_idx,var_idx])), axis=0)
            sd = np.nanstd(np.vstack((M2019_max_cI[:,field_idx,var_idx],
                                   M2021_max_cI[:,field_idx,var_idx],
                                   M2022_max_cI[:,field_idx,var_idx],
                                   M2023_max_cI[:,field_idx,var_idx],
                                   M2024_max_cI[:,field_idx,var_idx],
                                   M2025_max_cI[:,field_idx,var_idx],
                                   M2026_max_cI[:,field_idx,var_idx])), axis=0)
            
            ax[var_idx].plot(x_space, m, color = color_code[field_idx], label = field)
            ax[var_idx].fill_between(x_space, m-sd, m+sd, color = color_code[field_idx], alpha=0.25)
        
    ax[var_idx].set_ylabel('Structure Index',fontsize=14)
    ax[var_idx].set_ylim([0,1])
    ax[var_idx].set_xlim([x_space[0]-0.2, x_space[-1]+0.2])
    plt.setp(ax[var_idx].spines.values(), linewidth=2)
    ax[var_idx].spines['right'].set_visible(False)
    ax[var_idx].spines['top'].set_visible(False)        
    ax[var_idx].legend(fontsize=14)
    ax[var_idx].set_title(var_label,fontsize=16)
    ax[var_idx].set_xticks(x_space)
    ax[var_idx].set_xticklabels(["Day1-evening", "Day2-morning", "Day2-evening", "Day4-evening"],rotation = 45, ha="right")
fig.tight_layout()
plt.show()
#%%
###############################################################################
#                             decoder                            #
###############################################################################
R2s_M2019_dict = ltd.decoders_LT_Jercog(M2019_dict, x_base = "ML_rates", x_emb=["ML_pca", "ML_iso", "ML_umap"], 
                                        y_labels = ["posx", "posy","index_mat"], n_dims = 3, n_splits=10, 
                                        wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2019, mouse_M2019+ "_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2019_dict, save_file)
save_file.close()        

R2s_M2021_dict = ltd.decoders_LT_Jercog(M2021_dict, x_base = "ML_rates", x_emb=["ML_pca", "ML_iso", "ML_umap"], 
                                        y_labels = ["posx", "posy","index_mat"], n_dims = 3, n_splits=10, 
                                        wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2021, mouse_M2021+ "_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2021_dict, save_file)
save_file.close()        

R2s_M2022_dict = ltd.decoders_LT_Jercog(M2022_dict, x_base = "ML_rates", x_emb=["ML_pca", "ML_iso", "ML_umap"], 
                                        y_labels = ["posx", "posy","index_mat"], n_dims = 3, n_splits=10, 
                                        wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2022, mouse_M2022+ "_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2022_dict, save_file)
save_file.close()  

R2s_M2023_dict = ltd.decoders_LT_Jercog(M2023_dict, x_base = "ML_rates", x_emb=["ML_pca", "ML_iso", "ML_umap"], 
                                        y_labels = ["posx", "posy","index_mat"], n_dims = 3, n_splits=10, 
                                        wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2023, mouse_M2023+ "_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2023_dict, save_file)
save_file.close()  

R2s_M2024_dict = ltd.decoders_LT_Jercog(M2024_dict, x_base = "ML_rates", x_emb=["ML_pca", "ML_iso", "ML_umap"], 
                                        y_labels = ["posx", "posy","index_mat"], n_dims = 3, n_splits=10, 
                                        wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2024, mouse_M2024+ "_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2024_dict, save_file)
save_file.close()  

R2s_M2025_dict = ltd.decoders_LT_Jercog(M2025_dict, x_base = "ML_rates", x_emb=["ML_pca", "ML_iso", "ML_umap"], 
                                        y_labels = ["posx", "posy","index_mat"], n_dims = 3, n_splits=10, 
                                        wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2025, mouse_M2025+ "_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2025_dict, save_file)
save_file.close()  

R2s_M2026_dict = ltd.decoders_LT_Jercog(M2026_dict, x_base = "ML_rates", x_emb=["ML_pca", "ML_iso", "ML_umap"], 
                                        y_labels = ["posx", "posy","index_mat"], n_dims = 3, n_splits=10, 
                                        wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2026, mouse_M2026+ "_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2026_dict, save_file)
save_file.close()  

def get_max_R2s(cI_dict, skip = np.array([])):
    max_R2s = np.zeros((4,4,4,3,2))
    file_idx = -1
    
    new_dict = copy.deepcopy(cI_dict)
        
    for file, it in new_dict.items():
        if not np.any(skip ==it):
            if '_LT_' not in file or '_LT_1' in file:
                file_div = 1
                file_idx +=1
            else:
                file_div+=1
            field_idx = -1
            for field, it2 in it.items():
                field_idx +=1
                decoder_idx = -1
                for decoder, it3 in it2.items():
                    decoder_idx +=1
                    it3[it3>1e2] = np.nan
                    if '_LT_' not in file or '_LT_1' in file:
                        max_R2s[file_idx, field_idx,decoder_idx,:,:] = np.nanmean(it3, axis=0)
                    else:
                        max_R2s[file_idx, field_idx,decoder_idx,:,:] += (1/int(file[-1]))* (np.nanmean(it3, axis=0)-max_R2s[file_idx, field_idx,decoder_idx,:]) #online average
    return max_R2s

M2019_max_R2s = get_max_R2s(R2s_M2019_dict)
M2021_max_R2s = get_max_R2s(R2s_M2021_dict)
M2022_max_R2s = get_max_R2s(R2s_M2022_dict)
M2023_max_R2s = get_max_R2s(R2s_M2023_dict, skip = np.array([1]))
M2024_max_R2s = get_max_R2s(R2s_M2024_dict, skip = np.array([1,2]))
M2025_max_R2s = get_max_R2s(R2s_M2025_dict, skip = np.array([1]))
M2026_max_R2s = get_max_R2s(R2s_M2026_dict, skip = np.array([1]))


fig, ax = plt.subplots(2,4,figsize=(10,5))
fields = ['Rates','PCA', 'Isomap', 'Umap']
decList = ['wf', 'wc', 'xgb','svr']
x_dims = 4
x_space = np.linspace(1, x_dims, x_dims).astype(int)
color_code = [ 'gray', 'C4', 'C3','C0']
pred = 1
for train_test in range(2):
    for dec_idx, dec_label in enumerate(decList):
        for field_idx, field in enumerate(fields):
            if field:
                m = np.nanmean(np.vstack((M2019_max_R2s[:,field_idx,dec_idx, pred, train_test],
                                       M2021_max_R2s[:,field_idx,dec_idx, pred, train_test],
                                       M2023_max_R2s[:,field_idx,dec_idx, pred, train_test],
                                       M2024_max_R2s[:,field_idx,dec_idx, pred,train_test],
                                       M2025_max_R2s[:,field_idx,dec_idx, pred,train_test],
                                       M2026_max_R2s[:,field_idx,dec_idx, pred,train_test])), axis=0)
                sd = np.nanstd(np.vstack((M2019_max_R2s[:,field_idx,dec_idx, pred,train_test],
                                       M2021_max_R2s[:,field_idx,dec_idx, pred,train_test],
                                       M2023_max_R2s[:,field_idx,dec_idx, pred,train_test],
                                       M2024_max_R2s[:,field_idx,dec_idx, pred,train_test],
                                       M2025_max_R2s[:,field_idx,dec_idx, pred,train_test],
                                       M2026_max_R2s[:,field_idx,dec_idx, pred,train_test])), axis=0)/np.sqrt(6)
                
                ax[train_test, dec_idx].plot(x_space, m, color = color_code[field_idx], label = field)
                ax[train_test, dec_idx].fill_between(x_space, m-sd, m+sd, color = color_code[field_idx], alpha=0.25)
            
        ax[train_test, dec_idx].set_ylabel('Error [%trials]',fontsize=14)
        #ax[train_test, dec_idx].set_ylim([0,1])
        ax[train_test, dec_idx].set_xlim([x_space[0]-0.2, x_space[-1]+0.2])
        plt.setp(ax[train_test,dec_idx].spines.values(), linewidth=2)
        ax[train_test, dec_idx].spines['right'].set_visible(False)
        ax[train_test, dec_idx].spines['top'].set_visible(False)        
        ax[train_test, dec_idx].set_title(dec_label,fontsize=16)
        ax[train_test, dec_idx].set_xticks(x_space)
        ax[train_test, dec_idx].set_xticklabels(["Day1-evening", "Day2-morning", "Day2-evening", "Day4-evening"],rotation = 45, ha="right")

ax[-1, -1].legend(fontsize=14)
fig.tight_layout()
plt.show()

#%% CROSS SESSION ALIGNMENT DECODERS
import timeit
from datetime import datetime

save_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\results_pipeline\\move_data'
verbose = True

if load_data:
    save_dir = 'E:\\Users\\Usuario\\Documents\\Spatial_navigation_project\\Jercog_data\\LT\\results_pipeline\\move_data'
    verbose = True

    mouse_M2019 = 'M2019'
    results_dir_M2019 = os.path.join(save_dir, "M2019_170122_100510")
    M2019_dict = gu.load_files(results_dir_M2019, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    R2s_M2019_alignment_dict = gu.load_files(results_dir_M2019, '*_alignment_decoder_dict.pkl', verbose=verbose, struct_type = "pickle")

    mouse_M2021 = 'M2021'
    results_dir_M2021 = os.path.join(save_dir, "M2021_170122_131036")
    M2021_dict = gu.load_files(results_dir_M2021, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    R2s_M2021_alignment_dict = gu.load_files(results_dir_M2021, '*_alignment_decoder_dict.pkl', verbose=verbose, struct_type = "pickle")

    mouse_M2023 = 'M2023'
    results_dir_M2023 = os.path.join(save_dir, "M2023_170122_134606")
    M2023_dict = gu.load_files(results_dir_M2023, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    R2s_M2023_alignment_dict = gu.load_files(results_dir_M2023, '*_alignment_decoder_dict.pkl', verbose=verbose, struct_type = "pickle")

    mouse_M2024 = 'M2024'
    results_dir_M2024 = os.path.join(save_dir, "M2024_170122_141956")
    M2024_dict = gu.load_files(results_dir_M2024, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    R2s_M2024_alignment_dict = gu.load_files(results_dir_M2024, '*_alignment_decoder_dict.pkl', verbose=verbose, struct_type = "pickle")

    mouse_M2025 = 'M2025'
    results_dir_M2025 = os.path.join(save_dir, "M2025_170122_143435")
    M2025_dict = gu.load_files(results_dir_M2025, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    R2s_M2025_alignment_dict = gu.load_files(results_dir_M2025, '*_alignment_decoder_dict.pkl', verbose=verbose, struct_type = "pickle")

    mouse_M2026 = 'M2026'
    results_dir_M2026 = os.path.join(save_dir, "M2026_170122_150559")
    M2026_dict = gu.load_files(results_dir_M2026, '*_move_data_dict.pkl', verbose=verbose, struct_type = "pickle")
    R2s_M2026_alignment_dict = gu.load_files(results_dir_M2026, '*_alignment_decoder_dict.pkl', verbose=verbose, struct_type = "pickle")
elif initialize:
    R2s_M2019_alignment_dict = dict()
    R2s_M2021_alignment_dict = dict()
    R2s_M2023_alignment_dict = dict()
    R2s_M2024_alignment_dict = dict()
    R2s_M2025_alignment_dict = dict()
    R2s_M2026_alignment_dict = dict()


local_starttime = timeit.default_timer()
R2s_M2019_alignment_dict["M2019"] = ltd.cross_session_decoders_LT(copy.deepcopy(M2019_dict),copy.deepcopy(M2019_dict), x_base = "ML_rates", x_emb="ML_umap", y_signal = "pos", n_dims = 3, n_splits=10, wf = True, wc = True, xgb = True, svr = True, verbose = True)
print('\tTime: %.4f' %(timeit.default_timer()-local_starttime), 
      ' (', datetime.now().strftime("%H:%M:%S"),')', sep='')

save_file = open(os.path.join(results_dir_M2019, mouse_M2019+ "_alignment_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2019_alignment_dict, save_file)
save_file.close()  


local_starttime = timeit.default_timer()
R2s_M2021_alignment_dict["M2021"] = ltd.cross_session_decoders_LT(copy.deepcopy(M2021_dict),copy.deepcopy(M2021_dict), x_base = "ML_rates", x_emb="ML_umap", y_signal = "pos", n_dims = 3, n_splits=10, wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2021, mouse_M2021+ "_alignment_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2021_alignment_dict, save_file)
save_file.close()  
print('\tTime: %.4f' %(timeit.default_timer()-local_starttime), 
      ' (', datetime.now().strftime("%H:%M:%S"),')', sep='')


local_starttime = timeit.default_timer()
R2s_M2023_alignment_dict["M2023"] = ltd.cross_session_decoders_LT(copy.deepcopy(M2023_dict),copy.deepcopy(M2023_dict), x_base = "ML_rates", x_emb="ML_umap", y_signal = "pos", n_dims = 3, n_splits=10, wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2023, mouse_M2023+ "_alignment_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2023_alignment_dict, save_file)
save_file.close()  
print('\tTime: %.4f' %(timeit.default_timer()-local_starttime), 
      ' (', datetime.now().strftime("%H:%M:%S"),')', sep='')


local_starttime = timeit.default_timer()
R2s_M2024_alignment_dict["M2024"] = ltd.cross_session_decoders_LT(copy.deepcopy(M2024_dict),copy.deepcopy(M2024_dict), x_base = "ML_rates", x_emb="ML_umap", y_signal = "pos", n_dims = 3, n_splits=10, wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2024, mouse_M2024+ "_alignment_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2024_alignment_dict, save_file)
save_file.close()  
print('\tTime: %.4f' %(timeit.default_timer()-local_starttime), 
      ' (', datetime.now().strftime("%H:%M:%S"),')', sep='')


local_starttime = timeit.default_timer()
R2s_M2025_alignment_dict["M2025"] = ltd.cross_session_decoders_LT(copy.deepcopy(M2025_dict),copy.deepcopy(M2025_dict), x_base = "ML_rates", x_emb="ML_umap", y_signal = "pos", n_dims = 3, n_splits=10, wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2025, mouse_M2025+ "_alignment_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2025_alignment_dict, save_file)
save_file.close()  
print('\tTime: %.4f' %(timeit.default_timer()-local_starttime), 
      ' (', datetime.now().strftime("%H:%M:%S"),')', sep='')

local_starttime = timeit.default_timer()
R2s_M2026_alignment_dict["M2026"] = ltd.cross_session_decoders_LT(copy.deepcopy(M2026_dict),copy.deepcopy(M2026_dict), x_base = "ML_rates", x_emb="ML_umap", y_signal = "pos", n_dims = 3, n_splits=10, wf = True, wc = True, xgb = True, svr = True, verbose = True)
save_file = open(os.path.join(results_dir_M2026, mouse_M2026+ "_alignment_decoder_dict.pkl"), "wb")
pickle.dump(R2s_M2026_alignment_dict, save_file)
save_file.close()  
print('\tTime: %.4f' %(timeit.default_timer()-local_starttime), 
      ' (', datetime.now().strftime("%H:%M:%S"),')', sep='')

