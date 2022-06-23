#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:26:06 2022

@author: julio
"""

#%% IMPORTS
import numpy as np
from neural_manifold import general_utils as gu
from neural_manifold import dimensionality_reduction as dim_red
import pickle, os, copy

import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO

import seaborn as sns
import pandas as pd

#%% PLOT UMAP DIM
def plot_umap_dim_study(dim_dict, save_dir):
    cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]

    fnames = list(dim_dict.keys())
    
    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + f"<h1>Umap nn study - {fnames[0]}</h1>\n<br>\n"    #Add title
    html = html + f"<h2>signal: {dim_dict[fnames[0]]['params']['base_name']} - "
    html = html + f"<br>{datetime.now().strftime('%d/%m/%y %H:%M:%S')}</h2><br>\n"    #Add subtitle
    
    
    sI_vmin = np.inf
    sI_vmax = 0

    trust_vmin = np.inf
    trust_vmax = 0
    
    cont_vmin = np.inf
    cont_vmax = 0
    
    R2s_vmin = 0
    R2s_vmax = np.zeros((4,1))
    
    for file_idx, file_name in enumerate(fnames):
        sI_vmin = np.nanmin([sI_vmin, np.min(dim_dict[file_name]['sI'], axis= (0,1))])
        sI_vmax = np.nanmax([sI_vmax, np.max(dim_dict[file_name]['sI'], axis= (0,1))])
        
        trust_vmin = np.nanmin([trust_vmin, np.min(dim_dict[file_name]['trust'], axis= (0,1))])
        trust_vmax = np.nanmax([trust_vmax, np.max(dim_dict[file_name]['trust'], axis= (0,1))])
        
        cont_vmin = np.nanmin([cont_vmin, np.min(dim_dict[file_name]['cont'], axis= (0,1))])
        cont_vmax = np.nanmax([cont_vmax, np.max(dim_dict[file_name]['cont'], axis= (0,1))])
        
        temp_R2s_vmax = np.nanmax(np.mean(dim_dict[file_name]['R2s'], axis=2), axis= (0,1)).reshape(-1,1)
        temp_R2s_vmax[temp_R2s_vmax>25] = 25
        R2s_vmax = np.nanmax(np.concatenate((R2s_vmax, temp_R2s_vmax),axis=1), axis=1).reshape(-1,1)
    
    dec_list = ['wf','wc','xgb','svm']
    
    for file_idx, file_name in enumerate(fnames):

        fig= plt.figure(figsize = (16, 4))
        ytick_labels = [str(entry) for entry in dim_dict[file_name]['params']['nn_list']]
        xtick_labels =np.arange(dim_dict[file_name]['params']['max_dim'])
        fig.text(0.008, 0.5, f"{file_name}",horizontalalignment='center', 
                         rotation = 'vertical', verticalalignment='center', fontsize = 20)
        
        ax = plt.subplot(1,4,1)
        b = ax.imshow(dim_dict[file_name]['sI'][:,:,0].T, vmin = sI_vmin, vmax = sI_vmax, aspect = 'auto')
        ax.set_title(f"sI: {dim_dict[file_name]['params']['label_name'][0]}",fontsize=15)
        ax.set_ylabel('sI nn', labelpad = 5)
        ax.set_xlabel('dim', labelpad = -5)
        ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)
        ax.set_xticks(xtick_labels, labels=xtick_labels+1)
        fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=1)
        
        ax = plt.subplot(1,4,2)
        b = ax.imshow(dim_dict[file_name]['trust'][:,:].T, vmin = trust_vmin, vmax = trust_vmax, aspect = 'auto')
        ax.set_title("Trustworthiness",fontsize=15)
        ax.set_ylabel('nn', labelpad = 2)
        ax.set_xlabel('dim', labelpad = 0)
        ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)
        ax.set_xticks(xtick_labels, labels=xtick_labels+1)
        fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=1)
        
        ax = plt.subplot(1,4,3)
        b = ax.imshow(dim_dict[file_name]['cont'][:,:].T, vmin = cont_vmin, vmax = cont_vmax, aspect = 'auto')
        ax.set_title("Continuity",fontsize=15)
        ax.set_ylabel('nn', labelpad = 2)
        ax.set_xlabel('dim', labelpad = 0)
        ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)
        ax.set_xticks(xtick_labels, labels=xtick_labels+1)
        fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=1)


        for ii in range(4):
            if ii == 0:
                pos = 7
            elif ii == 1:
                pos = 8
            elif ii == 2:
                pos = 15
            elif ii == 3:
                pos = 16
    
            ax = plt.subplot(2,8,pos)
            for l_idx, ln in enumerate(dim_dict[file_name]['params']['label_name']):
                m = np.nanmean(dim_dict[file_name]['R2s'][:,:,l_idx,ii], axis=1)
                sd = np.nanstd(dim_dict[file_name]['R2s'][:,:,l_idx,ii], axis=1)
                
                ax.plot(m, c = cpal[2], label = ln)
                ax.fill_between(np.arange(len(m)), m-sd, m+sd, color = cpal[2], alpha = 0.3)
                
        
            ax.set_xlabel('nn', labelpad = -2)
            ax.set_xticks(xtick_labels, labels=xtick_labels+1, rotation = 90)
            ax.set_ylim([R2s_vmin, R2s_vmax[ii,0]])
            ax.set_yticks([R2s_vmin, R2s_vmax[0,0]/2, R2s_vmax[0,0]])
            ax.set_ylabel('R2s error', labelpad = 0)
            ax.set_title(dec_list[ii])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                

        plt.tight_layout()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
        plt.close(fig)
            
    with open(os.path.join(save_dir, f"{fnames[0][:5]}_umap_dim_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)
    
    return True
#%% GENERAL PARAMS
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/umap_params_study/dim'
params = {
    'nn_list': [3, 10, 20, 30, 60, 120, 200, 500, 1000],
    'max_dim': 12,
    'verbose': True,
    'n_splits': 5,
    'nn': 60
    }

signal_name = 'ML_rates'
label_names = ['posx']

#%% M2019
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2019' in f]
M2019 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2019_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2019.keys())
M2019_umap_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2019[fname])
    #compute hyperparameter study
    m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                     base_signal = signal_name, label_signal = label_names,
                                                     trial_signal = "index_mat", **params)
    m_params['base_name'] = signal_name
    m_params['label_name'] = label_names
    #save results
    M2019_umap_dim[fname] = {
        'trust': m_trust,
        'cont': m_cont,
        'sI': m_sI,
        'R2s': m_R2s,
        'params': m_params
        }
    
    save_ks = open(os.path.join(save_dir, "M2019_umap_dim_dict.pkl"), "wb")
    pickle.dump(M2019_umap_dim, save_ks)
    save_ks.close()
    
_ = plot_umap_dim_study(M2019_umap_dim, save_dir)

#%% M2021
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2021' in f]
M2021 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2021_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2021.keys())
M2021_umap_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2021[fname])
    #compute hyperparameter study
    m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                     base_signal = signal_name, label_signal = label_names,
                                                     trial_signal = "index_mat", **params)
    m_params['base_name'] = signal_name
    m_params['label_name'] = label_names
    #save results
    M2021_umap_dim[fname] = {
        'trust': m_trust,
        'cont': m_cont,
        'sI': m_sI,
        'R2s': m_R2s,
        'params': m_params
        }
    
    save_ks = open(os.path.join(save_dir, "M2021_umap_dim_dict.pkl"), "wb")
    pickle.dump(M2021_umap_dim, save_ks)
    save_ks.close()
    
_ = plot_umap_dim_study(M2021_umap_dim, save_dir)

#%% M2022
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2022' in f]
M2022 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2022_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2022.keys())
M2022_umap_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2022[fname])
    #compute hyperparameter study
    m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                     base_signal = signal_name, label_signal = label_names,
                                                     trial_signal = "index_mat", **params)
    m_params['base_name'] = signal_name
    m_params['label_name'] = label_names
    #save results
    M2022_umap_dim[fname] = {
        'trust': m_trust,
        'cont': m_cont,
        'sI': m_sI,
        'R2s': m_R2s,
        'params': m_params
        }
    
    save_ks = open(os.path.join(save_dir, "M2022_umap_dim_dict.pkl"), "wb")
    pickle.dump(M2022_umap_dim, save_ks)
    save_ks.close()
    
_ = plot_umap_dim_study(M2022_umap_dim, save_dir)

#%% M2023
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2023' in f]
M2023 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2023_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2023.keys())
M2023_umap_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2023[fname])
    #compute hyperparameter study
    m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                     base_signal = signal_name, label_signal = label_names,
                                                     trial_signal = "index_mat", **params)
    m_params['base_name'] = signal_name
    m_params['label_name'] = label_names
    #save results
    M2023_umap_dim[fname] = {
        'trust': m_trust,
        'cont': m_cont,
        'sI': m_sI,
        'R2s': m_R2s,
        'params': m_params
        }
    
    save_ks = open(os.path.join(save_dir, "M2023_umap_dim_dict.pkl"), "wb")
    pickle.dump(M2023_umap_dim, save_ks)
    save_ks.close()
    
_ = plot_umap_dim_study(M2023_umap_dim, save_dir)

#%% M2024
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2024' in f]
M2024 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2024_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2024.keys())
M2024_umap_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2024[fname])
    #compute hyperparameter study
    m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                     base_signal = signal_name, label_signal = label_names,
                                                     trial_signal = "index_mat", **params)
    m_params['base_name'] = signal_name
    m_params['label_name'] = label_names
    #save results
    M2024_umap_dim[fname] = {
        'trust': m_trust,
        'cont': m_cont,
        'sI': m_sI,
        'R2s': m_R2s,
        'params': m_params
        }
    
    save_ks = open(os.path.join(save_dir, "M2024_umap_dim_dict.pkl"), "wb")
    pickle.dump(M2024_umap_dim, save_ks)
    save_ks.close()
    
_ = plot_umap_dim_study(M2024_umap_dim, save_dir)

#%% M2025
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2025' in f]
M2025 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2025_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2025.keys())
M2025_umap_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2025[fname])
    #compute hyperparameter study
    m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                     base_signal = signal_name, label_signal = label_names,
                                                     trial_signal = "index_mat", **params)
    m_params['base_name'] = signal_name
    m_params['label_name'] = label_names
    #save results
    M2025_umap_dim[fname] = {
        'trust': m_trust,
        'cont': m_cont,
        'sI': m_sI,
        'R2s': m_R2s,
        'params': m_params
        }
    
    save_ks = open(os.path.join(save_dir, "M2025_umap_dim_dict.pkl"), "wb")
    pickle.dump(M2025_umap_dim, save_ks)
    save_ks.close()
    
_ = plot_umap_dim_study(M2025_umap_dim, save_dir)

#%% M2026
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2026' in f]
M2026 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2026_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2026.keys())
M2026_umap_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2026[fname])
    #compute hyperparameter study
    m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                     base_signal = signal_name, label_signal = label_names,
                                                     trial_signal = "index_mat", **params)
    m_params['base_name'] = signal_name
    m_params['label_name'] = label_names
    #save results
    M2026_umap_dim[fname] = {
        'trust': m_trust,
        'cont': m_cont,
        'sI': m_sI,
        'R2s': m_R2s,
        'params': m_params
        }
    
    save_ks = open(os.path.join(save_dir, "M2026_umap_dim_dict.pkl"), "wb")
    pickle.dump(M2026_umap_dim, save_ks)
    save_ks.close()
    
_ = plot_umap_dim_study(M2026_umap_dim, save_dir)

#%% LOAD DATA
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/umap_params_study/dim'
if "M2019_umap_dim" not in locals():
    M2019_umap_dim = gu.load_files(save_dir, '*M2019_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2021_umap_dim" not in locals():
    M2021_umap_dim = gu.load_files(save_dir, '*M2021_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2022_umap_dim" not in locals():
    M2022_umap_dim = gu.load_files(save_dir, '*M2022_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2023_umap_dim" not in locals():
    M2023_umap_dim = gu.load_files(save_dir, '*M2023_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2024_umap_dim" not in locals():
    M2024_umap_dim = gu.load_files(save_dir, '*M2024_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2025_umap_dim" not in locals():
    M2025_umap_dim = gu.load_files(save_dir, '*M2025_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2026_umap_dim" not in locals():
    M2026_umap_dim = gu.load_files(save_dir, '*M2026_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")