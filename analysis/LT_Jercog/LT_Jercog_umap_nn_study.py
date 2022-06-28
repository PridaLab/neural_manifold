#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 19:37:56 2022

@author: julio
"""

#%% IMPORTS
import numpy as np
from neural_manifold import general_utils as gu
from neural_manifold import dimensionality_reduction as dim_red
import pickle, os, copy, sys

import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO

import seaborn as sns
import pandas as pd
#%% PLOT UMAP NN
def plot_umap_nn_study(nn_dict, save_dir):
    cpal2 = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]

    fnames = list(nn_dict.keys())
    
    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + f"<h1>Umap nn study - {fnames[0]}</h1>\n<br>\n"    #Add title
    html = html + f"<h2>signal: {nn_dict[fnames[0]]['params']['base_name']} - "
    html = html + f"<br>{datetime.now().strftime('%d/%m/%y %H:%M:%S')}</h2><br>\n"    #Add subtitle
    
    
    sI_vmin_og = np.inf
    sI_vmax_og = 0
    sI_vmin_emb = np.inf
    sI_vmax_emb = 0
    
    max_dim = 0
    R2s_vmin = 0
    R2s_vmax = np.zeros((4,1))
    for file_idx, file_name in enumerate(fnames):
        sI_vmin_og = np.nanmin([sI_vmin_og, np.min(nn_dict[file_name]['sI_og'], axis= (0,1))])
        sI_vmax_og = np.nanmax([sI_vmax_og, np.max(nn_dict[file_name]['sI_og'], axis= (0,1))])
        
        sI_vmin_emb = np.nanmin([sI_vmin_emb, np.min(nn_dict[file_name]['sI_emb'], axis= (0,1,2))])
        sI_vmax_emb = np.nanmax([sI_vmax_emb, np.max(nn_dict[file_name]['sI_emb'], axis= (0,1,2))])
        
        temp_R2s_vmax = np.nanmax(np.mean(nn_dict[file_name]['R2s'], axis=2), axis= (0,1)).reshape(-1,1)
        temp_R2s_vmax[temp_R2s_vmax>25] = 25
        R2s_vmax = np.nanmax(np.concatenate((R2s_vmax, temp_R2s_vmax),axis=1), axis=1).reshape(-1,1)
        
        max_dim = np.nanmax([max_dim, np.max(nn_dict[file_name]['trust_dim']), np.max(nn_dict[file_name]['cont_dim'])])
    
    dec_list = ['wf','wc','xgb','svm']
    
    for file_idx, file_name in enumerate(fnames):
        
        fig= plt.figure(figsize = (15, 4))
        ytick_labels = [str(entry) for entry in nn_dict[file_name]['params']['nn_list']]
        xtick_labels = [str(entry) for entry in nn_dict[file_name]['params']['nn_list']]
        fig.text(0.008, 0.5, f"{file_name}",horizontalalignment='center', 
                         rotation = 'vertical', verticalalignment='center', fontsize = 20)
        ax = plt.subplot(1,4,1)
        ax.set_title("sI_og",fontsize=15)
        
        for l_idx, ln in enumerate(nn_dict[file_name]['params']['label_name']):
            ax.plot(ytick_labels,nn_dict[file_name]['sI_og'][:, l_idx],c = cpal2[1] , label = ln)
            
        ax.set_xlabel('nn', labelpad = -2)
        ax.set_ylabel('sI', labelpad = 5)
        ax.set_ylim([sI_vmin_og, sI_vmax_og])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()    
            
        ax = plt.subplot(1,4,2)
        
        b = ax.imshow(nn_dict[file_name]['sI_emb'][:,:,0], vmin = sI_vmin_emb, vmax = sI_vmax_emb, aspect = 'auto')
        ax.set_title(f"sI: {nn_dict[file_name]['params']['label_name'][0]}",fontsize=15)
        ax.set_xlabel('sI nn', labelpad = 5)
        ax.set_ylabel('nn', labelpad = -5)
        ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)
        ax.set_xticks(np.arange(len(ytick_labels)), labels=xtick_labels, rotation= 90)
        fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=1)
    
        ax = plt.subplot(1,4,3)
        ax.plot(ytick_labels, nn_dict[file_name]['trust_dim'], c = cpal2[2], label = 'trust')
        ax.plot(ytick_labels, nn_dict[file_name]['cont_dim'], c = cpal2[4], label = 'cont')
        ax.set_xlabel('nn', labelpad = -2)
        ax.set_ylabel('dim', labelpad = 5)
        ax.set_ylim([0, max_dim])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()    
    
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
            for l_idx, ln in enumerate(nn_dict[file_name]['params']['label_name']):
                m = np.nanmean(nn_dict[file_name]['R2s'][:,:,l_idx,ii], axis=1)
                sd = np.nanstd(nn_dict[file_name]['R2s'][:,:,l_idx,ii], axis=1)
                
                ax.plot(m, c = cpal2[2], label = ln)
                ax.fill_between(np.arange(len(m)), m-sd, m+sd, color = cpal2[2], alpha = 0.3)
                
        
            ax.set_xlabel('nn', labelpad = -2)
            ax.set_xticks(np.arange(len(m)), labels=ytick_labels, rotation = 90)
            ax.set_ylim([R2s_vmin, R2s_vmax[ii,0]])
            ax.set_yticks([R2s_vmin, R2s_vmax[0,0]/2, R2s_vmax[0,0]])
            ax.set_ylabel('R2s error', labelpad = 5)
            ax.set_title(dec_list[ii])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                

        plt.tight_layout()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
        plt.close(fig)
            
    with open(os.path.join(save_dir, f"{fnames[0][:5]}_umap_nn_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)
    
    return True

#%% GENERAL PARAMS
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/umap_params_study/nn'
params = {
    'nn_list': [3, 10, 20, 30, 60, 120, 200, 500, 1000],
    'dim': 5,
    'verbose': True,
    'n_splits': 5
    }

signal_name = 'ML_rates'
label_names = ['posx']
#%% M2019
f = open(os.path.join(save_dir,'M2019_umap_nn_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)
print(f"M2019 umap nn study: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2019' in f]
M2019 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2019_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2019.keys())
M2019_umap_nn = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2019[fname])
    #compute hyperparameter study
    M2019_umap_nn[fname] =  dim_red.compute_umap_nn(pd_object = pd_struct, base_signal = signal_name, 
                                                    label_signal = label_names, trial_signal = "index_mat", **params)
    M2019_umap_nn[fname]['params']['base_name'] = signal_name
    M2019_umap_nn[fname]['params']['label_name'] = label_names
    save_ks = open(os.path.join(save_dir, "M2019_umap_nn_dict.pkl"), "wb")
    pickle.dump(M2019_umap_nn, save_ks)
    save_ks.close()
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()    
_ = plot_umap_nn_study(M2019_umap_nn, save_dir)

#%% M2021
f = open(os.path.join(save_dir,'M2021_umap_nn_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)
print(f"M2021 umap nn study: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2021' in f]
M2021 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2021_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2021.keys())
M2021_umap_nn = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2021[fname])
    #compute hyperparameter study
    M2021_umap_nn[fname] =  dim_red.compute_umap_nn(pd_object = pd_struct, base_signal = signal_name, 
                                                    label_signal = label_names, trial_signal = "index_mat", **params)
    M2021_umap_nn[fname]['params']['base_name'] = signal_name
    M2021_umap_nn[fname]['params']['label_name'] = label_names
    save_ks = open(os.path.join(save_dir, "M2021_umap_nn_dict.pkl"), "wb")
    pickle.dump(M2021_umap_nn, save_ks)
    save_ks.close()

print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()    
_ = plot_umap_nn_study(M2021_umap_nn, save_dir)

#%% M2022
f = open(os.path.join(save_dir,'M2022_umap_nn_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)
print(f"M2022 umap nn study: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2022' in f]
M2022 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2022_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2022.keys())
M2022_umap_nn = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2022[fname])
    #compute hyperparameter study
    M2022_umap_nn[fname] =  dim_red.compute_umap_nn(pd_object = pd_struct, base_signal = signal_name, 
                                                    label_signal = label_names, trial_signal = "index_mat", **params)
    M2022_umap_nn[fname]['params']['base_name'] = signal_name
    M2022_umap_nn[fname]['params']['label_name'] = label_names
    save_ks = open(os.path.join(save_dir, "M2022_umap_nn_dict.pkl"), "wb")
    pickle.dump(M2022_umap_nn, save_ks)
    save_ks.close()

print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()    
_ = plot_umap_nn_study(M2022_umap_nn, save_dir)

#%% M2023
f = open(os.path.join(save_dir,'M2023_umap_nn_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)
print(f"M2023 umap nn study: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2023' in f]
M2023 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2023_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2023.keys())
M2023_umap_nn = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2023[fname])
    #compute hyperparameter study
    M2023_umap_nn[fname] =  dim_red.compute_umap_nn(pd_object = pd_struct, base_signal = signal_name, 
                                                    label_signal = label_names, trial_signal = "index_mat", **params)
    M2023_umap_nn[fname]['params']['base_name'] = signal_name
    M2023_umap_nn[fname]['params']['label_name'] = label_names
    save_ks = open(os.path.join(save_dir, "M2023_umap_nn_dict.pkl"), "wb")
    pickle.dump(M2023_umap_nn, save_ks)
    save_ks.close()

print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()    
_ = plot_umap_nn_study(M2023_umap_nn, save_dir)

#%% M2024
f = open(os.path.join(save_dir,'M2024_umap_nn_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)
print(f"M2024 umap nn study: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2024' in f]
M2024 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2024_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2024.keys())
M2024_umap_nn = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2024[fname])
    #compute hyperparameter study
    M2024_umap_nn[fname] =  dim_red.compute_umap_nn(pd_object = pd_struct, base_signal = signal_name, 
                                                    label_signal = label_names, trial_signal = "index_mat", **params)
    M2024_umap_nn[fname]['params']['base_name'] = signal_name
    M2024_umap_nn[fname]['params']['label_name'] = label_names
    save_ks = open(os.path.join(save_dir, "M2024_umap_nn_dict.pkl"), "wb")
    pickle.dump(M2024_umap_nn, save_ks)
    save_ks.close()
    
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()
_ = plot_umap_nn_study(M2024_umap_nn, save_dir)

#%% M2025
f = open(os.path.join(save_dir,'M2025_umap_nn_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)
print(f"M2025 umap nn study: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2025' in f]
M2025 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2025_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2025.keys())
M2025_umap_nn = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2025[fname])
    #compute hyperparameter study
    M2025_umap_nn[fname] =  dim_red.compute_umap_nn(pd_object = pd_struct, base_signal = signal_name, 
                                                    label_signal = label_names, trial_signal = "index_mat", **params)
    M2025_umap_nn[fname]['params']['base_name'] = signal_name
    M2025_umap_nn[fname]['params']['label_name'] = label_names
    save_ks = open(os.path.join(save_dir, "M2025_umap_nn_dict.pkl"), "wb")
    pickle.dump(M2025_umap_nn, save_ks)
    save_ks.close()

print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()
_ = plot_umap_nn_study(M2025_umap_nn, save_dir)

#%% M2026
f = open(os.path.join(save_dir,'M2026_umap_nn_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)
print(f"M2026 umap nn study: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2026' in f]
M2026 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2026_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2026.keys())
M2026_umap_nn = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2026[fname])
    #compute hyperparameter study
    M2026_umap_nn[fname] =  dim_red.compute_umap_nn(pd_object = pd_struct, base_signal = signal_name, 
                                                    label_signal = label_names, trial_signal = "index_mat", **params)
    M2026_umap_nn[fname]['params']['base_name'] = signal_name
    M2026_umap_nn[fname]['params']['label_name'] = label_names
    save_ks = open(os.path.join(save_dir, "M2026_umap_nn_dict.pkl"), "wb")
    pickle.dump(M2026_umap_nn, save_ks)
    save_ks.close()

print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()
_ = plot_umap_nn_study(M2026_umap_nn, save_dir)

#%% LOAD DATA
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/umap_params_study/nn'
if "M2019_umap_nn" not in locals():
    M2019_umap_nn = gu.load_files(save_dir, '*M2019_umap_nn_dict.pkl', verbose=True, struct_type = "pickle")
if "M2021_umap_nn" not in locals():
    M2021_umap_nn = gu.load_files(save_dir, '*M2021_umap_nn_dict.pkl', verbose=True, struct_type = "pickle")
if "M2022_umap_nn" not in locals():
    M2022_umap_nn = gu.load_files(save_dir, '*M2022_umap_nn_dict.pkl', verbose=True, struct_type = "pickle")
if "M2023_umap_nn" not in locals():
    M2023_umap_nn = gu.load_files(save_dir, '*M2023_umap_nn_dict.pkl', verbose=True, struct_type = "pickle")
if "M2024_umap_nn" not in locals():
    M2024_umap_nn = gu.load_files(save_dir, '*M2024_umap_nn_dict.pkl', verbose=True, struct_type = "pickle")
if "M2025_umap_nn" not in locals():
    M2025_umap_nn = gu.load_files(save_dir, '*M2025_umap_nn_dict.pkl', verbose=True, struct_type = "pickle")  
if "M2026_umap_nn" not in locals():
    M2026_umap_nn = gu.load_files(save_dir, '*M2026_umap_nn_dict.pkl', verbose=True, struct_type = "pickle")  
    
#%%
#Get kernel with better decoding performance
nn_list = M2019_umap_nn[list(M2019_umap_nn.keys())[0]]["params"]["nn_list"]
R2s = np.zeros((len(nn_list),4,4,6))
sI_og = np.zeros((len(nn_list), 4, 6))
sI_emb = np.zeros((len(nn_list), len(nn_list), 4, 6))

def get_nn_results(umap_nn_dict):
    nn_list = umap_nn_dict[list(umap_nn_dict.keys())[0]]["params"]["nn_list"]
    tR2s = np.zeros((len(nn_list),4,4))
    tsI_og = np.zeros((len(nn_list), 4))
    tsI_emb = np.zeros((len(nn_list), len(nn_list), 4))
    
    fnames = list(umap_nn_dict.keys())
    
    last_idx = -1
    count_idx = 0
    for s_idx, s_name in enumerate(fnames):
        if s_idx==0:
            last_idx+=1
            count_idx = 1
        else:
            old_s_name = fnames[s_idx-1]
            old_s_name = old_s_name[:old_s_name.find('_',-5)]
            new_s_name = s_name[:s_name.find('_',-5)]
            if new_s_name == old_s_name:
                count_idx += 1
            else:
                tsI_og[:,last_idx] = tsI_og[:,last_idx]/count_idx
                tsI_emb[:,:,last_idx] = tsI_emb[:,:,last_idx]/count_idx
                tR2s[:,:,last_idx] = tR2s[:,:,last_idx]/count_idx
                last_idx +=1
                count_idx = 1
                
        pd_struct = umap_nn_dict[s_name]
        tsI_og[:,last_idx] += pd_struct["sI_og"][:,0]
        tsI_emb[:,:,last_idx] += pd_struct["sI_emb"][:,:,0]
        tR2s[:,:,last_idx] += np.nanmean(pd_struct["R2s"][:, :,0,:], axis=1)
        
    tsI_og[:,last_idx] = tsI_og[:,last_idx]/count_idx
    tsI_emb[:,:,last_idx] = tsI_emb[:,:,last_idx]/count_idx
    tR2s[:,:,last_idx] = tR2s[:,:,last_idx]/count_idx
    
    return tsI_og, tsI_emb, tR2s

#M2019
sI_og[:,:,0], sI_emb[:,:,:,0],R2s[:,:,:,0] = get_nn_results(M2019_umap_nn)
#M2021
sI_og[:,:,1], sI_emb[:,:,:,1],R2s[:,:,:,1] = get_nn_results(M2021_umap_nn)
#M2022
#sI_og[:,:,2], sI_emb[:,:,:,2],R2s[:,:,:,2] = get_nn_results(M2022_umap_nn)
#M2023
sI_og[:,:,2], sI_emb[:,:,:,2],R2s[:,:,:,2] = get_nn_results(M2023_umap_nn)
#M2024
sI_og[:,:,3], sI_emb[:,:,:,3],R2s[:,:,:,3] = get_nn_results(M2024_umap_nn)
#M2025
sI_og[:,:,4], sI_emb[:,:,:,4],R2s[:,:,:,4] = get_nn_results(M2025_umap_nn)
#M2026
sI_og[:,:,5], sI_emb[:,:,:,5],R2s[:,:,:,5] = get_nn_results(M2026_umap_nn)


R2s_vmax = 25
dec_name = ["wf", "wc", "xgb", "svm"]

#%%
cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]
nn_list = M2019_umap_nn[list(M2019_umap_nn.keys())[0]]["params"]["nn_list"]
R2s_vmax = 25
dec_name = ["wf", "wc", "xgb", "svm"]

plt.figure(figsize=(10.7,3.7))
ax = plt.subplot(1,3,1)
m = np.nanmean(sI_og, axis=(1,2))
sd = np.nanstd(sI_og, axis=(1,2))/np.sqrt(sI_og.shape[1]*sI_og.shape[2])
ax.plot(m, c= cpal[0])
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= cpal[0], alpha = 0.3)
ax.set_xlabel('sI nn', labelpad = -2)
ax.set_xticks(np.arange(len(m)), labels=nn_list, rotation = 90)
ax.set_ylim([0, 1.1])
ax.axvline(x= 4, color='k', linestyle='--')
ax.set_ylabel('sI og-space xpos', labelpad = 5)
ax.set_title("sI og-space ")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = plt.subplot(1,3,2)
m = np.nanmean(sI_emb, axis=(2,3))
sd = np.nanstd(sI_emb, axis=(2,3))/np.sqrt(sI_emb.shape[2]*sI_emb.shape[3])
ax.plot(m[:,1], c= cpal[2], label = f'Local nn: {nn_list[1]}')
ax.fill_between(np.arange(len(m[:,1])), m[:,1]-sd[:,1], m[:,1]+sd[:,1], color=cpal[2], alpha = 0.3)
ax.plot(m[:,4], '--', c= cpal[2], label = f'Global nn: {nn_list[4]}')
ax.fill_between(np.arange(len(m[:,4])), m[:,4]-sd[:,4], m[:,4]+sd[:,4], color= cpal[2], alpha = 0.3)
ax.set_xlabel('umap nn', labelpad = -2)
ax.set_xticks(np.arange(len(m)), labels=nn_list, rotation = 45)
ax.set_ylim([0, 1])
ax.set_yticks([0,0.5, 1])
ax.axvline(x= 4, color='k', linestyle='--')
ax.legend()
ax.set_ylabel('sI umap xpos', labelpad = 5)
ax.set_title('sI umap xpos')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for dec_idx in range(4):
    if dec_idx <2:
        plot_idx = dec_idx+5
    else:
        plot_idx = dec_idx - 1 + 10
        
    ax = plt.subplot(2,6,plot_idx)
    m = np.nanmean(R2s[:,dec_idx,:,:], axis=(1,2))
    sd = np.nanstd(R2s[:,dec_idx,:,:], axis=(1,2))/np.sqrt(R2s.shape[2]*R2s.shape[3])
    ax.plot(m, c= cpal[2])
    ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= cpal[2], alpha = 0.3)
    ax.set_xlabel('umap nn', labelpad = -2)
    ax.set_xticks(np.arange(len(m)), labels=nn_list, rotation = 90)
    ax.set_ylim([0, R2s_vmax])
    ax.set_yticks([0, R2s_vmax/2, R2s_vmax])
    ax.set_ylabel('error xpos [cm]', labelpad = 5)
    ax.axvline(x= 4, color='k', linestyle='--')
    ax.set_title(dec_name[dec_idx])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir,'LT_Jercog_umap_nn_study.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'LT_Jercog_umap_nn_study.png'), dpi = 400,bbox_inches="tight")
#%%
cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]
nn_list = M2019_umap_nn[list(M2019_umap_nn.keys())[0]]["params"]["nn_list"]
R2s_vmax = 25
dec_name = ["wf", "wc", "xgb", "svm"]
dec_idx = 2

plt.figure(figsize=(10.7,3.7))
ax = plt.subplot(1,3,1)
m = np.nanmean(sI_og, axis=(1,2))
sd = np.nanstd(sI_og, axis=(1,2))/np.sqrt(sI_og.shape[1]*sI_og.shape[2])
ax.plot(m, c= cpal[0])
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= cpal[0], alpha = 0.3)
ax.set_xlabel('sI nn', labelpad = -2)
ax.set_xticks(np.arange(len(m)), labels=nn_list, rotation = 90)
ax.set_ylim([0, 1.1])
ax.axvline(x= 4, color='k', linestyle='--')
ax.set_ylabel('sI og-space xpos', labelpad = 5)
ax.set_title("sI og-space ")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = plt.subplot(1,3,2)
m = np.nanmean(sI_emb, axis=(2,3))
sd = np.nanstd(sI_emb, axis=(2,3))/np.sqrt(sI_emb.shape[2]*sI_emb.shape[3])

ax.plot(m[:,1], c= cpal[2], label = 'Local')
ax.fill_between(np.arange(len(m[:,1])), m[:,1]-sd[:,1], m[:,1]+sd[:,1], color=cpal[2], alpha = 0.3)
ax.plot(m[:,4], '--', c= cpal[2], label = 'Global')
ax.fill_between(np.arange(len(m[:,4])), m[:,4]-sd[:,4], m[:,4]+sd[:,4], color= cpal[2], alpha = 0.3)
ax.axvline(x= 4, color='k', linestyle='--')
ax.set_xlabel('umap nn', labelpad = -2)
ax.set_xticks(np.arange(len(m)), labels=nn_list, rotation = 45)
ax.set_ylim([0, 1])
ax.set_yticks([0,0.5, 1])
ax.legend()
ax.set_ylabel('sI umap xpos', labelpad = 5)
ax.set_title('sI umap xpos')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax = plt.subplot(1,3,3)
m = np.nanmean(R2s[:,dec_idx,:,:], axis=(1,2))
sd = np.nanstd(R2s[:,dec_idx,:,:], axis=(1,2))/np.sqrt(R2s.shape[2]*R2s.shape[3])
ax.plot(m, c= cpal[2])
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= cpal[2], alpha = 0.3)
ax.set_xlabel('umap nn', labelpad = -2)
ax.set_xticks(np.arange(len(m)), labels=nn_list, rotation = 90)
ax.axvline(x= 4, color='k', linestyle='--')
ax.set_ylim([0, R2s_vmax])
ax.set_yticks([0, R2s_vmax/2, R2s_vmax])
ax.set_ylabel('error xpos [cm]', labelpad = 5)
ax.set_title(dec_name[dec_idx])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()