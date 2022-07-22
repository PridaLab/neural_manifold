#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:20:36 2022

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
#%% PLOT UMAP NPOINTS

def plot_umap_ncell_study(ncells_dict, save_dir):
    cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]

    fnames = list(ncells_dict.keys())
    dec_list = ['wf','wc','xgb','svm']

    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + f"<h1>Umap npoints study - {fnames[0]}</h1>\n<br>\n"    #Add title
    html = html + f"<h2>signal: {ncells_dict[fnames[0]]['params']['base_name']} - "
    html = html + f"<br>{datetime.now().strftime('%d/%m/%y %H:%M:%S')}</h2><br>\n"    #Add subtitle
    
    sI_vmin = np.inf
    sI_vmax = 0

    trust_dmax = 0
    cont_dmax = 0
    
    inner_dmax = 0
    
    for file_idx, file_name in enumerate(fnames):
        
        sI_vmin = np.nanmin([sI_vmin, np.min(ncells_dict[file_name]['sI_values'])])
        sI_vmax = np.nanmax([sI_vmax, np.max(ncells_dict[file_name]['sI_values'])])
        
        inner_dmax = np.nanmax([inner_dmax, np.max(ncells_dict[file_name]['inner_dim'])])
        trust_dmax = np.nanmax([trust_dmax, np.max(ncells_dict[file_name]['trust_dim'])])
        cont_dmax = np.nanmax([cont_dmax, np.max(ncells_dict[file_name]['cont_dim'])])
        
    
    for file_idx, file_name in enumerate(fnames):

        fig= plt.figure(figsize = (16, 4))
        ytick_labels = [str(entry) for entry in ncells_dict[file_name]['params']['nn_list']]
        xtick_labels =[str(entry) for entry in ncells_dict[file_name]['params']['og_num_cells']]
        num_cells = ncells_dict[file_name]['params']['og_num_cells']
        fig.text(0.008, 0.5, f"{file_name}",horizontalalignment='center', 
                         rotation = 'vertical', verticalalignment='center', fontsize = 20)
        
        ax = plt.subplot(1,3,1)
        val = np.nanmean(ncells_dict[file_name]['sI_values'][:,:,2,:,0], axis = 1)
        b = ax.imshow(val.T, vmin = sI_vmin, vmax = sI_vmax, aspect = 'auto')
        ax.set_title(f"sI: {ncells_dict[file_name]['params']['label_name'][0]}",fontsize=15)
        ax.set_ylabel('sI nn', labelpad = 5)
        ax.set_xlabel('# cells', labelpad = -5)
        ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)
        ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels, rotation = 90)
        fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=1)
        
        ax = plt.subplot(1,3,2)
        trust_m = np.nanmean(ncells_dict[file_name]['trust_dim'], axis = (2,1))
        trust_sd = np.nanstd(ncells_dict[file_name]['trust_dim'], axis = (2,1))
        ax.plot(num_cells,trust_m, c = cpal[2], label = 'trust')
        ax.fill_between(num_cells, trust_m-trust_sd, trust_m+trust_sd, color = cpal[2], alpha = 0.3)
        
        cont_m = np.nanmean(ncells_dict[file_name]['cont_dim'], axis = (2,1))
        cont_sd = np.nanstd(ncells_dict[file_name]['cont_dim'], axis = (2,1))
        ax.plot(num_cells,cont_m, c = cpal[4], label = 'cont')
        ax.fill_between(num_cells, cont_m-cont_sd, cont_m+cont_sd, color = cpal[4], alpha = 0.3)
        
        inner_m = np.nanmean(ncells_dict[file_name]['inner_dim'], axis = 1)
        inner_sd = np.nanstd(ncells_dict[file_name]['inner_dim'], axis = 1)
        ax.plot(num_cells,inner_m, c = cpal[0], label = 'inner_dim')
        ax.fill_between(num_cells, inner_m-inner_sd, inner_m+inner_sd, color = cpal[0], alpha = 0.3)
        
    
        ax.set_xlabel('# cells', labelpad = -2)
        #ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels, rotation = 90)
        ax.set_ylim([0, 5])
        ax.set_ylabel('dim', labelpad = 0)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
     
        for ii in range(4):
            if ii == 0:
                pos = 5
            elif ii == 1:
                pos = 6
            elif ii == 2:
                pos = 11
            elif ii == 3:
                pos = 12
                
            ax = plt.subplot(2,6,pos)
            for l_idx, ln in enumerate(ncells_dict[file_name]['params']['label_name']):
                m = np.nanmean(ncells_dict[file_name]['R2s_values'][:,:,:,l_idx,ii], axis=(1,2))
                sd = np.nanstd(ncells_dict[file_name]['R2s_values'][:,:,:,l_idx,ii], axis=(1,2))
                
                ax.plot(num_cells,m, c = cpal[2], label = ln)
                ax.fill_between(num_cells, m-sd, m+sd, color = cpal[2], alpha = 0.3)
            
            ax.set_xlabel('# cells', labelpad = -2)
            #ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels, rotation = 90)
            ax.set_ylim([0, 25])
            ax.set_yticks([0, 12.5, 25])
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
            
    with open(os.path.join(save_dir, f"{fnames[0][:5]}_umap_ncells_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)
    
    return True
#%%
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/umap_params_study/ncells'
params = {
    'nn_list': [3, 10, 20, 30, 60, 120, 200],
    'max_dim': 10,
    'verbose': True,
    'n_splits': 10
    }

signal_name = 'ML_rates'
label_names = ['posx']

#%% M2019
#load data
f = open(os.path.join(save_dir,'M2019_umap_ncells_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2019 umap ncells: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2019' in f]
M2019 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2019_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2019.keys())
M2019_umap_ncells = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2019[fname])
    
    M2019_umap_ncells[fname] = dim_red.compute_umap_to_ncells(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,trial_signal = 'index_mat',**params)
    
    M2019_umap_ncells[fname]['params']['base_name'] = signal_name
    M2019_umap_ncells[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2019_umap_ncells_dict.pkl"), "wb")
    pickle.dump(M2019_umap_ncells, save_ks)
    save_ks.close()
      
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()
_ = plot_umap_ncell_study(M2019_umap_ncells, save_dir)

#%% M2025
#load data
f = open(os.path.join(save_dir,'M2025_umap_ncells_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2025 umap ncells: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2025' in f]
M2025 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2025_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2025.keys())
M2025_umap_ncells = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2025[fname])
    
    M2025_umap_ncells[fname] = dim_red.compute_umap_to_ncells(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,trial_signal = 'index_mat',**params)
    
    M2025_umap_ncells[fname]['params']['base_name'] = signal_name
    M2025_umap_ncells[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2025_umap_ncells_dict.pkl"), "wb")
    pickle.dump(M2025_umap_ncells, save_ks)
    save_ks.close()
      
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()
_ = plot_umap_ncell_study(M2025_umap_ncells, save_dir)