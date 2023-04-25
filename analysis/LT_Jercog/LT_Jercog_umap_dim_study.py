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
import pickle, os, copy, sys

import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO

import seaborn as sns
import pandas as pd
from kneed import KneeLocator
from scipy.stats import linregress

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
    sI_vmax = 0.8

    trust_vmin = np.inf
    trust_vmax = 0
    
    cont_vmin = np.inf
    cont_vmax = 0
    
    R2s_vmin = 0
    R2s_vmax = np.zeros((4,1))
    
    for file_idx, file_name in enumerate(fnames):
        sI_vmin = np.nanmin([sI_vmin, np.min(dim_dict[file_name]['sI'], axis= (0,1,2))])
        sI_vmax = np.nanmax([sI_vmax, np.max(dim_dict[file_name]['sI'], axis= (0,1,2))])
        
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
        if dim_dict[file_name]['sI'].shape[2]==1:
            b = ax.imshow(dim_dict[file_name]['sI'][:,:,0].T, vmin = sI_vmin, vmax = sI_vmax, aspect = 'auto')
            ax.set_title(f"sI: {dim_dict[file_name]['params']['label_name'][0]}",fontsize=15)
            ax.set_ylabel('sI nn', labelpad = 5)
            ax.set_xlabel('dim', labelpad = -5)
            ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)
            ax.set_xticks(xtick_labels, labels=xtick_labels+1)
            fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=1)
        else:
            for idx in range(dim_dict[file_name]['sI'].shape[2]):
                ax.plot(dim_dict[file_name]['sI'][:,1,idx], label = dim_dict[file_name]['params']['label_name'][idx])

            ax.set_xlabel('dim', labelpad = -5)
            ax.set_xticks(xtick_labels, labels=xtick_labels+1, rotation = 90)

            ax.set_ylabel('sI', labelpad = 5)
            ax.set_yticks([0, 0.25, 0.5, 1])
            ax.set_ylim([-0.05, 1.1])
            ax.legend()
            ax.set_title(f"sI nn: {dim_dict[file_name]['params']['nn_list'][1]}",fontsize=15)
            
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
                
                ax.plot(m, label = ln)
                ax.fill_between(np.arange(len(m)), m-sd, m+sd, alpha = 0.3)
                
        
            ax.set_xlabel('dim', labelpad = -2)
            ax.set_xticks(xtick_labels, labels=xtick_labels+1, rotation = 90)
            ax.set_ylim([R2s_vmin, R2s_vmax[ii,0]])
            ax.set_yticks([R2s_vmin, R2s_vmax[0,0]/2, R2s_vmax[0,0]])
            ax.set_ylabel('R2s error', labelpad = 0)
            ax.set_title(dec_list[ii])
            ax.legend()
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

#__________________________________________________________________________
#|                                                                        |#
#|                             UMAP DIM STUDY                              |#
#|________________________________________________________________________|#
#%% GENERAL PARAMS
signal_name = 'clean_traces'
case = 'clean_traces'

mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
base_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/'
save_dir = os.path.join(base_dir, case, 'umap_params_study', 'dim')
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
#%% GENERAL PARAMS
params = {
    'nn_list': [20, 60, 100, 200],
    'max_dim': 8,
    'verbose': True,
    'n_splits': 5,
    'nn': 50
    }
label_names = ['posx', 'dir_mat', 'posy', 'vel']

for mouse in mice_list:
    f = open(os.path.join(save_dir,mouse + '_umap_nn_logFile.txt'), 'w')
    original = sys.stdout
    sys.stdout = gu.Tee(sys.stdout, f)
    print(f"{mouse} umap nn study: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
    #load data
    sub_dir = next(os.walk(file_dir))[1]
    foi = [f for f in sub_dir if mouse in f]
    load_data_name = '*' + mouse + '_df_dict.pkl'
    animal_dict = gu.load_files(os.path.join(file_dir, foi[0]), load_data_name, verbose = True, struct_type = "pickle")

    fname_list = list(animal_dict.keys())
    umap_dim_dict = dict()
    save_name = mouse + '_umap_dim_dict.pkl'

    for f_idx, fname in enumerate(fname_list):
        print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
        pd_struct = copy.deepcopy(animal_dict[fname])
        #compute hyperparameter study
        m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                         base_signal = signal_name, label_signal = label_names,
                                                         trial_signal = "index_mat", **params)
        m_params['base_name'] = signal_name
        m_params['label_name'] = label_names
        #save results
        umap_dim_dict[fname] = {
            'trust': m_trust,
            'cont': m_cont,
            'sI': m_sI,
            'R2s': m_R2s,
            'params': m_params
            }
        save_ks = open(os.path.join(save_dir, save_name), "wb")
        pickle.dump(umap_dim_dict, save_ks)
        save_ks.close()

    print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
    sys.stdout = original
    f.close()    
    _ = plot_umap_dim_study(umap_dim_dict, save_dir)

#__________________________________________________________________________
#|                                                                        |#
#|                          PLOT UMAP DIM STUDY                            |#
#|________________________________________________________________________|#
if "M2019_umap_dim" not in locals():
    M2019_umap_dim = gu.load_files(save_dir, '*M2019_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2021_umap_dim" not in locals():
    M2021_umap_dim = gu.load_files(save_dir, '*M2021_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2023_umap_dim" not in locals():
    M2023_umap_dim = gu.load_files(save_dir, '*M2023_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2024_umap_dim" not in locals():
    M2024_umap_dim = gu.load_files(save_dir, '*M2024_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2025_umap_dim" not in locals():
    M2025_umap_dim = gu.load_files(save_dir, '*M2025_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")
if "M2026_umap_dim" not in locals():
    M2026_umap_dim = gu.load_files(save_dir, '*M2026_umap_dim_dict.pkl', verbose=True, struct_type = "pickle")

#%%
def get_dim_results(umap_dim_dict, session_list):
    nn_list = umap_dim_dict[list(umap_dim_dict.keys())[0]]["params"]["nn_list"]
    max_dim = umap_dim_dict[list(umap_dim_dict.keys())[0]]["params"]["max_dim"]

    tcont = np.zeros((max_dim, len(nn_list),3, 5))*np.nan
    ttrust = np.zeros((max_dim, len(nn_list),3, 5))*np.nan
    
    tcont_dim = np.zeros((len(nn_list)+1,3, 5))*np.nan
    ttrust_dim = np.zeros((len(nn_list)+1,3, 5))*np.nan
    meanh_dim = np.zeros((len(nn_list)+1,3, 5))*np.nan

    tsI = np.zeros((max_dim, len(nn_list),3,5))*np.nan

    tR2s = np.zeros((max_dim,4,3,5))*np.nan

    fnames = list(umap_dim_dict.keys())
    
    last_idx = -1
    for s_idx, s_name in enumerate(fnames):
        if s_idx==0:
            last_idx+=1
            count_idx = 0
        else:
            old_s_name = fnames[s_idx-1]
            old_s_name = old_s_name[:old_s_name.find('_',-5)]
            new_s_name = s_name[:s_name.find('_',-5)]
            if new_s_name == old_s_name:
                count_idx += 1
            else:
                last_idx +=1
                count_idx = 0
                
        pd_struct = umap_dim_dict[s_name]
        tcont[:,:,count_idx,session_list[last_idx]] = pd_struct["cont"]
        ttrust[:,:,count_idx,session_list[last_idx]] = pd_struct["trust"]
        
        for nn in range(len(nn_list)):
            kl = KneeLocator(np.arange(max_dim)+1, pd_struct["trust"][:, nn], curve = "concave", direction = "increasing")
            if kl.knee:
                ttrust_dim[nn,count_idx,session_list[last_idx]] = kl.knee

            kl = KneeLocator(np.arange(max_dim)+1, pd_struct["cont"][:, nn], curve = "concave", direction = "increasing")
            if kl.knee:
                tcont_dim[nn,count_idx,session_list[last_idx]] = kl.knee
            
            val = (2*pd_struct["trust"][:,nn]*pd_struct["cont"][:,nn])/(pd_struct["trust"][:,nn]+pd_struct["cont"][:,nn])
            kl = KneeLocator(np.arange(max_dim)+1, val, curve = "concave", direction = "increasing")
            if kl.knee:
                meanh_dim[nn,count_idx,session_list[last_idx]] = kl.knee
             
        last_nn = [1,3]
        kl = KneeLocator(np.arange(max_dim)+1, np.nanmean(pd_struct["trust"][:,last_nn],axis = 1), curve = "concave", direction = "increasing")
        if kl.knee:
            ttrust_dim[-1,count_idx,session_list[last_idx]] = kl.knee  
        kl = KneeLocator(np.arange(max_dim)+1, np.nanmean(pd_struct["cont"][:,last_nn],axis = 1), curve = "concave", direction = "increasing")
        if kl.knee:
            tcont_dim[-1,count_idx,session_list[last_idx]] = kl.knee
        val = (2*np.nanmean(pd_struct["trust"][:,last_nn],axis = 1)*np.nanmean(pd_struct["cont"][:,last_nn],axis = 1))/(np.nanmean(pd_struct["trust"][:,last_nn],axis = 1)+np.nanmean(pd_struct["cont"][:,last_nn],axis = 1))        
        kl = KneeLocator(np.arange(max_dim)+1,val, curve = "concave", direction = "increasing")
        if kl.knee:
            meanh_dim[-1,count_idx,session_list[last_idx]] = kl.knee    
        tsI[:,:,count_idx,session_list[last_idx]] = pd_struct["sI"][:,:,0]
        tR2s[:,:,count_idx,session_list[last_idx]] = np.nanmean(pd_struct["R2s"][:, :,0,:], axis=1)
        
    return np.nanmean(tcont,axis=-2), np.nanmean(tcont_dim,axis=-2), np.nanmean(ttrust,axis=-2),np.nanmean(ttrust_dim,axis=-2),np.nanmean(meanh_dim, axis=-2), np.nanmean(tsI,axis=-2), np.nanmean(tR2s,axis=-2)
#%%
#Get kernel with better decoding performance
nn_list = M2019_umap_dim[list(M2019_umap_dim.keys())[0]]["params"]["nn_list"]
max_dim = M2019_umap_dim[list(M2019_umap_dim.keys())[0]]["params"]["max_dim"]

cont = np.zeros((max_dim,len(nn_list),5,6))
cont_dim = np.zeros((len(nn_list)+1,5,6))

trust = np.zeros((max_dim,len(nn_list),5,6))
trust_dim = np.zeros((len(nn_list)+1,5,6))

meanh_dim = np.zeros((len(nn_list)+1,5,6))

sI = np.zeros((max_dim, len(nn_list), 5,6))
R2s = np.zeros((max_dim,4,5,6))

#M2019
cont[:,:,:,0], cont_dim[:,:,0],trust[:,:,:,0], trust_dim[:,:,0], meanh_dim[:,:,0], sI[:,:,:,0], R2s[:,:,:,0] = get_dim_results(M2019_umap_dim, [0,1,2,4])
#M2021
cont[:,:,:,1], cont_dim[:,:,1],trust[:,:,:,1], trust_dim[:,:,1], meanh_dim[:,:,1], sI[:,:,:,1], R2s[:,:,:,1] = get_dim_results(M2021_umap_dim, [0,1,2,4])
#M2023
cont[:,:,:,2], cont_dim[:,:,2],trust[:,:,:,2], trust_dim[:,:,2], meanh_dim[:,:,2], sI[:,:,:,2], R2s[:,:,:,2] = get_dim_results(M2023_umap_dim, [0,1,2,4])
#M2024
cont[:,:,:,3], cont_dim[:,:,3],trust[:,:,:,3], trust_dim[:,:,3], meanh_dim[:,:,3], sI[:,:,:,3], R2s[:,:,:,3] = get_dim_results(M2024_umap_dim, [0,1,2,3])
#M2025
cont[:,:,:,4], cont_dim[:,:,4],trust[:,:,:,4], trust_dim[:,:,4], meanh_dim[:,:,4], sI[:,:,:,4], R2s[:,:,:,4] = get_dim_results(M2025_umap_dim, [0,1,2,3])
#M2026
cont[:,:,:,5], cont_dim[:,:,5],trust[:,:,:,5], trust_dim[:,:,5], meanh_dim[:,:,5], sI[:,:,:,5], R2s[:,:,:,5] = get_dim_results(M2026_umap_dim, [0,1,2,3])
meanh = 2*cont*trust/(cont+trust)
#%%
nns =1
pd_umap_dim = pd.DataFrame(data={'Day': ['Day1', 'Day2-m', 'Day2-e', 'Day4', 'Day7']*(trust_dim.shape[-1])*3,
                                 'Measurement': ['trustworthiness']*(trust_dim.shape[-1]*trust_dim.shape[-2]) + 
                                 ['continuity']*(cont_dim.shape[-1]*cont_dim.shape[-2]) +
                                 ['harmonic_mean']*(meanh_dim.shape[-1]*meanh_dim.shape[-2]),
                                 'Index': np.concatenate((trust_dim[nns,:,:].T.reshape(-1,1).T, cont_dim[nns,:,:].T.reshape(-1,1).T, meanh_dim[nns,:,:].T.reshape(-1,1).T), axis = 1)[0,:]})
#%%
pd_umap_dim_v2 = pd.DataFrame(data={'Day': ['Day1', 'Day2-m', 'Day2-e', 'Day4', 'Day7']*(trust_dim.shape[-1])*3,
                                 'Measurement': ['trustworthiness']*(trust.shape[-1]*trust_dim.shape[-2]) + 
                                 ['continuity']*(cont_dim.shape[-1]*cont_dim.shape[-2]) +
                                 ['harmonic_mean']*(meanh_dim.shape[-1]*meanh_dim.shape[-2]),
                                 'Index': np.concatenate((trust[3,nns,:,:].T.reshape(-1,1).T, cont[3,nns,:,:].T.reshape(-1,1).T, meanh[3,nns,:,:].T.reshape(-1,1).T), axis = 1)[0,:]})
#%%
cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500", "#EE90FC"]
x_space = [1,2,2.5,4,7]
dim_space = np.arange(max_dim)+1
dec_name = ["wf", "wc", "xgb", "svm"]
nn_list = M2021_umap_dim[list(M2021_umap_dim.keys())[0]]["params"]["nn_list"]
lidx = 1
gidx = 3

#%%
fig = plt.figure(figsize=(15,8))
ax = plt.subplot(2,3,1)
m = np.nanmean(trust, axis = (2,3))
sd = np.nanstd(trust, axis = (2,3))
ax.plot(dim_space, m[:,lidx], color = cpal[1], label = f'local nn: {nn_list[lidx]}')
ax.fill_between(dim_space, m[:,lidx]-sd[:,lidx], m[:,lidx]+sd[:,lidx], color = cpal[1], alpha=0.3)
ax.plot(dim_space, m[:,gidx], color = cpal[1], linestyle='--', label = f'Global nn: {nn_list[gidx]}')
ax.fill_between(dim_space, m[:,gidx]-sd[:,gidx], m[:,gidx]+sd[:,gidx], color = cpal[1], alpha=0.3)
ax.set_ylabel('Trustworthiness', size=12)
ax.set_xticks(dim_space)
ax.set_xlabel('Dimension')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.axvline(x= 4, color='k', linestyle='--')
ax.set_ylim([0.75,1.02])
ax.legend()


ax = plt.subplot(2,3,2)
m = np.nanmean(cont, axis = (2,3))
sd = np.nanstd(cont, axis = (2,3))
ax.plot(dim_space, m[:,lidx], color = cpal[4], label = f'local nn: {nn_list[lidx]}')
ax.fill_between(dim_space, m[:,lidx]-sd[:,lidx], m[:,lidx]+sd[:,lidx], color = cpal[4], alpha=0.3)
ax.plot(dim_space, m[:,gidx], color = cpal[4], linestyle='--', label = f'Global nn: {nn_list[gidx]}')
ax.fill_between(dim_space, m[:,gidx]-sd[:,gidx], m[:,gidx]+sd[:,gidx], color = cpal[4], alpha=0.3)
ax.set_ylabel('Continuity', size=12)
ax.set_xticks(dim_space)
ax.set_xlabel('Dimension')
ax.axvline(x= 4, color='k', linestyle='--')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
ax.set_ylim([0.75,1.02])

ax = plt.subplot(2,3,3)
m = np.nanmean(meanh, axis = (2,3))
sd = np.nanstd(meanh, axis = (2,3))
ax.plot(dim_space, m[:,lidx], color = cpal[-1], label = f'local nn: {nn_list[lidx]}')
ax.fill_between(dim_space, m[:,lidx]-sd[:,lidx], m[:,lidx]+sd[:,lidx], color = cpal[-1], alpha=0.3)
ax.plot(dim_space, m[:,gidx], color = cpal[-1], linestyle='--', label = f'Global nn: {nn_list[gidx]}')
ax.fill_between(dim_space, m[:,gidx]-sd[:,gidx], m[:,gidx]+sd[:,gidx], color = cpal[-1], alpha=0.3)
ax.set_ylabel('Continuity', size=12)
ax.set_xticks(dim_space)
ax.set_xlabel('Dimension')
ax.axvline(x= 4, color='k', linestyle='--')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
ax.set_ylim([0.75,1.02])

ax = plt.subplot(2,3,4)
sns.barplot(ax=ax, x='Day', y='Index', hue = 'Measurement', data = pd_umap_dim_v2)
ax.set_ylabel('Index', size=12)
ax.axhline(y=1, color='k', linestyle= '--')
# ax.set_yticks([0,1,2,3,4])
ax.set_ylim([0.8,1.05])
ax.set_xlabel('Day')

ax = plt.subplot(2,3,5)
sns.barplot(ax=ax, x='Day', y='Index', hue = 'Measurement', data = pd_umap_dim)
ax.set_ylabel('Estimated dimension', size=12)
ax.axhline(y=4, color='k', linestyle= '--')
ax.set_yticks([0,1,2,3,4])
ax.set_ylim([0,4.5])
ax.set_xlabel('Day')

for dec_idx in range(4):
    if dec_idx <2:
        plot_idx = dec_idx+17
    else:
        plot_idx = dec_idx - 1 + 22
        
    ax = plt.subplot(4,6,plot_idx)
    m = np.nanmean(R2s[:,dec_idx,:,:], axis=(1,2))
    sd = np.nanstd(R2s[:,dec_idx,:,:], axis=(1,2))/np.sqrt(5*6)

    ax.plot(dim_space, m, c= cpal[2])
    ax.fill_between(dim_space, m-sd, m+sd, color= cpal[2], alpha = 0.3)
    ax.set_xlabel('Dimension', labelpad = -2)
    ax.set_ylim([0, 25])
    ax.set_yticks([0, 12.5, 25])
    ax.set_ylabel('error xpos [cm]', labelpad = 5)
    ax.axvline(x= 4, color='k', linestyle='--')
    ax.set_title(dec_name[dec_idx])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
#%%
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'umap_dims_both.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'umap_dims_both.svg'), dpi = 400,bbox_inches="tight",transparent=True)

