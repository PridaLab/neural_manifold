#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:37:59 2022

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

import pandas as pd

#%matplotlib qt

#%% COMPUTE INNER DIM
params = {
    'signal_name': 'deconvProb',
    'n_neigh': 60,
    'verbose': True}

save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/all/denoised_traces/inner_dim'
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/all/same_len_data/'

#%% M2019
if "M2019" not in locals():
    sub_dir = next(os.walk(data_dir))[1]
    foi = [f for f in sub_dir if 'M2019' in f]
    M2019 = gu.load_files(os.path.join(data_dir, foi[0]), '*M2019_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2019.keys())
M2019_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2019[fname])
    #signal
    signal = np.concatenate(pd_struct[params['signal_name']].values, axis = 0)
    #compute abids dim
    abids = dim_red.compute_abids(signal, params['n_neigh'])
    abids_dim = np.nanmean(abids)
    print(f"\tABIDS dim: {abids_dim:.2f}")
    #save results
    M2019_inner_dim[fname] = {
        'abids': abids,
        'abids_dim': abids_dim,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2019_inner_dim.pkl"), "wb")
    pickle.dump(M2019_inner_dim, save_ks)
    save_ks.close()


#%% M2021
if "M2021" not in locals():
    sub_dir = next(os.walk(data_dir))[1]
    foi = [f for f in sub_dir if 'M2021' in f]
    M2021 = gu.load_files(os.path.join(data_dir, foi[0]), '*M2021_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2021.keys())
M2021_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2021[fname])
    #signal
    signal = np.concatenate(pd_struct[params['signal_name']].values, axis = 0)
    #compute abids dim
    abids = dim_red.compute_abids(signal, params['n_neigh'])
    abids_dim = np.nanmean(abids)
    print(f"\tABIDS dim: {abids_dim:.2f}")
    #save results
    M2021_inner_dim[fname] = {
        'abids': abids,
        'abids_dim': abids_dim,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2021_inner_dim.pkl"), "wb")
    pickle.dump(M2021_inner_dim, save_ks)
    save_ks.close()


#%% M2023
if "M2023" not in locals():
    sub_dir = next(os.walk(data_dir))[1]
    foi = [f for f in sub_dir if 'M2023' in f]
    M2023 = gu.load_files(os.path.join(data_dir, foi[0]), '*M2023_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2023.keys())
M2023_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2023[fname])
    #signal
    signal = np.concatenate(pd_struct[params['signal_name']].values, axis = 0)
    #compute abids dim
    abids = dim_red.compute_abids(signal, params['n_neigh'])
    abids_dim = np.nanmean(abids)
    print(f"\tABIDS dim: {abids_dim:.2f}")
    #save results
    M2023_inner_dim[fname] = {
        'abids': abids,
        'abids_dim': abids_dim,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2023_inner_dim.pkl"), "wb")
    pickle.dump(M2023_inner_dim, save_ks)
    save_ks.close()


#%% M2024
if "M2024" not in locals():
    sub_dir = next(os.walk(data_dir))[1]
    foi = [f for f in sub_dir if 'M2024' in f]
    M2024 = gu.load_files(os.path.join(data_dir, foi[0]), '*M2024_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2024.keys())
M2024_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2024[fname])
    #signal
    signal = np.concatenate(pd_struct[params['signal_name']].values, axis = 0)
    #compute abids dim
    abids = dim_red.compute_abids(signal, params['n_neigh'])
    abids_dim = np.nanmean(abids)
    print(f"\tABIDS dim: {abids_dim:.2f}")
    #save results
    M2024_inner_dim[fname] = {
        'abids': abids,
        'abids_dim': abids_dim,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2024_inner_dim.pkl"), "wb")
    pickle.dump(M2024_inner_dim, save_ks)
    save_ks.close()


#%% M2025
if "M2025" not in locals():
    sub_dir = next(os.walk(data_dir))[1]
    foi = [f for f in sub_dir if 'M2025' in f]
    M2025 = gu.load_files(os.path.join(data_dir, foi[0]), '*M2025_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2025.keys())
M2025_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2025[fname])
    #signal
    signal = np.concatenate(pd_struct[params['signal_name']].values, axis = 0)
    #compute abids dim
    abids = dim_red.compute_abids(signal, params['n_neigh'])
    abids_dim = np.nanmean(abids)
    print(f"\tABIDS dim: {abids_dim:.2f}")
    #save results
    M2025_inner_dim[fname] = {
        'abids': abids,
        'abids_dim': abids_dim,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2025_inner_dim.pkl"), "wb")
    pickle.dump(M2025_inner_dim, save_ks)
    save_ks.close()


#%% M2026
if "M2026" not in locals():
    sub_dir = next(os.walk(data_dir))[1]
    foi = [f for f in sub_dir if 'M2026' in f]
    M2026 = gu.load_files(os.path.join(data_dir, foi[0]), '*M2026_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2026.keys())
M2026_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2026[fname])
    #signal
    signal = np.concatenate(pd_struct[params['signal_name']].values, axis = 0)
    #compute abids dim
    abids = dim_red.compute_abids(signal, params['n_neigh'])
    abids_dim = np.nanmean(abids)
    print(f"\tABIDS dim: {abids_dim:.2f}")
    #save results
    M2026_inner_dim[fname] = {
        'abids': abids,
        'abids_dim': abids_dim,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2026_inner_dim.pkl"), "wb")
    pickle.dump(M2026_inner_dim, save_ks)
    save_ks.close()
    save_ks.close()

#%% PLOT
if "M2019_inner_dim" not in locals():
    M2019_inner_dim = gu.load_files(save_dir, '*M2019_inner_dim.pkl', verbose=True, struct_type = "pickle")
if "M2021_inner_dim" not in locals():
    M2021_inner_dim = gu.load_files(save_dir, '*M2021_inner_dim.pkl', verbose=True, struct_type = "pickle")
if "M2022_inner_dim" not in locals():
    M2022_inner_dim = gu.load_files(save_dir, '*M2022_inner_dim.pkl', verbose=True, struct_type = "pickle")
if "M2023_inner_dim" not in locals():
    M2023_inner_dim = gu.load_files(save_dir, '*M2023_inner_dim.pkl', verbose=True, struct_type = "pickle")
if "M2024_inner_dim" not in locals():
    M2024_inner_dim = gu.load_files(save_dir, '*M2024_inner_dim.pkl', verbose=True, struct_type = "pickle")
if "M2025_inner_dim" not in locals():
    M2025_inner_dim = gu.load_files(save_dir, '*M2025_inner_dim.pkl', verbose=True, struct_type = "pickle")  
if "M2026_inner_dim" not in locals():
    M2026_inner_dim = gu.load_files(save_dir, '*M2026_inner_dim.pkl', verbose=True, struct_type = "pickle")  

import seaborn as sns
import pandas as pd

M2019_abids = [pd_struct['abids'] for _,pd_struct in M2019_inner_dim.items()]
M2019_abids_label = [M2019_abids[idx]*0 + idx for idx in range(len(M2019_abids))]

M2021_abids = [pd_struct['abids'] for _,pd_struct in M2021_inner_dim.items()]
M2021_abids_label = [M2021_abids[idx]*0 + idx + M2019_abids_label[-1][0]+1 for idx in range(len(M2021_abids))]

M2023_abids = [pd_struct['abids'] for _,pd_struct in M2023_inner_dim.items()]
M2023_abids_label = [M2023_abids[idx]*0 + idx + M2021_abids_label[-1][0]+1  for idx in range(len(M2023_abids))]

M2024_abids = [pd_struct['abids'] for _,pd_struct in M2024_inner_dim.items()]
M2024_abids_label = [M2024_abids[idx]*0 + idx + M2023_abids_label[-1][0]+1  for idx in range(len(M2024_abids))]

M2025_abids = [pd_struct['abids'] for _,pd_struct in M2025_inner_dim.items()]
M2025_abids_label = [M2025_abids[idx]*0 + idx + M2024_abids_label[-1][0]+1  for idx in range(len(M2025_abids))]

M2026_abids = [pd_struct['abids'] for _,pd_struct in M2026_inner_dim.items()]
M2026_abids_label = [M2026_abids[idx]*0 + idx + M2025_abids_label[-1][0]+1  for idx in range(len(M2026_abids))]


abids = np.concatenate(M2019_abids+M2021_abids+M2023_abids+M2024_abids+M2025_abids+M2026_abids)
abids_label =np.concatenate(M2019_abids_label+M2021_abids_label+M2023_abids_label+M2024_abids_label+M2025_abids_label+M2026_abids_label)

abids_pd = pd.DataFrame(data = {'label': abids_label,
                                'abids': abids.T})

f, ax = plt.subplots(1, 1, figsize=(10,6))
sns.kdeplot(data=abids_pd, x='abids', hue = 'label', fill=True, 
            common_norm=False, common_grid = True, palette = 'hls', ax=ax)  
plt.savefig(os.path.join(save_dir,'inner_dim_dist.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'inner_dim_dist.svg'), dpi = 400,bbox_inches="tight",transparent=True)


# #%%
# from scipy.stats import linregress

def get_inner_dim_results(inner_dim_dict, session_list):

    tinner_dim = np.zeros((3,5))*np.nan
    fnames = list(inner_dim_dict.keys())

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
            
        pd_struct = inner_dim_dict[s_name]

        tinner_dim[count_idx, session_list[last_idx]] =pd_struct['abids_dim']
    return np.nanmean(tinner_dim, axis = 0)

inner_dim = np.zeros((5,6))

inner_dim[:,0] = get_inner_dim_results(M2019_inner_dim,[0,1,2,4] )
inner_dim[:,1] = get_inner_dim_results(M2021_inner_dim,[0,1,2,4] )
inner_dim[:,2] = get_inner_dim_results(M2023_inner_dim,[0,1,2,4] )

inner_dim[:,3] = get_inner_dim_results(M2024_inner_dim,[0,1,2,3] )
inner_dim[:,4] = get_inner_dim_results(M2025_inner_dim,[0,1,2,3] )
inner_dim[:,5] = get_inner_dim_results(M2026_inner_dim,[0,1,2,3] )

mouse_list = ['2019']*5 + ['2021']*5 + ['2023']*5 + ['2024']*5 + ['2025']*5 + ['2026']*5
pd_inner_dim = pd.DataFrame(data={'Day': ['Day1', 'Day2-m', 'Day2-e', 'Day4', 'Day7']*(inner_dim.shape[-1]),
                                     'abids':inner_dim.T.reshape(-1,1).T[0,:],
                                     'mouse': mouse_list})
# #%%
cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500", "#EE90FC"]
x_space = [1,2,2.5,4,7]
f = plt.figure()
ax = plt.subplot(1,1,1)
sns.barplot(ax=ax, x='Day', y='abids', data = pd_inner_dim, color =cpal[0])
sns.lineplot(x = 'Day', y= 'abids', data = pd_inner_dim, units = 'mouse',
            ax = ax, estimator = None, color = ".5", markers = True)

ax.axhline(y=4, color='k', linestyle= '--')
ax.set_ylabel('Inner dimension', size=12)
ax.set_ylim([0,4.5])
ax.set_xlabel('Day')
#%%
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'inner_dim.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'inner_dim.svg'), dpi = 400,bbox_inches="tight",transparent=True)


  
