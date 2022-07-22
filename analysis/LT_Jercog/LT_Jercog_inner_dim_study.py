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

import seaborn as sns
import pandas as pd

#%% COMPUTE INNER DIM
params = {
    'base_signal': 'ML_rates',
    'min_neigh': 2,
    'max_neigh':2**5,
    'verbose': True}

save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/inner_dim'

#%% M2019
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
if "M2019" not in locals():
    sub_dir = next(os.walk(file_dir))[1]
    foi = [f for f in sub_dir if 'M2019' in f]
    M2019 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2019_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2019.keys())
M2019_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    pd_struct = copy.deepcopy(M2019[fname])
    #compute innerdim
    m, radius, neigh = dim_red.compute_inner_dim(pd_object = pd_struct,**params)
    #save results
    M2019_inner_dim[fname] = {
        'm': m,
        'radius': radius,
        'neigh': neigh,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2019_inner_dim.pkl"), "wb")
    pickle.dump(M2019_inner_dim, save_ks)
    save_ks.close()
    
#%% M2021
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
if "M2021" not in locals():
    sub_dir = next(os.walk(file_dir))[1]
    foi = [f for f in sub_dir if 'M2021' in f]
    M2021 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2021_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2021.keys())
M2021_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    pd_struct = copy.deepcopy(M2021[fname])
    #compute innerdim
    m, radius, neigh = dim_red.compute_inner_dim(pd_object = pd_struct,**params)
    #save results
    M2021_inner_dim[fname] = {
        'm': m,
        'radius': radius,
        'neigh': neigh,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2021_inner_dim.pkl"), "wb")
    pickle.dump(M2021_inner_dim, save_ks)
    save_ks.close()
#%% M2023
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
if "M2023" not in locals():
    sub_dir = next(os.walk(file_dir))[1]
    foi = [f for f in sub_dir if 'M2023' in f]
    M2023 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2023_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2023.keys())
M2023_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    pd_struct = copy.deepcopy(M2023[fname])
    #compute innerdim
    m, radius, neigh = dim_red.compute_inner_dim(pd_object = pd_struct,**params)
    #save results
    M2023_inner_dim[fname] = {
        'm': m,
        'radius': radius,
        'neigh': neigh,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2023_inner_dim.pkl"), "wb")
    pickle.dump(M2023_inner_dim, save_ks)
    save_ks.close()
#%% M2024
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
if "M2024" not in locals():
    sub_dir = next(os.walk(file_dir))[1]
    foi = [f for f in sub_dir if 'M2024' in f]
    M2024 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2024_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2024.keys())
M2024_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    pd_struct = copy.deepcopy(M2024[fname])
    #compute innerdim
    m, radius, neigh = dim_red.compute_inner_dim(pd_object = pd_struct,**params)
    #save results
    M2024_inner_dim[fname] = {
        'm': m,
        'radius': radius,
        'neigh': neigh,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2024_inner_dim.pkl"), "wb")
    pickle.dump(M2024_inner_dim, save_ks)
    save_ks.close()
#%% M2025
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
if "M2025" not in locals():
    sub_dir = next(os.walk(file_dir))[1]
    foi = [f for f in sub_dir if 'M2025' in f]
    M2025 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2025_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2025.keys())
M2025_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    pd_struct = copy.deepcopy(M2025[fname])
    #compute innerdim
    m, radius, neigh = dim_red.compute_inner_dim(pd_object = pd_struct,**params)
    #save results
    M2025_inner_dim[fname] = {
        'm': m,
        'radius': radius,
        'neigh': neigh,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2025_inner_dim.pkl"), "wb")
    pickle.dump(M2025_inner_dim, save_ks)
    save_ks.close()
#%% M2026
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
if "M2026" not in locals():
    sub_dir = next(os.walk(file_dir))[1]
    foi = [f for f in sub_dir if 'M2026' in f]
    M2026 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2026_df_dict.pkl', verbose = True, struct_type = "pickle")
    
fname_list = list(M2026.keys())
M2026_inner_dim = dict()
for f_idx, fname in enumerate(fname_list):
    pd_struct = copy.deepcopy(M2026[fname])
    #compute innerdim
    m, radius, neigh = dim_red.compute_inner_dim(pd_object = pd_struct,**params)
    #save results
    M2026_inner_dim[fname] = {
        'm': m,
        'radius': radius,
        'neigh': neigh,
        'params': params
        }
    save_ks = open(os.path.join(save_dir, "M2026_inner_dim.pkl"), "wb")
    pickle.dump(M2026_inner_dim, save_ks)
    save_ks.close()

#%%
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/inner_dim'
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
save_fig = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/poster_figures'

#%%
from scipy.stats import linregress

def get_inner_dim_results(inner_dim_dict, session_list):
    lower_bound = 2
    upper_bound = 7
    
    tinner_dim = np.zeros((3,5))*np.nan
    tneigh_all = np.zeros((1000,3,5))*np.nan
    tradius_all = np.zeros((1000,3,5))*np.nan

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
        tneigh = pd_struct["neigh"]
        tradius = pd_struct["radius"]
        
        tneigh_all[:tneigh.shape[0],count_idx, session_list[last_idx]] = tneigh[:,0]
        tradius_all[:tradius.shape[0],count_idx, session_list[last_idx]] = tradius[:,0]

        in_range_mask = np.all(np.vstack((tneigh[:,0]<upper_bound, tneigh[:,0]>lower_bound)).T, axis=1)
        radius_in_range = tradius[in_range_mask, :] 
        neigh_in_range = tneigh[in_range_mask, :]
        m = linregress(radius_in_range[:,0], neigh_in_range[:,0])[0]
        
        tinner_dim[count_idx, session_list[last_idx]] =m
    return np.nanmean(tinner_dim, axis = 0), np.nanmean(tneigh_all, axis = 1),np.nanmean(tradius_all, axis = 1)

inner_dim = np.zeros((5,6))
radius_dim = np.zeros((1000, 5,6))
neigh_dim = np.zeros((1000, 5,6))

inner_dim[:,0], neigh_dim[:,:,0], radius_dim[:,:,0] = get_inner_dim_results(M2019_inner_dim,[0,1,2,4] )
inner_dim[:,1], neigh_dim[:,:,1], radius_dim[:,:,1] = get_inner_dim_results(M2021_inner_dim,[0,1,2,4] )
inner_dim[:,2], neigh_dim[:,:,2], radius_dim[:,:,2] = get_inner_dim_results(M2023_inner_dim,[0,1,2,4] )

inner_dim[:,3], neigh_dim[:,:,3], radius_dim[:,:,3] = get_inner_dim_results(M2024_inner_dim,[0,1,2,3] )
inner_dim[:,4], neigh_dim[:,:,4], radius_dim[:,:,4] = get_inner_dim_results(M2025_inner_dim,[0,1,2,3] )
inner_dim[:,5], neigh_dim[:,:,5], radius_dim[:,:,5] = get_inner_dim_results(M2026_inner_dim,[0,1,2,3] )

pd_inner_dim = pd.DataFrame(data={'Day': ['Day1', 'Day2-m', 'Day2-e', 'Day4', 'Day7']*(inner_dim.shape[-1]),
                                    'slope':inner_dim.T.reshape(-1,1).T[0,:]})
#%%
cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500", "#EE90FC"]
x_space = [1,2,2.5,4,7]
#%%
miny = 2
maxy = 7
minx = -1
maxx = 5
m = 3

fig = plt.figure(figsize=(15,8))
ax = plt.subplot(1,2,1)
for mice in range(neigh_dim.shape[-1]):
    for sess in range(neigh_dim.shape[-2]):
        ax.plot(radius_dim[:,sess,mice],neigh_dim[:,sess,mice])
ax.set_ylim([miny,maxy])
ax.set_xlim([minx,maxx])

ax.set_xlabel('radius (log2)', size=12)
ax.set_ylabel('neighbors (log2)', size=12)

ns = np.linspace(miny-np.floor(maxx)*m, maxy, 10)
for ni in ns:
    x = np.linspace(np.max([((miny-ni)/m), minx]), np.min([((maxy-ni)/m),maxx]), 20).reshape(-1,1)
    y = m*x + ni
    plt.plot(x,y, color = [.5,.5,.5], linestyle = '--')
    
ax = plt.subplot(1,2,2)
sns.barplot(ax=ax, x='Day', y='slope', data = pd_inner_dim, color =cpal[0])
ax.axhline(y=3, color='k', linestyle= '--')
ax.set_ylabel('Inner dimension', size=12)
ax.set_ylim([0,4.5])
ax.set_xlabel('Day')
#%%
plt.tight_layout()
plt.savefig(os.path.join(save_fig,'inner_dim.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_fig,'inner_dim.svg'), dpi = 400,bbox_inches="tight",transparent=True)


  
