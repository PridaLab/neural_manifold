#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:37:59 2022

@author: julio
"""
#%% IMPORTS
import numpy as np
from neural_manifold import general_utils as gu
import pickle, os, copy
import matplotlib.pyplot as plt
import pandas as pd
from neural_manifold import dimensionality_reduction as dim_red
import seaborn as sns
#%matplotlib qt

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

def get_num_cells(animal_dict, session_list):
    num_cells = np.zeros((5,))*np.nan
    fnames = list(animal_dict.keys())
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
        num_cells[session_list[last_idx]] = animal_dict[s_name]['clean_traces'][0].shape[1]
    return num_cells
#__________________________________________________________________________
#|                                                                        |#
#|                               INNER DIM                                |#
#|________________________________________________________________________|#
mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']

params = {
    'signal_name': 'clean_traces',
    'n_neigh': 60,
    'verbose': True
}

base_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving'
save_dir = os.path.join(base_dir, params['signal_name'], 'inner_dim')
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'

for mouse in mice_list:
    print('')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = gu.load_files(file_path,'*'+file_name,verbose=True,struct_type="pickle")
    fnames = list(mouse_dict.keys())
    inner_dim_dict = dict()
    save_name = mouse + '_inner_dim.pkl'

    for f_idx, fname in enumerate(fnames):
        print(f"Working on session: {fname} ({f_idx+1}/{len(fnames)})")
        pd_struct= copy.deepcopy(mouse_dict[fname])
        #signal
        signal = np.concatenate(pd_struct[params['signal_name']].values, axis = 0)
        #compute abids dim
        abids = dim_red.compute_abids(signal, params['n_neigh'])
        abids_dim = np.nanmean(abids)
        print(f"\tABIDS dim: {abids_dim:.2f}")
        #save results
        inner_dim_dict[fname] = {
            'abids': abids,
            'abids_dim': abids_dim,
            'params': params
        }
        save_ks = open(os.path.join(save_dir, save_name), "wb")
        pickle.dump(inner_dim_dict, save_ks)
        save_ks.close()

#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT INNER DIM                             |#
#|________________________________________________________________________|#

M2019_inner_dim = gu.load_files(save_dir, '*M2019_inner_dim.pkl', verbose=True, struct_type = "pickle")
M2021_inner_dim = gu.load_files(save_dir, '*M2021_inner_dim.pkl', verbose=True, struct_type = "pickle")
M2022_inner_dim = gu.load_files(save_dir, '*M2022_inner_dim.pkl', verbose=True, struct_type = "pickle")
M2023_inner_dim = gu.load_files(save_dir, '*M2023_inner_dim.pkl', verbose=True, struct_type = "pickle")
M2024_inner_dim = gu.load_files(save_dir, '*M2024_inner_dim.pkl', verbose=True, struct_type = "pickle")
M2025_inner_dim = gu.load_files(save_dir, '*M2025_inner_dim.pkl', verbose=True, struct_type = "pickle")  
M2026_inner_dim = gu.load_files(save_dir, '*M2026_inner_dim.pkl', verbose=True, struct_type = "pickle")  

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


inner_dim = np.zeros((5,6))
inner_dim[:,0] = get_inner_dim_results(M2019_inner_dim,[0,1,2,4] )
inner_dim[:,1] = get_inner_dim_results(M2021_inner_dim,[0,1,2,4] )
inner_dim[:,2] = get_inner_dim_results(M2023_inner_dim,[0,1,2,4] )
inner_dim[:,3] = get_inner_dim_results(M2024_inner_dim,[0,1,2,3] )
inner_dim[:,4] = get_inner_dim_results(M2025_inner_dim,[0,1,2,3] )
inner_dim[:,5] = get_inner_dim_results(M2026_inner_dim,[0,1,2,3] )
num_cells = np.zeros((5,6))
for idx, mouse in enumerate(mice_list):
    if idx<3:
        session_list = [0,1,2,4]
    else:
        session_list = [0,1,2,3]

    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = gu.load_files(file_path,'*'+file_name,verbose=True,struct_type="pickle")
    num_cells[:,idx] = get_num_cells(mouse_dict,session_list)

animal_list = ['2019']*5 + ['2021']*5 + ['2023']*5 + ['2024']*5 + ['2025']*5 + ['2026']*5
pd_inner_dim = pd.DataFrame(data={'Day': ['Day1', 'Day2-m', 'Day2-e', 'Day4', 'Day7']*(inner_dim.shape[-1]),
                                     'abids':inner_dim.T.reshape(-1,1).T[0,:],
                                     'num_cells': num_cells.T.reshape(-1,1).T[0,:],
                                     'mouse': animal_list})
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


# f = plt.figure()
# ax = plt.subplot(1,1,1)
# sns.scatterplot(ax=ax, x='num_cells', y='abids', hue = 'Day', data = pd_inner_dim, color =cpal[0])
# r, p = sp.stats.spearmanr(pd_inner_dim['num_cells'], pd_inner_dim['abids'], nan_policy='omit')
# plt.text(.05, .8, 'r={:.2f}'.format(r), transform=ax.transAxes)
# plt.text(.05, .7, 'p={:.2f}'.format(p), transform=ax.transAxes)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir,'inner_dim_vs_num_cells.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
# plt.savefig(os.path.join(save_dir,'inner_dim_vs_num_cells.svg'), dpi = 400,bbox_inches="tight",transparent=True)