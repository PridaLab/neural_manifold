#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:42:23 2022

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

def plot_umap_npoint_study(npoints_dict, save_dir):
    cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]

    fnames = list(npoints_dict.keys())
    
    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + f"<h1>Umap npoints study - {fnames[0]}</h1>\n<br>\n"    #Add title
    html = html + f"<h2>signal: {npoints_dict[fnames[0]]['params']['base_name']} - "
    html = html + f"<br>{datetime.now().strftime('%d/%m/%y %H:%M:%S')}</h2><br>\n"    #Add subtitle
    
    
    sI_vmin = np.inf
    sI_vmax = 0

    trust_dmax = 0
    
    cont_dmax = 0
    for file_idx, file_name in enumerate(fnames):
        sI_vmin = np.nanmin([sI_vmin, np.min(npoints_dict[file_name]['sI_values'])])
        sI_vmax = np.nanmax([sI_vmax, np.max(npoints_dict[file_name]['sI_values'])])
        
        trust_dmax = np.nanmax([trust_dmax, np.nanmax(npoints_dict[file_name]['trust_dim'])])
        cont_dmax = np.nanmax([cont_dmax, np.nanmax(npoints_dict[file_name]['cont_dim'])])
        
    if trust_dmax<1:
        trust_dmax= 6
    if cont_dmax<1:
        cont_dmax= 6
    for file_idx, file_name in enumerate(fnames):

        fig= plt.figure(figsize = (16, 4))
        ytick_labels = [str(entry) for entry in npoints_dict[file_name]['params']['nn_list']]
        xtick_labels =[str(entry) for entry in npoints_dict[file_name]['params']['og_point_list']]
        
        fig.text(0.008, 0.5, f"{file_name}",horizontalalignment='center', 
                         rotation = 'vertical', verticalalignment='center', fontsize = 20)
        
        ax = plt.subplot(1,3,1)
        val = np.nanmean(npoints_dict[file_name]['sI_values'][:,:,2,:,0], axis = 1)
        b = ax.imshow(val.T, vmin = sI_vmin, vmax = sI_vmax, aspect = 'auto')
        ax.set_title(f"sI: {npoints_dict[file_name]['params']['label_name'][0]}",fontsize=15)
        ax.set_ylabel('sI nn', labelpad = 5)
        ax.set_xlabel('# points', labelpad = -5)
        ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)
        ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels, rotation = 90)
        fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=1)
        
        ax = plt.subplot(1,4,2)
        trust_m = np.nanmean(npoints_dict[file_name]['trust_dim'], axis = (2,1))
        trust_sd = np.nanstd(npoints_dict[file_name]['trust_dim'], axis = (2,1))
        ax.plot(trust_m, c = cpal[2], label = 'trust')
        ax.fill_between(np.arange(len(trust_m)), trust_m-trust_sd, trust_m+trust_sd, color = cpal[2], alpha = 0.3)
        ax.set_xlabel('# points', labelpad = -2)
        ax.set_title('Trustworthiness')

        ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels, rotation = 90)
        ax.set_ylim([0, trust_dmax])
        ax.set_ylabel('dim', labelpad = 0)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax = plt.subplot(1,4,3)
        cont_m = np.nanmean(npoints_dict[file_name]['cont_dim'], axis = (2,1))
        cont_sd = np.nanstd(npoints_dict[file_name]['cont_dim'], axis = (2,1))
        ax.plot(cont_m, c = cpal[2], label = 'cont')
        ax.fill_between(np.arange(len(cont_m)), cont_m-cont_sd, cont_m+cont_sd, color = cpal[2], alpha = 0.3)
        ax.set_xlabel('# points', labelpad = -2)
        ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels, rotation = 90)
        ax.set_ylim([0, cont_dmax])
        ax.set_title('Continuity')
        ax.set_ylabel('dim', labelpad = 0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax = plt.subplot(1,4,4)
        m = np.nanmean(npoints_dict[file_name]['inner_dim'], axis = 1)
        sd = np.nanstd(npoints_dict[file_name]['inner_dim'], axis = 1)
        ax.plot(m, c = cpal[0])
        ax.fill_between(np.arange(len(m)), m-sd, m+sd, color = cpal[0], alpha = 0.3)
        ax.set_xlabel('# points', labelpad = -2)
        ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels, rotation = 90)
        ax.set_ylim([0, np.nanmax(m)+1])
        ax.set_ylabel('inner dim', labelpad = 0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
        plt.close(fig)
            
    with open(os.path.join(save_dir, f"{fnames[0][:5]}_umap_npoints_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)
    
    return True
#%%
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/umap_params_study/npoints'
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
f = open(os.path.join(save_dir,'M2019_umap_npoints_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2019 umap npoints: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2019' in f]
M2019 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2019_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2019.keys())
M2019_umap_npoints = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2019[fname])
    
    M2019_umap_npoints[fname] = dim_red.compute_umap_to_npoints(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,**params)
    
    M2019_umap_npoints[fname]['params']['base_name'] = signal_name
    M2019_umap_npoints[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2019_umap_npoints_dict.pkl"), "wb")
    pickle.dump(M2019_umap_npoints, save_ks)
    save_ks.close()
      
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")  
sys.stdout = original
f.close()
_ = plot_umap_npoint_study(M2019_umap_npoints, save_dir)

#%% M2021
#load data
f = open(os.path.join(save_dir,'M2021_umap_npoints_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2021 umap npoints: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2021' in f]
M2021 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2021_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2021.keys())
M2021_umap_npoints = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2021[fname])
    
    M2021_umap_npoints[fname] = dim_red.compute_umap_to_npoints(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,**params)
    
    M2021_umap_npoints[fname]['params']['base_name'] = signal_name
    M2021_umap_npoints[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2021_umap_npoints_dict.pkl"), "wb")
    pickle.dump(M2021_umap_npoints, save_ks)
    save_ks.close()
      
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")  
sys.stdout = original
f.close()
_ = plot_umap_npoint_study(M2021_umap_npoints, save_dir)

#%% M2022
#load data
f = open(os.path.join(save_dir,'M2022_umap_npoints_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2022 umap npoints: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2022' in f]
M2022 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2022_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2022.keys())
M2022_umap_npoints = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2022[fname])
    
    M2022_umap_npoints[fname] = dim_red.compute_umap_to_npoints(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,**params)
    
    M2022_umap_npoints[fname]['params']['base_name'] = signal_name
    M2022_umap_npoints[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2022_umap_npoints_dict.pkl"), "wb")
    pickle.dump(M2022_umap_npoints, save_ks)
    save_ks.close()
    
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
sys.stdout = original
f.close()
_ = plot_umap_npoint_study(M2022_umap_npoints, save_dir)

#%% M2023
#load data
f = open(os.path.join(save_dir,'M2023_umap_npoints_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2023 umap npoints: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2023' in f]
M2023 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2023_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2023.keys())
M2023_umap_npoints = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2023[fname])
    
    M2023_umap_npoints[fname] = dim_red.compute_umap_to_npoints(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,**params)
    
    M2023_umap_npoints[fname]['params']['base_name'] = signal_name
    M2023_umap_npoints[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2023_umap_npoints_dict.pkl"), "wb")
    pickle.dump(M2023_umap_npoints, save_ks)
    save_ks.close()
    
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
sys.stdout = original
f.close()
_ = plot_umap_npoint_study(M2023_umap_npoints, save_dir)

#%% M2024
#load data
f = open(os.path.join(save_dir,'M2024_umap_npoints_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2024 umap npoints: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2024' in f]
M2024 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2024_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2024.keys())
M2024_umap_npoints = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2024[fname])
    
    M2024_umap_npoints[fname] = dim_red.compute_umap_to_npoints(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,**params)
    
    M2024_umap_npoints[fname]['params']['base_name'] = signal_name
    M2024_umap_npoints[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2024_umap_npoints_dict.pkl"), "wb")
    pickle.dump(M2024_umap_npoints, save_ks)
    save_ks.close()
    
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
sys.stdout = original
f.close()
_ = plot_umap_npoint_study(M2024_umap_npoints, save_dir)

#%% M2025
#load data
f = open(os.path.join(save_dir,'M2025_umap_npoints_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2025 umap npoints: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2025' in f]
M2025 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2025_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2025.keys())
M2025_umap_npoints = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2025[fname])
    
    M2025_umap_npoints[fname] = dim_red.compute_umap_to_npoints(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,**params)
    
    M2025_umap_npoints[fname]['params']['base_name'] = signal_name
    M2025_umap_npoints[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2025_umap_npoints_dict.pkl"), "wb")
    pickle.dump(M2025_umap_npoints, save_ks)
    save_ks.close()
    
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
sys.stdout = original
f.close()
_ = plot_umap_npoint_study(M2025_umap_npoints, save_dir)

#%% M2026
#load data
f = open(os.path.join(save_dir,'M2026_umap_npoints_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

print(f"M2026 umap npoints: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")

file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2026' in f]
M2026 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2026_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2026.keys())
M2026_umap_npoints = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2026[fname])
    
    M2026_umap_npoints[fname] = dim_red.compute_umap_to_npoints(pd_object = pd_struct,base_signal = signal_name, 
                                                  label_signal = label_names,**params)
    
    M2026_umap_npoints[fname]['params']['base_name'] = signal_name
    M2026_umap_npoints[fname]['params']['label_name'] = label_names
    
    #save results
    save_ks = open(os.path.join(save_dir, "M2026_umap_npoints_dict.pkl"), "wb")
    pickle.dump(M2026_umap_npoints, save_ks)
    save_ks.close()
    
    
print(f"\nCompleted: {datetime.now().strftime('%d/%m/%y %H:%M:%S')}\n")
sys.stdout = original
f.close()
_ = plot_umap_npoint_study(M2026_umap_npoints, save_dir)

#%% LOAD DATA
if "M2019_umap_npoints" not in locals():
    M2019_umap_npoints = gu.load_files(save_dir, '*M2019_umap_npoints_dict.pkl', verbose=True, struct_type = "pickle")
if "M2021_umap_npoints" not in locals():
    M2021_umap_npoints = gu.load_files(save_dir, '*M2021_umap_npoints_dict.pkl', verbose=True, struct_type = "pickle")
if "M2023_umap_npoints" not in locals():
    M2023_umap_npoints = gu.load_files(save_dir, '*M2023_umap_npoints_dict.pkl', verbose=True, struct_type = "pickle")
if "M2024_umap_npoints" not in locals():
    M2024_umap_npoints = gu.load_files(save_dir, '*M2024_umap_npoints_dict.pkl', verbose=True, struct_type = "pickle")
    
    
if "M2025_umap_npoints" not in locals():
    M2025_umap_npoints = gu.load_files(save_dir, '*M2025_umap_npoints_dict.pkl', verbose=True, struct_type = "pickle")  
if "M2026_umap_npoints" not in locals():
    M2026_umap_npoints = gu.load_files(save_dir, '*M2026_umap_npoints_dict.pkl', verbose=True, struct_type = "pickle")  

#%%
from kneed import KneeLocator

def get_npoints_params(npoints_dict, session_list):
    nn_list = npoints_dict[list(npoints_dict.keys())[0]]["params"]["nn_list"]
    max_dim = npoints_dict[list(npoints_dict.keys())[0]]["params"]["max_dim"]
    point_list = npoints_dict[list(npoints_dict.keys())[0]]["params"]["og_point_list"]
    final_point_list = npoints_dict[list(npoints_dict.keys())[0]]["params"]["point_list"]
    
    tcont_val = np.zeros((len(point_list),max_dim, len(nn_list),3, 5))*np.nan
    ttrust_val = np.zeros((len(point_list),max_dim, len(nn_list),3, 5))*np.nan
    
    tcont_dim = np.zeros((len(point_list),len(nn_list),3, 5))*np.nan
    ttrust_dim = np.zeros((len(point_list),len(nn_list),3, 5))*np.nan
    meanh_dim = np.zeros((len(point_list),len(nn_list),3, 5))*np.nan
    
    tsI = np.zeros((len(point_list),max_dim, len(nn_list),3, 5))*np.nan
    
    fnames = list(npoints_dict.keys())
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
                
        pd_struct = npoints_dict[s_name]
        tcont_dim[:,:,count_idx, session_list[last_idx]] = np.nanmean(pd_struct["cont_dim"],axis=1)
        ttrust_dim[:,:,count_idx, session_list[last_idx]] = np.nanmean(pd_struct["trust_dim"],axis=1)
        
        tcont_val[:,:,:,count_idx,session_list[last_idx]] = np.nanmean(pd_struct["cont_dim_values"],axis=1)
        ttrust_val[:,:,:,count_idx,session_list[last_idx]] = np.nanmean(pd_struct["trust_dim_values"],axis=1)
        
        for pts in range(len(final_point_list)):
            for nn in range(len(nn_list)):
                cont_val = np.nanmean(pd_struct["cont_dim_values"][pts,:,:,nn],axis=0)
                trust_val = np.nanmean(pd_struct["trust_dim_values"][pts,:,:,nn],axis=0)
                
                val = (2*trust_val*cont_val)/(trust_val+cont_val)
                kl = KneeLocator(np.arange(max_dim)+1, val, curve = "concave", direction = "increasing")
                if kl.knee:
                    meanh_dim[pts,nn,count_idx, session_list[last_idx]] = kl.knee
    
     
                
        tsI[:,:,:,count_idx,session_list[last_idx]] = np.nanmean(pd_struct["sI_values"], axis = 1)[:,:,:,0]
        
    return np.nanmean(tcont_val,axis=-2), np.nanmean(tcont_dim,axis=-2), np.nanmean(ttrust_val,axis=-2),np.nanmean(ttrust_dim,axis=-2),np.nanmean(meanh_dim,axis=-2),np.nanmean(tsI,axis=-2)


#%%

nn_list = M2019_umap_npoints[list(M2019_umap_npoints.keys())[0]]["params"]["nn_list"]
max_dim = M2019_umap_npoints[list(M2019_umap_npoints.keys())[0]]["params"]["max_dim"]
point_list = M2019_umap_npoints[list(M2019_umap_npoints.keys())[0]]["params"]["og_point_list"]

cont = np.zeros((len(point_list),max_dim,len(nn_list),5,6))*np.nan
cont_dim = np.zeros((len(point_list),len(nn_list),5,6))*np.nan

trust = np.zeros((len(point_list),max_dim,len(nn_list),5,6))*np.nan
trust_dim = np.zeros((len(point_list),len(nn_list),5,6))*np.nan

meanh_dim = np.zeros((len(point_list),len(nn_list),5,6))*np.nan

sI_val = np.zeros((len(point_list),max_dim, len(nn_list), 5,6))*np.nan

#M2019
cont[:,:,:,:,0], cont_dim[:,:,:,0],trust[:,:,:,:,0], trust_dim[:,:,:,0], meanh_dim[:,:,:,0], sI_val[:,:,:,:,0] = get_npoints_params(M2019_umap_npoints, [0,1,2,4])
#M2021
cont[:,:,:,:,1], cont_dim[:,:,:,1],trust[:,:,:,:,1], trust_dim[:,:,:,1], meanh_dim[:,:,:,1], sI_val[:,:,:,:,1] = get_npoints_params(M2021_umap_npoints, [0,1,2,4])
#M2023
cont[:,:,:,:,2], cont_dim[:,:,:,2],trust[:,:,:,:,2], trust_dim[:,:,:,2], meanh_dim[:,:,:,2], sI_val[:,:,:,:,2] = get_npoints_params(M2023_umap_npoints, [0,1,2,4])
#M2024
cont[:,:,:,:,3], cont_dim[:,:,:,3],trust[:,:,:,:,3], trust_dim[:,:,:,3], meanh_dim[:,:,:,3], sI_val[:,:,:,:,3] = get_npoints_params(M2024_umap_npoints, [0,1,2,3])
#M2025
#cont[:,:,:,:,4], cont_dim[:,:,:,4],trust[:,:,:,:,4], trust_dim[:,:,:,4], meanh_dim[:,:,:,4], sI[:,:,:,:,4] = get_npoints_params(M2025_umap_npoints, [0,1,2,3])
#M2026
#cont[:,:,:,:,5], cont_dim[:,:,:,5],trust[:,:,:,:,5], trust_dim[:,:,:,5], meanh_dim[:,:,:,5], sI[:,:,:,:,5] = get_npoints_params(M2026_umap_npoints, [0,1,2,3])

#%%
nns = 1
cpal = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500", "#EE90FC"]

#%%
pd_npoints = pd.DataFrame(data={'Points': [str(pts) for pts in point_list]*(trust_dim.shape[-1]*trust_dim.shape[-2])*3,
                                 'Measurement': ['trustworthiness']*(trust_dim.shape[-1]*trust_dim.shape[-2]*trust_dim.shape[0]) + 
                                 ['continuity']*(cont_dim.shape[-1]*cont_dim.shape[-2]*cont_dim.shape[0]) +
                                 ['harmonic_mean']*(meanh_dim.shape[-1]*meanh_dim.shape[-2]*meanh_dim.shape[0]),
                                 'Index': np.concatenate((trust_dim[:,nns,:,:].T.reshape(-1,1).T, cont_dim[:,nns,:,:].T.reshape(-1,1).T, meanh_dim[:,nns,:,:].T.reshape(-1,1).T), axis = 1)[0,:]})


dim_space = np.arange(9)

#%%
fig = plt.figure(figsize=(15,8))

ax = plt.subplot(1,2,1)
m = np.nanmean(sI_val[:8,2,nns,:,:], axis = (-2,-1))
sd = np.nanstd(sI_val[:8,2,nns,:,:], axis = (-2,-1))
ax.plot(point_list[:8], m, color = cpal[0])
ax.fill_between(point_list[:8], m-sd, m+sd, color = cpal[0], alpha=0.3)
ax.set_ylim([0.245, 1.05])
ax.set_ylabel('Structure Index (x-pos)', size=12)
ax.set_xlabel('Number of points', size=12)


ax = plt.subplot(1,2,2)
m = np.nanmean(trust_dim[:8,nns,:,:], axis = (-2,-1))
sd = np.nanstd(trust_dim[:8,nns,:,:], axis = (-2,-1))
ax.plot(point_list[:8], m, color = cpal[1], label = 'trust')
ax.fill_between(point_list[:8], m-sd, m+sd, color = cpal[1], alpha=0.3)

m = np.nanmean(cont_dim[:8,nns,:,:], axis = (-2,-1))
sd = np.nanstd(cont_dim[:8,nns,:,:], axis = (-2,-1))
ax.plot(point_list[:8], m, color = cpal[4], label = 'cont')
ax.fill_between(point_list[:8], m-sd, m+sd, color = cpal[4], alpha=0.3)

m = np.nanmean(meanh_dim[:8,nns,:,:], axis = (-2,-1))
sd = np.nanstd(meanh_dim[:8,nns,:,:], axis = (-2,-1))
ax.plot(point_list[:8], m, color = cpal[-1], label = 'HM')
ax.fill_between(point_list[:8], m-sd, m+sd, color = cpal[-1], alpha=0.3)

ax.set_yticks([0,1,2,3,4])
ax.set_ylim([0,4.5])
ax.set_ylabel('Dimensionality', size=12)
ax.set_xlabel('Number of points', size=12)

ax.legend()

#%%
ax = plt.subplot(1,3,3)
sns.barplot(ax=ax, x='Points', y='Index', hue = 'Measurement', data = pd_npoints)
ax.set_ylabel('Estimated dimension', size=12)
ax.axhline(y=3, color='k', linestyle= '--')
ax.set_yticks([0,1,2,3,4])
ax.set_ylim([0,4.5])
ax.set_xlabel('Day')
