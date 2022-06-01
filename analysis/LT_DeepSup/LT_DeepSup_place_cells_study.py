#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:39:11 2022

@author: julio
"""

import numpy as np
import copy
from neural_manifold import general_utils as gu
from neural_manifold import place_cells as pc
import pickle
import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#%% GENERAL PARAMS
save_dir =  '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/place_cells_study'
params = {
    "dim": 1,
    "save_dir":save_dir,
    "bin_width": 5,
    "std_pos": 0.025,
    "ignore_edges":10,
    "std_pdf": 2,
    "method": "spatial_info",
    "num_shuffles":1000,
    "min_shift":5,
    "sF":20,
    'th_metric': 99
    }

#%% CZ3: 
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/CZ3'
CZ3 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(CZ3.keys())
CZ3_place_cells_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(CZ3[fname])
    if 'dir_mat' not in pd_struct.columns:
            pd_struct["dir_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+
                             ('L' == pd_struct["dir"][idx])+ 2*('R' == pd_struct["dir"][idx])
                                 for idx in range(pd_struct.shape[0])]
    
    pos_signal = np.concatenate(pd_struct["pos"].values, axis= 0)
    vel_signal = np.concatenate(pd_struct["vel"].values, axis= 0)
    spikes_signal = np.concatenate(pd_struct["events_SNR3"].values, axis= 0)
    direction_signal = np.concatenate(pd_struct["dir_mat"].values, axis= 0)
    
    
    pos_signal = pos_signal[direction_signal[:,0]>0,:] 
    vel_signal = vel_signal[direction_signal[:,0]>0] 
    spikes_signal = spikes_signal[direction_signal[:,0]>0,:] 
    direction_signal = direction_signal[direction_signal[:,0]>0,:] 

    
    CZ3_place_cells_study[fname] = pc.get_place_cells(pos_signal, spikes_signal, vel_signal = vel_signal,
                          direction_signal = direction_signal, mouse = fname, **params)
   
    save_ks = open(os.path.join(params["save_dir"], "CZ3_place_cells_study.pkl"), "wb")
    pickle.dump(CZ3_place_cells_study, save_ks)
    save_ks.close()
    
#%% CZ4: 
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/CZ4'
CZ4 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(CZ4.keys())
CZ4_place_cells_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(CZ4[fname])
    if 'dir_mat' not in pd_struct.columns:
            pd_struct["dir_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+
                             ('L' == pd_struct["dir"][idx])+ 2*('R' == pd_struct["dir"][idx])
                                 for idx in range(pd_struct.shape[0])]
    
    pos_signal = np.concatenate(pd_struct["pos"].values, axis= 0)
    vel_signal = np.concatenate(pd_struct["vel"].values, axis= 0)
    spikes_signal = np.concatenate(pd_struct["events_SNR3"].values, axis= 0)
    direction_signal = np.concatenate(pd_struct["dir_mat"].values, axis= 0)
    
    pos_signal = pos_signal[direction_signal[:,0]>0,:] 
    vel_signal = vel_signal[direction_signal[:,0]>0] 
    spikes_signal = spikes_signal[direction_signal[:,0]>0,:] 
    direction_signal = direction_signal[direction_signal[:,0]>0,:] 

    CZ4_place_cells_study[fname] = pc.get_place_cells(pos_signal, spikes_signal, vel_signal = vel_signal,
                          direction_signal = direction_signal, mouse = fname, **params)
   
    save_ks = open(os.path.join(params["save_dir"], "CZ4_place_cells_study.pkl"), "wb")
    pickle.dump(CZ4_place_cells_study, save_ks)
    save_ks.close()

#%% CZ6: 
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/CZ6'
CZ6 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(CZ6.keys())
CZ6_place_cells_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(CZ6[fname])
    if 'dir_mat' not in pd_struct.columns:
            pd_struct["dir_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+
                             ('L' == pd_struct["dir"][idx])+ 2*('R' == pd_struct["dir"][idx])
                                 for idx in range(pd_struct.shape[0])]
    
    pos_signal = np.concatenate(pd_struct["pos"].values, axis= 0)
    vel_signal = np.concatenate(pd_struct["vel"].values, axis= 0)
    spikes_signal = np.concatenate(pd_struct["events_SNR3"].values, axis= 0)
    direction_signal = np.concatenate(pd_struct["dir_mat"].values, axis= 0)
    
    pos_signal = pos_signal[direction_signal[:,0]>0,:] 
    vel_signal = vel_signal[direction_signal[:,0]>0] 
    spikes_signal = spikes_signal[direction_signal[:,0]>0,:] 
    direction_signal = direction_signal[direction_signal[:,0]>0,:] 

    CZ6_place_cells_study[fname] = pc.get_place_cells(pos_signal, spikes_signal, vel_signal = vel_signal,
                          direction_signal = direction_signal, mouse = fname, **params)
   
    save_ks = open(os.path.join(params["save_dir"], "CZ6_place_cells_study.pkl"), "wb")
    pickle.dump(CZ6_place_cells_study, save_ks)
    save_ks.close()

#%% GC2: 
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/GC2'
GC2 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(GC2.keys())
GC2_place_cells_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(GC2[fname])
    if 'dir_mat' not in pd_struct.columns:
            pd_struct["dir_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+
                             ('L' == pd_struct["dir"][idx])+ 2*('R' == pd_struct["dir"][idx])
                                 for idx in range(pd_struct.shape[0])]
    
    pos_signal = np.concatenate(pd_struct["pos"].values, axis= 0)
    vel_signal = np.concatenate(pd_struct["vel"].values, axis= 0)
    spikes_signal = np.concatenate(pd_struct["events_SNR3"].values, axis= 0)
    direction_signal = np.concatenate(pd_struct["dir_mat"].values, axis= 0)
    
    pos_signal = pos_signal[direction_signal[:,0]>0,:] 
    vel_signal = vel_signal[direction_signal[:,0]>0] 
    spikes_signal = spikes_signal[direction_signal[:,0]>0,:] 
    direction_signal = direction_signal[direction_signal[:,0]>0,:] 

    GC2_place_cells_study[fname] = pc.get_place_cells(pos_signal, spikes_signal, vel_signal = vel_signal,
                          direction_signal = direction_signal, mouse = fname, **params)
   
    save_ks = open(os.path.join(params["save_dir"], "GC2_place_cells_study.pkl"), "wb")
    pickle.dump(GC2_place_cells_study, save_ks)
    save_ks.close()

#%% GC3: 
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/GC3'
GC3 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(GC3.keys())
GC3_place_cells_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(GC3[fname])
    if 'dir_mat' not in pd_struct.columns:
            pd_struct["dir_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+
                             ('L' == pd_struct["dir"][idx])+ 2*('R' == pd_struct["dir"][idx])
                                 for idx in range(pd_struct.shape[0])]
    
    pos_signal = np.concatenate(pd_struct["pos"].values, axis= 0)
    vel_signal = np.concatenate(pd_struct["vel"].values, axis= 0)
    spikes_signal = np.concatenate(pd_struct["events_SNR3"].values, axis= 0)
    direction_signal = np.concatenate(pd_struct["dir_mat"].values, axis= 0)
    
    pos_signal = pos_signal[direction_signal[:,0]>0,:] 
    vel_signal = vel_signal[direction_signal[:,0]>0] 
    spikes_signal = spikes_signal[direction_signal[:,0]>0,:] 
    direction_signal = direction_signal[direction_signal[:,0]>0,:] 

    GC3_place_cells_study[fname] = pc.get_place_cells(pos_signal, spikes_signal, vel_signal = vel_signal,
                          direction_signal = direction_signal, mouse = fname, **params)
   
    save_ks = open(os.path.join(params["save_dir"], "GC3_place_cells_study.pkl"), "wb")
    pickle.dump(GC3_place_cells_study, save_ks)
    save_ks.close()

#%% Load
if "CZ3_place_cells_study" not in locals():
    CZ3_place_cells_study = gu.load_files(save_dir, '*CZ3_place_cells_study.pkl', verbose=True, struct_type = "pickle")
if "CZ4_place_cells_study" not in locals():
    CZ4_place_cells_study = gu.load_files(save_dir, '*CZ4_place_cells_study.pkl', verbose=True, struct_type = "pickle")
if "CZ6_place_cells_study" not in locals():
    CZ6_place_cells_study = gu.load_files(save_dir, '*CZ6_place_cells_study.pkl', verbose=True, struct_type = "pickle")
if "GC2_place_cells_study" not in locals():
    GC2_place_cells_study = gu.load_files(save_dir, '*GC2_place_cells_study.pkl', verbose=True, struct_type = "pickle")
if "GC3_place_cells_study" not in locals():
    GC3_place_cells_study = gu.load_files(save_dir, '*GC3_place_cells_study.pkl', verbose=True, struct_type = "pickle")

#%%Num place cells (general), num place cells direction-dependent vs non-direction dependent
num_pc_sup = np.zeros((3,2))
num_cells_sup = np.zeros((3,2))
num_pc_deep = np.zeros((2,2))
num_cells_deep = np.zeros((2,2))

for s_idx, s_name in enumerate(list(CZ3_place_cells_study.keys())):
    pd_struct = CZ3_place_cells_study[s_name]
    num_pc_sup[0,s_idx] = len(pd_struct["place_cells_idx"])
    num_cells_sup[0,s_idx] =pd_struct["metric_val"].shape[0]

for s_idx, s_name in enumerate(list(CZ4_place_cells_study.keys())):
    pd_struct = CZ4_place_cells_study[s_name]
    num_pc_sup[1,s_idx] = len(pd_struct["place_cells_idx"])
    num_cells_sup[1,s_idx] =pd_struct["metric_val"].shape[0]

for s_idx, s_name in enumerate(list(CZ6_place_cells_study.keys())):
    pd_struct = CZ6_place_cells_study[s_name]
    num_pc_sup[2,s_idx] = len(pd_struct["place_cells_idx"])
    num_cells_sup[2,s_idx] =pd_struct["metric_val"].shape[0]

for s_idx, s_name in enumerate(list(GC2_place_cells_study.keys())):
    pd_struct = GC2_place_cells_study[s_name]
    num_pc_deep[0,s_idx] = len(pd_struct["place_cells_idx"])
    num_cells_deep[0,s_idx] =pd_struct["metric_val"].shape[0]

for s_idx, s_name in enumerate(list(GC3_place_cells_study.keys())):
    pd_struct = GC3_place_cells_study[s_name]
    num_pc_deep[1,s_idx] = len(pd_struct["place_cells_idx"])
    num_cells_deep[1,s_idx] =pd_struct["metric_val"].shape[0]

perc_pc_sup = num_pc_sup/num_cells_sup
perc_pc_deep = num_pc_deep/num_cells_deep

# plot boxes
pd_place_cells = pd.DataFrame(data={'Condition':['Sup', 'Sup', 'Sup','Sup', 'Sup', 'Sup', 'Deep', 'Deep','Deep', 'Deep'], 
                                    'Session': ['Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot'],
                                    'place_cells':100*np.vstack((perc_pc_sup.reshape(-1,1), perc_pc_deep.reshape(-1,1))).T[0,:]})

# creating boxplot
plt.figure(figsize = (3.8, 5))
b = sns.boxplot(x='Condition', y='place_cells', hue='Session', data=pd_place_cells,
                palette='Set2', linewidth = 1, width= .5)
# adding data points
sns.stripplot(x='Condition', y='place_cells', hue='Session', data=pd_place_cells, 
              jitter = True, dodge = True, linewidth=1,palette ='Set2',edgecolor='gray')
# display plot
b.axes.set_title("Place cells",fontsize=16)
b.set_xlabel(" ",fontsize=15)
b.set_ylabel("%",fontsize=15)
b.set_yticks([0, 20, 40, 60, 80])
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)
b.tick_params(labelsize=12)
plt.tight_layout()
handles, labels = b.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2])
plt.show()

#%%Num place cells (general), num place cells direction-dependent vs non-direction dependent
num_pc_sup = np.zeros((3,2,2))
num_cells_sup = np.zeros((3,2,2))
num_pc_deep = np.zeros((2,2,2))
num_cells_deep = np.zeros((2,2,2))

for s_idx, s_name in enumerate(list(CZ3_place_cells_study.keys())):
    pd_struct = CZ3_place_cells_study[s_name]
    num_pc_sup[0,:,s_idx] = np.sum(pd_struct["place_cells_dir"], axis= 0)
    num_cells_sup[0,:,s_idx] =pd_struct["metric_val"].shape[0]

for s_idx, s_name in enumerate(list(CZ4_place_cells_study.keys())):
    pd_struct = CZ4_place_cells_study[s_name]
    num_pc_sup[1,:,s_idx] = np.sum(pd_struct["place_cells_dir"], axis= 0)
    num_cells_sup[1,:,s_idx] =pd_struct["metric_val"].shape[0]

for s_idx, s_name in enumerate(list(CZ6_place_cells_study.keys())):
    pd_struct = CZ6_place_cells_study[s_name]
    num_pc_sup[2,:,s_idx] = np.sum(pd_struct["place_cells_dir"], axis= 0)
    num_cells_sup[2,:,s_idx] =pd_struct["metric_val"].shape[0]

for s_idx, s_name in enumerate(list(GC2_place_cells_study.keys())):
    pd_struct = GC2_place_cells_study[s_name]
    num_pc_deep[0,:,s_idx] = np.sum(pd_struct["place_cells_dir"], axis= 0)
    num_cells_deep[0,:,s_idx] =pd_struct["metric_val"].shape[0]

for s_idx, s_name in enumerate(list(GC3_place_cells_study.keys())):
    pd_struct = GC3_place_cells_study[s_name]
    num_pc_deep[1,:,s_idx] = np.sum(pd_struct["place_cells_dir"], axis= 0)
    num_cells_deep[1,:,s_idx] =pd_struct["metric_val"].shape[0]

perc_pc_sup = num_pc_sup/num_cells_sup
perc_pc_deep = num_pc_deep/num_cells_deep

# plot boxes
pd_place_cells = pd.DataFrame(data={'Condition':['Sup', 'Sup', 'Sup','Sup', 'Sup', 'Sup','Sup', 'Sup', 'Sup','Sup', 
                                                 'Sup', 'Sup', 'Deep', 'Deep','Deep', 'Deep','Deep', 'Deep','Deep', 'Deep'], 
                                    
                                    'Session': ['Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot',
                                                'Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot', 'Pre', 'Rot'],
                                    
                                    'Dir': ['Left', 'Left', 'Right', 'Right','Left', 'Left', 'Right', 'Right','Left', 'Left',
                                            'Left', 'Left', 'Right', 'Right','Left', 'Left', 'Right', 'Right','Left', 'Left'],
                                    'place_cells':100*np.vstack((perc_pc_sup.reshape(-1,1), perc_pc_deep.reshape(-1,1))).T[0,:]})

# creating boxplot
plt.figure(figsize = (3.8, 5))
b = sns.boxplot(x='Condition', y='place_cells', hue='Session', data=pd_place_cells,
                palette='Set2', linewidth = 1, width= .5)
# adding data points
sns.stripplot(x='Condition', y='place_cells', hue='Session', data=pd_place_cells, 
              jitter = True, dodge = True, linewidth=1,palette ='Set2',edgecolor='gray')
# display plot
b.axes.set_title("Place cells",fontsize=16)
b.set_xlabel(" ",fontsize=15)
b.set_ylabel("%",fontsize=15)
b.set_yticks([0, 20, 40, 60, 80])
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)
b.tick_params(labelsize=12)
plt.tight_layout()
handles, labels = b.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2])
plt.show()

#%% Convertion rate (of the total number of place cells pre-rot how many are place cells in both - and check their location)
hm_pc_sup = np.zeros((12,2,2,3))
hm_pc_deep = np.zeros((12,2,2,2))

for s_idx, s_name in enumerate(list(CZ3_place_cells_study.keys())):
    pd_struct = CZ3_place_cells_study[s_name]
    hm_pc_sup[:,:,s_idx, 0] =np.mean(pd_struct["place_fields"],axis=1)

for s_idx, s_name in enumerate(list(CZ4_place_cells_study.keys())):
    pd_struct = CZ4_place_cells_study[s_name]
    hm_pc_sup[:,:,s_idx, 1] =np.mean(pd_struct["place_fields"],axis=1)

for s_idx, s_name in enumerate(list(CZ6_place_cells_study.keys())):
    pd_struct = CZ6_place_cells_study[s_name]
    if s_idx == 0:
        hm_pc_sup[1:-1,:,s_idx, 2] =np.mean(pd_struct["place_fields"],axis=1)
    else:
        hm_pc_sup[1:,:,s_idx, 2] =np.mean(pd_struct["place_fields"],axis=1)

for s_idx, s_name in enumerate(list(GC2_place_cells_study.keys())):
    pd_struct = GC2_place_cells_study[s_name]
    hm_pc_deep[:,:,s_idx, 0] =np.mean(pd_struct["place_fields"],axis=1)

for s_idx, s_name in enumerate(list(GC3_place_cells_study.keys())):
    pd_struct = GC3_place_cells_study[s_name]
    hm_pc_deep[:,:,s_idx, 1] =np.mean(pd_struct["place_fields"],axis=1)

#%%correlation of distribution of place fields along linear track (maybe divide by direction) before and after rotating
hm_pc_sup = np.zeros((12,4,3))*np.nan
hm_pc_deep = np.zeros((12,4,2))*np.nan


s_names = list(CZ3_place_cells_study.keys())
neu_pdf_pre = CZ3_place_cells_study[s_names[0]]["neu_pdf"][:, CZ3_place_cells_study[s_names[0]]["place_cells_idx"],:]
neu_pdf_rot = CZ3_place_cells_study[s_names[1]]["neu_pdf"][:, CZ3_place_cells_study[s_names[0]]["place_cells_idx"],:]
for pos in range(np.min([neu_pdf_pre.shape[0], neu_pdf_rot.shape[0]])):
    hm_pc_sup[pos,0,0] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[pos, :, 0])[0]
    hm_pc_sup[pos,1,0] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[pos, :, 1])[0]
    hm_pc_sup[pos,2,0] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[-pos, :, 0])[0]
    hm_pc_sup[pos,3,0] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[-pos, :, 1])[0]

s_names = list(CZ4_place_cells_study.keys())
neu_pdf_pre = CZ4_place_cells_study[s_names[0]]["neu_pdf"][:, CZ4_place_cells_study[s_names[0]]["place_cells_idx"],:]
neu_pdf_rot = CZ4_place_cells_study[s_names[1]]["neu_pdf"][:, CZ4_place_cells_study[s_names[0]]["place_cells_idx"],:]
for pos in range(np.min([neu_pdf_pre.shape[0], neu_pdf_rot.shape[0]])):
    hm_pc_sup[pos,0,1] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[pos, :, 0])[0]
    hm_pc_sup[pos,1,1] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[pos, :, 1])[0]
    hm_pc_sup[pos,2,1] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[-pos, :, 0])[0]
    hm_pc_sup[pos,3,1] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[-pos, :, 1])[0]

s_names = list(CZ6_place_cells_study.keys())
neu_pdf_pre = CZ6_place_cells_study[s_names[0]]["neu_pdf"][:, CZ6_place_cells_study[s_names[0]]["place_cells_idx"],:]
neu_pdf_rot = CZ6_place_cells_study[s_names[1]]["neu_pdf"][:, CZ6_place_cells_study[s_names[0]]["place_cells_idx"],:]
for pos in range(np.min([neu_pdf_pre.shape[0], neu_pdf_rot.shape[0]])):
    hm_pc_sup[pos,0,2] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[pos, :, 0])[0]
    hm_pc_sup[pos,1,2] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[pos, :, 1])[0]
    hm_pc_sup[pos,2,2] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[-pos, :, 0])[0]
    hm_pc_sup[pos,3,2] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[-pos, :, 1])[0]
    
s_names = list(GC2_place_cells_study.keys())
neu_pdf_pre = GC2_place_cells_study[s_names[0]]["neu_pdf"][:, GC2_place_cells_study[s_names[0]]["place_cells_idx"],:]
neu_pdf_rot = GC2_place_cells_study[s_names[1]]["neu_pdf"][:, GC2_place_cells_study[s_names[0]]["place_cells_idx"],:]
for pos in range(np.min([neu_pdf_pre.shape[0], neu_pdf_rot.shape[0]])):
    hm_pc_deep[pos,0,0] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[pos, :, 0])[0]
    hm_pc_deep[pos,1,0] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[pos, :, 1])[0]
    hm_pc_deep[pos,2,0] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[-pos, :, 0])[0]
    hm_pc_deep[pos,3,0] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[-pos, :, 1])[0]

s_names = list(GC3_place_cells_study.keys())
neu_pdf_pre = GC3_place_cells_study[s_names[0]]["neu_pdf"][:, GC3_place_cells_study[s_names[0]]["place_cells_idx"],:]
neu_pdf_rot = GC3_place_cells_study[s_names[1]]["neu_pdf"][:, GC3_place_cells_study[s_names[0]]["place_cells_idx"],:]
for pos in range(np.min([neu_pdf_pre.shape[0], neu_pdf_rot.shape[0]])):
    hm_pc_deep[pos,0,1] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[pos, :, 0])[0]
    hm_pc_deep[pos,1,1] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[pos, :, 1])[0]
    hm_pc_deep[pos,2,1] = pearsonr(neu_pdf_pre[pos,:,0], neu_pdf_rot[-pos, :, 0])[0]
    hm_pc_deep[pos,3,1] = pearsonr(neu_pdf_pre[pos,:,1], neu_pdf_rot[-pos, :, 1])[0]
    

plt.figure(figsize = (5,4))
x = np.linspace(0,3,4).astype(int)

m = np.nanmean(hm_pc_sup, axis=(0,2))
sd = np.nanstd(hm_pc_sup, axis=(0,2))
plt.plot(x,m, c= 'C4', label = 'sup')
plt.fill_between(x, m-sd, m+sd, alpha = 0.25, color= 'C4')


m = np.nanmean(hm_pc_deep, axis=(0,2))
sd = np.nanstd(hm_pc_deep, axis=(0,2))
plt.plot(x,m, c= 'C5', label = 'sup')
plt.fill_between(x, m-sd, m+sd, alpha = 0.25, color= 'C5')

plt.xticks(ticks= x, labels = ['Left', 'Right', 'Left inv', 'Right inv'])
for s_idx, s_name in enumerate(list(CZ3_place_cells_study.keys())):
    pd_struct = CZ3_place_cells_study[s_name]
    hm_pc_sup[:,:,s_idx, 0] =np.mean(pd_struct["place_fields"],axis=1)

for s_idx, s_name in enumerate(list(CZ4_place_cells_study.keys())):
    pd_struct = CZ4_place_cells_study[s_name]
    hm_pc_sup[:,:,s_idx, 1] =np.mean(pd_struct["place_fields"],axis=1)

for s_idx, s_name in enumerate(list(CZ6_place_cells_study.keys())):
    pd_struct = CZ6_place_cells_study[s_name]
    if s_idx == 0:
        hm_pc_sup[1:-1,:,s_idx, 2] =np.mean(pd_struct["place_fields"],axis=1)
    else:
        hm_pc_sup[1:,:,s_idx, 2] =np.mean(pd_struct["place_fields"],axis=1)

for s_idx, s_name in enumerate(list(GC2_place_cells_study.keys())):
    pd_struct = GC2_place_cells_study[s_name]
    hm_pc_deep[:,:,s_idx, 0] =np.mean(pd_struct["place_fields"],axis=1)

for s_idx, s_name in enumerate(list(GC3_place_cells_study.keys())):
    pd_struct = GC3_place_cells_study[s_name]
    hm_pc_deep[:,:,s_idx, 1] =np.mean(pd_struct["place_fields"],axis=1)
    
    
    
vmax = 0.5
vmin = 0

fig = plt.figure(figsize=(9,5))
ax = plt.subplot(2,5,1)
ax.matshow(hm_pc_sup[:,:,0,0], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,6)
ax.matshow(hm_pc_sup[:,:,1,0], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,2)
ax.matshow(hm_pc_sup[:,:,0,1], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,7)
ax.matshow(hm_pc_sup[:,:,1,1], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,3)
ax.matshow(hm_pc_sup[:,:,0,2], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,8)
ax.matshow(hm_pc_sup[:,:,1,2], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,4)
ax.matshow(hm_pc_deep[:,:,0,0], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,9)
ax.matshow(hm_pc_deep[:,:,1,0], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,5)
ax.matshow(hm_pc_deep[:,:,0,1], vmax = vmax, vmin = vmin)

ax = plt.subplot(2,5,10)
ax.matshow(hm_pc_deep[:,:,1,1], vmax = vmax, vmin = vmin)

#study manifold continuity as we include o remove place cells

#check decoders on place cells vs all, and check for biases in prediction (probability of predicting position at cues is
#larger than probability animal is there. )

#field properties (size, number of fields)

