#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:08:20 2022

@author: julio
"""
#%% IMPORTS
import copy, os
from neural_manifold import general_utils as gu
import seaborn as sns
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
#%% PARAMS
signal_field = 'revents_SNR3'
save_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/event_rates_study'

kernel_std = 0.25
assymetry = True
events_rates = list()
#%% CZ3
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ3'
CZ3 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ3.keys())
CZ3_p= copy.deepcopy(CZ3[fnames[0]])
CZ3_r= copy.deepcopy(CZ3[fnames[1]])

CZ3_p = gu.add_firing_rates(CZ3_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ3_r = gu.add_firing_rates(CZ3_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ3_signal_p = copy.deepcopy(np.concatenate(CZ3_p[signal_field].values, axis=0))
CZ3_signal_r = copy.deepcopy(np.concatenate(CZ3_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(CZ3_signal_p,axis=0),np.mean(CZ3_signal_r,axis=0)),axis=1))

#%% CZ4
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ4'
CZ4 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ4.keys())
CZ4_p= copy.deepcopy(CZ4[fnames[0]])
CZ4_r= copy.deepcopy(CZ4[fnames[1]])

CZ4_p = gu.add_firing_rates(CZ4_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ4_r = gu.add_firing_rates(CZ4_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ4_signal_p = copy.deepcopy(np.concatenate(CZ4_p[signal_field].values, axis=0))
CZ4_signal_r = copy.deepcopy(np.concatenate(CZ4_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(CZ4_signal_p,axis=0),np.mean(CZ4_signal_r,axis=0)),axis=1))

#%% CZ6
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ6'
CZ6 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ6.keys())
CZ6_p= copy.deepcopy(CZ6[fnames[0]])
CZ6_r= copy.deepcopy(CZ6[fnames[1]])

CZ6_p = gu.add_firing_rates(CZ6_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ6_r = gu.add_firing_rates(CZ6_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ6_signal_p = copy.deepcopy(np.concatenate(CZ6_p[signal_field].values, axis=0))
CZ6_signal_r = copy.deepcopy(np.concatenate(CZ6_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(CZ6_signal_p,axis=0),np.mean(CZ6_signal_r,axis=0)),axis=1))

#%% CZ7
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ7'
CZ7 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ7.keys())
CZ7_p= copy.deepcopy(CZ7[fnames[0]])
CZ7_r= copy.deepcopy(CZ7[fnames[1]])

CZ7_p = gu.add_firing_rates(CZ7_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ7_r = gu.add_firing_rates(CZ7_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ7_signal_p = copy.deepcopy(np.concatenate(CZ7_p[signal_field].values, axis=0))
CZ7_signal_r = copy.deepcopy(np.concatenate(CZ7_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(CZ7_signal_p,axis=0),np.mean(CZ7_signal_r,axis=0)),axis=1))

#%% CZ8
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ8'
CZ8 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ8.keys())
CZ8_p= copy.deepcopy(CZ8[fnames[0]])
CZ8_r= copy.deepcopy(CZ8[fnames[1]])

CZ8_p = gu.add_firing_rates(CZ8_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ8_r = gu.add_firing_rates(CZ8_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ8_signal_p = copy.deepcopy(np.concatenate(CZ8_p[signal_field].values, axis=0))
CZ8_signal_r = copy.deepcopy(np.concatenate(CZ8_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(CZ8_signal_p,axis=0),np.mean(CZ8_signal_r,axis=0)),axis=1))

#%% CZ9
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ9'
CZ9 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ9.keys())
CZ9_p= copy.deepcopy(CZ9[fnames[0]])
CZ9_r= copy.deepcopy(CZ9[fnames[1]])

CZ9_p = gu.add_firing_rates(CZ9_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ9_r = gu.add_firing_rates(CZ9_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ9_signal_p = copy.deepcopy(np.concatenate(CZ9_p[signal_field].values, axis=0))
CZ9_signal_r = copy.deepcopy(np.concatenate(CZ9_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(CZ9_signal_p,axis=0),np.mean(CZ9_signal_r,axis=0)),axis=1))

#%% GC2
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/GC2'
GC2 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(GC2.keys())
GC2_p= copy.deepcopy(GC2[fnames[0]])
GC2_r= copy.deepcopy(GC2[fnames[1]])

GC2_p = gu.add_firing_rates(GC2_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
GC2_r = gu.add_firing_rates(GC2_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

GC2_signal_p = copy.deepcopy(np.concatenate(GC2_p[signal_field].values, axis=0))
GC2_signal_r = copy.deepcopy(np.concatenate(GC2_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(GC2_signal_p,axis=0),np.mean(GC2_signal_r,axis=0)),axis=1))

#%% GC3
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/GC3'
GC3 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(GC3.keys())
GC3_p= copy.deepcopy(GC3[fnames[0]])
GC3_r= copy.deepcopy(GC3[fnames[1]])

GC3_p = gu.add_firing_rates(GC3_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
GC3_r = gu.add_firing_rates(GC3_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

GC3_signal_p = copy.deepcopy(np.concatenate(GC3_p[signal_field].values, axis=0))
GC3_signal_r = copy.deepcopy(np.concatenate(GC3_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(GC3_signal_p,axis=0),np.mean(GC3_signal_r,axis=0)),axis=1))

#%% DDC
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/DDC'
DDC = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(DDC.keys())
DDC_p= copy.deepcopy(DDC[fnames[0]])
DDC_r= copy.deepcopy(DDC[fnames[1]])

DDC_p = gu.add_firing_rates(DDC_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
DDC_r = gu.add_firing_rates(DDC_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

DDC_signal_p = copy.deepcopy(np.concatenate(DDC_p[signal_field].values, axis=0))
DDC_signal_r = copy.deepcopy(np.concatenate(DDC_r[signal_field].values, axis=0))
events_rates.append(np.stack((np.mean(DDC_signal_p,axis=0),np.mean(DDC_signal_r,axis=0)),axis=1))

#%% GROUP DATA
sup_rates = np.concatenate(events_rates[:6], axis = 0)
deep_rates = np.concatenate(events_rates[7:], axis = 0)

pref_index_sup = (sup_rates[:,0]/np.median(sup_rates[:,0])) - (sup_rates[:,1]/np.median(sup_rates[:,1]))
pref_index_deep = (deep_rates[:,0]/np.mean(deep_rates[:,0])) - (deep_rates[:,1]/np.mean(deep_rates[:,1]))

sup_rates_pd = sup_rates.T.flatten()
deep_rates_pd = deep_rates.T.flatten()

sup_label_pd = ['sup']*sup_rates_pd.shape[0]
deep_label_pd = ['deep']*deep_rates_pd.shape[0]

sup_pre_label_pd = ['pre']*sup_rates.shape[0]
sup_rot_label_pd = ['rot']*sup_rates.shape[0]
deep_pre_label_pd = ['pre']*deep_rates.shape[0]
deep_rot_label_pd = ['rot']*deep_rates.shape[0]

rates_struct = pd.DataFrame(data={'rates':np.hstack((sup_rates_pd, deep_rates_pd)), 
                                  'cell_type':np.hstack((sup_label_pd, deep_label_pd)),
                                  'session_type':np.hstack((sup_pre_label_pd, sup_rot_label_pd,deep_pre_label_pd,deep_rot_label_pd))})

rates_struct_v2 = pd.DataFrame(data={'rates_pre':np.hstack((sup_rates[:,0], deep_rates[:,0])), 
                                     'rates_post':np.hstack((sup_rates[:,1], deep_rates[:,1])),
                                     'turn_index': np.hstack((pref_index_sup, pref_index_deep)),
                                     'cell_type': np.hstack((['sup']*sup_rates.shape[0],['deep']*deep_rates.shape[0]))})

#%% PLOT
# Update default settings
sns.set(style='whitegrid', context='talk', 
        palette=['#62C370', '#C360B4', '#6083C3', '#C3A060'])

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.kdeplot(x='rates', data=rates_struct,hue = 'cell_type', shade=True, 
            common_norm=False, clip = [0, None],  common_grid = True, ax=ax[0])  
ax[0].set_title("Event rates")
ax[0].set_xlim([-0.02, 0.602])
ax[0].set_xlabel("Event rates")

sns.kdeplot(x='turn_index', data=rates_struct_v2,hue = 'cell_type', shade=True, 
            common_norm=False,  common_grid = True, ax=ax[1])  
ax[1].set_title("Turning index")
ax[1].set_xlim([-5.2, 5.2])
ax[1].set_xlabel('Turning index')

plt.savefig(os.path.join(save_dir,'LT_DeepSup_event_rates_hist.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'LT_DeepSup_event_rates_hist.png'), dpi = 400,bbox_inches="tight")



