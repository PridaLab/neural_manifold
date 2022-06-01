#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:03:16 2022

@author: julio
"""

#%%
import numpy as np
from neural_manifold import general_utils as gu
from neural_manifold import dimensionality_reduction as dim_red
import copy
import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import umap

#%% GENERAL PARAMS
save_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/rotation_study'
signal_field = 'revents_SNR3'

#%% CZ3
mouse = 'CZ3'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/' + mouse
CZ3 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ3.keys())
CZ3_p= copy.deepcopy(CZ3[fnames[0]])
CZ3_r= copy.deepcopy(CZ3[fnames[1]])

CZ3_p = gu.add_firing_rates(CZ3_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ3_r = gu.add_firing_rates(CZ3_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ3_angle, CZ3_sI_nn, CZ3_params = dim_red.check_rotation_params(CZ3_p, CZ3_r, signal_field,save_dir, verbose = True)

CZ3_rotation_study = {
        'angle': CZ3_angle,
        'sI': CZ3_sI_nn,
        'params': CZ3_params
        }

save_rs = open(os.path.join(save_dir, "CZ3_rotation_study_dict.pkl"), "wb")
pickle.dump(CZ3_rotation_study, save_rs)
save_rs.close()

#%% CZ4
mouse = 'CZ4'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/' + mouse
CZ4 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ4.keys())
CZ4_p= copy.deepcopy(CZ4[fnames[0]])
CZ4_r= copy.deepcopy(CZ4[fnames[1]])

CZ4_p = gu.add_firing_rates(CZ4_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ4_r = gu.add_firing_rates(CZ4_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ4_angle, CZ4_sI_nn, CZ4_params = dim_red.check_rotation_params(CZ4_p, CZ4_r, signal_field,save_dir, verbose = True)

CZ4_rotation_study = {
        'angle': CZ4_angle,
        'sI': CZ4_sI_nn,
        'params': CZ4_params
        }

save_rs = open(os.path.join(save_dir, "CZ4_rotation_study_dict.pkl"), "wb")
pickle.dump(CZ4_rotation_study, save_rs)
save_rs.close()

#%% CZ6
mouse = 'CZ6'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/' + mouse
CZ6 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ6.keys())
CZ6_p= copy.deepcopy(CZ6[fnames[0]])
CZ6_r= copy.deepcopy(CZ6[fnames[1]])

CZ6_p = gu.add_firing_rates(CZ6_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ6_r = gu.add_firing_rates(CZ6_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ6_angle, CZ6_sI_nn, CZ6_params = dim_red.check_rotation_params(CZ6_p, CZ6_r, signal_field,save_dir, verbose = True)

CZ6_rotation_study = {
        'angle': CZ6_angle,
        'sI': CZ6_sI_nn,
        'params': CZ6_params
        }

save_rs = open(os.path.join(save_dir, "CZ6_rotation_study_dict.pkl"), "wb")
pickle.dump(CZ6_rotation_study, save_rs)
save_rs.close()

#%% GC2
mouse = 'GC2'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/' + mouse
GC2 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(GC2.keys())
GC2_p= copy.deepcopy(GC2[fnames[0]])
GC2_r= copy.deepcopy(GC2[fnames[1]])

GC2_p = gu.add_firing_rates(GC2_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
GC2_r = gu.add_firing_rates(GC2_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
GC2_angle, GC2_sI_nn, GC2_params = dim_red.check_rotation_params(GC2_p, GC2_r, signal_field,save_dir, verbose = True)

GC2_rotation_study = {
        'angle': GC2_angle,
        'sI': GC2_sI_nn,
        'params': GC2_params
        }

save_rs = open(os.path.join(save_dir, "GC2_rotation_study_dict.pkl"), "wb")
pickle.dump(GC2_rotation_study, save_rs)
save_rs.close()

#%% GC3
mouse = 'GC3'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/' + mouse
GC3 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(GC3.keys())
GC3_p= copy.deepcopy(GC3[fnames[0]])
GC3_r= copy.deepcopy(GC3[fnames[1]])

GC3_p = gu.add_firing_rates(GC3_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
GC3_r = gu.add_firing_rates(GC3_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
GC3_angle, GC3_sI_nn, GC3_params = dim_red.check_rotation_params(GC3_p, GC3_r, signal_field,save_dir, verbose = True)

GC3_rotation_study = {
        'angle': GC3_angle,
        'sI': GC3_sI_nn,
        'params': GC3_params
        }

save_rs = open(os.path.join(save_dir, "GC3_rotation_study_dict.pkl"), "wb")
pickle.dump(GC3_rotation_study, save_rs)
save_rs.close()

#%% Load
if "CZ3_rotation_study" not in locals():
    CZ3_rotation_study = gu.load_files(save_dir, '*CZ3_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "CZ4_rotation_study" not in locals():
    CZ4_rotation_study = gu.load_files(save_dir, '*CZ4_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "CZ6_rotation_study" not in locals():
    CZ6_rotation_study = gu.load_files(save_dir, '*CZ6_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "GC2_rotation_study" not in locals():
    GC2_rotation_study = gu.load_files(save_dir, '*GC2_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "GC3_rotation_study" not in locals():
    GC3_rotation_study = gu.load_files(save_dir, '*GC3_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
    
#%% PLOT BOXES ONLY
nn = 60
idx = [i for i in range(len(CZ3_rotation_study['params']['nn_list'])) if CZ3_rotation_study['params']['nn_list'][i] ==nn]
idx = idx[0]

sup_angle = np.array([CZ3_rotation_study['angle'][idx, 0],CZ4_rotation_study['angle'][idx, 0],CZ6_rotation_study['angle'][idx, 0]])
deep_angle = np.array([GC2_rotation_study['angle'][idx, 0],GC3_rotation_study['angle'][idx, 0]])
pd_angles = pd.DataFrame(data={'Condition':['Sup', 'Sup', 'Sup', 'Deep', 'Deep'], 'angle':np.abs(np.hstack((sup_angle, deep_angle)))})

# creating boxplot
plt.figure(figsize = (3.8, 5))
b = sns.boxplot(x='Condition', y='angle', data=pd_angles, linewidth = 2, width= .5)
# adding data points
sns.stripplot(x='Condition', y='angle', data=pd_angles, color="black")
# display plot
b.axes.set_title(f"Rotation Angle (nn: {nn})",fontsize=16)
b.set_xlabel(" ",fontsize=15)
b.set_ylabel("Degrees (º)",fontsize=15)
b.set_yticks([0, 45, 90, 135, 180])
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)
b.tick_params(labelsize=12)
plt.tight_layout()
plt.show()

#%% PREPARE PLOT EMBEDDING EXAMPLES + BOXES
if "GC2" not in  locals():
    mouse = 'GC2'
    load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/' + mouse
    GC2 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

    fnames = list(GC2.keys())
    GC2_p= copy.deepcopy(GC2[fnames[0]])
    GC2_r= copy.deepcopy(GC2[fnames[1]])

    GC2_p = gu.add_firing_rates(GC2_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
    GC2_r = gu.add_firing_rates(GC2_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)

if "CZ3" not in locals():
    mouse = 'CZ3'
    load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/' + mouse
    CZ3 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

    fnames = list(CZ3.keys())
    CZ3_p= copy.deepcopy(CZ3[fnames[0]])
    CZ3_r= copy.deepcopy(CZ3[fnames[1]])

    CZ3_p = gu.add_firing_rates(CZ3_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
    CZ3_r = gu.add_firing_rates(CZ3_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
    
GC2_signal_p = copy.deepcopy(np.concatenate(GC2_p[signal_field].values, axis=0))
GC2_pos_p = copy.deepcopy(np.concatenate(GC2_p['pos'].values, axis=0))
GC2_signal_r = copy.deepcopy(np.concatenate(GC2_r[signal_field].values, axis=0))
GC2_pos_r = copy.deepcopy(np.concatenate(GC2_r['pos'].values, axis=0))
GC2_concat_signal = np.vstack((GC2_signal_p, GC2_signal_r))
concat_index = np.vstack((np.zeros((GC2_signal_p.shape[0],1)),np.zeros((GC2_signal_r.shape[0],1))+1))

model = umap.UMAP(n_neighbors=nn, n_components = 3, min_dist=0.75)
concat_emb = model.fit_transform(GC2_concat_signal)
GC2_emb_p = concat_emb[concat_index[:,0]==0,:]
GC2_emb_r = concat_emb[concat_index[:,0]==1,:]


CZ3_signal_p = copy.deepcopy(np.concatenate(CZ3_p[signal_field].values, axis=0))
CZ3_pos_p = copy.deepcopy(np.concatenate(CZ3_p['pos'].values, axis=0))
CZ3_signal_r = copy.deepcopy(np.concatenate(CZ3_r[signal_field].values, axis=0))
CZ3_pos_r = copy.deepcopy(np.concatenate(CZ3_r['pos'].values, axis=0))
CZ3_concat_signal = np.vstack((CZ3_signal_p, CZ3_signal_r))
concat_index = np.vstack((np.zeros((CZ3_signal_p.shape[0],1)),np.zeros((CZ3_signal_r.shape[0],1))+1))

model = umap.UMAP(n_neighbors=nn, n_components = 3, min_dist=0.75)
concat_emb = model.fit_transform(CZ3_concat_signal)
CZ3_emb_p = concat_emb[concat_index[:,0]==0,:]
CZ3_emb_r = concat_emb[concat_index[:,0]==1,:]

#%% PLOT EMBEDDINGS + BOXES

fig = plt.figure(figsize=(9,5))
ax = plt.subplot(2,3,1, projection = '3d')
ax.set_title("Before rotating",fontsize=15)
ax.scatter(*CZ3_emb_p.T, c= CZ3_pos_p[:,0]/np.max(CZ3_pos_p[:,0]), cmap = plt.cm.magma)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 1', labelpad = -8)
ax.set_zlabel('Dim 1', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax = plt.subplot(2,3,2, projection = '3d')
ax.set_title("After rotating", fontsize=15)
ax.scatter(*CZ3_emb_r.T, c= CZ3_pos_r[:,0]/np.max(CZ3_pos_r[:,0]), cmap = plt.cm.magma)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 1', labelpad = -8)
ax.set_zlabel('Dim 1', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*GC2_emb_p.T, c= GC2_pos_p[:,0]/np.max(GC2_pos_p[:,0]), cmap = plt.cm.magma)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 1', labelpad = -8)
ax.set_zlabel('Dim 1', labelpad = -8)
fig.text(0.1, 0.75, 'Superficial',horizontalalignment='center', 
                 rotation = 'vertical', verticalalignment='center', fontsize = 15)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*GC2_emb_r.T, c= GC2_pos_r[:,0]/np.max(GC2_pos_r[:,0]), cmap = plt.cm.magma)
fig.text(0.1, 0.25, 'Deep',horizontalalignment='center', 
                 rotation = 'vertical', verticalalignment='center', fontsize = 15)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 1', labelpad = -8)
ax.set_zlabel('Dim 1', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


ax = plt.subplot(1,10,(9,10))
b = sns.boxplot(x='Condition', y='angle', data=pd_angles, linewidth = 2, width= .5, ax = ax)
# adding data points
sns.stripplot(x='Condition', y='angle', data=pd_angles, color="black", ax = ax)
# display plot
b.axes.set_title(f"Rotation Angle (nn: {nn})",fontsize=16)
b.set_xlabel(" ",fontsize=15)
b.set_ylabel("Degrees (º)",fontsize=12)
b.set_yticks([0, 45, 90, 135, 180])
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)
b.tick_params(labelsize=12)
plt.tight_layout()
plt.show()


