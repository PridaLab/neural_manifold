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
dim_deep = 3
dim_sup = 3
palette=['#62C370', '#C360B4', '#6083C3', '#C3A060']
#%% CZ3
mouse = 'CZ3'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
CZ3 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ3.keys())
CZ3_p= copy.deepcopy(CZ3[fnames[0]])
CZ3_r= copy.deepcopy(CZ3[fnames[1]])

CZ3_p = gu.add_firing_rates(CZ3_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ3_r = gu.add_firing_rates(CZ3_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ3_angle, CZ3_sI_nn, CZ3_params = dim_red.check_rotation_params(CZ3_p, CZ3_r, signal_field, save_dir, dim=dim_sup, verbose = True)

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
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
CZ4 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ4.keys())
CZ4_p= copy.deepcopy(CZ4[fnames[0]])
CZ4_r= copy.deepcopy(CZ4[fnames[1]])

CZ4_p = gu.add_firing_rates(CZ4_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ4_r = gu.add_firing_rates(CZ4_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ4_angle, CZ4_sI_nn, CZ4_params = dim_red.check_rotation_params(CZ4_p, CZ4_r, signal_field, save_dir, dim=dim_sup, verbose = True)

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
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
CZ6 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ6.keys())
CZ6_p= copy.deepcopy(CZ6[fnames[0]])
CZ6_r= copy.deepcopy(CZ6[fnames[1]])

CZ6_p = gu.add_firing_rates(CZ6_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ6_r = gu.add_firing_rates(CZ6_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ6_angle, CZ6_sI_nn, CZ6_params = dim_red.check_rotation_params(CZ6_p, CZ6_r, signal_field, save_dir, dim=dim_sup, verbose = True)

CZ6_rotation_study = {
        'angle': CZ6_angle,
        'sI': CZ6_sI_nn,
        'params': CZ6_params
        }

save_rs = open(os.path.join(save_dir, "CZ6_rotation_study_dict.pkl"), "wb")
pickle.dump(CZ6_rotation_study, save_rs)
save_rs.close()

#%% CZ7
mouse = 'CZ7'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
CZ7 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ7.keys())
CZ7_p= copy.deepcopy(CZ7[fnames[0]])
CZ7_r= copy.deepcopy(CZ7[fnames[1]])

CZ7_p = gu.add_firing_rates(CZ7_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ7_r = gu.add_firing_rates(CZ7_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ7_angle, CZ7_sI_nn, CZ7_params = dim_red.check_rotation_params(CZ7_p, CZ7_r, signal_field, save_dir, dim=dim_sup, verbose = True)

CZ7_rotation_study = {
        'angle': CZ7_angle,
        'sI': CZ7_sI_nn,
        'params': CZ7_params
        }

save_rs = open(os.path.join(save_dir, "CZ7_rotation_study_dict.pkl"), "wb")
pickle.dump(CZ7_rotation_study, save_rs)
save_rs.close()

#%% CZ8
mouse = 'CZ8'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
CZ8 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ8.keys())
CZ8_p= copy.deepcopy(CZ8[fnames[0]])
CZ8_r= copy.deepcopy(CZ8[fnames[1]])

CZ8_p = gu.add_firing_rates(CZ8_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ8_r = gu.add_firing_rates(CZ8_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ8_angle, CZ8_sI_nn, CZ8_params = dim_red.check_rotation_params(CZ8_p, CZ8_r, signal_field, save_dir, dim=dim_sup, verbose = True)

CZ8_rotation_study = {
        'angle': CZ8_angle,
        'sI': CZ8_sI_nn,
        'params': CZ8_params
        }

save_rs = open(os.path.join(save_dir, "CZ8_rotation_study_dict.pkl"), "wb")
pickle.dump(CZ8_rotation_study, save_rs)
save_rs.close()

#%% CZ9
mouse = 'CZ9'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
CZ9 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(CZ9.keys())
CZ9_p= copy.deepcopy(CZ9[fnames[0]])
CZ9_r= copy.deepcopy(CZ9[fnames[1]])

CZ9_p = gu.add_firing_rates(CZ9_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ9_r = gu.add_firing_rates(CZ9_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
CZ9_angle, CZ9_sI_nn, CZ9_params = dim_red.check_rotation_params(CZ9_p, CZ9_r, signal_field, save_dir, dim=dim_sup, verbose = True)

CZ9_rotation_study = {
        'angle': CZ9_angle,
        'sI': CZ9_sI_nn,
        'params': CZ9_params
        }

save_rs = open(os.path.join(save_dir, "CZ9_rotation_study_dict.pkl"), "wb")
pickle.dump(CZ9_rotation_study, save_rs)
save_rs.close()

#%% GC2
mouse = 'GC2'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
GC2 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(GC2.keys())
GC2_p= copy.deepcopy(GC2[fnames[0]])
GC2_r= copy.deepcopy(GC2[fnames[1]])

GC2_p = gu.add_firing_rates(GC2_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
GC2_r = gu.add_firing_rates(GC2_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
GC2_angle, GC2_sI_nn, GC2_params = dim_red.check_rotation_params(GC2_p, GC2_r, signal_field, save_dir, dim=dim_deep, verbose = True)

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
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
GC3 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(GC3.keys())
GC3_p= copy.deepcopy(GC3[fnames[0]])
GC3_r= copy.deepcopy(GC3[fnames[1]])

GC3_p = gu.add_firing_rates(GC3_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
GC3_r = gu.add_firing_rates(GC3_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
GC3_angle, GC3_sI_nn, GC3_params = dim_red.check_rotation_params(GC3_p, GC3_r, signal_field, save_dir, dim=dim_deep, verbose = True)

GC3_rotation_study = {
        'angle': GC3_angle,
        'sI': GC3_sI_nn,
        'params': GC3_params
        }

save_rs = open(os.path.join(save_dir, "GC3_rotation_study_dict.pkl"), "wb")
pickle.dump(GC3_rotation_study, save_rs)
save_rs.close()

#%% DDC
mouse = 'DDC'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
DDC = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

fnames = list(DDC.keys())
DDC_p= copy.deepcopy(DDC[fnames[0]])
DDC_r= copy.deepcopy(DDC[fnames[1]])

DDC_p = gu.add_firing_rates(DDC_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
DDC_r = gu.add_firing_rates(DDC_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
DDC_angle, DDC_sI_nn, DDC_params = dim_red.check_rotation_params(DDC_p, DDC_r, signal_field, save_dir, dim=dim_deep, verbose = True)

DDC_rotation_study = {
        'angle': DDC_angle,
        'sI': DDC_sI_nn,
        'params': DDC_params
        }

save_rs = open(os.path.join(save_dir, "DDC_rotation_study_dict.pkl"), "wb")
pickle.dump(DDC_rotation_study, save_rs)
save_rs.close()

#%% Load
save_dir2 = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/old/rotation_study_old/rotation_study'
if "CZ3_rotation_study" not in locals():
    CZ3_rotation_study = gu.load_files(save_dir, '*CZ3_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "CZ4_rotation_study" not in locals():
    CZ4_rotation_study = gu.load_files(save_dir, '*CZ4_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "CZ6_rotation_study" not in locals():
    CZ6_rotation_study = gu.load_files(save_dir, '*CZ6_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "CZ7_rotation_study" not in locals():
    CZ7_rotation_study = gu.load_files(save_dir, '*CZ7_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "CZ8_rotation_study" not in locals():
    CZ8_rotation_study = gu.load_files(save_dir, '*CZ8_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "CZ9_rotation_study" not in locals():
    CZ9_rotation_study = gu.load_files(save_dir, '*CZ9_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "GC2_rotation_study" not in locals():
    GC2_rotation_study = gu.load_files(save_dir2, '*GC2_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "GC3_rotation_study" not in locals():
    GC3_rotation_study = gu.load_files(save_dir2, '*GC3_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")
if "DDC_rotation_study" not in locals():
    DDC_rotation_study = gu.load_files(save_dir2, '*DDC_rotation_study_dict.pkl', verbose=True, struct_type = "pickle")  

#%% PLOT BOXES ONLY

nn = 60
idx = [i for i in range(len(CZ3_rotation_study['params']['nn_list'])) if CZ3_rotation_study['params']['nn_list'][i] ==nn]
idx = idx[0]

idx = list()
idx.append(np.argmax(np.mean(CZ3_rotation_study['sI'], axis=(1,2,3))))
idx.append(np.argmax(np.mean(CZ4_rotation_study['sI'], axis=(1,2,3))))
idx.append(np.argmax(np.mean(CZ6_rotation_study['sI'], axis=(1,2,3))))
idx.append(np.argmax(np.mean(CZ7_rotation_study['sI'], axis=(1,2,3))))
idx.append(np.argmax(np.mean(CZ8_rotation_study['sI'], axis=(1,2,3))))
idx.append(np.argmax(np.mean(CZ9_rotation_study['sI'], axis=(1,2,3))))


sup_angle = np.array([np.mean(CZ3_rotation_study['angle'][idx[0],3,:]),
                      #np.mean(CZ4_rotation_study['angle'][idx[1],3,:]),
                      np.mean(CZ6_rotation_study['angle'][idx[2],3,:]),
                      #np.mean(CZ7_rotation_study['angle'][idx[3],3,:]),
                      np.mean(CZ8_rotation_study['angle'][idx[4],3,:]),
                      np.mean(CZ9_rotation_study['angle'][idx[5],3,:])
                      ])
idx = list()
idx.append(np.argmax(np.mean(GC2_rotation_study['sI'], axis=(1,2))))
idx.append(np.argmax(np.mean(GC3_rotation_study['sI'], axis=(1,2))))
idx.append(np.argmax(np.mean(DDC_rotation_study['sI'], axis=(1,2))))

nn = 60
idx = [i for i in range(len(CZ3_rotation_study['params']['nn_list'])) if CZ3_rotation_study['params']['nn_list'][i] ==nn]
idx = idx[0]

deep_angle = np.array([GC2_rotation_study['angle'][idx, 0],
                       GC3_rotation_study['angle'][idx, 0],
                       DDC_rotation_study['angle'][idx, 0]])


pd_angles = pd.DataFrame(data={'Condition':['Sup']*4 + ['Deep']*3, 
                               'angle':np.abs(np.hstack((sup_angle, deep_angle)))})
#%% SNS PARAMS
sns.set(style='whitegrid', context='talk', 
        palette=['#62C370', '#C360B4', '#6083C3', '#C3A060'])
palette=['#62C370', '#C360B4', '#6083C3', '#C3A060']

#%%
# creating boxplot
plt.figure(figsize = (4, 6))
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
#%%
plt.savefig(os.path.join(save_dir,'LT_DeepSup_rotation_boxplot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'LT_DeepSup_rotation_boxplot.png'), dpi = 400,bbox_inches="tight")


#%% PREPARE PLOT EMBEDDING EXAMPLES + BOXES
if "GC2" not in  locals():
    mouse = 'GC2'
    load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
    GC2 = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")

    fnames = list(GC2.keys())
    GC2_p= copy.deepcopy(GC2[fnames[0]])
    GC2_r= copy.deepcopy(GC2[fnames[1]])

    GC2_p = gu.add_firing_rates(GC2_p, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
    GC2_r = gu.add_firing_rates(GC2_r, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)

if "CZ3" not in locals():
    mouse = 'CZ3'
    load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
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
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax = plt.subplot(2,3,2, projection = '3d')
ax.set_title("After rotating", fontsize=15)
ax.scatter(*CZ3_emb_r.T, c= CZ3_pos_r[:,0]/np.max(CZ3_pos_r[:,0]), cmap = plt.cm.magma)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*GC2_emb_p.T, c= GC2_pos_p[:,0]/np.max(GC2_pos_p[:,0]), cmap = plt.cm.magma)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
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
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


ax = plt.subplot(1,10,(9,10))
b = sns.boxplot(x='Condition', y='angle', data=pd_angles, linewidth = 2, width= .5, ax = ax)
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
#%%

plt.savefig(os.path.join(save_dir,'LT_DeepSup_rotation_all.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'LT_DeepSup_rotation_all.png'), dpi = 400,bbox_inches="tight")

#%% PLOT SI
sI_sup = np.zeros((2,2,6))
sI_sup[:,:,0] = np.max(np.mean(CZ3_rotation_study['sI'][:,:,[0,3],:], axis=3), axis = 0)
sI_sup[:,:,1] = np.max(np.mean(CZ4_rotation_study['sI'][:,:,[0,3],:], axis=3), axis = 0)
sI_sup[:,:,2] = np.max(np.mean(CZ6_rotation_study['sI'][:,:,[0,3],:], axis=3), axis = 0)
sI_sup[:,:,3] = np.max(np.mean(CZ7_rotation_study['sI'][:,:,[0,3],:], axis=3), axis = 0)
sI_sup[:,:,4] = np.max(np.mean(CZ8_rotation_study['sI'][:,:,[0,3],:], axis=3), axis = 0)
sI_sup[:,:,5] = np.max(np.mean(CZ9_rotation_study['sI'][:,:,[0,3],:], axis=3), axis = 0)

sI_deep = np.zeros((2,2,3))
sI_deep[:,:,0] = np.max(GC2_rotation_study['sI'][:,:,[0,3]], axis = 0)
sI_deep[:,:,1] = np.max(GC3_rotation_study['sI'][:,:,[0,3]], axis = 0)
sI_deep[:,:,2] = np.max(DDC_rotation_study['sI'][:,:,[0,3]], axis = 0)

fig = plt.figure(figsize=(9,5))


ax = plt.subplot(1,2,1)
m = np.mean(sI_sup[:,0,:], axis=1)
sd = np.std(sI_sup[:,0,:], axis=1)/np.sqrt(sI_sup.shape[2])
ax.plot(m, c= palette[0], label = 'Sup')
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= palette[0], alpha = 0.3)

m = np.mean(sI_deep[:,0,:], axis=1)
sd = np.std(sI_deep[:,0,:], axis=1)/np.sqrt(sI_deep.shape[2])
ax.plot(m, c= palette[1], label = 'Deep')
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= palette[1], alpha = 0.3)


ax.set_xticks(np.arange(len(m)), labels=['pre', 'rot'])
ax.set_ylim([0.5,1])
ax.set_xlim([-0.2, 1.2])

ax.set_ylabel('sI xpos', labelpad = 5)
ax.set_title('local')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = plt.subplot(1,2,2)
m = np.mean(sI_sup[:,1,:], axis=1)
sd = np.std(sI_sup[:,1,:], axis=1)/np.sqrt(sI_sup.shape[2])
ax.plot(m, c= palette[0], label = 'Sup')
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= palette[0], alpha = 0.3)

m = np.mean(sI_deep[:,1,:], axis=1)
sd = np.std(sI_deep[:,1,:], axis=1)/np.sqrt(sI_deep.shape[2])
ax.plot(m, c= palette[1], label = 'Deep')
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= palette[1], alpha = 0.3)


ax.set_xticks(np.arange(len(m)), labels=['pre', 'rot'])
ax.set_ylim([0.5,1])
ax.set_xlim([-0.2, 1.2])
ax.set_ylabel('sI xpos', labelpad = 5)
ax.set_title('global')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()

plt.savefig(os.path.join(save_dir,'LT_DeepSup_rotation_sI.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'LT_DeepSup_rotation_sI.png'), dpi = 400,bbox_inches="tight")

#%% PLOT BOXES ONLY
nn = 60
idx_CZ3 = np.argmax(np.mean(CZ3_rotation_study['sI'], axis=(2,1)))
idx_CZ4 = np.argmax(np.mean(CZ4_rotation_study['sI'], axis=(2,1)))
idx_CZ6 = np.argmax(np.mean(CZ6_rotation_study['sI'], axis=(2,1)))


idx_GC2 = np.argmax(np.mean(GC2_rotation_study['sI'], axis=(2,1)))
idx_GC3 = np.argmax(np.mean(GC3_rotation_study['sI'], axis=(2,1)))
idx_DDC = np.argmax(np.mean(DDC_rotation_study['sI'], axis=(2,1)))


sup_angle = np.array([CZ3_rotation_study['angle'][idx_CZ3, 4],CZ4_rotation_study['angle'][idx_CZ4, 4],CZ6_rotation_study['angle'][idx_CZ6, 4]])
deep_angle = np.array([GC2_rotation_study['angle'][idx_GC2, 0],GC3_rotation_study['angle'][idx_GC3, 4],DDC_rotation_study['angle'][idx_DDC, 0]])
pd_angles = pd.DataFrame(data={'Condition':['Sup', 'Sup', 'Sup', 'Deep', 'Deep','Deep'], 'angle':np.abs(np.hstack((sup_angle, deep_angle)))})

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
