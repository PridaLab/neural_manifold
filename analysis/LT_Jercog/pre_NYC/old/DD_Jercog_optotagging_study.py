#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:54:40 2022

@author: julio
"""

#%% IMPORTS
import numpy as np
from neural_manifold import general_utils as gu
import pickle, os, copy

import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO

import seaborn as sns
import pandas as pd

from neural_manifold import decoders as dec
import math

from neural_manifold import dimensionality_reduction as dim_red
from neural_manifold import structure_index as sI 
import umap

#%%
signal = 'revents_SNR3'
nn = 60
vel_th = 0.2
#%%
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/DREADDS/data/DD2_veh/DD2_veh_LTMb_s3/'
DD2_lt = gu.load_files(file_dir, '*DD2_veh_lt_s3_*.mat', verbose = True, struct_type = "PyalData")
DD2_rot = gu.load_files(file_dir, '*DD2_veh_rot_s3_*.mat', verbose = True, struct_type = "PyalData")

if 'dir_mat' not in DD2_lt.columns:
	DD2_lt["dir_mat"] = [np.zeros((DD2_lt["pos"][idx].shape[0],1)).astype(int)+
								('L' == DD2_lt["dir"][idx])+ 2*('R' == DD2_lt["dir"][idx])
								for idx in DD2_lt.index]

if 'dir_mat' not in DD2_rot.columns:
	DD2_rot["dir_mat"] = [np.zeros((DD2_rot["pos"][idx].shape[0],1)).astype(int)+
								('L' == DD2_rot["dir"][idx])+ 2*('R' == DD2_rot["dir"][idx])
								for idx in DD2_rot.index]
    
DD2_lt = gu.add_firing_rates(DD2_lt, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
DD2_rot = gu.add_firing_rates(DD2_rot, 'smooth', std=0.25, num_std = 5, assymetry = True, continuous = True)
DD2_lt, _ = gu.keep_only_moving(DD2_lt, vel_th)
DD2_rot, _ = gu.keep_only_moving(DD2_rot, vel_th)


calb_class = gu.load_files(file_dir, '*calb_class.mat', verbose = True, struct_type = "PyalData")
calb_pos = calb_class['calb_pos'][0]-1
calb_neg = calb_class['calb_neg'][0]-1
calb_unc = calb_class['calb_unc'][0]-1


revents_lt = np.concatenate(DD2_lt[signal].values, axis = 0)
pos_lt = np.concatenate(DD2_lt['pos'].values, axis = 0)
dir_lt = np.concatenate(DD2_lt['dir_mat'].values, axis = 0)

revents_rot = np.concatenate(DD2_rot[signal].values, axis = 0)
pos_rot = np.concatenate(DD2_rot['pos'].values, axis = 0)
dir_rot = np.concatenate(DD2_rot['dir_mat'].values, axis = 0)

index = np.vstack((np.zeros((revents_lt.shape[0],1)),np.zeros((revents_rot.shape[0],1))+1))
concat_signal = np.vstack((revents_lt, revents_rot))

#%% COMPUTE EMBEDDINGS

print('All model')
model_all = umap.UMAP(n_neighbors=nn, min_dist = 0.75, n_components = 3)
concat_emb = model_all.fit_transform(concat_signal)
emb_lt_all = concat_emb[index[:,0]==0,:]
emb_rot_all = concat_emb[index[:,0]==1,:]

print('Calb pos model')
model_calb_pos = umap.UMAP(n_neighbors=nn, min_dist = 0.75, n_components = 3)
concat_emb = model_calb_pos.fit_transform(concat_signal[:, calb_pos])
emb_lt_calb_pos = concat_emb[index[:,0]==0,:]
emb_rot_calb_pos = concat_emb[index[:,0]==1,:]

print('Calb neg model')
model_calb_neg = umap.UMAP(n_neighbors=nn, min_dist = 0.75, n_components = 3)
concat_emb = model_calb_neg.fit_transform(concat_signal[:, calb_neg])
emb_lt_calb_neg = concat_emb[index[:,0]==0,:]
emb_rot_calb_neg = concat_emb[index[:,0]==1,:]

print('Calb unc model')
model_calb_unc = umap.UMAP(n_neighbors=nn, min_dist = 0.75, n_components = 3)
concat_emb = model_calb_unc.fit_transform(concat_signal[:, calb_unc])
emb_lt_calb_unc = concat_emb[index[:,0]==0,:]
emb_rot_calb_unc = concat_emb[index[:,0]==1,:]

#%% COMPUTE ROTATION
TAB, RAB = dec.align_manifolds_1D(emb_lt_all, emb_rot_all, pos_lt[:,0], pos_rot[:,0], ndims = 3, nCentroids = 20)   
tr = (np.trace(RAB)-1)/2
if abs(tr)>1:
    tr = round(tr,2)
    if abs(tr)>1:
        tr = np.nan
angle_all = math.acos(tr)*180/np.pi
print(f"Umap All: {angle_all:.2f}º ")

TAB, RAB = dec.align_manifolds_1D(emb_lt_calb_pos, emb_rot_calb_pos, pos_lt[:,0], pos_rot[:,0], ndims = 3, nCentroids = 10)   
tr = (np.trace(RAB)-1)/2
if abs(tr)>1:
    tr = round(tr,2)
    if abs(tr)>1:
        tr = np.nan
angle_calb_pos = math.acos(tr)*180/np.pi
print(f"Umap Calb pos: {angle_calb_pos:.2f}º ")

TAB, RAB = dec.align_manifolds_1D(emb_lt_calb_neg, emb_rot_calb_neg, pos_lt[:,0], pos_rot[:,0], ndims = 3, nCentroids = 10)   
tr = (np.trace(RAB)-1)/2
if abs(tr)>1:
    tr = round(tr,2)
    if abs(tr)>1:
        tr = np.nan
angle_calb_neg = math.acos(tr)*180/np.pi
print(f"Umap Calb neg: {angle_calb_neg:.2f}º ")

TAB, RAB = dec.align_manifolds_1D(emb_lt_calb_unc, emb_rot_calb_unc, pos_lt[:,0], pos_rot[:,0], ndims = 3, nCentroids = 10)   
tr = (np.trace(RAB)-1)/2
if abs(tr)>1:
    tr = round(tr,2)
    if abs(tr)>1:
        tr = np.nan
angle_calb_unc = math.acos(tr)*180/np.pi
print(f"Umap Calb unc: {angle_calb_unc:.2f}º ")

#%% PLOT
fig = plt.figure()
ax = plt.subplot(2,4,1, projection= '3d')
ax.scatter(*emb_lt_all.T, c = pos_lt[:,0])
ax.set_title('All')
ax = plt.subplot(2,4,5, projection= '3d')
ax.scatter(*emb_rot_all.T, c = pos_rot[:,0])

ax = plt.subplot(2,4,2, projection= '3d')
ax.scatter(*emb_lt_calb_pos.T, c = pos_lt[:,0])
ax.set_title('Calb Pos')
ax = plt.subplot(2,4,6, projection= '3d')
ax.scatter(*emb_rot_calb_pos.T, c = pos_rot[:,0])

ax = plt.subplot(2,4,3, projection= '3d')
ax.scatter(*emb_lt_calb_neg.T, c = pos_lt[:,0])
ax.set_title('Calb Neg')
ax = plt.subplot(2,4,7, projection= '3d')
ax.scatter(*emb_rot_calb_neg.T, c = pos_rot[:,0])

ax = plt.subplot(2,4,4, projection= '3d')
ax.scatter(*emb_lt_calb_unc.T, c = pos_lt[:,0])
ax.set_title('Calb Unc')
ax = plt.subplot(2,4,8, projection= '3d')
ax.scatter(*emb_rot_calb_unc.T, c = pos_rot[:,0])
#%%
fig = plt.figure()
ax = plt.subplot(2,4,1, projection= '3d')
ax.scatter(*emb_lt_all.T, c = dir_lt)
ax.set_title('All')
ax = plt.subplot(2,4,5, projection= '3d')
ax.scatter(*emb_rot_all.T, c = dir_rot)

ax = plt.subplot(2,4,2, projection= '3d')
ax.scatter(*emb_lt_calb_pos.T, c = dir_lt)
ax.set_title('Calb Pos')
ax = plt.subplot(2,4,6, projection= '3d')
ax.scatter(*emb_rot_calb_pos.T, c = dir_rot)

ax = plt.subplot(2,4,3, projection= '3d')
ax.scatter(*emb_lt_calb_neg.T, c = dir_lt)
ax.set_title('Calb Neg')
ax = plt.subplot(2,4,7, projection= '3d')
ax.scatter(*emb_rot_calb_neg.T, c = dir_rot)

ax = plt.subplot(2,4,4, projection= '3d')
ax.scatter(*emb_lt_calb_unc.T, c = dir_lt)
ax.set_title('Calb Unc')
ax = plt.subplot(2,4,8, projection= '3d')
ax.scatter(*emb_rot_calb_unc.T, c = dir_rot)