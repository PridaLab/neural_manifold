#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:40:40 2022

@author: julio
"""

#%% IMPORTS
import copy
from neural_manifold import general_utils as gu
from neural_manifold import dimensionality_reduction as dim_red
import pickle
import os

#%% GENERAL PARAMS
params = {
    'nn': 60,
    'min_dist':0.75,
    'max_dim':12,
    'sI_nn_list':[3, 10,20, 50, 100, 200],
    'verbose': True,
    'signal_field':'revents_SNR3',
    'save_dir':'/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/emb_quality_study',
    'label_list':['posx', 'posy','dir_mat']
    }
save_dir = params["save_dir"]
kernel_std = 0.25
assymetry = True
#%% CZ3
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ3'
CZ3 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ3.keys())
CZ3_p= copy.deepcopy(CZ3[fnames[0]])
CZ3_r= copy.deepcopy(CZ3[fnames[1]])

CZ3_p = gu.add_firing_rates(CZ3_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ3_r = gu.add_firing_rates(CZ3_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ3_sI_dim, CZ3_trust_dim, CZ3_cont_dim, CZ3_params = dim_red.check_rotation_emb_quality(CZ3_p, CZ3_r, **params)

CZ3_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
CZ3_emb_quality_study = {
        'sI_dim': CZ3_sI_dim,
        'trust_dim': CZ3_trust_dim,
        'cont_dim':CZ3_cont_dim,
        'params': CZ3_params
        }

save_rs = open(os.path.join(save_dir, "CZ3_emb_quality_study.pkl"), "wb")
pickle.dump(CZ3_emb_quality_study, save_rs)
save_rs.close()

#%% CZ4
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ4'
CZ4 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ4.keys())
CZ4_p= copy.deepcopy(CZ4[fnames[0]])
CZ4_r= copy.deepcopy(CZ4[fnames[1]])

CZ4_p = gu.add_firing_rates(CZ4_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ4_r = gu.add_firing_rates(CZ4_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ4_sI_dim, CZ4_trust_dim, CZ4_cont_dim, CZ4_params = dim_red.check_rotation_emb_quality(CZ4_p, CZ4_r, **params)

CZ4_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
CZ4_emb_quality_study = {
        'sI_dim': CZ4_sI_dim,
        'trust_dim': CZ4_trust_dim,
        'cont_dim':CZ4_cont_dim,
        'params': CZ4_params
        }

save_rs = open(os.path.join(save_dir, "CZ4_emb_quality_study.pkl"), "wb")
pickle.dump(CZ4_emb_quality_study, save_rs)
save_rs.close()

#%% CZ6
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ6'
CZ6 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ6.keys())
CZ6_p= copy.deepcopy(CZ6[fnames[0]])
CZ6_r= copy.deepcopy(CZ6[fnames[1]])

CZ6_p = gu.add_firing_rates(CZ6_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ6_r = gu.add_firing_rates(CZ6_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ6_sI_dim, CZ6_trust_dim, CZ6_cont_dim, CZ6_params = dim_red.check_rotation_emb_quality(CZ6_p, CZ6_r, **params)

CZ6_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
CZ6_emb_quality_study = {
        'sI_dim': CZ6_sI_dim,
        'trust_dim': CZ6_trust_dim,
        'cont_dim':CZ6_cont_dim,
        'params': CZ6_params
        }

save_rs = open(os.path.join(save_dir, "CZ6_emb_quality_study.pkl"), "wb")
pickle.dump(CZ6_emb_quality_study, save_rs)
save_rs.close()

#%% CZ7
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ7'
CZ7 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ7.keys())
CZ7_p= copy.deepcopy(CZ7[fnames[0]])
CZ7_r= copy.deepcopy(CZ7[fnames[1]])

CZ7_p = gu.add_firing_rates(CZ7_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ7_r = gu.add_firing_rates(CZ7_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ7_sI_dim, CZ7_trust_dim, CZ7_cont_dim, CZ7_params = dim_red.check_rotation_emb_quality(CZ7_p, CZ7_r, **params)

CZ7_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
CZ7_emb_quality_study = {
        'sI_dim': CZ7_sI_dim,
        'trust_dim': CZ7_trust_dim,
        'cont_dim':CZ7_cont_dim,
        'params': CZ7_params
        }

save_rs = open(os.path.join(save_dir, "CZ7_emb_quality_study.pkl"), "wb")
pickle.dump(CZ7_emb_quality_study, save_rs)
save_rs.close()

#%% CZ8
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ8'
CZ8 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ8.keys())
CZ8_p= copy.deepcopy(CZ8[fnames[0]])
CZ8_r= copy.deepcopy(CZ8[fnames[1]])

CZ8_p = gu.add_firing_rates(CZ8_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ8_r = gu.add_firing_rates(CZ8_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ8_sI_dim, CZ8_trust_dim, CZ8_cont_dim, CZ8_params = dim_red.check_rotation_emb_quality(CZ8_p, CZ8_r, **params)

CZ8_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
CZ8_emb_quality_study = {
        'sI_dim': CZ8_sI_dim,
        'trust_dim': CZ8_trust_dim,
        'cont_dim':CZ8_cont_dim,
        'params': CZ8_params
        }

save_rs = open(os.path.join(save_dir, "CZ8_emb_quality_study.pkl"), "wb")
pickle.dump(CZ8_emb_quality_study, save_rs)
save_rs.close()

#%% CZ9
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/CZ9'
CZ9 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(CZ9.keys())
CZ9_p= copy.deepcopy(CZ9[fnames[0]])
CZ9_r= copy.deepcopy(CZ9[fnames[1]])

CZ9_p = gu.add_firing_rates(CZ9_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
CZ9_r = gu.add_firing_rates(CZ9_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

CZ9_sI_dim, CZ9_trust_dim, CZ9_cont_dim, CZ9_params = dim_red.check_rotation_emb_quality(CZ9_p, CZ9_r, **params)

CZ9_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
CZ9_emb_quality_study = {
        'sI_dim': CZ9_sI_dim,
        'trust_dim': CZ9_trust_dim,
        'cont_dim':CZ9_cont_dim,
        'params': CZ9_params
        }

save_rs = open(os.path.join(save_dir, "CZ9_emb_quality_study.pkl"), "wb")
pickle.dump(CZ9_emb_quality_study, save_rs)
save_rs.close()

#%% GC2
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/GC2'
GC2 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(GC2.keys())
GC2_p= copy.deepcopy(GC2[fnames[0]])
GC2_r= copy.deepcopy(GC2[fnames[1]])

GC2_p = gu.add_firing_rates(GC2_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
GC2_r = gu.add_firing_rates(GC2_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

GC2_sI_dim, GC2_trust_dim, GC2_cont_dim, GC2_params = dim_red.check_rotation_emb_quality(GC2_p, GC2_r, **params)

GC2_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
GC2_emb_quality_study = {
        'sI_dim': GC2_sI_dim,
        'trust_dim': GC2_trust_dim,
        'cont_dim':GC2_cont_dim,
        'params': GC2_params
        }

save_rs = open(os.path.join(save_dir, "GC2_emb_quality_study.pkl"), "wb")
pickle.dump(GC2_emb_quality_study, save_rs)
save_rs.close()

#%% GC3
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/GC3'
GC3 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(GC3.keys())
GC3_p= copy.deepcopy(GC3[fnames[0]])
GC3_r= copy.deepcopy(GC3[fnames[1]])

GC3_p = gu.add_firing_rates(GC3_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
GC3_r = gu.add_firing_rates(GC3_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

GC3_sI_dim, GC3_trust_dim, GC3_cont_dim, GC3_params = dim_red.check_rotation_emb_quality(GC3_p, GC3_r, **params)

GC3_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
GC3_emb_quality_study = {
        'sI_dim': GC3_sI_dim,
        'trust_dim': GC3_trust_dim,
        'cont_dim':GC3_cont_dim,
        'params': GC3_params
        }

save_rs = open(os.path.join(save_dir, "GC3_emb_quality_study.pkl"), "wb")
pickle.dump(GC3_emb_quality_study, save_rs)
save_rs.close()

#%% DDC
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/DDC'
DDC = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fnames = list(DDC.keys())
DDC_p= copy.deepcopy(DDC[fnames[0]])
DDC_r= copy.deepcopy(DDC[fnames[1]])

DDC_p = gu.add_firing_rates(DDC_p, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)
DDC_r = gu.add_firing_rates(DDC_r, 'smooth', std=kernel_std, num_std = 5, assymetry = assymetry, continuous = True)

DDC_sI_dim, DDC_trust_dim, DDC_cont_dim, DDC_params = dim_red.check_rotation_emb_quality(DDC_p, DDC_r, **params)

DDC_params.update({'kernel_std':kernel_std, 'assymetry':assymetry})
DDC_emb_quality_study = {
        'sI_dim': DDC_sI_dim,
        'trust_dim': DDC_trust_dim,
        'cont_dim':DDC_cont_dim,
        'params': DDC_params
        }

save_rs = open(os.path.join(save_dir, "DDC_emb_quality_study.pkl"), "wb")
pickle.dump(DDC_emb_quality_study, save_rs)
save_rs.close()

#%% Load
if "CZ3_emb_quality_study" not in locals():
    CZ3_emb_quality_study = gu.load_files(save_dir, '*CZ3_emb_quality_study.pkl', verbose=True, struct_type = "pickle")
if "CZ4_emb_quality_study" not in locals():
    CZ4_emb_quality_study = gu.load_files(save_dir, '*CZ4_emb_quality_study.pkl', verbose=True, struct_type = "pickle")
if "CZ6_emb_quality_study" not in locals():
    CZ6_emb_quality_study = gu.load_files(save_dir, '*CZ6_emb_quality_study.pkl', verbose=True, struct_type = "pickle")
if "CZ7_emb_quality_study" not in locals():
    CZ7_emb_quality_study = gu.load_files(save_dir, '*CZ7_emb_quality_study.pkl', verbose=True, struct_type = "pickle")
if "CZ8_emb_quality_study" not in locals():
    CZ8_emb_quality_study = gu.load_files(save_dir, '*CZ8_emb_quality_study.pkl', verbose=True, struct_type = "pickle")
if "CZ9_emb_quality_study" not in locals():
    CZ9_emb_quality_study = gu.load_files(save_dir, '*CZ9_emb_quality_study.pkl', verbose=True, struct_type = "pickle")
    
if "GC2_emb_quality_study" not in locals():
    GC2_emb_quality_study = gu.load_files(save_dir, '*GC2_emb_quality_study.pkl', verbose=True, struct_type = "pickle")
if "GC3_emb_quality_study" not in locals():
    GC3_emb_quality_study = gu.load_files(save_dir, '*GC3_emb_quality_study.pkl', verbose=True, struct_type = "pickle")
if "DDC_emb_quality_study" not in locals():
    DDC_emb_quality_study = gu.load_files(save_dir, '*DDC_emb_quality_study.pkl', verbose=True, struct_type = "pickle")

#%%
from kneed import KneeLocator
import numpy as np

trust_dim = np.zeros((5,2))*np.nan
cont_dim = np.zeros((5,2))*np.nan
sI_dim = np.zeros((5,3,2))*np.nan


max_dim = CZ3_emb_quality_study['params']['max_dim']
dim_space = np.linspace(1,max_dim, max_dim).astype(int)    

for idx in range(2):
    kl = KneeLocator(dim_space, CZ3_emb_quality_study["trust_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        trust_dim[0,idx] = kl.knee

    kl = KneeLocator(dim_space, CZ3_emb_quality_study["cont_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        cont_dim[0,idx] = kl.knee    
    
    for sI_idx in range(3):
        kl = KneeLocator(dim_space, np.mean(CZ3_emb_quality_study["sI_dim"][:,:,sI_idx,idx], axis=1), curve = "concave", direction = "increasing")
        if kl.knee:
            sI_dim[0,sI_idx, idx] = kl.knee    
        
for idx in range(2):
    kl = KneeLocator(dim_space, CZ4_emb_quality_study["trust_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        trust_dim[1,idx] = kl.knee

    kl = KneeLocator(dim_space, CZ4_emb_quality_study["cont_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        cont_dim[1,idx] = kl.knee    
    
    for sI_idx in range(3):
        kl = KneeLocator(dim_space, np.mean(CZ4_emb_quality_study["sI_dim"][:,:,sI_idx,idx], axis=1), curve = "concave", direction = "increasing")
        if kl.knee:
            sI_dim[1,sI_idx, idx] = kl.knee    

for idx in range(2):
    kl = KneeLocator(dim_space, CZ6_emb_quality_study["trust_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        trust_dim[2,idx] = kl.knee

    kl = KneeLocator(dim_space, CZ6_emb_quality_study["cont_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        cont_dim[2,idx] = kl.knee    
    
    for sI_idx in range(3):
        kl = KneeLocator(dim_space, np.mean(CZ6_emb_quality_study["sI_dim"][:,:,sI_idx,idx], axis=1), curve = "concave", direction = "increasing")
        if kl.knee:
            sI_dim[2,sI_idx, idx] = kl.knee  
            

for idx in range(2):
    kl = KneeLocator(dim_space, GC2_emb_quality_study["trust_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        trust_dim[3,idx] = kl.knee

    kl = KneeLocator(dim_space, GC2_emb_quality_study["cont_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        cont_dim[3,idx] = kl.knee    
    
    for sI_idx in range(3):
        kl = KneeLocator(dim_space, np.mean(GC2_emb_quality_study["sI_dim"][:,:,sI_idx,idx], axis=1), curve = "concave", direction = "increasing")
        if kl.knee:
            sI_dim[3,sI_idx, idx] = kl.knee    
            
            
for idx in range(2):
    kl = KneeLocator(dim_space, GC3_emb_quality_study["trust_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        trust_dim[4,idx] = kl.knee

    kl = KneeLocator(dim_space, GC3_emb_quality_study["cont_dim"][:,idx], curve = "concave", direction = "increasing")
    if kl.knee:
        cont_dim[4,idx] = kl.knee    
    
    for sI_idx in range(3):
        kl = KneeLocator(dim_space, np.mean(GC3_emb_quality_study["sI_dim"][:,:,sI_idx,idx], axis=1), curve = "concave", direction = "increasing")
        if kl.knee:
            sI_dim[4,sI_idx, idx] = kl.knee            