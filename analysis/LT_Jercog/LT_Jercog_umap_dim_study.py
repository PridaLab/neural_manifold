#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:26:06 2022

@author: julio
"""

#%% IMPORTS
import numpy as np
from neural_manifold import general_utils as gu
from neural_manifold import dimensionality_reduction as dim_red
import pickle, os, copy

import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO

import seaborn as sns
import pandas as pd

#%% PLOT UMAP NN

#%% GENERAL PARAMS
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/umap_params_study/dim'
params = {
    'nn_list': [3, 10, 20, 30, 60, 120, 200, 500, 1000],
    'max_dim': 12,
    'verbose': True,
    'n_splits': 5,
    'nn': 60
    }

signal_name = 'ML_rates'
label_names = ['posx']

#%% M2019
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2019' in f]
M2019 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2019_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2019.keys())
M2019_umap_dim = dict()
for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")
    pd_struct = copy.deepcopy(M2019[fname])
    #compute hyperparameter study
    m_trust, m_cont, m_sI, m_R2s, m_params = dim_red.compute_umap_dim(pd_object = pd_struct, 
                                                     base_signal = signal_name, label_signal = label_names,
                                                     trial_signal = "index_mat", **params)
    m_params['base_name'] = signal_name
    m_params['label_name'] = label_names
    #save results
    M2019_umap_dim[fname] = {
        'trust': m_trust,
        'cont': m_cont,
        'sI': m_sI,
        'R2s': m_R2s,
        'params': m_params
        }
    
    save_ks = open(os.path.join(save_dir, "M2019_umap_dim_dict.pkl"), "wb")
    pickle.dump(M2019_umap_dim, save_ks)
    save_ks.close()
    
#_ = plot_umap_nn_study(M2019_umap_dim, save_dir)