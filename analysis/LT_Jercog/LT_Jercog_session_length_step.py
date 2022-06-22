#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:15:44 2022

@author: julio
"""
#%% IMPORTS
from neural_manifold.pipelines.LT_Jercog_session_length import LT_session_length
#%% PARAMS
params = {
    'kernel_std': 0.3,
    'vel_th': 0.2,
    'kernel_num_std': 5,
    'min_session_len': 6000, #5 min
    'equalize_session_len': True
    }
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'

#%% M2019
data_dir =  '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/2019'
mouse = 'M2019'
LT_session_length(data_dir,mouse, save_dir, **params)
#%% M2021
data_dir =  '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/2021'
mouse = 'M2021'
LT_session_length(data_dir,mouse, save_dir, **params)
#%% M2022
data_dir =  '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/2022'
mouse = 'M2022'
LT_session_length(data_dir,mouse, save_dir, **params)
#%% M2023
data_dir =  '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/2023'
mouse = 'M2023'
LT_session_length(data_dir,mouse, save_dir, **params)
#%% M2024
data_dir =  '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/2024'
mouse = 'M2024'
LT_session_length(data_dir,mouse, save_dir, **params)
#%% M2025
data_dir =  '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/2025'
mouse = 'M2025'
LT_session_length(data_dir,mouse, save_dir, **params)
#%% M2026
data_dir =  '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/2026'
mouse = 'M2026'
LT_session_length(data_dir,mouse, save_dir, **params)
