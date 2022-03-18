#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:40:39 2022

@author: julio
"""

import numpy as np
import pandas as pd
import copy

from neural_manifold import general_utils as gu
from neural_manifold import structure_index as sI 
import neural_manifold.decoders as dec
import neural_manifold.dimensionality_reduction as dim_red


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import umap
from sklearn.metrics import median_absolute_error, explained_variance_score


def check_kernel_size(input_object = None, spikes_signal = None, traces_signal = None, behavioral_signal = None, 
                  fs = None, std_kernels = [0, 0.05, 0.1, 0.2, 0.4, 1, np.inf], n_neigh = 0.01, ndim= 3, 
                  trial_signal = None, n_splits = 5, assymetry = True, verbose = False):
    
    #check signal input
    if isinstance(input_object, pd.DataFrame):
        spikes_signal = gu.dataframe_to_1array_translator(input_object,spikes_signal)
    elif isinstance(spikes_signal,np.ndarray):
        spikes_signal = copy.deepcopy(spikes_signal)
    else:
        raise ValueError("Input object has to be a dataframe or a numpy array.")                 
                 
    #check traces signal
    if isinstance(input_object, pd.DataFrame) and isinstance(traces_signal, str):
        traces_signal = gu.dataframe_to_1array_translator(input_object,traces_signal)
    elif isinstance(traces_signal,np.ndarray):
        traces_signal = copy.deepcopy(traces_signal)
        
    #check behavioral signal
    if isinstance(input_object, pd.DataFrame):
        if isinstance(behavioral_signal, str):
            behavioral_signal = list([gu.dataframe_to_1array_translator(input_object,behavioral_signal)])
        elif isinstance(behavioral_signal, list):
             behavioral_signal = gu.dataframe_to_manyarray_translator(input_object,behavioral_signal)
        else:
            raise ValueError("If 'input_object' provided, behavioral signal must be a string or a list of string.")
    elif isinstance(behavioral_signal, np.ndarray):
        behavioral_signal = list(copy.deepcopy(behavioral_signal))
    elif isinstance(behavioral_signal, list):
        if isinstance(behavioral_signal[0], np.ndarray):
            behavioral_signal = copy.deepcopy(behavioral_signal)
        else:
            raise ValueError("If 'behavioral_signal' is a list and 'input_object' is not provided, it must " +
                             f"be composed of numpy arrays. However it was {type(behavioral_signal[0])}.")
    behLimits = [(np.percentile(beh,5), np.percentile(beh,95)) for beh in behavioral_signal]

    #check neighbours
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(spikes_signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print(f"Resulting in {n_neigh} neighbours.")
            
    #check sampling frequency
    if isinstance(fs, type(None)):
        if isinstance(input_object, pd.DataFrame):
            columns_name = [col for col in input_object.columns.values]
            lower_columns_name = [col.lower() for col in input_object.columns.values]
            if 'bin_size' in lower_columns_name:
                bin_size = input_object.iloc[0][columns_name[lower_columns_name.index("bin_size")]]
            elif 'fs' in lower_columns_name:
                bin_size = 1/input_object.iloc[0][columns_name[lower_columns_name.index("fs")]]
            elif 'sf' in lower_columns_name:
                bin_size = 1/input_object.iloc[0][columns_name[lower_columns_name.index("sf")]]
            else:
                raise ValueError('Dataframe does not contain binsize, sf, or fs field.')
    else:
        bin_size = 1/fs
        
    #check trial signal to divide train/test in decoders
    if isinstance(trial_signal, np.ndarray): #user inputs an independent array for sI
        trial_signal = copy.deepcopy(trial_signal)
    elif isinstance(trial_signal, str):
        trial_signal = gu.dataframe_to_1array_translator(input_object,trial_signal)
    else:
        trial_signal = None 
        
    #initialize outputs
    emb_space = np.linspace(0,ndim-1, ndim).astype(int)
    signal_comparison = np.zeros((3, len(std_kernels)))
    emb_memory = np.zeros((spikes_signal.shape[0], ndim, len(std_kernels)))
    trust = np.zeros((1, len(std_kernels)))
    sI_array = np.zeros((len(behavioral_signal), len(std_kernels)))
    R2s = np.zeros((n_splits, len(behavioral_signal), 2, 2, 4, len(std_kernels)))
    
    dim_dict = dict()
    dim_dict['inner_dim'] = np.zeros((len(std_kernels)))
    dim_dict['inner_dim_radii_vs_nn'] = np.zeros((1000,2, len(std_kernels)))*np.nan
    dim_dict['num_trust'] = np.zeros((2,len(std_kernels)))
    dim_dict['res_var'] = np.zeros((2,len(std_kernels)))
    dim_dict['rec_error'] = np.zeros((2,len(std_kernels)))
    #iterate over each kernel size
    for kernel_index, stdk in enumerate(std_kernels):
        if verbose:
            print(f'Kernel: {stdk:.2f} ms ({kernel_index+1}/{len(std_kernels)})')
        #Compute firing rate
        if stdk==0:
            X_signal = copy.deepcopy(spikes_signal)
        elif stdk==np.inf:
            X_signal = copy.deepcopy(traces_signal)
        else:
            win_size = np.round(stdk*bin_size*10).astype(np.uint32)
            win = gu.norm_gauss_window(bin_size, stdk, num_bins = win_size, assymetry = True)
            X_signal = gu.smooth_data(spikes_signal, win=win)/bin_size
            
        #compare signal to traces
        signal_comparison[0, kernel_index] = np.corrcoef(X_signal.reshape(-1,1).T, traces_signal.reshape(-1,1).T)[0,1]
        signal_comparison[1, kernel_index] = median_absolute_error(X_signal, traces_signal)
        signal_comparison[2, kernel_index] = explained_variance_score(X_signal, traces_signal)
        
        #check inner dimensionality
        m , radius, neigh = dim_red.compute_inner_dim(X_signal)
        dim_dict['inner_dim'][kernel_index] = m
        dim_dict['inner_dim_radii_vs_nn'][radius.shape[0],:,kernel_index] = np.hstack((radius, neigh))
        
        
        #check umap trustworthiness dimensionality
        dim, num_trust = dim_red.compute_umap_trust_dim(X_signal, n_neigh = n_neigh, max_dim = 8,verbose=False)
        dim_dict['num_trust'][0,kernel_index] = dim
        if not np.isnan(dim):
            dim_dict['num_trust'][1,kernel_index] = num_trust[dim-1,0]
        else:
            dim_dict['num_trust'][1,kernel_index] = np.nan
        
        #compute isomap resvar dim
        dim, res_var = dim_red.compute_isomap_resvar_dim(X_signal, n_neigh = n_neigh, max_dim = 8,verbose=False)
        dim_dict['res_var'][0,kernel_index] = dim
        if not np.isnan(dim):
            dim_dict['res_var'][1,kernel_index] = res_var[dim-1,0]
        else:
            dim_dict['res_var'][1,kernel_index] = np.nan
        
        #compute isomap reconstruction error
        dim, rec_error = dim_red.compute_isomap_recerror_dim(X_signal, n_neigh = n_neigh, max_dim = 8,verbose=False)
        dim_dict['rec_error'][0,kernel_index] = dim
        if not np.isnan(dim):
            dim_dict['rec_error'][1,kernel_index] = rec_error[dim-1,0]
        else:
            dim_dict['rec_error'][1,kernel_index] = np.nan
        
        #project data
        model = umap.UMAP(n_neighbors=n_neigh, n_components =ndim, min_dist=0.75)
        emb_memory[:,:,kernel_index] = model.fit_transform(X_signal)
        
        #compute trustworthiness
        #trust[0, kernel_index] = validation.trustworthiness_vector(X_signal, emb_memory[:,:,kernel_index] ,n_neigh)[-1]
        
        #compute structure index
        for behavioral_index, behavioral_values in enumerate(behavioral_signal):
            if behavioral_values.ndim>1:
                behavioral_values = behavioral_values[:,0]
            minVal, maxVal = behLimits[behavioral_index]
            sI_array[behavioral_index, kernel_index], _, _ = sI.compute_structure_index(emb_memory[:,:,kernel_index],
                                                                               behavioral_values, 20, emb_space,
                                                                               0, vmin= minVal, vmax = maxVal)
            
        #compute decoder error
        R2s_temp = dec.decoders_1D(X_signal, input_label = behavioral_signal, emb_list = ["umap"], input_trial = trial_signal,
                                    n_dims = ndim, n_splits = n_splits, verbose = verbose)
                                    
        for signal_idx, signal_name in enumerate(list(R2s_temp.keys())):
            for dec_idx, dec_name in enumerate(list(R2s_temp[signal_name].keys())):
                R2s[:,:,:, signal_idx, dec_idx, kernel_index] = R2s_temp[signal_name][dec_name]
                
                
    return signal_comparison, dim_dict, emb_memory, trust, sI_array, R2s