#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:40:39 2022

@author: julio
"""

import copy
import numpy as np
from neural_manifold import general_utils as gu
from neural_manifold import decoders as dec 
from neural_manifold import structure_index as sI 
from neural_manifold import dimensionality_reduction as dim_red


def check_kernel_size(pd_struct, spikes_field, traces_field, **kwargs):
	#CHECK INPUTS
	if 'ks_list' in kwargs:
		ks_list = kwargs['ks_list']
	else:
		ks_list = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10,np.inf]

	if 'assymetry_list' in kwargs:
		assymetry_list = kwargs['assymetry_list']
	else:
		assymetry_list = [False, True]

	if 'sI_nn_list' in kwargs:
		sI_nn_list = kwargs['sI_nn_list']
	else:
		sI_nn_list = [3, 10, 20, 50, 100, 200]

	if 'decoder_list' in kwargs:
		decoder_list = kwargs['decoder_list']
	else:
		decoder_list = ["wf", "wc", "xgb", "svr"]

	if 'n_splits' in kwargs:
		n_splits = kwargs['n_splits']
	else:
		n_splits = 10

	if 'verbose' in kwargs:
		verbose = kwargs['verbose']
	else:
		verbose = False
 
	#START 
	rates_field =  spikes_field.replace("spikes", "rates")
	sI_kernel = np.zeros((len(ks_list), len(assymetry_list), len(sI_nn_list)))
	R2s_kernel = np.zeros((len(ks_list), len(assymetry_list), n_splits, len(decoder_list)))
	dim_kernel = np.zeros((len(ks_list), len(assymetry_list)))
	if 'index_mat' not in pd_struct:
		pd_struct["index_mat"] = [np.zeros((pd_struct[spikes_field][idx].shape[0],1))+
									pd_struct["trial_id"][idx] for idx in pd_struct.index]

	for ks_idx, ks_val in enumerate(ks_list):
		if verbose:
			print(f"Kernel_size: {ks_val} ({ks_idx+1}/{len(ks_list)})")
		for assy_idx, assy_val in enumerate(assymetry_list):
			if verbose:
				print(f"\tAssymetry: {assy_val} ({assy_idx+1}/{len(assymetry_list)})")

			#compute firing rates with new ks_val
			if ks_val == 0:
				pd_struct_mov = gu.select_trials(pd_struct, "dir == ['L', 'R']")
				rates =  copy.deepcopy(np.concatenate(pd_struct_mov[spikes_field].values, axis=0))
			elif ks_val == np.inf:
				pd_struct_mov = gu.select_trials(pd_struct, "dir == ['L', 'R']")
				rates =  copy.deepcopy(np.concatenate(pd_struct_mov[traces_field].values, axis=0))
			else:
				pd_struct = gu.add_firing_rates(pd_struct,'smooth', std=ks_val, num_std=5, 
															assymetry=assy_val, continuous=True)
				pd_struct_mov = gu.select_trials(pd_struct, "dir == ['L', 'R']")
				rates = copy.deepcopy(np.concatenate(pd_struct_mov[rates_field].values, axis=0))

			pos = copy.deepcopy(np.concatenate(pd_struct_mov['pos'].values, axis=0))

			#1. Check decoder ability
			trial_idx = copy.deepcopy(np.concatenate(pd_struct_mov['index_mat'].values, axis=0))
			R2s_temp, _ = dec.decoders_1D(x_base_signal=rates,y_signal_list=pos[:,0],n_splits=n_splits,
									decoder_list=decoder_list,trial_signal=trial_idx,verbose=verbose)

			for dec_idx, dec_name in enumerate(decoder_list):
				R2s_kernel[ks_idx, assy_idx,:, dec_idx] = R2s_temp['base_signal'][dec_name][:,0,0]

			#2. Check structure index
			rates_space = np.linspace(0,rates.shape[1]-1, rates.shape[1]).astype(int)
			posLimits = [np.percentile(pos[:,0],5), np.percentile(pos[:,0],95)]
			if verbose:
				print("\t\tSI: X/X", end = '', sep = '')
				pre_del = '\b\b\b'
			for nn_idx, nn_val in enumerate(sI_nn_list):
				if verbose:
					print(f"{pre_del}{nn_idx+1}/{len(sI_nn_list)}", sep = '', end = '')
					pre_del =  (len(str(nn_idx+1))+len(str(len(sI_nn_list)))+1)*'\b'
				temp,_ , _ = sI.compute_structure_index(rates,pos[:,0],10,rates_space,0,nn=nn_val,
	        													vmin=posLimits[0],vmax=posLimits[1])
				sI_kernel[ks_idx, assy_idx, nn_idx] = temp

			#3. Check inner dimension
			if verbose:
				print("\t\tInner dimension")
			dim_kernel[ks_idx, assy_idx],_,_ = dim_red.compute_inner_dim(base_signal=rates,min_neigh=2, 
																max_neigh = int(rates.shape[0]*0.1))

	return R2s_kernel, sI_kernel, dim_kernel
