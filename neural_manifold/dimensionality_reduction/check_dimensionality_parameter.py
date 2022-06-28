#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:40:39 2022

@author: julio
"""

import numpy as np
from neural_manifold import general_utils as gu
from neural_manifold import decoders as dec 
from neural_manifold import structure_index as sI 

from neural_manifold.dimensionality_reduction import validation as dim_validation 
from neural_manifold.dimensionality_reduction import compute_dimensionality as compute_dim

from kneed import KneeLocator

import copy, os, math, random, umap, base64
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

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

	if 'vel_th' in kwargs:
		vel_th = kwargs['vel_th']
	else:
		vel_th = 0
		kwargs['vel_th'] = vel_th
		if 'sF' in kwargs:
			sF = kwargs['sF']
		else:
			if 'sF' in pd_struct.columns:
				sF = pd_struct['sF'][0]
			elif 'Fs' in pd_struct.columns:
				sF = pd_struct['Fs'][0]
			else:
				assert True, "you must provide the sampling frequency ('sF')"
			kwargs['sF'] = sF

	if 'verbose' in kwargs:
		verbose = kwargs['verbose']
	else:
		verbose = False

	#START 
	if 'spikes' in spikes_field:
		rates_field =  spikes_field.replace("spikes", "rates")
	else:
		rates_field =  spikes_field.replace("events", "revents")
        
	sI_kernel = np.zeros((len(ks_list), len(assymetry_list), len(sI_nn_list)))
	R2s_kernel = np.zeros((len(ks_list), len(assymetry_list), n_splits, len(decoder_list)))
	dim_kernel = np.zeros((len(ks_list), len(assymetry_list)))

	if 'index_mat' not in pd_struct:
		pd_struct["index_mat"] = [np.zeros((pd_struct[spikes_field][idx].shape[0],1))+
									pd_struct["trial_id"][idx] for idx in pd_struct.index]

	if ('vel' not in pd_struct) and (vel_th>0):
		pos = copy.deepcopy(np.concatenate(pd_struct['pos'].values, axis=0))
		index_mat = np.concatenate(pd_struct["index_mat"].values, axis=0)

		vel = np.linalg.norm(np.diff(pos, axis= 0), axis=1)*sF
		vel = np.hstack((vel[0], vel))
		pd_struct['vel'] = [vel[index_mat[:,0]==pd_struct["trial_id"][idx]] 
							           for idx in pd_struct.index]

	for ks_idx, ks_val in enumerate(ks_list):
		if verbose:
			print(f"Kernel_size: {ks_val} ({ks_idx+1}/{len(ks_list)})")
		for assy_idx, assy_val in enumerate(assymetry_list):
			if verbose:
				print(f"\tAssymetry: {assy_val} ({assy_idx+1}/{len(assymetry_list)})")

			#compute firing rates with new ks_val
			if ks_val == 0:
				rates =  copy.deepcopy(np.concatenate(pd_struct[spikes_field].values, axis=0))
			elif ks_val == np.inf:
				rates =  copy.deepcopy(np.concatenate(pd_struct[traces_field].values, axis=0))
			else:
				pd_struct = gu.add_firing_rates(pd_struct,'smooth', std=ks_val, num_std=5, 
															assymetry=assy_val, continuous=True)
				rates = copy.deepcopy(np.concatenate(pd_struct[rates_field].values, axis=0))

			pos = copy.deepcopy(np.concatenate(pd_struct['pos'].values, axis=0))
			trial_idx = copy.deepcopy(np.concatenate(pd_struct['index_mat'].values, axis=0))
			if vel_th>0:
				vel = copy.deepcopy(np.concatenate(pd_struct['vel'].values, axis=0))
				pos = pos[vel>=vel_th, :]
				rates = rates[vel>=vel_th,:]
				trial_idx = trial_idx[vel>=vel_th,:]

			#1. Check decoder ability
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
				print("\n\t\tInner dimension: ",  sep = '', end = '')
			dim_kernel[ks_idx, assy_idx],_,_ = compute_dim.compute_inner_dim(base_signal=rates,min_neigh=2, 
																max_neigh = int(rates.shape[0]*0.1))
			if verbose and not np.isnan(dim_kernel[ks_idx, assy_idx]):
				print(f"{dim_kernel[ks_idx, assy_idx]}")
			    
	return R2s_kernel, sI_kernel, dim_kernel


@gu.check_inputs_for_pd
def compute_umap_nn(base_signal=None, label_signal = None, trial_signal = None, **kwargs):

	if 'nn_list' in kwargs:
		nn_list = kwargs['nn_list']
	else:
		nn_list = [3, 10, 20, 30, 40, 50, 60, 80, 100, 200, 500, 1000]
		kwargs['nn_list'] = nn_list

	if 'min_dist' in kwargs:
		min_dist = kwargs['min_dist']
	else:
		min_dist = 0.75
		kwargs['min_dist'] = min_dist

	if 'dim' in kwargs:
		dim = kwargs['dim']
	else:
		dim = 10
		kwargs['dim'] = dim

	if 'max_dim' in kwargs:
		max_dim = kwargs['max_dim']
	else:
		max_dim = 10
		kwargs['max_dim'] = max_dim

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
		kwargs['verbose'] = verbose
	label_list = list()
	for label in label_signal:
		if label.ndim == 2:
			label = label[:,0]
		label_list.append(label)
	label_limits = [(np.percentile(label,5), np.percentile(label,95)) for label in label_list]    

	sI_og = np.zeros((len(nn_list), len(label_list)))*np.nan
	trust_dim = np.zeros((len(nn_list)))*np.nan
	trust_dim_val = np.zeros((len(nn_list),max_dim))*np.nan
	cont_dim = np.zeros((len(nn_list)))*np.nan
	cont_dim_val = np.zeros((len(nn_list),max_dim))*np.nan
	sI_emb = np.zeros((len(nn_list), len(nn_list), len(label_list)))*np.nan
	R2s_nn = np.zeros((len(nn_list), n_splits, len(label_list), len(decoder_list)))*np.nan

	og_space = np.arange(base_signal.shape[1])
	if verbose:
		print("Computing sI in og space:")

	for label_idx, label in enumerate(label_list):
		label_lim = label_limits[label_idx]
		if verbose:
			print(f"\tsI {label_idx+1}/{len(label_list)}: X/X", end = '', sep = '')
			pre_del = '\b\b\b'
		if len(np.unique(label))<10:
			nbins = len(np.unique(label))
		else:
			nbins = 10
		for sI_nn_idx, sI_nn in enumerate(nn_list):
			if verbose:
				print(f"{pre_del}{sI_nn_idx+1}/{len(nn_list)}", sep = '', end = '')
				pre_del = (len(str(sI_nn_idx+1))+len(str(len(nn_list)))+1)*'\b'
			temp,_ , _ = sI.compute_structure_index(base_signal,label,nbins,og_space,0,nn=sI_nn,
																vmin=label_lim[0], vmax=label_lim[1])
			sI_og[sI_nn_idx,label_idx] = temp

	if verbose:
		print('')
	emb_space = np.arange(dim)
	emb_list = list()
	for nn_idx, nn in enumerate(nn_list):
		if verbose:
			print(f"NN: {nn} ({nn_idx+1}/{len(nn_list)}):")
		#0. Model
		model = umap.UMAP(n_neighbors = nn, n_components =dim, min_dist=0.75)
		if verbose:
			print("\tFitting model...", sep= '', end = '')
		emb_signal = model.fit_transform(base_signal)
		emb_list.append(emb_signal)
		if verbose:
			print("Done")
		#1. Trustworthiness dim
		if verbose:
			print("\tComputing trustworthiness dim...", sep= '', end = '')
		dim, num_trust = compute_dim.compute_umap_trust_dim(base_signal, n_neigh = nn, max_dim = max_dim)
		trust_dim[nn_idx] = dim
		trust_dim_val[nn_idx, :len(num_trust)] = num_trust[:,0]
		if verbose:
			print(f"\b\b\b: {dim}")

		#2. Continuity dim
		if verbose:
			print("\tComputing continuity dim...", sep= '', end = '')
		dim, num_cont = compute_dim.compute_umap_continuity_dim(base_signal, n_neigh = nn, max_dim = max_dim)
		cont_dim[nn_idx] = dim
		cont_dim_val[nn_idx, :len(num_cont)] = num_cont[:,0]
		if verbose:
			print(f"\b\b\b: {dim}")

		#3. sI
		for label_idx, label in enumerate(label_list):
			label_lim = label_limits[label_idx]
			if verbose:
				print(f"\tComputing sI {label_idx+1}/{len(label_list)}: X/X", sep='', end = '')
				pre_del = '\b'*3
			if len(np.unique(label))<10:
				nbins = len(np.unique(label))
			else:
				nbins = 10
			for sI_nn_idx, sI_nn in enumerate(nn_list):
				if verbose:
					print(f"{pre_del}{sI_nn_idx+1}/{len(nn_list)}", sep = '', end = '')
					pre_del = (len(str(sI_nn_idx+1))+len(str(len(nn_list)))+1)*'\b'
				temp,_ , _ = sI.compute_structure_index(emb_signal,label,nbins,emb_space,0,nn=sI_nn,
																	vmin=label_lim[0], vmax=label_lim[1])
				sI_emb[nn_idx,sI_nn_idx,label_idx] = temp

		if verbose:
			print("\n\tComputing R2s:")

		#1. Check decoder ability
		R2s_temp, _ = dec.decoders_1D(x_base_signal=base_signal,y_signal_list=label_list,n_splits=n_splits, n_dims = dim,
								nn = nn, emb_list = ['umap'],decoder_list = decoder_list, 
								trial_signal=trial_signal,verbose=verbose)
		for dec_idx, dec_name in enumerate(decoder_list):
			R2s_nn[nn_idx,:, :,dec_idx] = R2s_temp['umap'][dec_name][:,:,0]

	output_dict = {
		'sI_og': sI_og,
		'trust_dim': trust_dim,
		'trust_dim_val': trust_dim_val,
		'cont_dim': cont_dim,
		'cont_dim_val': cont_dim_val,
		'sI_emb': sI_emb,
		'R2s': R2s_nn,
		'emb_list': emb_list,
		'params':kwargs
	}
	return output_dict


@gu.check_inputs_for_pd
def compute_umap_dim(base_signal=None, label_signal = None, trial_signal = None, **kwargs):

	if 'nn_list' in kwargs:
		nn_list = kwargs['nn_list']
	else:
		nn_list = [3, 10, 20, 30, 60, 100, 200]
		kwargs['nn_list'] = nn_list

	if 'min_dist' in kwargs:
		min_dist = kwargs['min_dist']
	else:
		min_dist = 0.75
		kwargs['min_dist'] = min_dist

	if 'nn' in kwargs:
		nn = kwargs['nn']
	else:
		nn = 60
		kwargs['nn'] = nn

	if 'max_dim' in kwargs:
		max_dim = kwargs['max_dim']
	else:
		max_dim = 12
		kwargs['max_dim'] = max_dim

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
		kwargs['verbose'] = verbose

	label_list = list()
	for label in label_signal:
		if label.ndim == 2:
			label = label[:,0]
		label_list.append(label)
	label_limits = [(np.percentile(label,5), np.percentile(label,95)) for label in label_list]    

	sI_dim = np.zeros((max_dim, len(nn_list), len(label_list)))
	R2s_dim = np.zeros((max_dim, n_splits, len(label_list), len(decoder_list)))
	trust_dim = np.zeros((max_dim,len(nn_list)))
	cont_dim = np.zeros((max_dim, len(nn_list)))


	if verbose:
		print("Computing rank indices og space...", end = '', sep = '')
	base_signal_indices = dim_validation.compute_rank_indices(base_signal)
	if verbose:
		print("\b\b\b: Done")

	for dim in range(max_dim):
		emb_space = np.arange(dim+1)
		if verbose:
			print(f"Dim: {dim+1} ({dim+1}/{max_dim})")

		model = umap.UMAP(n_neighbors = nn, n_components =dim+1, min_dist=0.75)
		if verbose:
			print("\tFitting model...", sep= '', end = '')
		emb_signal = model.fit_transform(base_signal)
		if verbose:
			print("\b\b\b: Done")
		#1. Compute trustworthiness
		if verbose:
			print("\tComputing trustworthiness...", sep= '', end = '')
		temp = dim_validation.trustworthiness_vector(base_signal, emb_signal ,nn_list[-1], indices_source = base_signal_indices)
		trust_dim[dim,:] = temp[nn_list]
		if verbose:
			print(f"\b\b\b: {np.mean(trust_dim[dim,:]):.2f}")

		#2. Compute continuity
		if verbose:
			print("\tComputing continuity...", sep= '', end = '')
		temp = dim_validation.continuity_vector(base_signal, emb_signal ,nn_list[-1])
		cont_dim[dim,:] = temp[nn_list]
		if verbose:
			print(f"\b\b\b: {np.mean(cont_dim[dim,:]):.2f}")

		#3. Compute sI
		for label_idx, label in enumerate(label_list):
			label_lim = label_limits[label_idx]
			if verbose:
				print(f"\tComputing sI {label_idx+1}/{len(label_list)}: X/X", sep='', end = '')
				pre_del = '\b'*3

			if len(np.unique(label))<10:
				nbins = len(np.unique(label))
			else:
				nbins = 10
			for sI_nn_idx, sI_nn in enumerate(nn_list):
				if verbose:
					print(f"{pre_del}{sI_nn_idx+1}/{len(nn_list)}", sep = '', end = '')
					pre_del = (len(str(sI_nn_idx+1))+len(str(len(nn_list)))+1)*'\b'
				temp,_ , _ = sI.compute_structure_index(emb_signal,label,nbins,emb_space,0,nn=sI_nn,
																	vmin=label_lim[0], vmax=label_lim[1])
				sI_dim[dim,sI_nn_idx,label_idx] = temp
			if verbose:
				print(f" - Mean result: {np.nanmean(sI_dim[dim,:,label_idx]):.2f}")

		#4. Check decoder ability
		if verbose:
			print("\tComputing R2s:")
		R2s_temp, _ = dec.decoders_1D(x_base_signal=base_signal,y_signal_list=label_list,n_splits=n_splits, n_dims = dim+1,
								nn = nn, emb_list = ['umap'] ,decoder_list = decoder_list, trial_signal=trial_signal,verbose=verbose)
		for dec_idx, dec_name in enumerate(decoder_list):
			R2s_dim[dim,:, :,dec_idx] = R2s_temp['umap'][dec_name][:,:,0]

		if verbose:
			print(f"\t\tMean result: {np.nanmean(R2s_dim[dim,:, :, :]):.2f}")	
	return trust_dim, cont_dim, sI_dim, R2s_dim, kwargs


def check_rotation_params(pd_struct_pre, pd_struct_rot, signal_field, save_dir, **kwargs):

	if 'nn_list' in kwargs:
		nn_list = kwargs['nn_list']
	else:
		nn_list = [5,10,15,20,30,40,50,60,80,100,200,500,1000]
		kwargs['nn_list'] = nn_list

	if 'sI_nn_list' in kwargs:
		sI_nn_list = kwargs['sI_nn_list']
	else:
		sI_nn_list = [3, 10, 20, 50, 100, 200]
		kwargs['sI_nn_list'] = sI_nn_list
    
	if 'dim' in kwargs:
		dim = kwargs['dim']
	else:
		dim = 3
		kwargs['dim'] = 3

	if 'n_iter' in kwargs:
		n_iter = kwargs['n_iter']
	else:
		n_iter = 5
		kwargs['n_iter'] = n_iter    

	if 'verbose' in kwargs:
		verbose = kwargs['verbose']
	else:
		verbose = False
		kwargs['verbose'] = verbose
	kwargs["signal_field"] = signal_field
	sI_nn = np.zeros((len(nn_list), 2, len(sI_nn_list), n_iter))
	angle_nn = np.zeros((len(nn_list), 4, n_iter))

	if 'dir_mat' not in pd_struct_pre.columns:
		pd_struct_pre["dir_mat"] = [np.zeros((pd_struct_pre["pos"][idx].shape[0],1)).astype(int)+
									('L' == pd_struct_pre["dir"][idx])+ 2*('R' == pd_struct_pre["dir"][idx])
									for idx in pd_struct_pre.index]

	if 'dir_mat' not in pd_struct_rot.columns:
		pd_struct_rot["dir_mat"] = [np.zeros((pd_struct_rot["pos"][idx].shape[0],1)).astype(int)+
									('L' == pd_struct_rot["dir"][idx])+ 2*('R' == pd_struct_rot["dir"][idx])
									for idx in pd_struct_rot.index]


	signal_p = copy.deepcopy(np.concatenate(pd_struct_pre[signal_field].values, axis=0))
	pos_p = copy.deepcopy(np.concatenate(pd_struct_pre['pos'].values, axis=0))
	dir_mat_p = copy.deepcopy(np.concatenate(pd_struct_pre['dir_mat'].values, axis=0))

	signal_r = copy.deepcopy(np.concatenate(pd_struct_rot[signal_field].values, axis=0))
	pos_r = copy.deepcopy(np.concatenate(pd_struct_rot['pos'].values, axis=0))
	dir_mat_r = copy.deepcopy(np.concatenate(pd_struct_rot['dir_mat'].values, axis=0))

	index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
	concat_signal = np.vstack((signal_p, signal_r))

	html = '<HTML>\n'
	html = html + '<style>\n'
	html = html + 'h1 {text-align: center;}\n'
	html = html + 'h2 {text-align: center;}\n'
	html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
	html = html + '</style>\n'
	html = html + f"<h1>Rotation params - {pd_struct_pre['mouse'][0]}</h1>\n<br>\n"    #Add title
	html = html + f"<br><h2>{datetime.now().strftime('%d/%m/%y %H:%M:%S')}</h2><br>\n"    #Add subtitle

	for nn_idx, nn_val in enumerate(nn_list):
		if verbose:
			print(f"Neighbours: {nn_val} ({nn_idx+1}/{len(nn_list)})")

		for ite in range(n_iter):
			if verbose:
				print(f"\tIteration {ite+1}/{n_iter}")

			#1. Project data
			model = umap.UMAP(n_neighbors = nn_val, n_components =dim, min_dist=0.75)
			if verbose:
				print("\tFitting model...", sep= '', end = '')
			concat_emb = model.fit_transform(concat_signal)

			emb_p = concat_emb[index[:,0]==0,:]
			emb_r = concat_emb[index[:,0]==1,:]
			if verbose:
				print("Done")

			#2. Compute structure index
			emb_space = np.linspace(0,emb_p.shape[1]-1, emb_p.shape[1]).astype(int)
			posLimits_p = [np.percentile(pos_p[:,0],5), np.percentile(pos_p[:,0],95)]
			posLimits_r = [np.percentile(pos_r[:,0],5), np.percentile(pos_r[:,0],95)]

			if verbose:
				print("\tSI: X/X", end = '', sep = '')
				pre_del = '\b\b\b'

			for sI_nn_idx, sI_nn_val in enumerate(sI_nn_list):
				if verbose:
					print(f"{pre_del}{sI_nn_idx+1}/{len(sI_nn_list)}", sep = '', end = '')
					pre_del =  (len(str(nn_idx+1))+len(str(len(sI_nn_list)))+1)*'\b'

				temp,_ , _ = sI.compute_structure_index(emb_p,pos_p[:,0],10,emb_space,0,nn=sI_nn_val,
																	vmin=posLimits_p[0],vmax=posLimits_p[1])
				sI_nn[nn_idx, 0, sI_nn_idx, ite] = temp

				temp,_ , _ = sI.compute_structure_index(emb_r,pos_r[:,0],10,emb_space,0,nn=sI_nn_val,
																	vmin=posLimits_r[0],vmax=posLimits_r[1])
				sI_nn[nn_idx, 1, sI_nn_idx, ite] = temp

			#3. Align and compute rotation-angle
			if verbose:
				print("\n\tAligning manifolds...", sep= '', end = '')
			#TAB, RAB = dec.align_manifolds_1D(emb_p, emb_r, pos_p[:,0], pos_r[:,0],ndims = 2, nCentroids = 10)
			#_,_,Z = get_angle_from_rot(RAB)
			#angle_nn[nn_idx, 0] = Z 

			TAB, RAB = dec.align_manifolds_1D(emb_p, emb_r, pos_p[:,0], pos_r[:,0], ndims = dim, nCentroids = 10)
			X,Y,Z = get_angle_from_rot(RAB)
			angle_nn[nn_idx, 0, ite] = X
			angle_nn[nn_idx, 1, ite] = Y
			angle_nn[nn_idx, 2, ite] = Z
	        
			tr = (np.trace(RAB)-1)/2
			if abs(tr)>1:
				tr = round(tr,2)
				if abs(tr)>1:
					tr = np.nan
			angle_nn[nn_idx, 3, ite] = math.acos(tr)*180/np.pi
			if verbose:
				print(f"\b\b\b: {angle_nn[nn_idx, 3, ite]:.2f}º - Done")

		fig= plt.figure(figsize = (12, 3))

		ax = plt.subplot(1,4,1, projection='3d')
		ax.scatter(*emb_p[:,:3].T, c=dir_mat_p[:,0])
		ax.set_title('num neigh: ' + str(nn_val))
		ax.set_xlabel('Dim 1', labelpad= -8)
		ax.set_ylabel('Dim 2', labelpad= -8)
		ax.set_zlabel('Dim 3', labelpad= -8)

		ax = plt.subplot(1,4,2, projection='3d')
		ax.set_title(f"SI pre:{np.mean(sI_nn[nn_idx, 0, :, :]):2f}")

		ax.scatter(*emb_p[:,:3].T, c=pos_p[:,0], cmap = plt.cm.magma)
		ax.set_xlabel('Dim 1', labelpad= -8)
		ax.set_ylabel('Dim 2', labelpad= -8)
		ax.set_zlabel('Dim 3', labelpad= -8)

		ax = plt.subplot(1,4,3, projection='3d')
		ax.scatter(*emb_r[:,:3].T, c=dir_mat_r[:,0])
		ax.set_title(f"Rot angle: {np.nanmean(angle_nn[nn_idx, 3, :]):.2f}")
		ax.set_xlabel('Dim 1', labelpad= -8)
		ax.set_ylabel('Dim 2', labelpad= -8)
		ax.set_zlabel('Dim 3', labelpad= -8)

		ax = plt.subplot(1,4,4, projection='3d')
		ax.scatter(*emb_r[:,:3].T, c=pos_r[:,0], cmap = plt.cm.magma)
		ax.set_title(f"SI pos:{np.mean(sI_nn[nn_idx, 1, :, :]):2f}")
		ax.set_xlabel('Dim 1', labelpad= -8)
		ax.set_ylabel('Dim 2', labelpad= -8)
		ax.set_zlabel('Dim 3', labelpad= -8)
		plt.tight_layout() 
		
		tmpfile = BytesIO()
		fig.savefig(tmpfile, format='png')
		encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
		html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
		plt.close(fig)

	with open(os.path.join(save_dir, f"{pd_struct_pre['mouse'][0]}_rotation_study_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
		f.write(html)
	return angle_nn, sI_nn, kwargs


def get_angle_from_rot(R21):
	if R21.shape[0]>=3:
		X = math.atan2(R21[2,1], R21[2,2])*180/np.pi 
		Y = math.atan2(-R21[2,0], np.sqrt(R21[2,1]**2+R21[2,2]**2))*180/np.pi
		Z = math.atan2(R21[1,0], R21[0,0])*180/np.pi
	elif R21.shape[0]==2:
		X = 0
		Y = 0
		Z = math.atan2(R21[1,0], R21[0,0])*180/np.pi
	return X,Y,Z


def check_rotation_emb_quality(pd_struct_pre, pd_struct_rot, signal_field,label_list, save_dir, **kwargs):

	if 'nn' in kwargs:
		nn = kwargs['nn']
	else:
		nn = 60
		kwargs['nn'] = nn

	if 'min_dist' in kwargs:
		min_dist = kwargs['min_dist']
	else:
		min_dist = 0.75
		kwargs['min_dist'] = min_dist

	if 'max_dim' in kwargs:
		max_dim = kwargs['max_dim']
	else:
		max_dim = 10
		kwargs['max_dim'] = max_dim

	if 'sI_nn_list' in kwargs:
		sI_nn_list = kwargs['sI_nn_list']
	else:
		sI_nn_list = [3, 10, 20, 50, 100, 200]
		kwargs['sI_nn_list'] = sI_nn_list

	if 'verbose' in kwargs:
		verbose = kwargs['verbose']
	else:
		verbose = False
		kwargs['verbose'] = verbose

	kwargs["signal_field"] = signal_field

	sI_dim = np.zeros((max_dim, len(sI_nn_list),len(label_list),2))
	trust_dim = np.zeros((max_dim,2))
	cont_dim = np.zeros((max_dim,2))

	if 'dir_mat' not in pd_struct_pre.columns:
		pd_struct_pre["dir_mat"] = [np.zeros((pd_struct_pre["pos"][idx].shape[0],1)).astype(int)+
									('L' == pd_struct_pre["dir"][idx])+ 2*('R' == pd_struct_pre["dir"][idx])
									for idx in pd_struct_pre.index]

	if 'dir_mat' not in pd_struct_rot.columns:
		pd_struct_rot["dir_mat"] = [np.zeros((pd_struct_rot["pos"][idx].shape[0],1)).astype(int)+
									('L' == pd_struct_rot["dir"][idx])+ 2*('R' == pd_struct_rot["dir"][idx])
									for idx in pd_struct_rot.index]

	signal_p = gu.pd_to_array_translator(pd_struct_pre, signal_field)
	label_p = list()
	for sub_label_idx, sub_label in enumerate(label_list):
		temp_label = gu.pd_to_array_translator(pd_struct_pre, sub_label)
		if temp_label.ndim == 2:
			temp_label = temp_label[:,0]
		label_p.append(temp_label)
	label_p_limits = [(np.percentile(label,5), np.percentile(label,95)) for label in label_p]

	signal_r = gu.pd_to_array_translator(pd_struct_rot, signal_field)
	label_r = list()
	for sub_label_idx, sub_label in enumerate(label_list):
		temp_label = gu.pd_to_array_translator(pd_struct_rot, sub_label)
		if temp_label.ndim == 2:
			temp_label = temp_label[:,0]
		label_r.append(temp_label)
	label_r_limits = [(np.percentile(label,5), np.percentile(label,95)) for label in label_r]    

	index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
	concat_signal = np.vstack((signal_p, signal_r))

	if verbose:
		print("Computing rank indices pre...", end = '', sep = '')
	signal_indices_p = dim_validation.compute_rank_indices(signal_p)
	if verbose:
		print("Done\nComputing rank indices rot...", end = '', sep = '')
	signal_indices_r = dim_validation.compute_rank_indices(signal_r)
	if verbose:
		print("Done")

	for dim in range(max_dim):
		emb_space = np.linspace(0, dim, dim+1).astype(int)
		if verbose:
			print(f"Dim: {dim+1}/{max_dim}")

		#0. Fit model
		model = umap.UMAP(n_neighbors = nn, n_components =dim+1, min_dist=0.75)
		if verbose:
			print("\tFitting model...", sep= '', end = '')
		concat_emb = model.fit_transform(concat_signal)

		emb_p = concat_emb[index[:,0]==0,:]
		emb_r = concat_emb[index[:,0]==1,:]
		if verbose:
			print("Done")

		#1. Structure index
		if verbose:
			print("\tSI: X/X", end = '', sep = '')
			pre_del = '\b\b\b'

		for sub_label_idx in range(len(label_list)):
			label_lim_p = label_p_limits[sub_label_idx]
			label_lim_r = label_r_limits[sub_label_idx]
			if verbose:
				print(f"{pre_del}{label_list[sub_label_idx]} {sub_label_idx+1}/{len(label_list)}", sep = '', end = '')
				pre_del =  (len(label_list[sub_label_idx])+1+len(str(sub_label_idx+1))+len(str(len(label_list)))+1)*'\b'

			if 'dir_mat' in label_list[sub_label_idx]:
				nbin = 3
			else:
				nbin = 10
			for sI_nn_idx, sI_nn_val in enumerate(sI_nn_list):
				temp,_ , _ = sI.compute_structure_index(emb_p,label_p[sub_label_idx],nbin,emb_space,0,nn=sI_nn_val,
																	vmin=label_lim_p[0], vmax=label_lim_p[1])

				sI_dim[dim, sI_nn_idx,sub_label_idx,0] = temp

				temp,_ , _ = sI.compute_structure_index(emb_r,label_r[sub_label_idx],nbin,emb_space,0,nn=sI_nn_val,
																	vmin=label_lim_r[0], vmax=label_lim_r[1])
				sI_dim[dim, sI_nn_idx,sub_label_idx,1] = temp

		#2. Trustworthiness
		if verbose:
			print("\n\tComputing trustworthiness...", sep= '', end = '')
		trust_dim[dim, 0] = dim_validation.trustworthiness_vector(signal_p, emb_p ,nn, indices_source = signal_indices_p)[-1]
		trust_dim[dim, 1] = dim_validation.trustworthiness_vector(signal_r, emb_r ,nn, indices_source = signal_indices_r)[-1]
		if verbose:
			print(f"\b\b\b:{np.mean(trust_dim[dim, :]):.2f}")
		#3. Continuity
		if verbose:
			print("\tComputing continuity...", sep= '', end = '')
		cont_dim[dim, 0] = dim_validation.continuity_vector(signal_p, emb_p ,nn)[-1]
		cont_dim[dim, 1] = dim_validation.continuity_vector(signal_r, emb_r ,nn)[-1]
		if verbose:
			print(f"\b\b\b:{np.mean(cont_dim[dim, :]):.2f}")

	
	html = '<HTML>\n'
	html = html + '<style>\n'
	html = html + 'h1 {text-align: center;}\n'
	html = html + 'h2 {text-align: center;}\n'
	html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
	html = html + '</style>\n'
	html = html + f"<h1>Embedding quality - {pd_struct_pre['mouse'][0]}</h1>\n<br>\n"    #Add title
	html = html + f"<br><h2>{datetime.now().strftime('%d/%m/%y %H:%M:%S')}</h2><br>\n"    #Add subtitle


	fig= plt.figure(figsize = (13, 5))
	dim_space = np.linspace(1,max_dim, max_dim).astype(int)
	fig.text(0, 0.75, "pre",horizontalalignment='center', 
			rotation = 'vertical', verticalalignment='center', fontsize = 20)

	fig.text(0, 0.25, "rot",horizontalalignment='center', 
			rotation = 'vertical', verticalalignment='center', fontsize = 20)

	for label_idx in range(len(label_list)):
		ax = plt.subplot(2,len(label_list)+1,label_idx+1)
		b = ax.imshow(sI_dim[:,:,label_idx, 0].T, vmin = 0.25, vmax = 1, aspect = 'auto', cmap = 'plasma')
		ax.set_ylabel('Range (local vs global)')
		ax.set_yticks(np.linspace(0, len(sI_nn_list)-1, len(sI_nn_list)), labels = sI_nn_list)
		ax.set_xlabel('Dimensions')
		ax.set_xticks(dim_space-1, labels = dim_space)
		ax.set_title(f"sI: {label_list[label_idx]}")

		if label_idx == (len(label_list)-1):
			fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)

		ax = plt.subplot(2,len(label_list)+1,label_idx+1+len(label_list)+1)
		b = ax.imshow(sI_dim[:,:,label_idx, 1].T, vmin = 0.25, vmax = 1, aspect = 'auto', cmap = 'plasma')
		ax.set_ylabel('Range (local vs global)')
		ax.set_yticks(np.linspace(0, len(sI_nn_list)-1, len(sI_nn_list)), labels = sI_nn_list)
		ax.set_xlabel('Dimensions')
		ax.set_xticks(dim_space-1, labels = dim_space)

		if label_idx == (len(label_list)-1):
			fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)

	ax = plt.subplot(2, len(label_list)+1,len(label_list)+1)
	ax.plot(dim_space, trust_dim[:,0], label = 'trustworthiness')
	ax.plot(dim_space, cont_dim[:,0], label = 'continuity')
	ax.set_ylim([0.25,1])
	ax.set_xlim([0.8, dim_space[-1]+0.2])
	ax.set_xticks(dim_space.astype(int))
	ax.set_xlabel('Dimensions')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.legend()

	ax = plt.subplot(2, len(label_list)+1,2*(len(label_list)+1))
	ax.plot(dim_space, trust_dim[:,1], label = 'trustworthiness')
	ax.plot(dim_space, cont_dim[:,1], label = 'continuity')
	ax.set_ylim([0.25,1])
	ax.set_xlim([0.8, dim_space[-1]+0.2])
	ax.set_xticks(dim_space.astype(int))
	ax.set_xlabel('Dimensions')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.legend(fontsize=8)

	plt.tight_layout() 
	tmpfile = BytesIO()
	fig.savefig(tmpfile, format='png')
	encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
	html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
	plt.close(fig)

	with open(os.path.join(save_dir, f"{pd_struct_pre['mouse'][0]}_emb_quality_study_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
		f.write(html)
        
	return sI_dim, trust_dim, cont_dim, kwargs


@gu.check_inputs_for_pd
def compute_umap_to_npoints(base_signal=None, label_signal = None, **kwargs):
	import warnings
	warnings.filterwarnings("ignore")

	if 'nn_list' in kwargs:
		nn_list = kwargs['nn_list']
	else:
		nn_list = [3, 10, 20, 30, 60, 100, 200]
		kwargs['nn_list'] = nn_list

	if 'min_dist' in kwargs:
		min_dist = kwargs['min_dist']
	else:
		min_dist = 0.75
		kwargs['min_dist'] = min_dist

	if 'nn' in kwargs:
		nn = kwargs['nn']
	else:
		nn = 60
		kwargs['nn'] = nn

	if 'max_dim' in kwargs:
		max_dim = kwargs['max_dim']
	else:
		max_dim = 12
		kwargs['max_dim'] = max_dim

	if 'min_points' in kwargs:
		min_points = kwargs['min_points']
	else:
		min_points = 1200
		kwargs['min_points'] = min_points

	if 'max_points' in kwargs:
		max_points = kwargs['max_points']
	else:
		max_points = 12000
		kwargs['max_points'] = max_points

	if 'n_steps' in kwargs:
		n_steps = kwargs['n_steps']
	else:
		n_steps = 10
		kwargs['n_steps'] = n_steps

	if 'n_splits' in kwargs:
		n_splits = kwargs['n_splits']
	else:
		n_splits = 10
		kwargs['n_splits'] = n_splits

	if 'verbose' in kwargs:
		verbose = kwargs['verbose']
	else:
		verbose = False
		kwargs['verbose'] = verbose

	#get point_list containing the number of points to be included on each iteration
	num_points = base_signal.shape[0]
	point_list = np.unique(np.linspace(min_points, max_points,n_steps,dtype=int))
	kwargs["og_point_list"] = point_list
	point_list = point_list[point_list<=num_points]

	kwargs["point_list"] = point_list
	label_list = list()
	for label in label_signal:
		if label.ndim == 2:
			label = label[:,0]
		label_list.append(label)

	points_picked_list = np.zeros((n_steps,n_splits,num_points)).astype(bool)

	inner_dim = np.zeros((n_steps, n_splits))*np.nan
	inner_dim_radii_vs_nn = np.zeros((n_steps, n_splits, 1000, 2))*np.nan

	trust_dim = np.zeros((n_steps, n_splits,len(nn_list)))*np.nan
	trust_dim_values = np.zeros((n_steps, n_splits, max_dim,len(nn_list)))*np.nan

	cont_dim = np.zeros((n_steps, n_splits,len(nn_list)))*np.nan
	cont_dim_values = np.zeros((n_steps, n_splits, max_dim,len(nn_list)))*np.nan

	sI_values = np.zeros((n_steps, n_splits, max_dim,len(nn_list), len(label_list)))*np.nan
    
	#Iterate over each one
	dim_space = np.linspace(1,max_dim, max_dim).astype(int)
	for npoint_idx, npoint_val in enumerate(point_list):
		if verbose:
			print(f"Checking number of points: {npoint_val} ({npoint_idx+1}/{n_steps}):")
			print("\tIteration X/X", sep= '', end = '')
			pre_del = '\b\b\b'

		for split_idx in range(n_splits):
			if verbose:
				print(pre_del, f"{split_idx+1}/{n_splits}", sep = '', end = '')
				pre_del = (len(str(split_idx+1))+len(str(n_splits))+1)*'\b'

			points_picked = random.sample(list(np.arange(num_points).astype(int)), npoint_val)
			points_picked_list[npoint_idx, split_idx, points_picked] = True 
			it_signal = copy.deepcopy(base_signal[points_picked,:])
			it_label_list = [label[points_picked] for label in label_list]
			it_label_limits = [(np.percentile(label,5), np.percentile(label,95)) for label in it_label_list]

			#1. Inner dim
			m , radius, neigh = compute_dim.compute_inner_dim(it_signal)
			inner_dim[npoint_idx, split_idx] = m
			inner_dim_radii_vs_nn[npoint_idx, split_idx, :radius.shape[0],:] = np.hstack((radius, neigh))

			base_signal_indices = dim_validation.compute_rank_indices(it_signal)

			for dim in range(max_dim):
				#0. Model
				emb_space = np.arange(dim+1)
				model = umap.UMAP(n_neighbors = nn, n_components =dim+1, min_dist=0.75)
				emb_signal = model.fit_transform(it_signal)
				#1. Compute trustworthiness
				temp = dim_validation.trustworthiness_vector(it_signal, emb_signal ,nn_list[-1], indices_source = base_signal_indices)
				trust_dim_values[npoint_idx, split_idx, dim,:] = temp[nn_list]

				#2. Compute continuity
				temp = dim_validation.continuity_vector(it_signal, emb_signal ,nn_list[-1])
				cont_dim_values[npoint_idx, split_idx, dim,:] = temp[nn_list]

				#3. Compute sI
				for label_idx, label in enumerate(it_label_list):
					label_lim = it_label_limits[label_idx]
					if len(np.unique(label))<10:
						nbins = len(np.unique(label))
					else:
						nbins = 10
					for sI_nn_idx, sI_nn in enumerate(nn_list):
						try:
							temp,_ , _ = sI.compute_structure_index(emb_signal,label,nbins,emb_space,0,nn=sI_nn,
																				vmin=label_lim[0], vmax=label_lim[1])
						except:
							temp = np.nan
						sI_values[npoint_idx,split_idx,dim,sI_nn_idx, label_idx] = temp
						
			#Compute trust and cont dims
			for nn_idx in range(len(nn_list)):
				kl = KneeLocator(dim_space, trust_dim_values[npoint_idx, split_idx, :,nn_idx], curve = "concave", direction = "increasing")
				if kl.knee:
					dim = kl.knee
				else:
					dim = np.nan
				trust_dim[npoint_idx, split_idx, nn_idx] = dim

				kl = KneeLocator(dim_space, cont_dim_values[npoint_idx, split_idx, :,nn_idx], curve = "concave", direction = "increasing")
				if kl.knee:
					dim = kl.knee
				else:
					dim = np.nan
				cont_dim[npoint_idx, split_idx, nn_idx] = dim
	
		if verbose:
			print(": Mean results: ")
			print(f"\t\tInner dim: {np.nanmean(inner_dim[npoint_idx,:]):.2f} \u00B1 {np.nanstd(inner_dim[npoint_idx,:]):.2f}")	
			print(f"\t\tTrust dim: {np.nanmean(trust_dim[npoint_idx,:,:]):.2f} \u00B1 {np.nanstd(trust_dim[npoint_idx,:,:]):.2f}")	
			print(f"\t\tCont dim: {np.nanmean(cont_dim[npoint_idx,:,:]):.2f} \u00B1 {np.nanstd(cont_dim[npoint_idx,:,:]):.2f}")
			print(f"\t\tsI: {np.nanmean(sI_values[npoint_idx,:,:,0]):.2f} \u00B1 {np.nanstd(sI_values[npoint_idx,:,:,0]):.2f}")
 
	output_dict = {
		'inner_dim':inner_dim,
		'trust_dim': trust_dim,
		'trust_dim_values': trust_dim_values,
		'cont_dim': cont_dim,
		'cont_dim_values': cont_dim_values,
		'sI_values': sI_values,
		'points_picked_list':points_picked_list,
		'params': kwargs}

	return output_dict


@gu.check_inputs_for_pd
def compute_umap_to_ncells(base_signal=None, label_signal = None,trial_signal = None, **kwargs):
	import warnings
	warnings.filterwarnings("ignore")

	if 'nn_list' in kwargs:
		nn_list = kwargs['nn_list']
	else:
		nn_list = [3, 10, 20, 30, 60, 100, 200]
		kwargs['nn_list'] = nn_list

	if 'min_dist' in kwargs:
		min_dist = kwargs['min_dist']
	else:
		min_dist = 0.75
		kwargs['min_dist'] = min_dist

	if 'nn' in kwargs:
		nn = kwargs['nn']
	else:
		nn = 60
		kwargs['nn'] = nn

	if 'max_dim' in kwargs:
		max_dim = kwargs['max_dim']
	else:
		max_dim = 12
		kwargs['max_dim'] = max_dim

	if 'min_cells' in kwargs:
		min_cells = kwargs['min_cells']
	else:
		min_cells = 5
		kwargs['min_cells'] = min_cells

	if 'max_cells' in kwargs:
		max_cells = kwargs['max_cells']
	else:
		max_cells = 200
		kwargs['max_cells'] = max_cells

	if 'n_steps' in kwargs:
		n_steps = kwargs['n_steps']
	else:
		n_steps = 10
		kwargs['n_steps'] = n_steps

	if 'n_splits' in kwargs:
		n_splits = kwargs['n_splits']
	else:
		n_splits = 10
		kwargs['n_splits'] = n_splits

	if 'n_folds' in kwargs:
		n_folds = kwargs['n_folds']
	else:
		n_folds = 5
		kwargs['n_folds'] = n_folds

	if 'decoder_list' in kwargs:
		decoder_list = kwargs['decoder_list']
	else:
		decoder_list = ["wf", "wc", "xgb", "svr"]

	if 'verbose' in kwargs:
		verbose = kwargs['verbose']
	else:
		verbose = False
		kwargs['verbose'] = verbose

	#get cell_list containing the number of points to be included on each iteration
	num_cells = base_signal.shape[1]
	cell_list = np.unique(np.logspace(np.log10(min_cells), np.log10(max_cells),n_steps,dtype=int))
	kwargs["og_num_cells"] = cell_list
	cell_list = cell_list[cell_list<=num_cells]
	kwargs["cell_list"] = cell_list

	label_list = list()
	for label in label_signal:
		if label.ndim == 2:
			label = label[:,0]
		label_list.append(label)
	label_limits = [(np.percentile(label,5), np.percentile(label,95)) for label in label_list]

	cells_picked_list = np.zeros((n_steps,n_splits,num_cells)).astype(bool)

	inner_dim = np.zeros((n_steps, n_splits))*np.nan
	inner_dim_radii_vs_nn = np.zeros((n_steps, n_splits, 1000, 2))*np.nan

	trust_dim = np.zeros((n_steps, n_splits,len(nn_list)))*np.nan
	trust_dim_values = np.zeros((n_steps, n_splits, max_dim,len(nn_list)))*np.nan

	cont_dim = np.zeros((n_steps, n_splits,len(nn_list)))*np.nan
	cont_dim_values = np.zeros((n_steps, n_splits, max_dim,len(nn_list)))*np.nan

	sI_values = np.zeros((n_steps, n_splits, max_dim,len(nn_list), len(label_list)))*np.nan
	R2s_values = np.zeros((n_steps, n_splits, n_folds, len(label_list), len(decoder_list)))*np.nan

	#Iterate over each one
	dim_space = np.linspace(1,max_dim, max_dim).astype(int)
	for ncell_idx, ncell_val in enumerate(cell_list):
		if verbose:
			print(f"Checking number of cells: {ncell_val} ({ncell_idx+1}/{n_steps}):")
			print("\tIteration X/X", sep= '', end = '')
			pre_del = '\b\b\b'

		for split_idx in range(n_splits):
			if verbose:
				print(pre_del, f"{split_idx+1}/{n_splits}", sep = '', end = '')
				pre_del = (len(str(split_idx+1))+len(str(n_splits))+1)*'\b'

			cells_picked = random.sample(list(np.arange(num_cells).astype(int)), ncell_val)
			cells_picked_list[ncell_idx, split_idx, cells_picked] = True 
			it_signal = copy.deepcopy(base_signal[:, cells_picked])

			#1. Inner dim
			m , radius, neigh = compute_dim.compute_inner_dim(it_signal)
			inner_dim[ncell_idx, split_idx] = m
			inner_dim_radii_vs_nn[ncell_idx, split_idx, :radius.shape[0],:] = np.hstack((radius, neigh))

			base_signal_indices = dim_validation.compute_rank_indices(it_signal)

			for dim in range(max_dim):
				#0. Model
				emb_space = np.arange(dim+1)
				model = umap.UMAP(n_neighbors = nn, n_components =dim+1, min_dist=0.75)
				emb_signal = model.fit_transform(it_signal)

				#1. Compute trustworthiness
				temp = dim_validation.trustworthiness_vector(it_signal, emb_signal ,nn_list[-1], indices_source = base_signal_indices)
				trust_dim_values[ncell_idx, split_idx, dim,:] = temp[nn_list]

				#2. Compute continuity
				temp = dim_validation.continuity_vector(it_signal, emb_signal ,nn_list[-1])
				cont_dim_values[ncell_idx, split_idx, dim,:] = temp[nn_list]

				#3. Compute sI
				for label_idx, label in enumerate(label_list):
					label_lim = label_limits[label_idx]
					if len(np.unique(label))<10:
						nbins = len(np.unique(label))
					else:
						nbins = 10
					for sI_nn_idx, sI_nn in enumerate(nn_list):
						try:
							temp,_ , _ = sI.compute_structure_index(emb_signal,label,nbins,emb_space,0,nn=sI_nn,
																				vmin=label_lim[0], vmax=label_lim[1])
						except:
							temp = np.nan
						sI_values[ncell_idx,split_idx,dim,sI_nn_idx, label_idx] = temp
						
			#Compute trust and cont dims
			for nn_idx in range(len(nn_list)):
				kl = KneeLocator(dim_space, trust_dim_values[ncell_idx, split_idx, :,nn_idx], curve = "concave", direction = "increasing")
				if kl.knee:
					dim = kl.knee
				else:
					dim = np.nan
				trust_dim[ncell_idx, split_idx, nn_idx] = dim

				kl = KneeLocator(dim_space, cont_dim_values[ncell_idx, split_idx, :,nn_idx], curve = "concave", direction = "increasing")
				if kl.knee:
					dim = kl.knee
				else:
					dim = np.nan
				cont_dim[ncell_idx, split_idx, nn_idx] = dim
			#Compute R2s
			R2s_temp, _ = dec.decoders_1D(x_base_signal=it_signal,y_signal_list=label_list,n_splits=n_folds, n_dims = 3, nn = nn,
								emb_list = ['umap'] ,decoder_list = decoder_list, trial_signal=trial_signal,verbose=False)

			for dec_idx, dec_name in enumerate(decoder_list):
				R2s_values[ncell_idx,split_idx, :,:,dec_idx] = R2s_temp['umap'][dec_name][:,:,0]

		if verbose:
			print(": Mean results: ")
			print(f"\t\tInner dim: {np.nanmean(inner_dim[ncell_idx,:]):.2f} \u00B1 {np.nanstd(inner_dim[ncell_idx,:]):.2f}")	
			print(f"\t\tTrust dim: {np.nanmean(trust_dim[ncell_idx,:,:]):.2f} \u00B1 {np.nanstd(trust_dim[ncell_idx,:,:]):.2f}")	
			print(f"\t\tCont dim: {np.nanmean(cont_dim[ncell_idx,:,:]):.2f} \u00B1 {np.nanstd(cont_dim[ncell_idx,:,:]):.2f}")
			print(f"\t\tsI: {np.nanmean(sI_values[ncell_idx,:,:,0]):.2f} \u00B1 {np.nanstd(sI_values[ncell_idx,:,:,0]):.2f}")
			print(f"\t\tR2s xgb: {np.nanmean(R2s_values[ncell_idx,:,:,0,2]):.2f} \u00B1 {np.nanstd(R2s_values[ncell_idx,:,:,0,2]):.2f}")

	output_dict = {
		'inner_dim':inner_dim,
		'trust_dim': trust_dim,
		'trust_dim_values': trust_dim_values,
		'cont_dim': cont_dim,
		'cont_dim_values': cont_dim_values,
		'sI_values': sI_values,
		'R2s_values': R2s_values,
		'cells_picked_list':cells_picked_list,
		'params': kwargs}

	return output_dict