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

from neural_manifold.dimensionality_reduction import validation as dim_validation 
from neural_manifold.dimensionality_reduction import compute_dimensionality as compute_dim


import math
import matplotlib.pyplot as plt
import umap
from datetime import datetime
import base64
from io import BytesIO
import os

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
				print("\n\t\tInner dimension: ",  sep = '', end = '')
			dim_kernel[ks_idx, assy_idx],_,_ = compute_dim.compute_inner_dim(base_signal=rates,min_neigh=2, 
																max_neigh = int(rates.shape[0]*0.1))
			if verbose and not np.isnan(dim_kernel[ks_idx, assy_idx]):
				print(f"{dim_kernel[ks_idx, assy_idx]}")
                
	return R2s_kernel, sI_kernel, dim_kernel


def check_rotation_params(pd_struct_pre, pd_struct_rot, signal_field,save_dir, **kwargs):

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

	if 'verbose' in kwargs:
		verbose = kwargs['verbose']
	else:
		verbose = False
		kwargs['verbose'] = verbose
	kwargs["signal_field"] = signal_field
	sI_nn = np.zeros((len(nn_list), 2, len(sI_nn_list)))
	angle_nn = np.zeros((len(nn_list), 4))

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

		#1. Project data
		model = umap.UMAP(n_neighbors = nn_val, n_components =3, min_dist=0.75)
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
			sI_nn[nn_idx, 0, sI_nn_idx] = temp

			temp,_ , _ = sI.compute_structure_index(emb_r,pos_r[:,0],10,emb_space,0,nn=sI_nn_val,
																vmin=posLimits_r[0],vmax=posLimits_r[1])
			sI_nn[nn_idx, 1, sI_nn_idx] = temp


		#3. Align and compute rotation-angle
		if verbose:
			print("\n\tAligning manifolds...", sep= '', end = '')
		TAB, RAB = dec.align_manifolds_1D(emb_p, emb_r, pos_p[:,0], pos_r[:,0],ndims = 2, nCentroids = 10)
		_,_,Z = get_angle_from_rot(RAB)
		angle_nn[nn_idx, 0] = Z 

		TAB, RAB = dec.align_manifolds_1D(emb_p, emb_r, pos_p[:,0], pos_r[:,0],ndims = 3, nCentroids = 10)
		X,Y,Z = get_angle_from_rot(RAB)
		angle_nn[nn_idx, 1] = X
		angle_nn[nn_idx, 2] = Y
		angle_nn[nn_idx, 3] = Z

		if verbose:
			print("Done")
		fig= plt.figure(figsize = (12, 3))

		ax = plt.subplot(1,4,1, projection='3d')
		p = ax.scatter(*emb_p.T, c=dir_mat_p[:,0])
		ax.set_title('NN: ' + str(nn_val))
		ax.set_xlabel('Dim 1', labelpad= -8)
		ax.set_ylabel('Dim 2', labelpad= -8)
		ax.set_zlabel('Dim 3', labelpad= -8)

		ax = plt.subplot(1,4,2, projection='3d')
		ax.set_title(f"SI pre:{np.mean(sI_nn[nn_idx, 0, :]):2f}")

		p = ax.scatter(*emb_p.T, c=pos_p[:,0], cmap = plt.cm.magma)
		ax.set_xlabel('Dim 1', labelpad= -8)
		ax.set_ylabel('Dim 2', labelpad= -8)
		ax.set_zlabel('Dim 3', labelpad= -8)

		ax = plt.subplot(1,4,3, projection='3d')
		p = ax.scatter(*emb_r.T, c=dir_mat_r[:,0])
		ax.set_title(f"2D angle: {angle_nn[nn_idx, 0]:.2f}")
		ax.set_xlabel('Dim 1', labelpad= -8)
		ax.set_ylabel('Dim 2', labelpad= -8)
		ax.set_zlabel('Dim 3', labelpad= -8)

		ax = plt.subplot(1,4,4, projection='3d')
		p = ax.scatter(*emb_r.T, c=pos_r[:,0], cmap = plt.cm.magma)
		ax.set_title(f"SI pos:{np.mean(sI_nn[nn_idx, 1, :]):2f}")
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
	if R21.shape[0]==3:
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