# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:58:00 2022

@author: Usuario
"""
import pandas as pd
import numpy as np
import copy
import warnings
from sklearn.metrics import pairwise_distances
from scipy.stats import linregress

from sklearn.manifold import Isomap
from scipy.stats import pearsonr

import umap
from umap import validation
from kneed import KneeLocator
import random

#INNER PACKAGE IMPORTS
from neural_manifold import general_utils as gu #translators (from Dataframe to np.ndarray)
from neural_manifold import structure_index as sI 
from neural_manifold.decoders import decoders_1D


def compute_inner_dim(input_object, field = None, min_neigh = 2, max_neigh = 2**5, verbose=False):
    '''Compute internal dimensionality of data array as described in: https://doi.org/10.1038/s41467-019-12724-2
    
    Parameters:
    ----------
        input_object (DataFrame or numpy Array): object containing the signal one wants to estimate the 
                        internal dimensionality.
        
    Optional parameters:
    --------------------
        field (string): if 'input_object' is a dataframe, name of column with the signal (otherwise set it 
                        to None, as per default).
        
        min_neigh (float): minimum number neighbours used to defined the range in which to compute the 
                        internal dimmensionality. If value smaller than 1 and the same for max_neigh, it is 
                        interpreted as fracion of total number of samples. 
                        
        max_neigh (float): maximum number neighbours used to defined the range in which to compute the 
                        internal dimmensionality. If value smaller than 1 and the same for min_neigh, it is 
                        interpreted as fracion of total number of samples. 
                           
    Returns:
    -------
        m (float): internal dimensionality of the data.
        
        radius (array): array containing the radius of the ball used to compute the number of neighbours in 
                        log2 base.
                        
        neigh  (array): mean number of neighbours for all samples for each ball radius (specified in radius)
                        in log2 base.
    '''
    #Check inputs
    if isinstance(input_object, pd.DataFrame):
        signal = gu.dataframe_to_1array_translator(input_object,field)
    elif isinstance(input_object,np.ndarray):
        signal = copy.deepcopy(input_object)
    else:
        raise ValueError("Input object has to be a dataframe or a numpy array.")
    
    #Check shape ratio
    if signal.shape[0]<signal.shape[1]:
        warnings.warn("Shape[0] of signal is smaller than Shape[1]. Note that " +
                      "function considers rows to be samples and cols to be features.",SyntaxWarning)
    #Check that array is 1D or 2D
    if signal.ndim>2:
        raise ValueError("Signal has more than 2 dimensions. Will only consider the first two.")
    #Check min_neigh, max_neigh numbers
    if min_neigh>=max_neigh:
        raise ValueError("'min_neigh' argument (%.2f) must be smaller than 'max_neigh' (%.2f) argument." %(min_neigh, max_neigh))
    if min_neigh<1 and max_neigh<1:
        if verbose:
            print("'min_neigh' and 'max_neigh' arguments smaller than 1. Interpreting them as fraction of total number of samples.")
        min_neigh = signal.shape[0]*min_neigh
        max_neigh = signal.shape[0]*max_neigh
        if verbose:
            print("Resulting in min_neigh=%i and max_neigh=%i" %(min_neigh, max_neigh))
    #Compute pairwise distances between all samples
    D_pw = pairwise_distances(signal,metric = 'euclidean')
    #Compute min distance between samples
    min_dist = np.min(D_pw)
    if min_dist == 0:
        min_dist = 1e-10
    #Compute max distance between samples
    max_dist = np.max(D_pw)
    #Define space of growing radius of balls in log2 base
    radius = np.logspace(np.log2(min_dist), np.log2(max_dist),1000,endpoint = True, base=2).reshape(-1,1)
    #For each radius check number of neighbours for each sample
    neigh = np.zeros((radius.shape[0],1))
    for idx, rad in enumerate(radius):
        temp = D_pw <= rad
        neigh[idx,0] = np.mean(np.sum(temp,axis=1)-1)
    #get rid of initial 0s (no neighbours)
    radius = np.log2(radius[neigh[:,0]>0,:])
    neigh = np.log2(neigh[neigh[:,0]>0,:])
    #Compute slope in the range of neighbours between points given by (min_neigh, max_neigh)
    lower_bound = np.log2(min_neigh)
    upper_bound = np.log2(max_neigh)
    if upper_bound<np.min(neigh):
        if verbose:
            print("'max_neigh' (%d) is smaller than the experimental minimum " %(max_neigh) +
                          "number of neighbours (%d). Redifining it to 50 percentile. " %(2**np.min(neigh)) +
                          "Resulting in %d neighbours." %(2**np.percentile(neigh, 50)))
        upper_bound = np.percentile(neigh, 50)  
    try:
        in_range_mask = np.all(np.vstack((neigh[:,0]<upper_bound, neigh[:,0]>lower_bound)).T, axis=1)
        radius_in_range = radius[in_range_mask, :] 
        neigh_in_range = neigh[in_range_mask, :]
        m = linregress(radius_in_range[:,0], neigh_in_range[:,0])[0]
        return m, radius, neigh
    except:
        print('Could not compute internal dimension. Probably check range (min_neigh, max_neigh)')
        return np.nan, radius, neigh
    

def compute_umap_trust_dim(input_object, field = None, n_neigh= 0.01, max_dim=10, min_dist = 0.75, return_emb = False, verbose = False):
    '''Compute dimensionality of data array according to UMAP trustworthiness (see Venna, Jarkko, and Samuel Kaski.
    "Local multidimensional scaling with controlled tradeoff between trustworthiness and continuity." Proceedings 
    of 5th Workshop on Self-Organizing Maps. 2005.)
    
    Parameters:
    -----------
        input_object (DataFrame or numpy Array): object containing the signal one wants to project using UMAP 
                        to estimate dimensionality.
        
    Optional parameters:
    -------------------
        field (string): if 'input_object' is a dataframe, name of column with the signal (otherwise set it 
                        to None, as per default).
        
        n_neigh (float): number of neighbours used to compute UMAP projection.
        
        max_dim (int): maximum number of dimensions to check (note this parameters is the main time-expensive)
        
        min_dist (float): minimum distance used to compute UMAP projection (values in (0,1]))
    
        return_emb (boolean): boolean specifing whether or not to return all UMAP embeddings computed.
        
        verbose (boolean): boolean indicating whether to pring verbose or not.
                           
    Returns:
    --------
        dim (int): dimensionality of the data according to UMAP trustworthiness.
        
        num_trust (array): array containing the trustworthiness values of each dimensionality iteration. This
                        array is the one used to find the knee defining the dimensionality of the data
                        
        return_emb_list (list): list containing the UMAP projections computed for each of the dimensionalities.
                        (only applicable if input parameter 'return_emb' set to True)
    '''
    #Check inputs
    if isinstance(input_object, pd.DataFrame):
        signal = gu.dataframe_to_1array_translator(input_object,field)
    elif isinstance(input_object,np.ndarray):
        signal = copy.deepcopy(input_object)
    else:
        raise ValueError("Input object has to be a dataframe or a numpy array.")
    #Check consistency of maximum dimensionality
    if max_dim>signal.shape[1]:
        if verbose:
            print("Maximum number of dimensions (%i) larger than original number of " %(max_dim)+
                          "dimensions (%i). Setting the first to the later." %(signal.shape[1]))
        max_dim = signal.shape[1]
    #Check number of neighbours
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("Resulting in %i neighbours." %(n_neigh))
    #check if user wants the different embedings backs
    if return_emb:
        return_emb_list = list()
    #initialize variables 
    num_trust = np.zeros((max_dim,1))*np.nan
    for dim in range(1, max_dim+1):
        if verbose:
            print('Checking dimension %i ' %(dim), sep='', end = '')
        emb = umap.UMAP(n_neighbors = n_neigh, n_components = dim, min_dist=min_dist).fit_transform(signal)
        num_trust[dim-1,0] = validation.trustworthiness_vector(signal, emb ,n_neigh)[-1]
        if return_emb:
            return_emb_list.append(emb)
        if dim>1:
            red_error = abs(num_trust[dim-1,0]-num_trust[dim-2,0])
            if verbose:
                print(': Error improvement of %.4f' %red_error, sep='')
        else:
            if verbose:
                print('')
    dim_space = np.linspace(1,max_dim, max_dim).astype(int)        
    kl = KneeLocator(dim_space, num_trust[:,0], curve = "concave", direction = "increasing")
    if kl.knee:
        dim = kl.knee
        if verbose:
            print('Final dimension: %d - Final error of %.4f' %(dim, 1-num_trust[dim-1,0]))
    else:
        dim = np.nan
        if verbose:
            print('Could estimate final dimension (knee not found). Returning nan.')
    if return_emb:
        return dim, num_trust, return_emb_list
    else:
        return dim, num_trust


def compute_isomap_resvar_dim(input_object, field = None, n_neigh= 0.01, max_dim=10, return_emb = False, verbose = False):
    '''Compute dimensionality of data array according to isomap residual variance.
    
    Parameters:
    -----------
        input_object (DataFrame or numpy Array): object containing the signal one wants to project using Isomap
                        to estimate dimensionality.
        
    Optional parameters:
    -------------------
        field (string): if 'input_object' is a dataframe, name of column with the signal (otherwise set it 
                        to None, as per default).
        
        n_neigh (float): number of neighbours used to compute Isomap projection.
        
        max_dim (int): maximum number of dimensions to check.
        
        return_emb (boolean): boolean specifing whether or not to return Isomap embeddings computed.
        verbose (boolean): boolean indicating whether to pring verbose or not.
                           
    Returns:
    --------
        dim (int): dimensionality of the data according to Isomap residual variance.
        
        res_var (array): array containing the residual variance values of each dimensionality iteration. This
                        array is the one used to find the knee defining the dimensionality of the data.
                        
        emb (array): Isomap embedding computed for each of the dimensionalities. (only applicable if input 
                        parameter 'return_emb' set to True)
    '''
    #Check inputs
    if isinstance(input_object, pd.DataFrame):
        signal = gu.dataframe_to_1array_translator(input_object,field)
    elif isinstance(input_object,np.ndarray):
        signal = copy.deepcopy(input_object)
    else:
        raise ValueError("Input object has to be a dataframe or a numpy array.")
    #Check consistency of maximum dimensionality
    if max_dim>signal.shape[1]:
        if verbose:
            print("Maximum number of dimensions (%i) larger than original number of " %(max_dim)+
                      "dimensions (%i). Setting the first to the later." %(signal.shape[1]))
        max_dim = signal.shape[1]
    #Check number of neighbours
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("Resulting in %i neighbours." %(n_neigh))
    #initialize isomap object
    iso_model = Isomap(n_neighbors = n_neigh, n_components = max_dim)
    #fit and project data
    emb = iso_model.fit_transform(signal)
    #compute residual variance
    res_var = np.zeros((max_dim, 1))
    for dim in range(1, max_dim+1):
        D_emb = pairwise_distances(emb[:,:dim], metric = 'euclidean') 
        res_var[dim-1,0] = 1 - pearsonr(np.matrix.flatten(iso_model.dist_matrix_),
                                            np.matrix.flatten(D_emb.astype('float32')))[0]**2
    #find knee in residual variance
    dim_space = np.linspace(1,max_dim, max_dim).astype(int)        
    kl = KneeLocator(dim_space, res_var[:,0], curve = "convex", direction = "decreasing")                                                      
    if kl.knee:
        dim = kl.knee
        if verbose:
            print('Final dimension: %d - Final residual variance: %.4f' %(dim, res_var[-1,0]))
    else:
        dim = np.nan
        if verbose:
            print('Could estimate final dimension (knee not found). Returning nan.')
    if return_emb:
        return dim, res_var, emb
    else:
        return dim, res_var
    
    
def compute_isomap_recerror_dim(input_object, field = None, n_neigh= 0.01, max_dim=10, return_emb = False, verbose = False):
    '''Compute dimensionality of data array according to isomap reconstruction error.
    
    Parameters:
    -----------
        input_object (DataFrame or numpy Array): object containing the signal one wants to project using Isomap
                        to estimate dimensionality.
        
    Optional parameters:
    -------------------
        field (string): if 'input_object' is a dataframe, name of column with the signal (otherwise set it 
                        to None, as per default).
        
        n_neigh (float): number of neighbours used to compute Isomap projection.
        
        max_dim (int): maximum number of dimensions to check.
        
        return_emb (boolean): boolean specifing whether or not to return Isomap embeddings computed.
        
        verbose (boolean): boolean indicating whether to pring verbose or not.
                           
    Returns:
    --------
        dim (int): dimensionality of the data according to Isomap reconstruction error.
        
        rec_error (array): array containing the reconstruction error of each dimensionality iteration. This
                        array is the one used to find the knee defining the dimensionality of the data
                        
        emb (array): Isomap embedding computed for each of the dimensionalities. (only applicable if input 
                        parameter 'return_emb' set to True)
    '''
    #Check inputs
    if isinstance(input_object, pd.DataFrame):
        signal = gu.dataframe_to_1array_translator(input_object,field)
    elif isinstance(input_object,np.ndarray):
        signal = copy.deepcopy(input_object)
    else:
        raise ValueError("Input object has to be a dataframe or a numpy array.")
    #Check consistency of maximum dimensionality
    if max_dim>signal.shape[1]:
        if verbose:
            print("Maximum number of dimensions (%i) larger than original number of " %(max_dim)+
                          "dimensions (%i). Setting the first to the later." %(signal.shape[1]))
        max_dim = signal.shape[1]
    #Check number of neighbours
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("Resulting in %i neighbours." %(n_neigh))
    #define kernel function
    K = lambda D: -0.5*((np.eye(D.shape[0])-(1/D.shape[0])).dot(np.square(D))).dot(np.eye(D.shape[0])-(1/D.shape[0]))
    #initialize isomap object
    iso_model = Isomap(n_neighbors = n_neigh, n_components = max_dim)
    #fit and project data
    emb = iso_model.fit_transform(signal)
    #compute Isomap kernel for input data once before going into the loop
    KD_signal = K(iso_model.dist_matrix_)
    #compute residual error
    rec_error = np.zeros((max_dim, 1))
    n_samples = signal.shape[0]
    for dim in range(1, max_dim+1):
        D_emb = pairwise_distances(emb[:,:dim], metric = 'euclidean')
        KD_emb = K(D_emb)
        rec_error[dim-1,0] = np.linalg.norm((KD_signal-KD_emb)/n_samples, 'fro')
    #find knee in reconstruction error
    dim_space = np.linspace(1,max_dim, max_dim).astype(int)        
    kl = KneeLocator(dim_space, rec_error[:,0], curve = "convex", direction = "decreasing")                                                      
    if kl.knee:
        dim = kl.knee
        if verbose:
            print('Final dimension: %d - Final reconstruction error: %.4f' %(dim, rec_error[-1,0]))
    else:
        dim = np.nan
        if verbose:
            print('Could estimate final dimension (knee not found). Returning nan.')
    if return_emb:
        return dim, rec_error, emb
    else:
        return dim, rec_error


def dim_to_number_cells(input_signal, field_signal = None, n_neigh = 0.01, n_steps = 15, input_label=None, 
                            min_cells = 5, n_splits = 5, verbose = True, input_trial = None, cells_to_include = 'all'):
    #check signal input
    if isinstance(input_signal, pd.DataFrame):
        signal = gu.dataframe_to_1array_translator(input_signal,field_signal)
    elif isinstance(input_signal,np.ndarray):
        signal = copy.deepcopy(input_signal)
    else:
        raise ValueError("Input object has to be a dataframe or a numpy array.")
    #check label list input
    if isinstance(input_label, np.ndarray): #user inputs an independent array for sI
            label_list = list([copy.deepcopy(input_label)])
    elif isinstance(input_label, str): #user specifies field_signal
        label_list = list([gu.dataframe_to_1array_translator(input_signal,input_label)])
    elif isinstance(input_label, list):
        if isinstance(input_label[0], str):
            label_list = gu.dataframe_to_manyarray_translator(input_signal,input_label)
        elif isinstance(input_label[0], np.ndarray):
            label_list = copy.deepcopy(input_label)
        else:
            raise ValueError("If 'input_label' is a list, it must be composed of either strings " +
                             "(referring to the columns of the input_signal dataframe), or arrays. " +
                             "However it was %s" %type(input_label[0]))
    else:
        raise ValueError(" 'input_label' must be either an array or a string (referring to the columns "+
                         "of the input_signal dataframe) (or a list composed of those). However it was %s"
                         %type(input_label))
    #Check if trial mat to use when spliting training/test for decoders
    if isinstance(input_trial, np.ndarray): #user inputs an independent array for sI
        trial_signal = copy.deepcopy(input_trial)
    elif isinstance(input_trial, str): #user specifies field_signal for sI but not a dataframe (then use input_signal)
        trial_signal = gu.dataframe_to_1array_translator(input_signal,input_trial)
    else:
        trial_signal = None 
    #Check if specific cells to include
    if isinstance(cells_to_include,str):
        if 'all' in cells_to_include:
            cells_to_include = np.arange(0, signal.shape[1],1)
    elif isinstance(cells_to_include, type(None)):
            cells_to_include = np.arange(0, signal.shape[1],1)
    #check number of neigh
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("Resulting in %i neighbours." %(n_neigh))
    #get total number of cells
    tnum_cells = len(cells_to_include)
    #get array with number of cells to include on each iteration
    num_cells = np.unique(np.logspace(np.log10(min_cells), np.log10(tnum_cells),n_steps,dtype=int))
    #initilize dictionary where all results will be saved
    dim_dict = dict()
    dim_dict["num_cells"] = num_cells
    dim_dict["cells_picked"] = dict()
    dim_dict['inner_dim'] = np.zeros((n_steps,n_splits))
    dim_dict['inner_dim_radii_vs_nn'] = np.zeros((n_steps, 1000,2, n_splits))*np.nan
    dim_dict['num_trust'] = np.zeros((n_steps,2,n_splits))
    dim_dict['res_var'] = np.zeros((n_steps,2,n_splits))
    dim_dict['rec_error'] = np.zeros((n_steps,2,n_splits))
    dim_dict['n_neigh'] = n_neigh

    if label_list:
        #define limit of variables used in sI if applicable
        varLimits = [(np.percentile(label,5), np.percentile(label,95)) for label in label_list]
        dim_dict['sI_val'] = np.zeros((n_steps,len(label_list),n_splits))
        
        dim_dict['R2s_base'] = np.zeros((n_steps,n_splits, 5, len(label_list),2))*np.nan
        dim_dict['R2s_umap'] = np.zeros((n_steps,n_splits, 5, len(label_list),2))*np.nan
        
    if verbose:
        print('Checking number of cells idx X/X',end='', sep='')
        pre_del = '\b\b\b'
    for num_cells_idx, num_cells_val in enumerate(num_cells):
        if verbose:
            print(pre_del,"%d/%d" %(num_cells_idx+1, n_steps), sep = '', end='')
            pre_del = (len(str(num_cells_idx+1))+len(str(n_steps))+1)*'\b'
            
        dim_dict['cells_picked'][str(num_cells_idx)] = np.zeros((num_cells_val,n_splits)).astype(int)
        for split_idx in range(n_splits):
            cells_picked = random.sample(list(cells_to_include), num_cells_val)
            dim_dict['cells_picked'][str(num_cells_idx)][:,split_idx] = cells_picked
            signal_new = copy.deepcopy(signal[:, cells_picked])
            #1.CHECK INNER DIM
            m , radius, neigh = compute_inner_dim(signal_new)
            dim_dict['inner_dim'][num_cells_idx, split_idx] = m
            dim_dict['inner_dim_radii_vs_nn'][num_cells_idx, :radius.shape[0],:,split_idx] = np.hstack((radius, neigh))
            
            #2.CHECK UMAP DIM TRUSTWORTHINESS
            dim, num_trust = compute_umap_trust_dim(signal_new, n_neigh = n_neigh, max_dim = 8,verbose=False)
            dim_dict['num_trust'][num_cells_idx,0,split_idx] = dim
            dim_dict['num_trust'][num_cells_idx,1,split_idx] = num_trust[dim-1,0]
            
            #3.CHECK ISOMAP RESVAR DIM
            dim, res_var = compute_isomap_resvar_dim(signal_new, n_neigh = n_neigh, max_dim = 8,verbose=False)
            dim_dict['res_var'][num_cells_idx,0,split_idx] = dim
            dim_dict[num_cells_idx,1,split_idx] = res_var[dim-1,0]
            
            #4.CHECK ISOMAP REC ERROR DIM
            dim, rec_error = compute_isomap_recerror_dim(signal_new, n_neigh = n_neigh, max_dim = 8,verbose=False)
            dim_dict['rec_error'][num_cells_idx,0,split_idx] = dim
            dim_dict['rec_error'][num_cells_idx,1,split_idx] = rec_error[dim-1,0]
            
            #CHECK sI index in 3 dims
            if label_list:
                emb = umap.UMAP(n_neighbors = n_neigh, n_components = 3, min_dist=0.75).fit_transform(signal_new)
                emb_space = np.linspace(0,3-1, 3).astype(int)              
                for labels_idx, labels_values in enumerate(label_list):
                    if labels_values.ndim>1:
                        labels_values = labels_values[:,0]
                    minVal, maxVal = varLimits[labels_idx]
                    dim_dict['sI_val'][num_cells_idx, labels_idx, split_idx],_ , _ = sI.compute_structure_index(emb, 
                                                                                      labels_values, 20, emb_space, 0,
                                                                                      vmin= minVal, vmax = maxVal)
            #Check decoding in 3 dims
            if label_list:
                dim_dict['R2s_base'][str(num_cells_idx)+'_'+str(split_idx)]  = decoders_1D(signal_new, input_label=label_list, emb_list = ["umap"], input_trial = trial_signal,
                                       n_dims = 3, n_splits=5, decoder_list = ["wf", "wc", "xgb", "svr"], verbose = False)
     
    return dim_dict