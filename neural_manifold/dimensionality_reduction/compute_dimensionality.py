# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:58:00 2022

@author: Usuario
"""
import numpy as np
import os, copy, pickle, warnings, random

from sklearn.metrics import pairwise_distances
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
import scipy.io

from tqdm import tqdm

from sklearn.manifold import Isomap

import umap
from kneed import KneeLocator

#INNER PACKAGE IMPORTS
from neural_manifold import general_utils as gu #translators (from Dataframe to np.ndarray)
from neural_manifold import structure_index as sI 
from neural_manifold.dimensionality_reduction import validation as dim_validation 

@gu.check_inputs_for_pd
def compute_inner_dim(base_signal = None, min_neigh = 2, max_neigh = 2**5, num_iter = 1000, verbose=False):
    '''Compute internal dimensionality of data array as described in: https://doi.org/10.1038/s41467-019-12724-2
    
    Parameters:
    ----------
        base_signal (Dnumpy Array): array containing the signal one wants to estimate the 
                        internal dimensionality.
        
    Optional parameters:
    --------------------
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
    #Check shape ratio
    if base_signal.shape[0]<base_signal.shape[1]:
        warnings.warn("Shape[0] of base_signal is smaller than Shape[1]. Note that " +
                      "function considers rows to be samples and cols to be features.",SyntaxWarning)
    #Check that array is 1D or 2D
    if base_signal.ndim>2:
        raise ValueError("Signal has more than 2 dimensions. Will only consider the first two.")
    #Check min_neigh, max_neigh numbers
    if min_neigh>=max_neigh:
        raise ValueError("'min_neigh' argument (%.2f) must be smaller than 'max_neigh' (%.2f) argument." %(min_neigh, max_neigh))
    if min_neigh<1 and max_neigh<1:
        if verbose:
            print("'min_neigh' and 'max_neigh' arguments smaller than 1. Interpreting them as fraction of total number of samples.")
        min_neigh = base_signal.shape[0]*min_neigh
        max_neigh = base_signal.shape[0]*max_neigh
        if verbose:
            print("Resulting in min_neigh=%i and max_neigh=%i" %(min_neigh, max_neigh))
    #Compute pairwise distances between all samples
    D_pw = pairwise_distances(base_signal,metric = 'euclidean')
    #Compute min distance between samples
    min_dist = np.min(D_pw)
    if min_dist == 0:
        min_dist = 1e-10
    #Compute max distance between samples
    max_dist = np.max(D_pw)
    #Define space of growing radius of balls in log2 base
    radius = np.logspace(np.log2(min_dist), np.log2(max_dist),num_iter,endpoint = True, base=2).reshape(-1,1)
    #For each radius check number of neighbours for each sample
    neigh = np.zeros((radius.shape[0],1))
    if verbose:
        bar=tqdm(total=int(len(radius)), desc='Computing Inner dim')
    for idx, rad in enumerate(radius):
        temp = D_pw <= rad
        neigh[idx,0] = np.mean(np.sum(temp,axis=1)-1)
        if verbose: bar.update(1)  
    #get rid of initial 0s (no neighbours)
    if verbose: bar.close()
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
    
@gu.check_inputs_for_pd
def compute_umap_trust_dim(base_signal = None, n_neigh= 0.01, max_dim=10, min_dist = 0.75, return_emb = False, verbose = False):
    '''Compute dimensionality of data array according to UMAP trustworthiness (see Venna, Jarkko, and Samuel Kaski.
    "Local multidimensional scaling with controlled tradeoff between trustworthiness and continuity." Proceedings 
    of 5th Workshop on Self-Organizing Maps. 2005.)
    
    Parameters:
    -----------
        base_signal (array): array containing the signal one wants to project using UMAP 
                        to estimate dimensionality.
        
    Optional parameters:
    ------------------- 
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
    #Check consistency of maximum dimensionality
    if max_dim>base_signal.shape[1]:
        if verbose:
            print("Maximum number of dimensions (%i) larger than original number of " %(max_dim)+
                          "dimensions (%i). Setting the first to the later." %(base_signal.shape[1]))
        max_dim = base_signal.shape[1]
    #Check number of neighbours
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(base_signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("Resulting in %i neighbours." %(n_neigh))
    #check if user wants the different embedings backs
    if return_emb:
        return_emb_list = list()
    #compute ranking order of source base_signal
    if verbose:
        print('Computing ranking order of base_signal')
    signal_indices = dim_validation.compute_rank_indices(base_signal)
    #initialize variables 
    num_trust = np.zeros((max_dim,1))*np.nan
    for dim in range(1, max_dim+1):
        if verbose:
            print('Checking dimension %i' %(dim), sep='', end = '')
        emb = umap.UMAP(n_neighbors = n_neigh, n_components = dim, min_dist=min_dist).fit_transform(base_signal)
        num_trust[dim-1,0] = dim_validation.trustworthiness_vector(base_signal, emb ,n_neigh, indices_source = signal_indices)[-1]
        if return_emb:
            return_emb_list.append(emb)
        if dim>1:
            red_error = abs(num_trust[dim-1,0]-num_trust[dim-2,0])
            if verbose:
                print(': error improvement of %.4f' %red_error, sep='')
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

@gu.check_inputs_for_pd
def compute_umap_continuity_dim(base_signal = None, n_neigh= 0.01, max_dim=10, min_dist = 0.75, return_emb = False, verbose = False):
    '''Compute dimensionality of data array according to UMAP continuity (see Venna, Jarkko, and Samuel Kaski.
    "Local multidimensional scaling with controlled tradeoff between trustworthiness and continuity." Proceedings 
    of 5th Workshop on Self-Organizing Maps. 2005.)
    
    Parameters:
    -----------
        base_signal (numpy Array): array containing the signal one wants to project using UMAP 
                        to estimate dimensionality.
        
    Optional parameters:
    -------------------
        n_neigh (float): number of neighbours used to compute UMAP projection.
        
        max_dim (int): maximum number of dimensions to check (note this parameters is the main time-expensive)
        
        min_dist (float): minimum distance used to compute UMAP projection (values in (0,1]))
    
        return_emb (boolean): boolean specifing whether or not to return all UMAP embeddings computed.
        
        verbose (boolean): boolean indicating whether to pring verbose or not.
                           
    Returns:
    --------
        dim (int): dimensionality of the data according to UMAP trustworthiness.
        
        num_cont (array): array containing the trustworthiness values of each dimensionality iteration. This
                        array is the one used to find the knee defining the dimensionality of the data
                        
        return_emb_list (list): list containing the UMAP projections computed for each of the dimensionalities.
                        (only applicable if input parameter 'return_emb' set to True)
    '''
    #Check consistency of maximum dimensionality
    if max_dim>base_signal.shape[1]:
        if verbose:
            print("Maximum number of dimensions (%i) larger than original number of " %(max_dim)+
                          "dimensions (%i). Setting the first to the later." %(base_signal.shape[1]))
        max_dim = base_signal.shape[1]
    #Check number of neighbours
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(base_signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("Resulting in %i neighbours." %(n_neigh))
    #check if user wants the different embedings backs
    if return_emb:
        return_emb_list = list()
    #initialize variables 
    num_cont = np.zeros((max_dim,1))*np.nan
    for dim in range(1, max_dim+1):
        if verbose:
            print('Checking dimension %i' %(dim), sep='', end = '')
        emb = umap.UMAP(n_neighbors = n_neigh, n_components = dim, min_dist=min_dist).fit_transform(base_signal)
        num_cont[dim-1,0] = dim_validation.continuity_vector(base_signal, emb ,n_neigh)[-1]
        if return_emb:
            return_emb_list.append(emb)
        if dim>1:
            red_error = abs(num_cont[dim-1,0]-num_cont[dim-2,0])
            if verbose:
                print(': error improvement of %.4f' %red_error, sep='')
        else:
            if verbose:
                print('')
    dim_space = np.linspace(1,max_dim, max_dim).astype(int)        
    kl = KneeLocator(dim_space, num_cont[:,0], curve = "concave", direction = "increasing")
    if kl.knee:
        dim = kl.knee
        if verbose:
            print('Final dimension: %d - Final error of %.4f' %(dim, 1-num_cont[dim-1,0]))
    else:
        dim = np.nan
        if verbose:
            print('Could estimate final dimension (knee not found). Returning nan.')
    if return_emb:
        return dim, num_cont, return_emb_list
    else:
        return dim, num_cont

@gu.check_inputs_for_pd
def compute_isomap_resvar_dim(base_signal = None, n_neigh= 0.01, max_dim=10, return_emb = False, verbose = False):
    '''Compute dimensionality of data array according to isomap residual variance.
    
    Parameters:
    -----------
        base_signal (numpy Array): array containing the signal one wants to project using Isomap
                        to estimate dimensionality.
        
    Optional parameters:
    -------------------
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
    #Check consistency of maximum dimensionality
    if max_dim>base_signal.shape[1]:
        if verbose:
            print("Maximum number of dimensions (%i) larger than original number of " %(max_dim)+
                      "dimensions (%i). Setting the first to the later." %(base_signal.shape[1]))
        max_dim = base_signal.shape[1]
    #Check number of neighbours
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(base_signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("Resulting in %i neighbours." %(n_neigh))
    #initialize isomap object
    iso_model = Isomap(n_neighbors = n_neigh, n_components = max_dim)
    #fit and project data
    emb = iso_model.fit_transform(base_signal)
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
    
@gu.check_inputs_for_pd    
def compute_isomap_recerror_dim(base_signal = None, n_neigh= 0.01, max_dim=10, return_emb = False, verbose = False):
    '''Compute dimensionality of data array according to isomap reconstruction error.
    
    Parameters:
    -----------
        base_signal (numpy Array): array containing the signal one wants to project using Isomap
                        to estimate dimensionality.
        
    Optional parameters:
    -------------------
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
    #Check consistency of maximum dimensionality
    if max_dim>base_signal.shape[1]:
        if verbose:
            print("Maximum number of dimensions (%i) larger than original number of " %(max_dim)+
                          "dimensions (%i). Setting the first to the later." %(base_signal.shape[1]))
        max_dim = base_signal.shape[1]
    #Check number of neighbours
    if n_neigh<1:
        if verbose:
           print("'n_neigh' argument smaller than 1 (%.4f). Interpreting them as fraction of total number of samples." %(n_neigh), end='')
        n_neigh = np.round(base_signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("Resulting in %i neighbours." %(n_neigh))
    #define kernel function
    K = lambda D: -0.5*((np.eye(D.shape[0])-(1/D.shape[0])).dot(np.square(D))).dot(np.eye(D.shape[0])-(1/D.shape[0]))
    #initialize isomap object
    iso_model = Isomap(n_neighbors = n_neigh, n_components = max_dim)
    #fit and project data
    emb = iso_model.fit_transform(base_signal)
    #compute Isomap kernel for input data once before going into the loop
    KD_signal = K(iso_model.dist_matrix_)
    #compute residual error
    rec_error = np.zeros((max_dim, 1))
    n_samples = base_signal.shape[0]
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


@gu.check_inputs_for_pd
def compute_umap_sI_dim(base_signal = None, label_signal = None, n_neigh= 0.01, max_dim=10, min_dist = 0.75, return_emb = False, verbose = False):
    '''Compute dimensionality of data array according to Structure Index in UMAP
    
    Parameters:
    -----------
        base_signal (numpy Array): array containing the signal one wants to project using UMAP 
                        to estimate dimensionality.
        label_signal (numpy Array): array containing the label one wants to compute the structure
        
    Optional parameters:
    -------------------
        n_neigh (float): number of neighbours used to compute UMAP projection.
        
        max_dim (int): maximum number of dimensions to check (note this parameters is the main time-expensive)
        
        min_dist (float): minimum distance used to compute UMAP projection (values in (0,1]))
    
        return_emb (boolean): boolean specifing whether or not to return all UMAP embeddings computed.
        
        verbose (boolean): boolean indicating whether to pring verbose or not.
                           
    Returns:
    --------
        dim (int): dimensionality of the data according to UMAP trustworthiness.
        
        sI_val (array): array containing the structure index values of each dimensionality iteration. This
                        array is the one used to find the knee defining the dimensionality of the data
                        
        return_emb_list (list): list containing the UMAP projections computed for each of the dimensionalities.
                        (only applicable if input parameter 'return_emb' set to True)
    '''
    #Check consistency of maximum dimensionality
    if max_dim>base_signal.shape[1]:
        if verbose:
            print("Maximum number of dimensions (%i) larger than original number of " %(max_dim)+
                          "dimensions (%i). Setting the first to the later." %(base_signal.shape[1]))
        max_dim = base_signal.shape[1]
    #check label_signal
    if label_signal.ndim>1:
        label_signal = label_signal[:,0]
    n_bins = len(np.unique(label_signal))
    if n_bins > 20:
        n_bins = 10
    min_val = np.percentile(label_signal,5)
    max_val = np.percentile(label_signal,95)
    
    #Check number of neighbours
    if n_neigh<1:
        n_neigh = np.round(base_signal.shape[0]*n_neigh).astype(np.uint32)
        if verbose:
            print("'n_neigh' argument smaller than 1 (%.4f). Interpreting it as fraction of number of samples." %(n_neigh), end='')
            print("Resulting in %i neighbours." %(n_neigh))
    #check if user wants the different embedings backs
    if return_emb:
        return_emb_list = list()
    
    #initialize variables 
    sI_val = np.zeros((max_dim,1))*np.nan
    for dim in range(1, max_dim+1):
        if verbose:
            print('Checking dimension %i ' %(dim), sep='', end = '')
        emb = umap.UMAP(n_neighbors = n_neigh, n_components = dim, min_dist=min_dist).fit_transform(base_signal)
        emb_space = np.arange(dim).astype(int)
        sI_val[dim-1,0],_,_ = sI.compute_structure_index(emb ,label_signal, n_bins, emb_space, 0,
                                                         vmin = min_val, vmax = max_val, nn = n_neigh)[-1]
        if return_emb:
            return_emb_list.append(emb)
        if dim>1:
            sI_improvement = sI_val[dim-1,0]-sI_val[dim-2,0]
            if verbose:
                print(f': sI improvement of {sI_improvement:.4f}', sep='')
        else:
            if verbose:
                print('')
                
    dim_space = np.linspace(1,max_dim, max_dim).astype(int)        
    kl = KneeLocator(dim_space, sI_val[:,0], curve = "concave", direction = "increasing")
    if kl.knee:
        dim = kl.knee
        if verbose:
            print(f'Final dimension: {dim} - Final sI of {sI_val[dim-1,0]:.4f}')
    else:
        dim = np.nan
        if verbose:
            print('Could estimate final dimension (knee not found). Returning nan.')
    if return_emb:
        return dim, sI_val, return_emb_list
    else:
        return dim, sI_val


@gu.check_inputs_for_pd
def compute_abids(base_signal = None, n_neigh= 50, verbose = True):
    '''Compute dimensionality of data array according to Structure Index in UMAP
    
    Parameters:
    -----------
        base_signal (numpy Array): 
            array containing the signal one wants to estimate dimensionality.
        n_neigh (int): 
            number of neighbours used to compute angle.                     
    Returns:
    --------
        (array): 
            array containing the estimated dimensionality for each point in the
            cloud.
        
    '''

    def abid(X,k,x,search_struct,offset=1):
        neighbor_norms, neighbors = search_struct.query(x,k+offset)
        neighbors = X[neighbors[offset:]] - x
        normed_neighbors = neighbors / neighbor_norms[offset:,None]
        # Original publication version that computes all cosines
        # coss = normed_neighbors.dot(normed_neighbors.T)
        # return np.mean(np.square(coss))**-1
        # Using another product to get the same values with less effort
        para_coss = normed_neighbors.T.dot(normed_neighbors)
        return k**2 / np.sum(np.square(para_coss))

    search_struct = cKDTree(base_signal)
    if verbose:
        return np.array([
            abid(base_signal,n_neigh,x,search_struct)
            for x in tqdm(base_signal,desc="abids",leave=False)
        ])
    else:
        return np.array([abid(base_signal,n_neigh,x,search_struct) for x in base_signal])




