# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:24:23 2022

@author: Usuario
"""
import numpy as np
import copy

import scipy.signal as scs
from scipy.ndimage import convolve1d
import warnings

#Adapted from PyalData (19/10/21) (add lower case, and continuous option)
def add_firing_rates(data_frame, method, std=None, hw=None, win=None, continuous = False, num_std = 5, assymetry = False):
    """
    Add firing rate fields calculated from spikes fields

    Parameters
    ----------
    trial_data : pd.DataFrame
        trial_data dataframe
    method : str
        'bin' or 'smooth'
    std : float (optional)
        standard deviation of the Gaussian window to smooth with
        default 0.05 seconds
    hw : float (optional)
        half-width of the of the Gaussian window to smooth with
    win : 1D array
        smoothing window

    Returns
    -------
    td : pd.DataFrame
        trial_data with '_rates' fields added
    """
    out_frame = copy.deepcopy(data_frame)
    spike_fields = [name for name in out_frame.columns.values if name.lower().__contains__("spikes")]
    rate_fields = [name.replace("spikes", "rates") for name in spike_fields]
    columns_name = [col for col in out_frame.columns.values]
    lower_columns_name = [col.lower() for col in out_frame.columns.values]
    if 'bin_size' in lower_columns_name:
        bin_size = out_frame.iloc[0][columns_name[lower_columns_name.index("bin_size")]]
    elif 'fs' in lower_columns_name:
        bin_size = 1/out_frame.iloc[0][columns_name[lower_columns_name.index("fs")]]
    elif 'sf' in lower_columns_name:
        bin_size = 1/out_frame.iloc[0][columns_name[lower_columns_name.index("sf")]]
    else:
        raise ValueError('Dataframe does not contain binsize, sf, or fs field.')
        
    assert sum([arg is not None for arg in [win, hw, std]]) == 1, "only give win, hw or std"
    if method == "smooth":
        if win is None:
            if hw is not None:
                std = hw_to_std(hw)
                
            win = norm_gauss_window(bin_size, std, num_std = num_std, assymetry = assymetry)
            
        def get_rate(spikes):
            return smooth_data(spikes, win=win)/bin_size

    elif method == "bin":
        assert all([x is None for x in [std, hw, win]]), "If binning is used, then std, hw, and win have no effect, so don't provide them."
        def get_rate(spikes):
            return spikes/bin_size
    # calculate rates for every spike field
    if not continuous:
        for (spike_field, rate_field) in zip(spike_fields, rate_fields):
            out_frame[rate_field] = [get_rate(spikes) for spikes in out_frame[spike_field]]
    else:
        out_frame["index_mat"] = [np.zeros((out_frame[spike_fields[0]][idx].shape[0],1))+out_frame["trial_id"][idx] 
                                  for idx in range(out_frame.shape[0])]
        index_mat = np.concatenate(out_frame["index_mat"].values, axis=0)
        
        for (spike_field, rate_field) in zip(spike_fields, rate_fields):
            spikes = np.concatenate(out_frame[spike_field], axis = 0)
            rates = get_rate(spikes)
            out_frame[rate_field] = [rates[index_mat[:,0]==out_frame["trial_id"][idx] ,:] 
                                                                for idx in range(out_frame.shape[0])]
    return out_frame


#Adapted from PyalData package (19/10/21) (added assymetry, and variable win_length)
def norm_gauss_window(bin_size, std, num_std = 5, assymetry = False):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_size (float): binning size of the array we want to smooth in same time 
                units as the std
    
    std (float): standard deviation of the window use hw_to_std to calculate 
                std based from half-width (same time units as bin_size)
                
    num_std (int): size of the window to convolve in #of stds

    Returns
    -------
    win (1D np.array): Gaussian kernel with length: num_bins*std/bin_length
                mass normalized to 1
    """
    win_len = int(num_std*std/bin_size)
    if win_len%2==0:
        win_len = win_len+1
    win = scs.gaussian(win_len, std/bin_size)
    if assymetry:
        win_2 = scs.gaussian(win_len, 0.5*std/bin_size)
        win[:int((win_len-1)/2)] = win_2[:int((win_len-1)/2)]
        
    return win / np.sum(win)

#Copied from PyalData package (19/10/21)
def hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))

#Copied from PyalData package (19/10/21)
def smooth_data(mat, bin_size=None, std=None, hw=None, win=None, assymetry = False, axis=0):
    """
    Smooth a 1D array or every column of a 2D array

    Parameters
    ----------
    mat : 1D or 2D np.array
        vector or matrix whose columns to smooth
        e.g. recorded spikes in a time x neuron array
    bin_size : float
        length of the timesteps in seconds
    std : float (optional)
        standard deviation of the smoothing window
    hw : float (optional)
        half-width of the smoothing window
    win : 1D array-like (optional)
        smoothing window to convolve with

    Returns
    -------
    np.array of the same size as mat
    """
    #assert mat.ndim == 1 or mat.ndim == 2, "mat has to be a 1D or 2D array"
    assert  sum([arg is not None for arg in [win, hw, std]]) == 1, "only give win, hw or std"
    
    if win is None:
        assert bin_size is not None, "specify bin_size if not supplying window"
        if std is None:
            std = hw_to_std(hw)
        win = norm_gauss_window(bin_size, std, assymetry = assymetry)
    return convolve1d(mat, win, axis=axis, output=np.float32, mode='reflect')

#Copied from PyalData package (19/10/21)
def select_trials(trial_data, query, reset_index=True):
    """
    Select trials based on some criteria

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    query : str, function, array-like
        if array-like, the dataframe is indexed with this
            can be either a list of indices or a mask
        if str, it should express a condition
            it is passed to trial_data.query
        if function/callable, it should take a trial as argument
            and return True for trials you want to keep
    reset_index : bool, optional, default True
        whether to reset the dataframe index to [0,1,2,...]
        or keep the original indices of the kept trials

    Returns
    -------
    trial_data with only the selected trials

    Examples
    --------
    succ_td = select_trials(td, "result == 'R'")

    succ_mask = (td.result == 'R')
    succ_td = select_trials(td, succ_mask)

    train_idx = np.arange(10)
    train_trials = select_trials(td, train_idx)

    right_trials = select_trials(td, lambda trial: np.cos(trial.target_direction) > np.finfo(float).eps)
    """
    if isinstance(query, str):
        trials_to_keep = trial_data.query(query).index
    elif callable(query):
        trials_to_keep = [query(trial) for (i, trial) in trial_data.iterrows()]
    else:
        trials_to_keep = query

    if reset_index:
        return trial_data.loc[trials_to_keep, :].reset_index(drop=True)
    else:
        return trial_data.loc[trials_to_keep, :]
    
    
def remove_low_firing_neurons(trial_data, signal, threshold=None, divide_by_bin_size=None, verbose=False, mask= None):
    """
    Remove neurons from signal whose average firing rate
    across all trials is lower than a threshold
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        signal from which to calculate the average firing rates
        ideally spikes or rates
    threshold : float
        threshold in Hz
    divide_by_bin_size : bool, optional
        whether to divide by the bin size when calculating the firing rates
    verbose : bool, optional, default False
        print a message about how many neurons were removed

    Returns
    -------
    trial_data with the low-firing neurons removed from the
    signal and the corresponding unit_guide
    """

    if not np.any(mask):
        av_rates = np.mean(np.concatenate(trial_data[signal].values, axis=0), axis=0)
        if divide_by_bin_size:
            av_rates = av_rates/trial_data.bin_size[0]
        mask = av_rates >= threshold
        
    neuronal_fields = _get_neuronal_fields(trial_data, ref_field= signal)
    for nfield in neuronal_fields:
        trial_data[nfield] = [arr[:, mask] for arr in trial_data[nfield]]
    
    if signal.endswith("_spikes"):
        suffix = "_spikes"
    elif signal.endswith("_rates"):
        suffix = "_rates"
    else:
        warnings.warn("Could not determine which unit_guide to modify.")
    unit_guide = signal[:-len(suffix)] + "_unit_guide"
    if unit_guide in trial_data.columns:
        trial_data[unit_guide] = [arr[mask, :] for arr in trial_data[unit_guide]]
    if verbose:
        print(f"Removed {np.sum(~mask)} neurons from {signal}.")
    return trial_data
  
    
def _get_neuronal_fields(trial_data, ref_field=None):
    """
    Identify time-varying fields in the dataset
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    neuronal_fields : list of str
        list of fieldnames that store time-varying signals
    """
    if ref_field is None:
        # look for a spikes field
        ref_field = [col for col in trial_data.columns.values
                     if col.endswith("spikes") or col.endswith("rates")][0]

    # identify candidates based on the first trial
    first_trial = trial_data.iloc[0]
    T = first_trial[ref_field].shape[1]
    neuronal_fields = []
    for col in first_trial.index:
        try:
            if first_trial[col].shape[1] == T:
                neuronal_fields.append(col)
        except:
            pass

    # but check the rest of the trials, too
    ref_lengths = np.array([arr.shape[1] for arr in trial_data[ref_field]])
    for col in neuronal_fields:
        col_lengths = np.array([arr.shape[1] for arr in trial_data[col]])
        assert np.all(col_lengths == ref_lengths), f"not all lengths in {col} match the reference {ref_field}"

    return neuronal_fields