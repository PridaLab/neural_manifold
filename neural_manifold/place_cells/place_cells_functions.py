#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:08:16 2022

@author: julio
"""

import copy
import numpy as np
from neural_manifold import general_utils as gu

@gu.check_inputs_for_pd
def get_place_cells_LT(pos_signal = None, rates_signal = None, traces_signal = None, **kwargs):
    #pos_signal
    #rates_signal
    #spikes_signal
    #traces_signal
    #vel_signal
    #direction_signal

    #bin_num (a.u.)
    #bin_width (cm)
    #sF (Hz)
    #std_pos (s)
    #std_traces (s)
    #ignore_edges (%)
    #std_pdf (cm)
    
    #num_shuffles
    #min_shift (s)
    assert  sum([arg not in kwargs for arg in ['bin_width', 'bin_num']]) < 2, \
                                                "only give bin_width or bin_num"
    assert  sum([not isinstance(arg, type(None)) for arg in [rates_signal, 
            traces_signal]]) < 2, "only give rates or traces"

    assert 'sF' in kwargs, "provide sampling frequency ('sF')"
    
    #bin length for the 1-dimensional rate map in centimeters
    if 'bin_num' not in kwargs and 'bin_width' not in kwargs:
        kwargs["bin_width"] = 2 #cm
    #std to smooth position
    if 'std_pos' not in kwargs:
        kwargs['std_pos'] = 0.025 #s

    #std to smooth traces
    if ('std_traces' not in kwargs) and (not isinstance(traces_signal, type(None))):
        kwargs['std_traces'] = 0.25 #s

    if 'std_pdf' not in kwargs:
        kwargs['std_pdf'] = 2 #cm

    #ignore edges
    if 'ignore_edges' not in kwargs:
        kwargs['ignore_edges'] = 10 #%
    
    if 'method' not in kwargs:
        kwargs['method'] = 'spatial_info'

    if 'num_shuffles' not in kwargs:
        kwargs['num_shuffles'] = 1000
    
    if 'min_shift' not in kwargs:
        kwargs['min_shift'] = 5 
    
    #Compute velocity if not provided
    if 'vel_signal' not in kwargs:
        vel = np.linalg.norm(np.diff(pos_signal, axis= 0), axis=1)*kwargs['sF']
        vel = np.hstack((vel[0], vel))
    else:
        vel = copy.deepcopy(kwargs['vel_signal'])
    vel = vel.reshape(-1)

    #smooth pos signal if applicable
    if kwargs['std_pos']>0:
        assert kwargs['std_pos']>0, "std to smooth position must be positive"
        assert kwargs['sF'] is not None, "to be able to smooth the position 'sF' (Hz) must be provided"
        pos = gu.smooth_data(pos_signal, std = kwargs['std_pos'], bin_size = 1/kwargs['sF'], assymetry = False)
    else:
        pos = copy.deepcopy(pos_signal)
    pos = pos[:,0].reshape(-1,1)
    #if traces provided, preprocess it 
    if not isinstance(traces_signal, type(None)):
        if kwargs['std_traces']>0:
            traces = gu.smooth_data(traces_signal, std = kwargs['std_traces'], bin_size = 1/kwargs['sF'], assymetry = False)
        else:
            traces = copy.deepcopy(traces_signal)
        #Threshold \DeltaF/F so that values less than 2 robust σ across the time series are set to 0
        traces_std = np.std(traces, axis=0)
        for cell in range(traces.shape[1]):
            traces[traces[:,cell]<2*traces_std[cell], cell] = 0
        neu_sig = traces
    else:
        neu_sig = copy.deepcopy(rates_signal)

    #compute moving epochs
    #TODO: consider minimum epoch duration and merging those with little time in between
    if 'vel_th' not in kwargs:
        kwargs['vel_th'] = 5 #cm/s
    move_epochs = vel>=kwargs['vel_th'] 

    #keep only moving epochs
    pos = pos[move_epochs]
    neu_sig = neu_sig[move_epochs]

    if 'spikes_signal' in kwargs:
        spikes = kwargs["spikes_signal"][move_epochs]
    else:
        spikes = None

    if 'direction_signal' in kwargs:
        direction = kwargs["direction_signal"][move_epochs]
    else:
        direction = None

    #Discard edges
    if kwargs['ignore_edges']>0:
        pos_boolean = _get_edges(pos, kwargs['ignore_edges'], '1D')
        pos = pos[pos_boolean]
        neu_sig = neu_sig[pos_boolean]

        if not isinstance(spikes, type(None)):
            spikes = spikes[pos_boolean]

        if not isinstance(direction, type(None)):
            direction = direction[pos_boolean]

    #Create grid along x-position
    min_pos, max_pos = [np.min(pos,axis=0), np.max(pos,axis = 0)]
    obs_length = max_pos - min_pos
    if kwargs['bin_width']:
        bin_width = kwargs['bin_width']
        nbins = np.ceil(obs_length/bin_width)
    elif kwargs['bin_num']:
        nbins = kwargs['bin_num']
        bin_width = np.round(obs_length/nbins,4)
        mapAxisX = np.arange(min_pos[0]+bin_width/2,max_pos[0]-bin_width/2, bin_width);
    #Compute probability density function
    pos_pdf, neu_pdf = _get_pdf(pos, neu_sig, mapAxisX, None, '1D', direction)
    #smooth pdfs
    pos_pdf = gu.smooth_data(pos_pdf, std = kwargs['std_pdf'], bin_size = bin_width, assymetry = False)
    neu_pdf = gu.smooth_data(neu_pdf, std = kwargs['std_pdf'], bin_size = bin_width, assymetry = False)

    metric_val = _compute_metric(pos_pdf, neu_pdf, kwargs["method"],'1D',direction)
    
    #do shuffling
    if not isinstance(direction, type(None)):
        shuffled_metric_val = np.zeros((kwargs["num_shuffles"], metric_val.shape[0], metric_val.shape[1]))
    else:
        shuffled_metric_val = np.zeros((kwargs["num_shuffles"], metric_val.shape[0]))

    min_shift = np.ceil(kwargs['min_shift']*kwargs['sF'])
    max_shift = pos.shape[0] - min_shift
    time_shift = np.random.randint(min_shift, max_shift, kwargs["num_shuffles"])

    for idx, shift in enumerate(time_shift):
        shifted_pos = np.zeros(pos.shape)
        shifted_pos[:-shift,:] = copy.deepcopy(pos[shift:,:])
        shifted_pos[-shift:,:] = copy.deepcopy(pos[:shift,:])
        if not isinstance(direction, type(None)):
            shifted_direction = np.zeros(direction.shape)
            shifted_direction[:-shift,:] = copy.deepcopy(direction[shift:,:])
            shifted_direction[-shift:,:] = copy.deepcopy(direction[:shift,:])
        else:
            shifted_direction = None

        shifted_pos_pdf, shifted_neu_pdf = _get_pdf(shifted_pos, neu_sig, mapAxisX, None, '1D', shifted_direction)
        shifted_pos_pdf = gu.smooth_data(shifted_pos_pdf, std = kwargs['std_pdf'], bin_size = bin_width, assymetry = False)
        shifted_neu_pdf = gu.smooth_data(shifted_neu_pdf, std = kwargs['std_pdf'], bin_size = bin_width, assymetry = False)
        shuffled_metric_val[idx] = _compute_metric(shifted_pos_pdf, shifted_neu_pdf, kwargs["method"],'1D',direction)
        
    th_metric_val = np.percentile(shuffled_metric_val, 99, axis=0)
    place_cells = np.linspace(0, metric_val.shape[0]-1, metric_val.shape[0])
    place_cells = place_cells[np.any(metric_val>th_metric_val,axis=1)].astype(int)
    
    #TODO: plot all cells 
    
def _compute_metric(pos_pdf, neu_pdf, method, dim, direction):
    if 'spatial_info' in method:
        return _compute_spatial_info(pos_pdf, neu_pdf, dim, direction)
    elif 'response_profile' in method:
        if '1D' in dim:
            min_pos_pdf = np.min([1/pos_pdf.shape[0], np.nanpercentile(pos_pdf,1)])
            nan_pos_pdf = copy.deepcopy(pos_pdf)
            nan_pos_pdf[pos_pdf<min_pos_pdf] = np.nan
            if isinstance(direction, type(None)):
                return np.nanmax(neu_pdf/nan_pos_pdf[:,0][:,None], axis=0)
            else:
                return np.nanmax(np.divide(neu_pdf,nan_pos_pdf), axis=0)
        elif '2D' in dim:
            min_pos_pdf = np.min([1/(pos_pdf.shape[0]*pos_pdf.shape[1]), np.nanpercentile(pos_pdf,1)])
            nan_pos_pdf = copy.deepcopy(pos_pdf)
            nan_pos_pdf[pos_pdf<min_pos_pdf] = np.nan
            return np.nanmax(np.divide(neu_pdf,nan_pos_pdf[:,:,0][:,:,None]), axis=(0,1))
            
def _compute_spatial_info(pos_pdf, neu_pdf, dim, direction):
    if '1D' in dim:
        if isinstance(direction, type(None)):
            norm_neu_pdf = neu_pdf/np.mean(neu_pdf,axis=0)[None,:]
            norm_neu_pdf[norm_neu_pdf==0] = np.nan
            return np.nansum(np.multiply(pos_pdf, np.multiply(norm_neu_pdf, np.log2(norm_neu_pdf))), axis=0)
        else:
            norm_neu_pdf = np.divide(neu_pdf,np.mean(neu_pdf,axis=0)[None,:,:])
            norm_neu_pdf[norm_neu_pdf==0] = np.nan
            return np.nansum(np.multiply(pos_pdf, np.multiply(norm_neu_pdf, np.log2(norm_neu_pdf))), axis=0)

    elif '2D' in dim:
        norm_neu_pdf = neu_pdf/np.mean(neu_pdf,axis=(0,1))[None,None,:]
        norm_neu_pdf[norm_neu_pdf==0] = np.nan
        return np.nansum(np.multiply(pos_pdf, np.multiply(norm_neu_pdf, np.log2(norm_neu_pdf))), axis=(0,1))

def _get_edges(pos, limit, dim):
    limit = limit/100
    minpos = np.min(pos, axis=0)
    maxpos = np.max(pos,axis=0)
    
    trackLength = maxpos - minpos
    trackLimits = np.vstack((minpos + limit*trackLength,
                             maxpos - limit*trackLength))
    
    if '1D' in dim:
        points_inside_lim = ~np.any(np.vstack((pos[:,0]<trackLimits[0,0],
                                       pos[:,0]>trackLimits[1,0])).T, axis=1)
    elif '2D' in dim:
        points_inside_lim = ~np.any(np.vstack((pos[:,0]<trackLimits[0,0],
                                       pos[:,1]<trackLimits[0,1],
                                       pos[:,0]>trackLimits[1,0],
                                       pos[:,1]>trackLimits[1,1])).T, axis=1)
    return points_inside_lim

def _get_pdf(pos, neu_signal, mapAxisX, mapAxisY, dim, dir_mat = None):
    if '1D' in dim:
        if not np.any(dir_mat):
            neu_map = np.zeros((mapAxisX.shape[0], neu_signal.shape[1]))
            pdf_map = np.zeros((mapAxisX.shape[0],1))
            for sample in range(pos.shape[0]):
                pos_entry = np.where(pos[sample,0]>=mapAxisX)
                if np.any(pos_entry):
                    pos_entry = pos_entry[0][-1]
                else:
                    pos_entry = 0
                pdf_map[pos_entry,0] +=1 
                neu_map[pos_entry,:]+= (1/pdf_map[pos_entry,0])*(neu_signal[sample,:]-neu_map[pos_entry,:]) #online average
                    
            pdf_map = pdf_map/np.sum(pdf_map) #normalize probability density function
        else:
            neu_map = np.zeros((mapAxisX.shape[0], neu_signal.shape[1],2))
            pdf_map = np.zeros((mapAxisX.shape[0],1,2))
            for dire in range(2):
                temp_pos = pos[dir_mat[:,0] == dire+1,:]
                temp_neu_signal = neu_signal[dir_mat[:,0] == dire+1,:]
                for sample in range(temp_pos.shape[0]):
                    pos_entry = np.where(temp_pos[sample,0]>=mapAxisX)
                    if np.any(pos_entry):
                        pos_entry = pos_entry[0][-1]
                    else:
                        pos_entry = 0
                    pdf_map[pos_entry,0, dire] +=1 
                    neu_map[pos_entry,:,dire]+= (1/pdf_map[pos_entry,0, dire])*(temp_neu_signal[sample,:]-neu_map[pos_entry,:, dire]) #online average
    
            pdf_map = np.divide(pdf_map, np.sum(pdf_map, axis=0)) #normalize probability density function
    elif '2D' in dim:
        neu_map = np.zeros((mapAxisX.shape[0], mapAxisY.shape[0], neu_signal.shape[1]))
        pdf_map = np.zeros((mapAxisX.shape[0], mapAxisY.shape[0],1))
       
        for sample in range(pos.shape[0]):
            pos_entry_x = np.where(pos[sample,0]>=mapAxisX)
            pos_entry_y = np.where(pos[sample,1]>=mapAxisY)
            if np.any(pos_entry_x):
                pos_entry_x = pos_entry_x[0][-1]
            else:
                pos_entry_x = 0
            if np.any(pos_entry_y):
                pos_entry_y = pos_entry_y[0][-1]
            else:
                pos_entry_y = 0
                
            pdf_map[pos_entry_x,pos_entry_y,0] +=1 
            neu_map[pos_entry_x,pos_entry_y, :] += (1/pdf_map[pos_entry_x,pos_entry_y, 0])*\
                                                (neu_signal[sample,:]- neu_map[pos_entry_x,pos_entry_y, :]) #online average
                
        pdf_map = pdf_map/np.sum(pdf_map) #normalize probability density function
        
    return pdf_map, neu_map




    







