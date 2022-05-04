#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:08:16 2022

@author: julio
"""

#spatial info loosely adapted from https://doi.org/10.1016/j.xpro.2021.100759 & https://doi.org/10.1038/s41467-019-10139-7
#response profile adapted from https://doi.org/10.1016/j.cub.2020.07.006
#entropy to be adapted from https://doi.org/10.3389/fnbeh.2020.00064

import copy
import numpy as np
from neural_manifold import general_utils as gu
import matplotlib.pyplot as plt
import os
from datetime import datetime
import base64
from io import BytesIO

from scipy import ndimage

@gu.check_inputs_for_pd
def get_place_cells(pos_signal = None, rates_signal = None, traces_signal = None, dim = 2,save_dir = None,**kwargs):
    """
    Parameters
    ----------
    pos_signal : TYPE, optional
        DESCRIPTION. The default is None.
    rates_signal : TYPE, optional
        DESCRIPTION. The default is None.
    traces_signal : TYPE, optional
        DESCRIPTION. The default is None.
    dim : TYPE, optional
        DESCRIPTION. The default is 2.
    save_dir : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    place_cells : TYPE
        DESCRIPTION.
    metric_val : TYPE
        DESCRIPTION.
    th_metric_val : TYPE
        DESCRIPTION.
    pos_pdf : TYPE
        DESCRIPTION.
    neu_pdf : TYPE
        DESCRIPTION.

    """
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
    #method (spatial_info, response_profile)
    #num_shuffles
    #min_shift (s)
    
    assert sum([arg in kwargs for arg in ['bin_width', 'bin_num']]) < 2,\
                                                "only give bin_width or bin_num"
    assert sum([not isinstance(arg, type(None)) for arg in [rates_signal, 
                                traces_signal]]) < 2, "only give rates or traces"

    assert 'sF' in kwargs, "you must provide sampling frequency ('sF')"
    
    assert dim>0, f"dim must be a positive integer but it was {dim}"

    assert pos_signal.shape[1]>= dim, f"pos_signal has less dimensions ({pos_signal.shape[1]}) " + \
                                f"than the indicated in dim ({dim})"
    
    #bin length for the 1-dimensional rate map in centimeters
    if 'bin_num' not in kwargs and 'bin_width' not in kwargs:
        kwargs["bin_width"] = 5 #cm
        
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
    if pos.ndim == 1:
        pos = pos.reshape(-1,1)
    #if traces provided, preprocess it 
    if not isinstance(traces_signal, type(None)):
        if kwargs['std_traces']>0:
            neu_sig = gu.smooth_data(traces_signal, std = kwargs['std_traces'], bin_size = 1/kwargs['sF'], assymetry = False)
        else:
            neu_sig = copy.deepcopy(traces_signal)
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
        direction = np.zeros((pos.shape[0],1))
    
    #Discard edges
    if kwargs['ignore_edges']>0:
        pos_boolean = _get_edges(pos, kwargs['ignore_edges'], 1)
        pos = pos[pos_boolean]
        neu_sig = neu_sig[pos_boolean]
        direction = direction[pos_boolean]
        if not isinstance(spikes, type(None)):
            spikes = spikes[pos_boolean]
            th_spikes = np.sum(spikes, axis=0)>20
    #Create grid along each dimensions
    min_pos, max_pos = [np.min(pos,axis=0), np.max(pos,axis = 0)] #(pos.shape[1],)
    obs_length = max_pos - min_pos #(pos.shape[1],)
    if kwargs['bin_width']:
        bin_width = kwargs['bin_width']
        if isinstance(bin_width, list):
            bin_width = np.array(bin_width)
        else:
            bin_width = np.repeat(bin_width, dim, axis=0) #(pos.shape[1],)
        nbins = np.ceil(obs_length[:dim]/bin_width).astype(int) #(pos.shape[1],)
    elif kwargs['bin_num']:
        nbins = kwargs['bin_num']
        if isinstance(nbins, list):
            nbins = np.array(nbins)
        else:
            nbins = np.repeat(nbins, dim, axis=0) #(pos.shape[1],)
        bin_width = np.round(obs_length[:dim]/nbins,4) #(pos.shape[1],)
    
    mapAxis = list()
    for d in range(dim):
        mapAxis.append(np.arange(min_pos[d],max_pos[d], bin_width[d])); #(nbins[d],)
    
    #Compute probability density function
    pos_pdf, neu_pdf = _get_pdf(pos, neu_sig, mapAxis, dim, direction)
    
    #smooth pdfs
    for d in range(dim):
        pos_pdf = gu.smooth_data(pos_pdf, std = kwargs['std_pdf'], bin_size = bin_width[d],axis = d, assymetry = False)
        neu_pdf = gu.smooth_data(neu_pdf, std = kwargs['std_pdf'], bin_size = bin_width[d], axis = d, assymetry = False)
    
    #compute metric
    metric_val = _compute_metric(pos_pdf, neu_pdf, kwargs["method"])
    
    #do shuffling
    shuffled_metric_val = np.zeros((kwargs["num_shuffles"], metric_val.shape[0], metric_val.shape[1]))
    min_shift = np.ceil(kwargs['min_shift']*kwargs['sF'])
    max_shift = pos.shape[0] - min_shift
    time_shift = np.random.randint(min_shift, max_shift, kwargs["num_shuffles"])

    for idx, shift in enumerate(time_shift):
        shifted_pos = np.zeros(pos.shape)
        shifted_pos[:-shift,:] = copy.deepcopy(pos[shift:,:])
        shifted_pos[-shift:,:] = copy.deepcopy(pos[:shift,:])
        
        shifted_direction = np.zeros(direction.shape)
        shifted_direction[:-shift,:] = copy.deepcopy(direction[shift:,:])
        shifted_direction[-shift:,:] = copy.deepcopy(direction[:shift,:])
        
        shifted_pos_pdf, shifted_neu_pdf = _get_pdf(shifted_pos, neu_sig, mapAxis, dim, shifted_direction)
        for d in range(dim):
            shifted_pos_pdf = gu.smooth_data(shifted_pos_pdf, std = kwargs['std_pdf'], bin_size = bin_width[d], axis = d, assymetry = False)
            shifted_neu_pdf = gu.smooth_data(shifted_neu_pdf, std = kwargs['std_pdf'], bin_size = bin_width[d], axis = d, assymetry = False)
        
        shuffled_metric_val[idx] = _compute_metric(shifted_pos_pdf, shifted_neu_pdf, kwargs["method"])
        
    th_metric_val = np.percentile(shuffled_metric_val, 99, axis=0)
    
    place_cells = np.linspace(0, metric_val.shape[0]-1, metric_val.shape[0])
    place_cells = place_cells[np.any(metric_val>th_metric_val,axis=1)*th_spikes].astype(int)
    
    place_fields, place_fields_id = _check_placefields(pos_pdf, neu_pdf, place_cells)
    if dim==1:
        _plot_place_cells_1D(pos, neu_sig, neu_pdf, metric_val, place_cells, 
                                     spikes, kwargs["method"], save_dir)
    elif dim==2:
        _plot_place_cells_2D(pos, neu_sig, neu_pdf, metric_val, place_cells, 
                                     spikes, kwargs["method"], save_dir)
        
    return place_cells, metric_val, th_metric_val, pos_pdf, neu_pdf


def _plot_place_cells_2D(pos, neu_sig, neu_pdf, metric_val, place_cells, spikes, method, save_dir):
    ###########################################################################
    
    
    
    ###########################################################################
    min_pos, max_pos = [np.min(pos,axis=0), np.max(pos,axis = 0)] #(pos.shape[1],)
    cells_per_row = 4
    cells_per_col = 4
    num_dir = neu_pdf.shape[-1]
    cells_per_plot = cells_per_row*cells_per_col
    num_figures = np.ceil(neu_sig.shape[1]/cells_per_plot).astype(int)
    
    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + "<h1>Place cells - {method}</h1>\n<br>\n"    #Add title
    html = html + f"<br><h2>{datetime.now().strftime('%d%m%y_%H%M%S')}</h2><br>\n"    #Add subtitle

    for cell_group in range(num_figures):
        fig= plt.figure(figsize = (12,12))
        cell_st = cell_group*cells_per_plot
        cells_to_plot = np.min([neu_sig.shape[1]-cell_st, cells_per_plot])
        for cell_idx in range(cells_to_plot):
            gcell_idx = cell_idx + cell_st #global cell index
            row_idx = cell_idx//cells_per_row
            col_idx = cell_idx%cells_per_col
            
            if gcell_idx in place_cells:
                color_cell = [.2,.6,.2]
            else:
                color_cell = 'k'
                
            ax  = plt.subplot2grid((cells_per_row*6, cells_per_col*num_dir), 
                   (row_idx*6, col_idx*num_dir), rowspan=2, colspan = num_dir)
            
            ax.plot(pos[:,0], pos[:,1], color = [.5,.5,.5], alpha = 0.3)
            if isinstance(spikes, type(None)):
                th = np.nanstd(neu_sig[:,gcell_idx])
                active_ts = neu_sig[:,gcell_idx] > 2*th
                t_neu_sig = neu_sig[active_ts, gcell_idx]
                t_pos = pos[active_ts,:]
                signal_inds = t_neu_sig.argsort()
                sorted_pos = t_pos[signal_inds,:]
                sorted_neu_sig = t_neu_sig[signal_inds]
                ax.scatter(*sorted_pos[:,:2].T, c = sorted_neu_sig, s= 8)
            else:
                ax.scatter(*pos[spikes[:,gcell_idx]>0,:2].T, color = color_cell, s= 5)
            ax.set_xlim([min_pos[0], max_pos[0]])
            ax.set_ylim([min_pos[1], max_pos[1]])
            title = list()
            title.append(f"Cell: {gcell_idx} -")
            [title.append(f"{mval:.2f} ") for mval in metric_val[gcell_idx]];
            ax.set_title(' '.join(title), color = color_cell)
            
            for dire in range(num_dir):
                ax  = plt.subplot2grid((cells_per_row*6, cells_per_col*num_dir), (row_idx*6+2, col_idx*num_dir+dire), rowspan=2)
                p = ax.matshow(np.flipud(neu_pdf[:,:,gcell_idx,dire].T), aspect = 'auto')
                ax.set_yticks([])
                ax.set_xticks([])
                cbar = fig.colorbar(p, ax=ax,fraction=0.3, pad=0.08, location = 'bottom')
                cbar.ax.set_ylabel(method, rotation=270, size=10)
            
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
        plt.close(fig)
    #Save html file
    with open(os.path.join(save_dir, f"PlaceCells_{method}__{datetime.now().strftime('%d%m%y_%H%M%S')}.htm"),'w') as f:
        f.write(html)

   
def _plot_place_cells_1D(pos, neu_sig, neu_pdf, metric_val, place_cells, spikes, method, save_dir):
    min_pos, max_pos = [np.min(pos,axis=0), np.max(pos,axis = 0)] #(pos.shape[1],)
    cells_per_row = 4
    cells_per_col = 4
    
    cells_per_plot = cells_per_row*cells_per_col
    num_figures = np.ceil(neu_sig.shape[1]/cells_per_plot).astype(int)
    
    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + "<h1>Place cells - {method}</h1>\n<br>\n"    #Add title
    html = html + f"<br><h2>{datetime.now().strftime('%d%m%y_%H%M%S')}</h2><br>\n"    #Add subtitle
    
    for cell_group in range(num_figures):
        fig= plt.figure(figsize = (12, 12))
        
        cell_st = cell_group*cells_per_plot
        cells_to_plot = np.min([neu_sig.shape[1]-cell_st, cells_per_plot])
        for cell_idx in range(cells_to_plot):
            gcell_idx = cell_idx + cell_st #global cell index
            row_idx = cell_idx//cells_per_row
            col_idx = cell_idx%cells_per_col
            
            if gcell_idx in place_cells:
                color_cell = [.2,.6,.2]
            else:
                color_cell = 'k'
                
            ax  = plt.subplot2grid((cells_per_row*6, cells_per_col), (row_idx*6, col_idx), rowspan=2)
            ax.plot(pos[:,0], pos[:,1], color = [.5,.5,.5], alpha = 0.3)
            
            if isinstance(spikes, type(None)):
                active_ts = neu_sig[:,gcell_idx] > 0
                t_neu_sig = neu_sig[active_ts, gcell_idx]
                t_pos = pos[active_ts,:]
                signal_inds = t_neu_sig.argsort()
                sorted_pos = t_pos[signal_inds,:]
                sorted_neu_sig = t_neu_sig[signal_inds]
                ax.scatter(*sorted_pos[:,:2].T, c = sorted_neu_sig, s= 8)
            else:
                ax.scatter(*pos[spikes[:,gcell_idx]>0,:2].T, color = color_cell, s= 5)
            ax.set_xlim([min_pos[0], max_pos[0]])
            ax.set_ylim([min_pos[1], max_pos[1]])
            title = list()
            title.append(f"Cell: {gcell_idx} -")
            [title.append(f"{mval:.2f} ") for mval in metric_val[gcell_idx]];
            ax.set_title(' '.join(title), color = color_cell)
            
            ax  = plt.subplot2grid((cells_per_row*6, cells_per_col), (row_idx*6+2, col_idx), rowspan=2)
            p = ax.matshow(neu_pdf[:,gcell_idx].T, aspect = 'auto')
            ax.set_yticks([])
            ax.set_xticks([])
            cbar = fig.colorbar(p, ax=ax,fraction=0.3, pad=0.08, location = 'bottom')
            cbar.ax.set_ylabel(method, rotation=270, size=10)
            
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
        plt.close(fig)
    #Save html file
    with open(os.path.join(save_dir, f"PlaceCells_{method}__{datetime.now().strftime('%d%m%y_%H%M%S')}.htm"),'w') as f:
        f.write(html)

def _compute_metric(pos_pdf, neu_pdf, method):
    if 'spatial_info' in method:
        return _compute_spatial_info(pos_pdf, neu_pdf)
    elif 'response_profile' in method:
        return _compute_response_profile(pos_pdf, neu_pdf)

def _check_placefields(pos_pdf, neu_pdf, place_cells = None):
    """Compute place fields of already detected place cells. Adapted from:
    https://doi.org/10.1038/nn.2648
    
    1-Potential place fields: contiguous regions in which all points were 
        greater than 25% of the difference between the peak fluo value and the
        baseline value (20 percentile)
        
    2-Potential place field have to satisfy:
        a) >18cm wide (Does not apply)
        b) At least one bin with > 0.1 fluo (changed to 0.01Hz)
        c) Mean in-field fluo > 3* mean out-field fluo
        d) Significant calcium traces must be present 30% of the time (Does not apply)

    Parameters
    ----------
    pos_pdf: numpy array (n1, n2,...,nm,1,d)
        Numpy array with the position probability density functions. Its shape is
        first all the bining of the space, then an empty dimension, and lastly the
        direction dim. 
            e.g. 
            1. If we divide a box in 10 x-bins, 6 y-bins, and no direction, 
            then pos_pdf has a shape of (10,6,1,1).
            2. If we divide a box in 10 x-bins, no y-bins, and 2 directions, 
            pos_pdf has a shape of (10,1,2)
            3. If we divide a 3D manifold, in 10 'x-bins', 6 'y-bins', 5 'z-bins'
            and 3 directions, then pos_pdf has a shape of (10,6,5,1,3)
            
    neu_pdf: numpy array (n1, n2,...,nm,neu,d)
        Numpy array with the mean neural activity per bin. Its shape is
        first all the bining of the space, then the number of neurons, and lastly the
        direction dim. 
            e.g. 
            1. If we divide a box in 10 x-bins, 6 y-bins, and no direction, 
            and we have 100 neurons then neu_pdf has a shape of (10,6,100,1).
            2. If we divide a box in 10 x-bins, no y-bins, 100 neurons and 2 
            directions, neu_pdf has a shape of (10,100,2)
            3. If we divide a 3D manifold, in 10 'x-bins', 6 'y-bins', 5 'z-bins'
            and 3 directions, and we have 150 neurons, then neu_pdf has a shape 
            of (10,6,5,150,3)
       
    Returns
    -------
    place_fields: numpy array (space_dims, nn, nd)
        Numpy array containing 0 on the bins with no place field, and x for 
        the bins that correspond to a place field (where x is the id of the 
        field it belongs to) for each neuron (nn) and each direction.
        
    place_fields_id: numpy array (nn,nd)
        Numpy array containing the number of place fields for each neuron 
        and direction. The final shape is (nn, nd) where nn is the number of 
        neurons, and nd the number of directions
            e.g. 
            For case 1, spatial_info has a shape of (100,1)
            For case 2, spatial info has a shape of (100,2)
            For case 3, spatial info has a shape of (150,3)
    """
    if isinstance(place_cells, type(None)):
        place_cells = np.linspace(0,neu_pdf.shape[-2]-1,neu_pdf.shape[-2]).astype(int)
        
    space_dims = tuple(range(neu_pdf.ndim - 2)) #idx of axis related to space
    subAxis_length =  [neu_pdf.shape[d] for d in range(neu_pdf.ndim-2)] #number of bins on each space dim
    pneu_pdf = eval("neu_pdf[" + ":,"*(neu_pdf.ndim - 2)+ "place_cells,:]")
    max_response = np.nanmax(pneu_pdf, axis=space_dims)
    baseline_th = np.percentile(pneu_pdf, 25, axis=space_dims)
    baseline_th = np.tile(baseline_th, tuple(subAxis_length+[1,1]))
    baseline_response = copy.deepcopy(pneu_pdf)
    baseline_response[pneu_pdf>baseline_th] = np.nan
    baseline_response = np.nanmean(baseline_response, axis= space_dims)
    
    putative_th = 0.25*(max_response-baseline_response)    
    putative_th = np.tile(putative_th, tuple(subAxis_length+[1,1]))
    putative_fields = pneu_pdf>putative_th
    
    place_fields = np.zeros(putative_fields.shape)
    place_fields_id = np.zeros(pneu_pdf.shape[-2:])
    for cell in range(len(place_cells)):
        for dire in range(pneu_pdf.shape[-1]):
            temp_neu = eval("putative_fields[" + ":,"*(pneu_pdf.ndim - 2)+ "cell, dire]")
            labeled, nr_objects = ndimage.label(temp_neu) 
            flag_1 = False
            flag_2 = False
            flag_3 = False
            flag_4 = False
            for field_idx in range(nr_objects):
                field_im = labeled == field_idx+1
                #condition 1: minimum size
                #condition 2: at least one bin with >0.01
                flag_2 = np.sum(pneu_pdf[field_im, cell, dire]>0.02)==0
                #condition 3: Mean in-field fluo > 3* mean out-field fluo
                m_infield = np.nanmean(pneu_pdf[field_im, cell, dire])
                m_outfield = np.nanmean(pneu_pdf[labeled==0, cell, dire])
                flag_3 = m_infield<3*m_outfield
                #condition 4:                 
                if (flag_1+flag_2+flag_3+flag_4):
                    putative_fields[field_im,cell,dire] = False       
                    labeled[field_im] = 0
            #see if need merging (fields apart by less than 3 steps)
            #programmed from: https://cs.stackexchange.com/questions/117989/hausdorff-distance-between-two-binary-images-according-to-distance-maps
            if len(np.unique(labeled))-1>1:
                n_fields = len(np.unique(labeled))-1
                field_og = 1
                while field_og < n_fields:
                    field_im = labeled == field_og
                    dist_field = get_distance_to_object(field_im)
                        
                    merged = 0
                    for field_to in range(field_og+1, n_fields+1):
                        dist_og_to = dist_field*(labeled==field_to)
                        dist_og_to[dist_og_to==0] = np.nan
                        min_dist = np.nanmin(dist_og_to)
                        if min_dist<3:
                            field_to_im = labeled == field_to
                            dist_to_field = get_distance_to_object(field_to_im)
                            dist_to_field[dist_to_field==0]
                            
                            in_between_bins = (dist_field<3)*(dist_to_field<3)
                            
                            labeled[labeled==field_to] = field_og
                            labeled[in_between_bins == True] = field_og
                            merged +=1
                            n_fields = len(np.unique(labeled))-1
                    if merged>0:
                        labeled[labeled>field_og] -=merged
                    else:
                        field_og +=1
                    
            place_fields[labeled>-1,cell,dire] = labeled
            place_fields_id[cell, dire] = len(np.unique(labeled))-1
            
    return place_fields, place_fields_id


def get_distance_to_object(mat):
    #Adapted from https://www.geeksforgeeks.org/distance-nearest-cell-1-binary-matrix/
    #TODO: generalize to N dimensions
    og_dim = False
    if mat.ndim==1:
        og_dim = True
        mat = mat.reshape(-1,1)
        
    # Initialize the answer matrix 
    D = np.zeros(mat.shape)+np.inf
    # For each cell
    for from_row in range(D.shape[0]):
        for from_col in range(D.shape[1]):
            if mat[from_row, from_col]==1:
                D[from_row, from_col] = 0
            else:
                # Traversing the whole matrix to find the minimum distance.
                for to_row in range(D.shape[0]):
                    for to_col in range(D.shape[1]):
                        # If cell contain 1, check for minimum distance.
                        if mat[to_row, to_col] == 1:
                            D[from_row, from_col] = min(D[from_row, from_col],
                                        abs(from_row - to_row) + abs(from_col - to_col))
    if og_dim:
        D = D[:,0]
                       
    return D


def _compute_response_profile(pos_pdf, neu_pdf):
    """Compute response profile adapted from
    https://doi.org/10.1016/j.cub.2020.07.006
    
    Parameters
    ----------
    pos_pdf: numpy array (n1, n2,...,nm,1,d)
        Numpy array with the position probability density functions. Its shape is
        first all the bining of the space, then an empty dimension, and lastly the
        direction dim. 
            e.g. 
            1. If we divide a box in 10 x-bins, 6 y-bins, and no direction, 
            then pos_pdf has a shape of (10,6,1,1).
            2. If we divide a box in 10 x-bins, no y-bins, and 2 directions, 
            pos_pdf has a shape of (10,1,2)
            3. If we divide a 3D manifold, in 10 'x-bins', 6 'y-bins', 5 'z-bins'
            and 3 directions, then pos_pdf has a shape of (10,6,5,1,3)
            
    neu_pdf: numpy array (n1, n2,...,nm,neu,d)
        Numpy array with the mean neural activity per bin. Its shape is
        first all the bining of the space, then the number of neurons, and lastly the
        direction dim. 
            e.g. 
            1. If we divide a box in 10 x-bins, 6 y-bins, and no direction, 
            and we have 100 neurons then neu_pdf has a shape of (10,6,100,1).
            2. If we divide a box in 10 x-bins, no y-bins, 100 neurons and 2 
            directions, neu_pdf has a shape of (10,100,2)
            3. If we divide a 3D manifold, in 10 'x-bins', 6 'y-bins', 5 'z-bins'
            and 3 directions, and we have 150 neurons, then neu_pdf has a shape 
            of (10,6,5,150,3)
       
    Returns
    -------
    response_profile: numpy array
        Response profile for each neuron and direction, which corresponds to the
        maximum activity across all space-bins. The final shape is (nn, nd) 
        where nn is the number of neurons, and nd the number of directions
            e.g. 
            For case 1, response_profile has a shape of (100,1)
            For case 2, response_profile  info has a shape of (100,2)
            For case 3, response_profile  info has a shape of (150,3)
            
    """
    space_dims = tuple(range(neu_pdf.ndim - 2)) #idx of axis related to space
    #n_neu = neu_pdf.shape[-2]
    #subAxis_length =  [neu_pdf.shape[d] for d in range(neu_pdf.ndim-2)] #number of bins on each space dim
    #nbins = np.prod(subAxis_length) #total number of space bins
    
    #min_pos_pdf = np.min([0.1/nbins, np.nanpercentile(pos_pdf,1)])
    #min_pos_pdf = np.min([0.1/nbins, np.nanpercentile(pos_pdf,1)])
    
    #tile_pos_pdf = np.tile(pos_pdf, tuple(list(np.tile(1, len(space_dims)))+[n_neu,1]))
    #tile_pos_pdf[tile_pos_pdf<=min_pos_pdf] = np.nan
    max_response = np.nanmax(neu_pdf, axis=space_dims)
    return max_response
    

def _compute_spatial_info(pos_pdf, neu_pdf):
    """Compute spatial information adapted from
    https://doi.org/10.1038/s41467-019-10139-7
    $$ Spatial information = \sum_{x,y}^{Nbins}p(x,y)\cdot(\frac{r(x,y)}{\bar(r)})
                                            \cdot\log_{2}(\frac{r(x,y)}{\bar(r)}) $$ 
    
    Parameters
    ----------
    pos_pdf: numpy array (n1, n2,...,nm,1,d)
        Numpy array with the position probability density functions. Its shape is
        first all the bining of the space, then an empty dimension, and lastly the
        direction dim. 
            e.g. 
            1. If we divide a box in 10 x-bins, 6 y-bins, and no direction, 
            then pos_pdf has a shape of (10,6,1,1).
            2. If we divide a box in 10 x-bins, no y-bins, and 2 directions, 
            pos_pdf has a shape of (10,1,2)
            3. If we divide a 3D manifold, in 10 'x-bins', 6 'y-bins', 5 'z-bins'
            and 3 directions, then pos_pdf has a shape of (10,6,5,1,3)
            
    neu_pdf: numpy array (n1, n2,...,nm,neu,d)
        Numpy array with the mean neural activity per bin. Its shape is
        first all the bining of the space, then the number of neurons, and lastly the
        direction dim. 
            e.g. 
            1. If we divide a box in 10 x-bins, 6 y-bins, and no direction, 
            and we have 100 neurons then neu_pdf has a shape of (10,6,100,1).
            2. If we divide a box in 10 x-bins, no y-bins, 100 neurons and 2 
            directions, neu_pdf has a shape of (10,100,2)
            3. If we divide a 3D manifold, in 10 'x-bins', 6 'y-bins', 5 'z-bins'
            and 3 directions, and we have 150 neurons, then neu_pdf has a shape 
            of (10,6,5,150,3)
       
    Returns
    -------
    spatial_info: numpy array
        Spatial information for each neuron and direction. The final shape is 
        (nn, nd) where nn is the number of neurons, and nd the number of directions
            e.g. 
            For case 1, spatial_info has a shape of (100,1)
            For case 2, spatial info has a shape of (100,2)
            For case 3, spatial info has a shape of (150,3)
            
    """
    space_dims = tuple(range(neu_pdf.ndim - 2)) #idx of axis related to space
    subAxis_length =  [neu_pdf.shape[d] for d in range(neu_pdf.ndim-2)] #number of bins on each space dim
    neu_pdf_min = copy.deepcopy(neu_pdf)
    neu_pdf_min[neu_pdf<1e-5] = 1e-5 
    m_neu_pdf = np.mean(neu_pdf_min, axis= space_dims) #mean neuronal activity for each neuron and each direction along all space
    m_neu_pdf = np.expand_dims(m_neu_pdf, axis = space_dims) #create empty dimensions along space axis
    m_neu_pdf = np.tile(m_neu_pdf, tuple(subAxis_length+[1,1])) #copy mean neuronal activity along all bin spaces
    norm_neu_pdf = np.divide(neu_pdf_min, m_neu_pdf) #element wise division (r(x,y)/mean(r))
    norm_neu_pdf[norm_neu_pdf==0] = np.nan #set no activity to 0 to avoid log2(0)
    
    n_neu = neu_pdf.shape[-2]
    
    tile_pos_pdf = np.tile(pos_pdf, tuple(list(np.tile(1, len(space_dims)))+[n_neu,1]))
    
    return np.nansum(np.multiply(tile_pos_pdf, np.multiply(norm_neu_pdf, 
                  np.log2(norm_neu_pdf))), axis=tuple(range(neu_pdf.ndim - 2)))


def _get_edges(pos, limit, dim):
    """Obtain which points of 'pos' are inside the limits of the border defined
    by limit (%).
    
    Parameters
    ----------
    pos: numpy array
        Numpy array with the position
        
    limit: numerical (%)
        Limit (%) used to decided which points are kepts and which ones are 
        discarded. (i.e. if limit=10 and pos=[0,100], only the points inside the
                    range [10,90] will be kepts).
    
    dim: integer
        Dimensionality of the division.    
                
    Returns
    -------
    signal: numpy array
        Concatenated dataframe column into numpy array along axis=0.
    """  
    assert pos.shape[1]>=dim, f"pos has less dimensions ({pos.shape[1]}) " + \
                                f"than the indicated in dim ({dim})"
    norm_limit = limit/100
    minpos = np.min(pos, axis=0)
    maxpos = np.max(pos,axis=0)
    
    trackLength = maxpos - minpos
    trackLimits = np.vstack((minpos + norm_limit*trackLength,
                             maxpos - norm_limit*trackLength))
    
    points_inside_lim = list()
    for d in range(dim):
        points_inside_lim.append(np.vstack((pos[:,d]<trackLimits[0,d],
                                       pos[:,d]>trackLimits[1,d])).T)
        
    points_inside_lim = np.concatenate(points_inside_lim, axis=1)
    points_inside_lim = ~np.any(points_inside_lim ,axis=1)
    
    return points_inside_lim


def _get_pdf(pos, neu_signal, mapAxis, dim, dir_mat = None):
    """Obtain probability density function of 'pos' along each bin defined by 
    the limits mapAxis as well as the mean signal of 'neu_signal' on those bins.
    If dir_mat is provided, the above is computed for each possible direction.

    Parameters
    ----------
    pos: numpy array (T, space_dims)
        Numpy array with the position
        
    neu_signal: numpy array (T, nn)
        Numpy array containing the neural signal (spikes, rates, or traces).

    mapAxis: list
        List with as many entries as spatial dimensions. Each entry contains 
        the lower limits of the bins one wants to divide that spatial dimension 
        into. 
            e.g.
            If we want to divide our 2D space, then mapAxis will contain 2 
            entries. The first entry will have the lower limits of the x-bins 
            and the second entry those of the y-bins.
    
    dim: integer
        Integer indicating the number of spatial dimenions of the bins. That
        is, 
            dim = len(mapAxis) <= pos.shape[1]
            
    Optional parameters:
    --------------------
    dir_mat: numpy array (T,1)
        Numpy array containing the labels of the direction the animal was 
        moving on each timestamp. For each label, a separate pdf will be 
        computed.
        
    Returns
    -------
    pos_pdf: numpy array (space_dims_bins, 1, dir_labels)
        Numpy array containing the probability the animal is at each bin for
        each of the directions if dir_mat was indicated.
    
    neu_pdf: numpy array (space_dims_bins, nn, dir_labels)
        Numpy array containing the mean neural activity of each neuron at each 
        bin for each of the directions if dir_mat was indicated.
    """
    if isinstance(dir_mat, type(None)):
        dir_mat = np.zeros((pos.shape[0],1))
        
    val_dirs = np.array(np.unique(dir_mat))
    num_dirs = len(val_dirs)
                   
    subAxis_length =  [len(mapsubAxis) for mapsubAxis in mapAxis] #nbins for each axis
    access_neu = np.linspace(0,neu_signal.shape[1]-1,neu_signal.shape[1]).astype(int) #used to later access neu_pdf easily

    neu_pdf = np.zeros(subAxis_length+[neu_signal.shape[1]]+[num_dirs])
    pos_pdf = np.zeros(subAxis_length+[1]+[num_dirs])
    
    for sample in range(pos.shape[0]):
        pos_entry_idx = list()
        for d in range(dim):
            temp_entry = np.where(pos[sample,d]>=mapAxis[d])
            if np.any(temp_entry):
                temp_entry = temp_entry[0][-1]
            else:
                temp_entry = 0
            pos_entry_idx.append(temp_entry)
        
        dir_idx = np.where(dir_mat[sample]==val_dirs)[0][0]
        pos_idxs = tuple(pos_entry_idx + [0]+[dir_idx])
        neu_idxs = tuple(pos_entry_idx + [access_neu]+[dir_idx])
        
        pos_pdf[pos_idxs] += 1
        neu_pdf[neu_idxs] += (1/pos_pdf[pos_idxs])* \
                                   (neu_signal[sample,:]-neu_pdf[neu_idxs]) #online average
                                   
    pos_pdf = pos_pdf/np.sum(pos_pdf,axis=tuple(range(pos_pdf.ndim - 1))) #normalize probability density function
    
    return pos_pdf, neu_pdf




    







