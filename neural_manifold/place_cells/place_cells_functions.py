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

import scipy.signal as scs
from scipy.ndimage import convolve1d
from scipy import ndimage


@gu.check_inputs_for_pd
def get_place_cells(pos_signal=None, neu_signal=None, dim=2, save_dir=None,**kwargs):
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
    
    assert dim>0, "dim must be a positive integer"

    assert pos_signal.shape[1]>= dim, f"pos_signal has less dimensions ({pos_signal.shape[1]})" +\
                                        f" than the indicated in dim ({dim})"
    
    assert kwargs['std_pos']>=0, "std to smooth position must be positive"

    assert sum([arg in kwargs for arg in ['bin_width', 'bin_num']]) < 2,\
                                            "provide bin_width or bin_num but not both of them"

    assert 'sF' in kwargs, "you must provide the sampling frequency ('sF')"

    kwargs = _fill_missing_kwargs(kwargs)


    neu_sig = copy.deepcopy(neu_signal)
    #smooth pos signal if applicable
    if kwargs['std_pos']>0:
        pos = _smooth_data(pos_signal, std = kwargs['std_pos'], bin_size = 1/kwargs['sF'])
    else:
        pos = copy.deepcopy(pos_signal)
    if pos.ndim == 1:
        pos = pos.reshape(-1,1)

    #Compute velocity if not provided
    if 'vel_signal' not in kwargs:
        vel = np.linalg.norm(np.diff(pos_signal, axis= 0), axis=1)*kwargs['sF']
        vel = np.hstack((vel[0], vel))
    else:
        vel = copy.deepcopy(kwargs['vel_signal'])
    vel = vel.reshape(-1)

    #Compute direction if not provided
    if 'direction_signal' in kwargs:
        direction = kwargs["direction_signal"]
    else:
        direction = np.zeros((pos.shape[0],1))
    
    #Discard edges
    if kwargs['ignore_edges']>0:
        pos_boolean = _get_edges(pos, kwargs['ignore_edges'], 1)
        pos = pos[pos_boolean]
        neu_sig = neu_sig[pos_boolean]
        direction = direction[pos_boolean]
        vel = vel[pos_boolean]

    #compute moving epochs
    move_epochs = vel>=kwargs['vel_th'] 
    #keep only moving epochs
    pos = pos[move_epochs]
    neu_sig = neu_sig[move_epochs]
    direction = direction[move_epochs]
    vel = vel[move_epochs]
        
    #Create grid along each dimensions
    min_pos = np.percentile(pos,1, axis=0) #(pos.shape[1],)
    max_pos = np.percentile(pos,99, axis = 0) #(pos.shape[1],)
    obs_length = max_pos - min_pos #(pos.shape[1],)
    if 'bin_width' in kwargs:
        bin_width = kwargs['bin_width']
        if isinstance(bin_width, list):
            bin_width = np.array(bin_width)
        else:
            bin_width = np.repeat(bin_width, dim, axis=0) #(pos.shape[1],)
        nbins = np.ceil(obs_length[:dim]/bin_width).astype(int) #(pos.shape[1],)
        kwargs['nbins'] = nbins
    elif 'bin_num' in kwargs:
        nbins = kwargs['bin_num']
        if isinstance(nbins, list):
            nbins = np.array(nbins)
        else:
            nbins = np.repeat(nbins, dim, axis=0) #(pos.shape[1],)
        bin_width = np.round(obs_length[:dim]/nbins,4) #(pos.shape[1],)
        kwargs['bin_width'] = bin_width
    mapAxis = list()
    for d in range(dim):
        mapAxis.append(np.linspace(min_pos[d], max_pos[d], nbins[d]+1)[:-1].reshape(-1,1)); #(nbins[d],)
    
    #Compute probability density function
    pos_pdf, neu_pdf = _get_pdf(pos, neu_sig, mapAxis, dim, direction)
    
    #smooth pdfs
    for d in range(dim):
        pos_pdf = _smooth_data(pos_pdf, std=kwargs['std_pdf'], bin_size=bin_width[d], axis=d)
        neu_pdf = _smooth_data(neu_pdf, std=kwargs['std_pdf'], bin_size=bin_width[d], axis=d)
    
    #compute metric
    metric_val = _compute_metric(pos_pdf, neu_pdf, kwargs["method"])

    #check which cells do not have a minimum activity
    val_dirs = np.array(np.unique(direction))
    num_dirs = len(val_dirs)
    total_firing_neurons = np.zeros((neu_sig.shape[1], num_dirs))
    if len(np.unique(neu_sig))==2:
        for dr in range(num_dirs):
            total_firing_neurons[:,dr] = np.sum(neu_sig[direction[:,0]==val_dirs[dr],:]>0., axis=0)
    else:
        if 'fluo_th' not in kwargs:
            kwargs['fluo_th'] = 0.1
        for dr in range(num_dirs):
            total_firing_neurons[:,dr] = (np.sum(np.diff(neu_sig[direction[:,0]==val_dirs[dr],:]\
                                            >kwargs['fluo_th'], axis=0)>0, axis= 0)/2).astype(int)

    low_firing_neurons = total_firing_neurons<3
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
            shifted_pos_pdf = _smooth_data(shifted_pos_pdf, std = kwargs['std_pdf'], bin_size = bin_width[d], axis = d)
            shifted_neu_pdf = _smooth_data(shifted_neu_pdf, std = kwargs['std_pdf'], bin_size = bin_width[d], axis = d)
        shuffled_metric_val[idx] = _compute_metric(shifted_pos_pdf, shifted_neu_pdf, kwargs["method"])
        
    th_metric_val = np.percentile(shuffled_metric_val, kwargs['th_metric'], axis=0)
    th_metric_val[th_metric_val<0] = 0
    place_cells_idx = np.linspace(0, metric_val.shape[0]-1, metric_val.shape[0])
    place_cells_idx = place_cells_idx[np.any((~low_firing_neurons)*metric_val>th_metric_val,axis=1)].astype(int)
    place_cells_dir = (metric_val*~low_firing_neurons>th_metric_val).astype(int)

    if dim==1:
        _plot_place_cells_1D(pos, neu_sig, neu_pdf, metric_val, direction, place_cells_idx, 
                                        place_cells_dir, kwargs["method"], save_dir, kwargs["mouse"])
    elif dim==2:
        _plot_place_cells_2D(pos, neu_sig, neu_pdf, metric_val, place_cells_idx, 
                                    kwargs["method"], save_dir)
    
    # place_fields, place_fields_id = _check_placefields(pos_pdf, neu_pdf, place_cells_idx)
    
    output_dict = {
        "place_cells_idx": place_cells_idx,
        "place_cells_dir": place_cells_dir,
        # "place_fields": place_fields,
        "metric_val": metric_val,
        "shuffled_metric_val": shuffled_metric_val,
        "th_metric_val": th_metric_val,
        "low_firing_neurons": low_firing_neurons,
        "total_firing_neurons": total_firing_neurons,
        "pos_pdf": pos_pdf,
        "neu_pdf": neu_pdf,
        "mapAxis": mapAxis,
        'neu_sig': neu_sig,
        'pos_sig': pos,
        'vel_sig': vel,
        'dir_sig': direction,
        "params": kwargs
    }

    return output_dict

def _fill_missing_kwargs(kwargs):
    if 'bin_num' not in kwargs and 'bin_width' not in kwargs:
        kwargs["bin_width"] = 2.5 #cm
    if 'std_pos' not in kwargs:
        kwargs['std_pos'] = 0.025 #s
    if 'ignore_edges' not in kwargs:
        kwargs['ignore_edges'] = 5 #%
    if 'std_pdf' not in kwargs:
        kwargs['std_pdf'] = 5 #cm
    if 'method' not in kwargs:
        kwargs['method'] = 'spatial_info'
    if 'num_shuffles' not in kwargs:
        kwargs['num_shuffles'] = 1000
    if 'min_shift' not in kwargs:
        kwargs['min_shift'] = 5  #s
    if 'th_metric' not in kwargs:
        kwargs['th_metric'] = 99 #percentile
    if 'mouse' not in kwargs:
        kwargs['mouse'] = 'unknown'
    if 'vel_th' not in kwargs:
        kwargs['vel_th'] = 5 #cm/s
    return kwargs

def _plot_place_cells_2D(pos, neu_sig, neu_pdf, metric_val, place_cells, method, save_dir):
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

            th = np.nanstd(neu_sig[:,gcell_idx])
            active_ts = neu_sig[:,gcell_idx] > 2*th
            t_neu_sig = neu_sig[active_ts, gcell_idx]
            t_pos = pos[active_ts,:]
            signal_inds = t_neu_sig.argsort()
            sorted_pos = t_pos[signal_inds,:]
            sorted_neu_sig = t_neu_sig[signal_inds]
            ax.scatter(*sorted_pos[:,:2].T, c = sorted_neu_sig, s= 8)

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
    with open(os.path.join(save_dir, f"PlaceCells_{method}__{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)

def _plot_place_cells_1D(pos, neu_sig, neu_pdf, metric_val, direction, place_cells, place_cells_dir, method, save_dir, mouse):
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
    html = html + f"<h1>{mouse} Place Cells - {method}</h1>\n<br>\n"    #Add title
    html = html + f"<br><h2>{datetime.now().strftime('%d%m%y_%H%M%S')}</h2><br>\n"    #Add subtitle
    
    cells_per_plot = cells_per_row*cells_per_col
    num_figures = np.ceil(neu_sig.shape[1]/cells_per_plot).astype(int)
    for cell_group in range(num_figures):
        fig= plt.figure(figsize = (12, 10))

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
            if len(np.unique(neu_sig[:,gcell_idx]))==2:
                color= ['C9' if l == 1 else 'C1' for l in direction[neu_sig[:,gcell_idx]>0]]
                ax.scatter(*pos[neu_sig[:,gcell_idx]>0,:2].T, c = color, s= 5)
            else:
                idxs = np.argsort(neu_sig[:,gcell_idx], axis = 0)
                temp_pos = pos[idxs,:2]    
                temp_signal = neu_sig[idxs,gcell_idx]
                ax.scatter(*temp_pos.T, c = temp_signal, s= 5, alpha=0.3)
            ax.set_xlim([min_pos[0], max_pos[0]])
            ax.set_ylim([min_pos[1], max_pos[1]])
            title = list()
            title.append(f"Cell: {gcell_idx} -")
            [title.append(f"{midx}: {mval:.2f} ") for midx, mval in enumerate(metric_val[gcell_idx]) 
                                                             if place_cells_dir[gcell_idx, midx]==1];

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
    with open(os.path.join(save_dir, f"{mouse}_placeCells_{method}_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)

def _compute_metric(pos_pdf, neu_pdf, method):
    if 'spatial_info' in method:
        return _compute_spatial_info(pos_pdf, neu_pdf)
    elif 'response_profile' in method:
        return _compute_response_profile(pos_pdf, neu_pdf)
    elif 'moransI' in method:
        return _compute_morans_i(pos_pdf, neu_pdf)

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
    baseline_th = np.percentile(pneu_pdf, 20, axis=space_dims)
    baseline_th = np.tile(baseline_th, tuple(subAxis_length+[1,1]))
    baseline_response = copy.deepcopy(pneu_pdf)
    baseline_response[pneu_pdf>baseline_th] = np.nan
    baseline_response = np.nanmean(baseline_response, axis= space_dims)
    
    putative_th = 0.25*(max_response-baseline_response)+baseline_response    
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

def _compute_morans_i(pos_pdf, neu_pdf):
    #include pos_pdf for consistency although not needed)
    num_bins = pos_pdf.shape[0]

    w_sigma = 10  # cm
    w_sigma = w_sigma / 5
    indices = np.arange(num_bins)
    distances = indices[:, None] - indices[None, :]
    weights = np.exp(-distances**2 / (2 * w_sigma**2))
    np.fill_diagonal(weights, 0)

    space_dims = tuple(range(neu_pdf.ndim - 2)) #idx of axis related to space
    subAxis_length =  [neu_pdf.shape[d] for d in range(neu_pdf.ndim-2)] #number of bins on each space dim
    m_neu_pdf = np.mean(neu_pdf, axis= space_dims) #mean neuronal activity for each neuron and each direction along all space
    neu_pdf_centered = neu_pdf - m_neu_pdf

    morans_i = np.zeros((neu_pdf.shape[-2],neu_pdf.shape[-1]))*np.nan
    for cell in range(neu_pdf.shape[-2]):
        r_centered = neu_pdf_centered[:,cell,0].reshape(-1,)
        morans_i[cell,0] = np.sum(weights * (r_centered[:, None] * r_centered[None, :])) / np.sum(weights) / np.sum(r_centered**2) * num_bins

        r_centered = neu_pdf_centered[:,cell,1]
        morans_i[cell,1] = np.sum(weights * (r_centered[:, None] * r_centered[None, :])) / np.sum(weights) / np.sum(r_centered**2) * num_bins

    return morans_i

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
                    range [10,90] will be kept).
    
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

#Adapted from PyalData package (19/10/21) (added variable win_length)
def _norm_gauss_window(bin_size, std, num_std = 5):
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
    return win / np.sum(win)

#Copied from PyalData package (19/10/21)
def _hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))

#Copied from PyalData package (19/10/21)
def _smooth_data(mat, bin_size=None, std=None, hw=None, win=None, axis=0):
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
    assert  sum([arg is not None for arg in [win, hw, std]]) == 1, "only give win, hw, or std"
    if win is None:
        assert bin_size is not None, "specify bin_size if not supplying window"
        if std is None:
            std = _hw_to_std(hw)
        win = _norm_gauss_window(bin_size, std)
    return convolve1d(mat, win, axis=axis, output=np.float32, mode='reflect')
