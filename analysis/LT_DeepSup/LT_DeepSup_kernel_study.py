#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:17:28 2022

@author: julio
"""

#%% IMPORTS
import numpy as np
import copy
from neural_manifold import general_utils as gu
from neural_manifold import dimensionality_reduction as dim_red
import pickle
import os

import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO

#%%
def plot_kernel_study(kernel_dict, save_dir):
    fnames = list(kernel_dict.keys())
    
    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + f"<h1>Kernel study - {fnames[0][:3]}</h1>\n<br>\n"    #Add title
    html = html + f"<h2>traces: {kernel_dict[fnames[0]]['params']['traces_field']} - "
    html = html + f"spikes: {kernel_dict[fnames[0]]['params']['spikes_field']} - "
    html = html + f"<br>{datetime.now().strftime('%d/%m/%y %H:%M:%S')}</h2><br>\n"    #Add subtitle
    
    sI_vmin = np.inf
    sI_vmax = 0
    
    R2s_vmin = 0
    R2s_vmax = np.zeros((4,1))
    for file_idx, file_name in enumerate(fnames):
        sI_vmin = np.min([sI_vmin, np.min(kernel_dict[file_name]['sI'], axis= (0,1,2))])
        sI_vmax = np.max([sI_vmax, np.max(kernel_dict[file_name]['sI'], axis= (0,1,2))])
        
        temp_R2s_vmax = np.max(np.mean(kernel_dict[file_name]['R2s'], axis=2), axis= (0,1)).reshape(-1,1)
        temp_R2s_vmax[temp_R2s_vmax>25] = 25
        R2s_vmax = np.max(np.concatenate((R2s_vmax, temp_R2s_vmax),axis=1), axis=1).reshape(-1,1)
    
    dec_list = ['wf','wc','xgb','svm']
    
    fig= plt.figure(figsize = (20, 10))
    
    for file_idx, file_name in enumerate(fnames):
    
        ytick_labels = [str(entry) for entry in kernel_dict[file_name]['params']['ks_list']]
        xtick_labels = [str(entry) for entry in kernel_dict[file_name]['params']['sI_nn_list']]
        
        ax = plt.subplot(2,6,1+6*file_idx)
        if file_idx == 0:
            pos = 0.75
        else:
            pos = 0.25
            
        fig.text(0.008, pos, f"{file_name}",horizontalalignment='center', 
                         rotation = 'vertical', verticalalignment='center', fontsize = 20)
    
        ax.set_title("sI: symmetric",fontsize=15)
        ax.imshow(kernel_dict[file_name]['sI'][:,0,:], vmin = sI_vmin, vmax = sI_vmax, aspect = 'auto')
        ax.set_xlabel('nn', labelpad = 5)
        ax.set_ylabel('kernel (ms)', labelpad = -5)
        ax.set_yticks(np.arange(kernel_dict[file_name]['sI'].shape[0]), labels=ytick_labels)
        ax.set_xticks(np.arange(kernel_dict[file_name]['sI'].shape[2]), labels=xtick_labels)
    
        ax = plt.subplot(2,6,2+6*file_idx)
        ax.set_title("sI: assymmetric",fontsize=15)
        b = ax.imshow(kernel_dict[file_name]['sI'][:,1,:], vmin = sI_vmin, vmax = sI_vmax, aspect = 'auto')
        fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
        
        ax.set_xlabel('nn', labelpad = 5)
        ax.set_ylabel('kernel (ms)', labelpad = -5)
        ax.set_yticks(np.arange(kernel_dict[file_name]['sI'].shape[0]), labels=ytick_labels)
        ax.set_xticks(np.arange(kernel_dict[file_name]['sI'].shape[2]), labels=xtick_labels)
        
        for ii in range(4):
            
            if ii == 0:
                pos = 3+file_idx*12
            elif ii == 1:
                pos = 4+file_idx*12
            elif ii == 2:
                pos = 9+file_idx*12
            elif ii == 3:
                pos = 10+file_idx*12
    
            ax = plt.subplot(4,6,pos)
            m = np.mean(kernel_dict[file_name]['R2s'][:,0,:,ii], axis=1)
            sd = np.std(kernel_dict[file_name]['R2s'][:,0,:,ii], axis=1)
            ax.plot(m, '--', c= 'b', label = 'symmetric')
            ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= 'b', alpha = 0.3)
            
            m = np.mean(kernel_dict[file_name]['R2s'][:,1,:,ii], axis=1)
            sd = np.std(kernel_dict[file_name]['R2s'][:,1,:,ii], axis=1)
            ax.plot(m, c= 'g', label = 'asymmetric')
            ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= 'g', alpha = 0.3)
            ax.set_xlabel('kernel (ms)', labelpad = -2)
            ax.set_xticks(np.arange(len(m)), labels=ytick_labels)
            ax.set_ylim([R2s_vmin, R2s_vmax[0,0]])
            ax.set_yticks([R2s_vmin, R2s_vmax[0,0]/2, R2s_vmax[0,0]])
            ax.set_ylabel('error xpos [cm]', labelpad = 5)
            ax.set_title(dec_list[ii])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                
        
        ax = plt.subplot(2,3,3+3*file_idx)
        ax.plot(kernel_dict[file_name]['inner_dim'][:,0], '--', c='b', label = 'symmetric')
        ax.plot(kernel_dict[file_name]['inner_dim'][:,1], c='g', label = 'asymmetric')
        ax.set_xlabel('kernel (ms)', labelpad = -2)
        ax.set_xticks(np.arange(len(m)), labels=ytick_labels)
        ax.set_ylabel('inner_dim', labelpad = 5)
        ax.set_ylim([0, 5])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
    plt.close(fig)
    
    with open(os.path.join(save_dir, f"{fnames[0][:3]}_kernel_study_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)
    
    return True
#%% GENERAL PARAMS
save_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/kernel_study'
params = {
    'ks_list':[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10,np.inf],
    'assymetry_list':[False, True],
    'sI_nn_list':[3, 10, 20, 50, 100, 200],
    'verbose': True
    }
spikes_field = 'events_SNR3'
traces_field = 'denoised_traces'

save_params = {}
save_params.update(params)
save_params.update({'spikes_field':spikes_field, 'traces_field':traces_field})
#%% CZ3
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/CZ3'
CZ3 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(CZ3.keys())
CZ3_kernel_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(CZ3[fname])
    #compute hyperparameter study
    CZ3_R2s_kernel, CZ3_sI_kernel, CZ3_dim_kernel = dim_red.check_kernel_size(pd_struct, spikes_field, 
                                                                              traces_field, **params)
    #save results
    CZ3_kernel_study[fname] = {
        'R2s': CZ3_R2s_kernel,
        'sI': CZ3_sI_kernel,
        'inner_dim': CZ3_dim_kernel,
        'params': save_params
        }
    save_ks = open(os.path.join(save_dir, "CZ3_kernel_study_dict.pkl"), "wb")
    pickle.dump(CZ3_kernel_study, save_ks)
    save_ks.close()

_ = plot_kernel_study(CZ3_kernel_study, save_dir)

#%% CZ4
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/CZ4'
CZ4 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(CZ4.keys())
CZ4_kernel_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(CZ4[fname])
    #compute hyperparameter study
    CZ4_R2s_kernel, CZ4_sI_kernel, CZ4_dim_kernel = dim_red.check_kernel_size(pd_struct, spikes_field, 
                                                                              traces_field, **params)
    #save results
    CZ4_kernel_study[fname] = {
        'R2s': CZ4_R2s_kernel,
        'sI': CZ4_sI_kernel,
        'inner_dim': CZ4_dim_kernel,
        'params': save_params
        }
    save_ks = open(os.path.join(save_dir, "CZ4_kernel_study_dict.pkl"), "wb")
    pickle.dump(CZ4_kernel_study, save_ks)
    save_ks.close()

_ = plot_kernel_study(CZ4_kernel_study, save_dir)

#%% CZ6
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/CZ6'
CZ6 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(CZ6.keys())
CZ6_kernel_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(CZ6[fname])
    #compute hyperparameter study
    CZ6_R2s_kernel, CZ6_sI_kernel, CZ6_dim_kernel = dim_red.check_kernel_size(pd_struct, spikes_field, 
                                                                              traces_field, **params)
    #save results
    CZ6_kernel_study[fname] = {
        'R2s': CZ6_R2s_kernel,
        'sI': CZ6_sI_kernel,
        'inner_dim': CZ6_dim_kernel,
        'params': save_params
        }
    save_ks = open(os.path.join(save_dir, "CZ6_kernel_study_dict.pkl"), "wb")
    pickle.dump(CZ6_kernel_study, save_ks)
    save_ks.close()

_ = plot_kernel_study(CZ6_kernel_study, save_dir)

#%% GC2
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/GC2'
GC2 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(GC2.keys())
GC2_kernel_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(GC2[fname])
    #compute hyperparameter study
    GC2_R2s_kernel, GC2_sI_kernel, GC2_dim_kernel = dim_red.check_kernel_size(pd_struct, spikes_field, 
                                                                              traces_field, **params)
    #save results
    GC2_kernel_study[fname] = {
        'R2s': GC2_R2s_kernel,
        'sI': GC2_sI_kernel,
        'inner_dim': GC2_dim_kernel,
        'params': save_params
        }
    save_ks = open(os.path.join(save_dir, "GC2_kernel_study_dict.pkl"), "wb")
    pickle.dump(GC2_kernel_study, save_ks)
    save_ks.close()
    
_ = plot_kernel_study(GC2_kernel_study, save_dir)

#%% GC3
#load data
file_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/GC3'
GC3 = gu.load_files(file_dir, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")

fname_list = list(GC3.keys())
GC3_kernel_study = dict()
for fname in fname_list:
    pd_struct = copy.deepcopy(GC3[fname])
    #compute hyperparameter study
    GC3_R2s_kernel, GC3_sI_kernel, GC3_dim_kernel = dim_red.check_kernel_size(pd_struct, spikes_field, 
                                                                              traces_field, **params)
    #save results
    GC3_kernel_study[fname] = {
        'R2s': GC3_R2s_kernel,
        'sI': GC3_sI_kernel,
        'inner_dim': GC3_dim_kernel,
        'params': save_params
        }
    save_ks = open(os.path.join(save_dir, "GC3_kernel_study_dict.pkl"), "wb")
    pickle.dump(GC3_kernel_study, save_ks)
    save_ks.close()

_ = plot_kernel_study(GC3_kernel_study, save_dir)
