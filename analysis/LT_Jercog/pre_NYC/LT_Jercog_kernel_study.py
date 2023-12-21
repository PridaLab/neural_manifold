#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:26:41 2022

@author: julio
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:08:20 2022

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
from kneed import KneeLocator
#%%
def plot_kernel_study(kernel_dict, save_dir):
    fnames = list(kernel_dict.keys())
    
    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + f"<h1>Kernel study - {fnames[0][:5]}</h1>\n<br>\n"    #Add title
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
    
    
    for file_idx, file_name in enumerate(fnames):
        fig= plt.figure(figsize = (18, 5))

        ytick_labels = [str(entry) for entry in kernel_dict[file_name]['params']['ks_list']]
        xtick_labels = [str(entry) for entry in kernel_dict[file_name]['params']['sI_nn_list']]
        
        ax = plt.subplot(1,6,1)
        fig.text(0.008,0.5, f"{file_name}",horizontalalignment='center', 
                         rotation = 'vertical', verticalalignment='center', fontsize = 20)
    
        ax.set_title("sI: symmetric",fontsize=15)
        ax.imshow(kernel_dict[file_name]['sI'][:,0,:], vmin = sI_vmin, vmax = sI_vmax, aspect = 'auto')
        ax.set_xlabel('nn', labelpad = 5)
        ax.set_ylabel('kernel (ms)', labelpad = -5)
        ax.set_yticks(np.arange(kernel_dict[file_name]['sI'].shape[0]), labels=ytick_labels)
        ax.set_xticks(np.arange(kernel_dict[file_name]['sI'].shape[2]), labels=xtick_labels)
    
        ax = plt.subplot(1,6,2)
        ax.set_title("sI: assymmetric",fontsize=15)
        b = ax.imshow(kernel_dict[file_name]['sI'][:,1,:], vmin = sI_vmin, vmax = sI_vmax, aspect = 'auto')
        fig.colorbar(b, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
        
        ax.set_xlabel('nn', labelpad = 5)
        ax.set_ylabel('kernel (ms)', labelpad = -5)
        ax.set_yticks(np.arange(kernel_dict[file_name]['sI'].shape[0]), labels=ytick_labels)
        ax.set_xticks(np.arange(kernel_dict[file_name]['sI'].shape[2]), labels=xtick_labels)
        
        for ii in range(4):
            
            if ii == 0:
                pos = 3
            elif ii == 1:
                pos = 4
            elif ii == 2:
                pos = 9
            elif ii == 3:
                pos = 10
    
            ax = plt.subplot(2,6,pos)
            m = np.mean(kernel_dict[file_name]['R2s'][:,0,:,ii], axis=1)
            sd = np.std(kernel_dict[file_name]['R2s'][:,0,:,ii], axis=1)
            ax.plot(m, '--', c= 'b', label = 'symmetric')
            ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= 'b', alpha = 0.3)
            
            m = np.mean(kernel_dict[file_name]['R2s'][:,1,:,ii], axis=1)
            sd = np.std(kernel_dict[file_name]['R2s'][:,1,:,ii], axis=1)
            ax.plot(m, c= 'g', label = 'asymmetric')
            ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= 'g', alpha = 0.3)
            ax.set_xlabel('kernel (ms)', labelpad = -2)
            ax.set_xticks(np.arange(len(m)), labels=ytick_labels, rotation = 90)
            ax.set_ylim([R2s_vmin, R2s_vmax[0,0]])
            ax.set_yticks([R2s_vmin, R2s_vmax[0,0]/2, R2s_vmax[0,0]])
            ax.set_ylabel('error xpos [cm]', labelpad = 5)
            ax.set_title(dec_list[ii])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                
        
        ax = plt.subplot(1,3,3)
        ax.plot(kernel_dict[file_name]['inner_dim'][:,0], '--', c='b', label = 'symmetric')
        ax.plot(kernel_dict[file_name]['inner_dim'][:,1], c='g', label = 'asymmetric')
        ax.set_xlabel('kernel (ms)', labelpad = -2)
        ax.set_xticks(np.arange(len(m)), labels=ytick_labels, rotation = 45)
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
        
    with open(os.path.join(save_dir, f"{fnames[0][:5]}_kernel_study_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)
    
    return True
#__________________________________________________________________________
#|                                                                        |#
#|                                KERNEL STUDY                            |#
#|________________________________________________________________________|#

#%% GENERAL PARAMS
mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/'

save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/kernel_study'
params = {
    'ks_list':[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10,np.inf],
    'assymetry_list':[False, True],
    'sI_nn_list':[3, 10, 20, 50, 100, 200],
    'verbose': True,
    'vel_th': 3
    }

spikes_field = 'ML_spikes'
traces_field = 'deconvProb'
save_params = {}
save_params.update(params)
save_params.update({'spikes_field':spikes_field, 'traces_field':traces_field})


for mouse in mice_list:
    print('')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir,mouse)
    mouse_dict = gu.load_files(file_path, '*_PyalData_struct.mat', verbose = True, struct_type = "PyalData")
    fnames = list(mouse_dict.keys())
    kernel_study_dict = dict()
    save_name = mouse + '_kernel_study_dict.pkl'
    for fname in fname_list:
        print(f"\nWorking on session: {fname}")
        pd_struct = copy.deepcopy(mouse_dict[fname])
        #compute hyperparameter study
        mouse_R2s_kernel, mouse_sI_kernel, mouse_dim_kernel = dim_red.check_kernel_size(pd_struct, spikes_field, 
                                                                                  traces_field, **params)
        #save results
        kernel_study_dict[fname] = {
            'R2s': mouse_R2s_kernel,
            'sI': mouse_sI_kernel,
            'inner_dim': mouse_dim_kernel,
            'params': save_params
        }
        save_ks = open(os.path.join(save_dir, save_name), "wb")
        pickle.dump(kernel_study_dict, save_ks)
        save_ks.close()

        _ = plot_kernel_study(kernel_study_dict, save_dir)

#__________________________________________________________________________
#|                                                                        |#
#|                           PLOT KERNEL STUDY                            |#
#|________________________________________________________________________|#

#%% LOAD DATA
if "M2019_kernel_study" not in locals():
    M2019_kernel_study = gu.load_files(save_dir, '*M2019_kernel_study_dict.pkl', verbose=True, struct_type = "pickle")
if "M2021_kernel_study" not in locals():
    M2021_kernel_study = gu.load_files(save_dir, '*M2021_kernel_study_dict.pkl', verbose=True, struct_type = "pickle")
if "M2023_kernel_study" not in locals():
    M2023_kernel_study = gu.load_files(save_dir, '*M2023_kernel_study_dict.pkl', verbose=True, struct_type = "pickle")
if "M2024_kernel_study" not in locals():
    M2024_kernel_study = gu.load_files(save_dir, '*M2024_kernel_study_dict.pkl', verbose=True, struct_type = "pickle")
if "M2025_kernel_study" not in locals():
    M2025_kernel_study = gu.load_files(save_dir, '*M2025_kernel_study_dict.pkl', verbose=True, struct_type = "pickle")  
if "M2026_kernel_study" not in locals():
    M2026_kernel_study = gu.load_files(save_dir, '*M2026_kernel_study_dict.pkl', verbose=True, struct_type = "pickle")  

#%% PLOT 1. 
#Get kernel with better decoding performance
assymetry = 1
kernel_std = M2019_kernel_study[list(M2019_kernel_study.keys())[0]]["params"]["ks_list"]
sI_nn = M2019_kernel_study[list(M2019_kernel_study.keys())[0]]["params"]["sI_nn_list"]

R2s = np.zeros((len(kernel_std),4,4,6))*np.nan
idim = np.zeros((len(kernel_std),4,6))*np.nan
sI = np.zeros((len(kernel_std), len(sI_nn), 4, 6))*np.nan

for s_idx, s_name in enumerate(M2019_kernel_study.keys()):
    pd_struct = M2019_kernel_study[s_name]
    R2s[:,:,s_idx,0] = np.mean(pd_struct["R2s"][:, assymetry,:,:], axis=1)
    idim[:,s_idx,0] = pd_struct["inner_dim"][:,assymetry]
    sI[:,:, s_idx,0] = pd_struct["sI"][:,assymetry,:]
    
for s_idx, s_name in enumerate(M2021_kernel_study.keys()):
    pd_struct = M2021_kernel_study[s_name]
    R2s[:,:,s_idx,1] = np.mean(pd_struct["R2s"][:, assymetry,:,:], axis=1)
    idim[:,s_idx,1] = pd_struct["inner_dim"][:,assymetry]
    sI[:,:, s_idx,0] = pd_struct["sI"][:,assymetry,:]
for s_idx, s_name in enumerate(M2023_kernel_study.keys()):
    pd_struct = M2023_kernel_study[s_name]
    R2s[:,:,s_idx,2] = np.mean(pd_struct["R2s"][:, assymetry,:,:], axis=1)
    idim[:,s_idx,2] = pd_struct["inner_dim"][:,assymetry]
    sI[:,:, s_idx,2] = pd_struct["sI"][:,assymetry,:]
    
for s_idx, s_name in enumerate(M2024_kernel_study.keys()):
    pd_struct = M2024_kernel_study[s_name]
    R2s[:,:,s_idx,3] = np.mean(pd_struct["R2s"][:, assymetry,:,:], axis=1)
    idim[:,s_idx,3] = pd_struct["inner_dim"][:,assymetry]
    sI[:,:, s_idx,3] = pd_struct["sI"][:,assymetry,:]
    
for s_idx, s_name in enumerate(M2025_kernel_study.keys()):
    pd_struct = M2025_kernel_study[s_name]
    R2s[:,:,s_idx,4] = np.mean(pd_struct["R2s"][:, assymetry,:,:], axis=1)
    idim[:,s_idx,4] = pd_struct["inner_dim"][:,assymetry]
    sI[:,:, s_idx,4] = pd_struct["sI"][:,assymetry,:]
    
for s_idx, s_name in enumerate(M2026_kernel_study.keys()):
    pd_struct = M2026_kernel_study[s_name]
    R2s[:,:,s_idx,5] = np.mean(pd_struct["R2s"][:, assymetry,:,:], axis=1)
    idim[:,s_idx,5] = pd_struct["inner_dim"][:,assymetry]
    sI[:,:, s_idx,5] = pd_struct["sI"][:,assymetry,:]
    
R2s_vmax = 25
dec_name = ["wf", "wc", "xgb", "svm"]


#%% PLOT
cpal2 = ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]
cpal3 = ["#CDB4DB", "#FFC8DD", "#FFAFCC", "#BDE0FE", "#A2D2FF"]
cpal1 = ["#5d93ef","#f8b242", "#97e297", "#f19b91", "#92bbef"]
plt.figure(figsize=(10.7,3.7))
dec_idx = 2
ax = plt.subplot(1,3,1)
m = np.nanmean(R2s[:,dec_idx,:], axis=(1,2))
sd = np.nanstd(R2s[:,dec_idx,:], axis=(1,2))/np.sqrt(R2s.shape[3]*R2s.shape[1])
ax.plot(m, c= cpal2[0])
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= cpal2[0], alpha = 0.3)
ax.set_xlabel('kernel (ms)', labelpad = -2)
ax.set_xticks(np.arange(len(m)), labels=kernel_std, rotation = 90)
ax.set_ylim([0, R2s_vmax])
ax.axvline(x= 4, color='k', linestyle='--')
ax.set_yticks([0, R2s_vmax/2, R2s_vmax])
ax.set_ylabel('error xpos [cm]', labelpad = 5)
ax.set_title(dec_name[dec_idx])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax = plt.subplot(1,3,2)
m = np.nanmean(idim, axis=(1,2))
sd = np.nanstd(idim, axis=(1,2))
ax.plot(m, c= cpal2[0])
ax.fill_between(np.arange(len(m)), m-sd, m+sd, color= cpal2[0], alpha = 0.3)
ax.set_xlabel('kernel (ms)', labelpad = -2)
ax.set_xticks(np.arange(len(m)), labels=kernel_std, rotation = 45)
ax.set_ylim([0, 5.5])
ax.set_yticks([0,1,2,3,4,5])
ax.set_ylabel('Inner Dim', labelpad = 5)
ax.axvline(x= 4, color='k', linestyle='--')
ax.set_title('Inner dim')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax = plt.subplot(1,3,3)
m = np.nanmean(sI, axis=(2,3))
sd = np.nanstd(sI, axis=(2,3))/np.sqrt(sI.shape[2]*sI.shape[3])
ax.plot(m[:,1], c= cpal2[3], label = 'Local: ' + str(sI_nn[1]))
ax.fill_between(np.arange(len(m[:,1])), m[:,1]-sd[:,1], m[:,1]+sd[:,1], color=cpal2[3], alpha = 0.3)
ax.plot(m[:,3], c= cpal2[1], label = 'Global: ' + str(sI_nn[3]))
ax.fill_between(np.arange(len(m[:,3])), m[:,3]-sd[:,3], m[:,3]+sd[:,3], color= cpal2[1], alpha = 0.3)
ax.set_xlabel('kernel (ms)', labelpad = -2)
ax.set_xticks(np.arange(len(m)), labels=kernel_std, rotation = 45)
kl = KneeLocator(kernel_std[:-1], np.mean(m[:-1,:], axis = -1), curve = "concave", direction = "increasing")
if kl.knee:
    dim = kl.knee
    dim = np.where(kernel_std==kl.knee)[0]
    ax.axvline(x= dim, color='k', linestyle='--', label= 'Knee')
ax.set_ylim([0, 1])
ax.set_yticks([0,0.5, 1])
ax.legend()
ax.set_ylabel('sI xpos', labelpad = 5)
ax.set_title('sI')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'LT_Jercog_Kernel_study.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'LT_Jercog_Kernel_study.png'), dpi = 400,bbox_inches="tight")