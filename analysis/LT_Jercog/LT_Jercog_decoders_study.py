#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 08:51:45 2022

@author: julio
"""
#%% IMPORTS
import numpy as np
from neural_manifold import general_utils as gu
import pickle, os, copy

import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO

import seaborn as sns
import pandas as pd
from neural_manifold import decoders as dec
#%%
def plot_decoders_study(dec_dict, save_dir):
    
    fnames = list(dec_dict.keys())
    dec_list = ['wf','wc','xgb','svr']

    html = '<HTML>\n'
    html = html + '<style>\n'
    html = html + 'h1 {text-align: center;}\n'
    html = html + 'h2 {text-align: center;}\n'
    html = html + 'img {display: block; width: 80%; margin-left: auto; margin-right: auto;}'
    html = html + '</style>\n'
    html = html + f"<h1>Decoders study - {fnames[0]}</h1>\n<br>\n"    #Add title
    html = html + f"<h2>signal: {dec_dict[fnames[0]]['params']['x_base_signal']} - "
    html = html + f"<br>{datetime.now().strftime('%d/%m/%y %H:%M:%S')}</h2><br>\n"    #Add subtitle
    
    my_pal = {dec_dict[fnames[0]]['params']['x_base_signal']: 'grey',
               'pca': '#5bb95bff', 'isomap': '#ff8a17ff', 'umap':'#249aefff'}

    for file_idx, file_name in enumerate(fnames):
        pd_struct = copy.deepcopy(dec_dict[file_name])

         
        fig= plt.figure(figsize = (16, 4))
        
        pd_struct = copy.deepcopy(dec_dict[file_name])
    
        signals= list(pd_struct.keys())
        signals.remove('params')
        fig.text(0.008, 0.5, f"{file_name}",horizontalalignment='center', 
                         rotation = 'vertical', verticalalignment='center', fontsize = 20)
        
        for dec_idx, dec_name in enumerate(dec_list):
            num_folds = pd_struct[signals[0]][dec_name].shape[0]

            temp =  ['base_signal' in s for s in signals]
            t_signals = copy.deepcopy(signals)
            t_signals[np.where(temp)[0][0]] = pd_struct['params']['x_base_signal']
            
            pd_signal = list()
            for s in t_signals:
                pd_signal += [s]*num_folds
            
            
            pd_train_data = pd_struct[signals[0]][dec_name][:,0,1]
            pd_test_data = pd_struct[signals[0]][dec_name][:,0,0]
            
            for s in signals[1:]:
                pd_train_data = np.hstack((pd_train_data,pd_struct[s][dec_name][:,0,1]))
                pd_test_data = np.hstack((pd_test_data,pd_struct[s][dec_name][:,0,0]))

            
            
            pd_train = pd.DataFrame(data={'Signal':pd_signal, 
                                          'R2s': pd_train_data})
                                          
                                          
            pd_test = pd.DataFrame(data={'Signal':pd_signal, 
                                          'R2s': pd_test_data})         
                                          
            
            ax = plt.subplot(2,len(dec_list),dec_idx+1)
            b = sns.boxplot(x = 'Signal', y = 'R2s', data = pd_train, width = .5, palette = my_pal)
            ax.set_title(f"{dec_name}",fontsize=15)
            ax.set_ylabel('train R2s (cm)', labelpad = 5)
            
            
            ax = plt.subplot(2,len(dec_list),dec_idx+1+len(dec_list))
            b = sns.boxplot(x = 'Signal', y = 'R2s', data = pd_test, width = .5, palette = my_pal)
            ax.set_ylabel('test R2s (cm)', labelpad = 5)
      
        
        #plt.tight_layout()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
        plt.close(fig)
            
    with open(os.path.join(save_dir, f"{fnames[0][:5]}_decoders_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)
    
    return True
#%% PARAMS
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/traces/decoders'
params = {
    'x_base_signal': 'deconvProb',
    'y_signal_list': ['posx'],
    'verbose': True,
    'trial_signal': 'index_mat',
    'nn': 60,
    'n_splits': 10,
    'n_dims': 3,
    'emb_list': ['pca', 'isomap', 'umap']
    }
#%% M2019
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2019' in f]
M2019 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2019_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2019.keys())
M2019_R2s = dict()
M2019_pred = dict()

for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")

    pd_struct = copy.deepcopy(M2019[fname])
    M2019_R2s[fname], M2019_pred[fname] = dec.decoders_1D(pd_object = pd_struct, **params)
    M2019_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2019_dec_dict.pkl"), "wb")
    pickle.dump(M2019_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2019_dec_pred.pkl"), "wb")
    pickle.dump(M2019_pred, save_ks)
    save_ks.close()
    
_ = plot_decoders_study(M2019_R2s, save_dir)
#%% M2021
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2021' in f]
M2021 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2021_df_dict.pkl', verbose = True, struct_type = "pickle")


fname_list = list(M2021.keys())
M2021_R2s = dict()
M2021_pred = dict()

for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")

    pd_struct = copy.deepcopy(M2021[fname])
    M2021_R2s[fname], M2021_pred[fname] = dec.decoders_1D(pd_object = pd_struct, **params)
    M2021_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2021_dec_dict.pkl"), "wb")
    pickle.dump(M2021_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2021_dec_pred.pkl"), "wb")
    pickle.dump(M2021_pred, save_ks)
    save_ks.close()
    
    
_ = plot_decoders_study(M2021_R2s, save_dir)

#%% M2023
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2023' in f]
M2023 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2023_df_dict.pkl', verbose = True, struct_type = "pickle")


fname_list = list(M2023.keys())
M2023_R2s = dict()
M2023_pred = dict()

for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")

    pd_struct = copy.deepcopy(M2023[fname])
    M2023_R2s[fname], M2023_pred[fname] = dec.decoders_1D(pd_object = pd_struct, **params)
    M2023_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2023_dec_dict.pkl"), "wb")
    pickle.dump(M2023_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2023_dec_pred.pkl"), "wb")
    pickle.dump(M2023_pred, save_ks)
    save_ks.close()

_ = plot_decoders_study(M2023_R2s, save_dir)

#%% M2024
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2024' in f]
M2024 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2024_df_dict.pkl', verbose = True, struct_type = "pickle")


fname_list = list(M2024.keys())
M2024_R2s = dict()
M2024_pred = dict()

for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")

    pd_struct = copy.deepcopy(M2024[fname])
    M2024_R2s[fname], M2024_pred[fname] = dec.decoders_1D(pd_object = pd_struct, **params)
    M2024_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2024_dec_dict.pkl"), "wb")
    pickle.dump(M2024_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2024_dec_pred.pkl"), "wb")
    pickle.dump(M2024_pred, save_ks)
    save_ks.close()

_ = plot_decoders_study(M2024_R2s, save_dir)

#%% M2025
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2025' in f]
M2025 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2025_df_dict.pkl', verbose = True, struct_type = "pickle")


fname_list = list(M2025.keys())
M2025_R2s = dict()
M2025_pred = dict()

for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")

    pd_struct = copy.deepcopy(M2025[fname])
    M2025_R2s[fname], M2025_pred[fname] = dec.decoders_1D(pd_object = pd_struct, **params)
    M2025_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2025_dec_dict.pkl"), "wb")
    pickle.dump(M2025_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2025_dec_pred.pkl"), "wb")
    pickle.dump(M2025_pred, save_ks)
    save_ks.close()

_ = plot_decoders_study(M2025_R2s, save_dir)

#%% M2026
file_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
sub_dir = next(os.walk(file_dir))[1]
foi = [f for f in sub_dir if 'M2026' in f]
M2026 = gu.load_files(os.path.join(file_dir, foi[0]), '*M2026_df_dict.pkl', verbose = True, struct_type = "pickle")

fname_list = list(M2026.keys())
M2026_R2s = dict()
M2026_pred = dict()

for f_idx, fname in enumerate(fname_list):
    print(f"\nWorking on session: {fname} ({f_idx+1}/{len(fname_list)})")

    pd_struct = copy.deepcopy(M2026[fname])
    M2026_R2s[fname], M2026_pred[fname] = dec.decoders_1D(pd_object = pd_struct, **params)
    M2026_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2026_dec_dict.pkl"), "wb")
    pickle.dump(M2026_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2026_dec_pred.pkl"), "wb")
    pickle.dump(M2026_pred, save_ks)
    save_ks.close()

_ = plot_decoders_study(M2026_R2s, save_dir)

#%% LOAD DATA
if "M2019_R2s" not in locals():
    M2019_R2s = gu.load_files(save_dir, '*M2019_dec_dict.pkl', verbose=True, struct_type = "pickle")
if "M2021_R2s" not in locals():
    M2021_R2s = gu.load_files(save_dir, '*M2021_dec_dict.pkl', verbose=True, struct_type = "pickle")
if "M2023_R2s" not in locals():
    M2023_R2s = gu.load_files(save_dir, '*M2023_dec_dict.pkl', verbose=True, struct_type = "pickle")
if "M2024_R2s" not in locals():
    M2024_R2s = gu.load_files(save_dir, '*M2024_dec_dict.pkl', verbose=True, struct_type = "pickle")
if "M2025_R2s" not in locals():
    M2025_R2s = gu.load_files(save_dir, '*M2025_dec_dict.pkl', verbose=True, struct_type = "pickle")
if "M2026_R2s" not in locals():
    M2026_R2s = gu.load_files(save_dir, '*M2026_dec_dict.pkl', verbose=True, struct_type = "pickle")
#%%
def get_decoder_R2s(R2s_dict, session_list):
    dec_list = ['wf','wc','xgb','svr']

    tR2s = np.zeros((4,4,2,3,5))*np.nan
    fnames = list(R2s_dict.keys())
    
    last_idx = -1
    for s_idx, s_name in enumerate(fnames):
        
        if s_idx==0:
            last_idx+=1
            count_idx = 0
        else:
            old_s_name = fnames[s_idx-1]
            old_s_name = old_s_name[:old_s_name.find('_',-5)]
            new_s_name = s_name[:s_name.find('_',-5)]
            if new_s_name == old_s_name:
                count_idx += 1
            else:
                last_idx +=1
                count_idx = 0
                
        pd_struct = copy.deepcopy(R2s_dict[s_name])
        for dec_idx, dec_name in enumerate(dec_list):
            
            tR2s[0,dec_idx,:,count_idx, session_list[last_idx]] = np.nanmean(pd_struct['base_signal'][dec_name][:,0,:], axis= 0)
            tR2s[1,dec_idx,:,count_idx, session_list[last_idx]] = np.nanmean(pd_struct['pca'][dec_name][:,0,:], axis= 0)
            tR2s[2,dec_idx,:,count_idx, session_list[last_idx]] = np.nanmean(pd_struct['isomap'][dec_name][:,0,:], axis= 0)
            tR2s[3,dec_idx,:,count_idx, session_list[last_idx]] = np.nanmean(pd_struct['umap'][dec_name][:,0,:], axis= 0)
   
    return tR2s


#%%
R2s_val = np.zeros((4,4,2,3,5,6))*np.nan
R2s_val[:,:,:,:,:,0] = get_decoder_R2s(M2019_R2s, [0,1,2,4])
R2s_val[:,:,:,:,:,1] = get_decoder_R2s(M2021_R2s, [0,1,2,4])
R2s_val[:,:,:,:,:,2] = get_decoder_R2s(M2023_R2s, [0,1,2,4])
R2s_val[:,:,:,:,:,3] = get_decoder_R2s(M2024_R2s, [0,1,2,3])
R2s_val[:,:,:,:,:,4] = get_decoder_R2s(M2025_R2s, [0,1,2,3])
R2s_val[:,:,:,:,:,5] = get_decoder_R2s(M2026_R2s, [0,1,2,3])
#%%
plt.figure()
dec_list = ['wf','wc','xgb','svm']

color_list = ['grey', '#5bb95bff', '#ff8a17ff', '#249aefff']
signal_list = ['rates', 'pca', 'iso', 'umap']
space = [1,2,2.5,4,7]
for dec_idx, dec_name in enumerate(dec_list):
    ax = plt.subplot(2,4,dec_idx+1)

    for sig_idx, sig_name in enumerate(signal_list):
        m = np.nanmean(R2s_val[sig_idx, dec_idx, 1,:,:,:], axis = (-1,-3))
        sd = np.nanstd(R2s_val[sig_idx, dec_idx, 1,:,:,:], axis = (-1,-3))/np.sqrt(np.sum(np.invert(np.isnan(R2s_val[sig_idx, dec_idx, 1,:,:,:])), axis = (-1,-3)))
        
        ax.plot(space,m, c = color_list[sig_idx], label = sig_name)
        ax.fill_between(space, m-sd, m+sd, color = color_list[sig_idx], alpha = 0.3)
        
    ax.set_title(dec_name)
    ax.set_xlabel('session', labelpad = -2)
    #ax.set_ylim([R2s_vmin, R2s_vmax[ii,0]])
    #ax.set_yticks([R2s_vmin, R2s_vmax[0,0]/2, R2s_vmax[0,0]])
    ax.set_ylabel('Train Position error (cm)', labelpad = 0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
   
    ax = plt.subplot(2,4,dec_idx+1+4)

    for sig_idx, sig_name in enumerate(signal_list):
        m = np.nanmean(R2s_val[sig_idx, dec_idx, 0,:,:], axis = (-1,-3))
        sd = np.nanstd(R2s_val[sig_idx, dec_idx, 0,:,:], axis = (-1,-3))/np.sqrt(np.sum(np.invert(np.isnan(R2s_val[sig_idx, dec_idx, 0,:,:])), axis = (-1,-3)))
        
        ax.plot(space,m, c = color_list[sig_idx], label = sig_name)
        ax.fill_between(space, m-sd, m+sd, color = color_list[sig_idx], alpha = 0.3)
        
    ax.set_xlabel('session', labelpad = -2)
    #ax.set_ylim([R2s_vmin, R2s_vmax[ii,0]])
    #ax.set_yticks([R2s_vmin, R2s_vmax[0,0]/2, R2s_vmax[0,0]])
    ax.set_ylabel('Test Position error (cm)', labelpad = 0)
    ax.set_xticks(space, label = space)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)           
    
    if dec_idx == 4:
        ax.legend()
plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.05)

#%%
save_fig = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/poster_figures'
plt.savefig(os.path.join(save_fig,'poster_decoders.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_fig,'poster_decoders.svg'), dpi = 400,bbox_inches="tight",transparent=True)    


