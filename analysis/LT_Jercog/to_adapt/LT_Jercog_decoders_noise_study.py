#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:55:15 2022

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
def plot_decoders_noise_study(dec_dict, save_dir):
    
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
    
    my_pal = ['grey','#5bb95bff', '#ff8a17ff','#249aefff']
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
            
            
            
            pd_train_data = np.zeros((pd_struct[signals[0]][dec_name].shape[0], pd_struct[signals[0]][dec_name].shape[1], len(signals)))
            pd_test_data = np.zeros((pd_struct[signals[0]][dec_name].shape[0], pd_struct[signals[0]][dec_name].shape[1], len(signals)))

            for s_idx, s in enumerate(signals):
                pd_train_data[:,:,s_idx] = pd_struct[s][dec_name][:,:,0,1]
                pd_test_data[:,:,s_idx] = pd_struct[s][dec_name][:,:,0,0]

            
            ax = plt.subplot(2,len(dec_list),dec_idx+1)
            space = pd_struct['params']['noise_list']
            m = np.nanmean(pd_train_data, axis = 0)
            sd = np.nanstd(pd_train_data, axis = 0)
            for sig_idx, sig_name in enumerate(signals):
                ax.plot(space,m[:,sig_idx], c = my_pal[sig_idx], label = sig_name)
                ax.fill_between(space, m[:,sig_idx]-sd[:,sig_idx], m[:,sig_idx]+sd[:,sig_idx], color = my_pal[sig_idx], alpha = 0.3)
                
            ax.set_ylabel('train R2s (cm)', labelpad = 5)
            ax.set_xticks(space, labels = space, rotation =90);
            ax.set_xlabel('noise sd')
            
            ax = plt.subplot(2,len(dec_list),dec_idx+1+len(dec_list))
            m = np.nanmean(pd_test_data, axis = 0)
            sd = np.nanstd(pd_test_data, axis = 0)
            for sig_idx, sig_name in enumerate(signals):
                ax.plot(space,m[:,sig_idx], c = my_pal[sig_idx], label = t_signals[sig_idx])
                ax.fill_between(space, m[:,sig_idx]-sd[:,sig_idx], m[:,sig_idx]+sd[:,sig_idx], color = my_pal[sig_idx], alpha = 0.3)
            ax.set_ylabel('test R2s (cm)', labelpad = 5)
            ax.set_xticks(space, labels = space, rotation =90);
            ax.set_xlabel('noise sd')
            
            if dec_idx == len(dec_list)-1:
                ax.legend()
                
        #plt.tight_layout()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
        plt.close(fig)
            
    with open(os.path.join(save_dir, f"{fnames[0][:5]}_decoders_noise_{datetime.now().strftime('%d%m%y_%H%M%S')}.html"),'w') as f:
        f.write(html)
    
    return True
#%% PARAMS
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/spikes/decoders/noise'
params = {
    'x_base_signal': 'ML_rates',
    'y_signal_list': ['posx'],
    'verbose': True,
    'trial_signal': 'index_mat',
    'nn': 60,
    'n_splits': 10,
    'n_dims': 3,
    'noise_list': [0,0.1,0.2,1,2,5],
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
    M2019_R2s[fname], M2019_pred[fname] = dec.decoders_noise_1D(pd_object = pd_struct, **params)
    M2019_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2019_dec_noise_dict.pkl"), "wb")
    pickle.dump(M2019_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2019_dec_noise_pred.pkl"), "wb")
    pickle.dump(M2019_pred, save_ks)
    save_ks.close()
    
_ = plot_decoders_noise_study(M2019_R2s, save_dir)

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
    M2021_R2s[fname], M2021_pred[fname] = dec.decoders_noise_1D(pd_object = pd_struct, **params)
    M2021_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2021_dec_noise_dict.pkl"), "wb")
    pickle.dump(M2021_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2021_dec_noise_pred.pkl"), "wb")
    pickle.dump(M2021_pred, save_ks)
    save_ks.close()
    
_ = plot_decoders_noise_study(M2021_R2s, save_dir)

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
    M2023_R2s[fname], M2023_pred[fname] = dec.decoders_noise_1D(pd_object = pd_struct, **params)
    M2023_R2s[fname]['params'] = params


    save_ks = open(os.path.join(save_dir, "M2023_dec_noise_dict.pkl"), "wb")
    pickle.dump(M2023_R2s, save_ks)
    save_ks.close()
    
    save_ks = open(os.path.join(save_dir, "M2023_dec_noise_pred.pkl"), "wb")
    pickle.dump(M2023_pred, save_ks)
    save_ks.close()
    
_ = plot_decoders_noise_study(M2023_R2s, save_dir)
