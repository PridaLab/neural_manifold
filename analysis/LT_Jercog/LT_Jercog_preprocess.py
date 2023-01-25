import umap
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import neural_manifold.general_utils as gu
import neural_manifold.general_utils as gu
import copy
import matplotlib.pyplot as plt
import os, pickle
import seaborn as sns
import pandas as pd

#__________________________________________________________________________
#|                                                                        |#
#|                       MAKE SESSIONS SAME LENGTH                        |#
#|________________________________________________________________________|#
mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
#%% IMPORTS
from neural_manifold.pipelines.LT_Jercog_session_length import LT_session_length
#%% PARAMS
params = {
    'kernel_std': 0.3,
    'vel_th': 3,
    'kernel_num_std': 5,
    'min_session_len': 6000, #5 min
    'equalize_session_len': True
    }
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
og_data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/'
for mouse in mice_list:
    LT_session_length(os.path.join(og_data_dir,mouse), mouse, data_dir, **params)

#__________________________________________________________________________
#|                                                                        |#
#|                               CLEAN TRACES                             |#
#|________________________________________________________________________|#

def add_dir_mat_field(pd_struct):
    out_pd = copy.deepcopy(pd_struct)
    if 'dir_mat' not in out_pd.columns:
        out_pd["dir_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            ('L' == out_pd["dir"][idx])+ 2*('R' == out_pd["dir"][idx])
                            for idx in out_pd.index]
    return out_pd

def preprocess_traces(pd_struct, signal_field, sigma = 5, MAD_th = 5):

    out_pd = copy.deepcopy(pd_struct)

    out_pd["index_mat"] = [np.zeros((out_pd[signal_field][idx].shape[0],1))+out_pd["trial_id"][idx] 
                                  for idx in range(out_pd.shape[0])]                     
    index_mat = np.concatenate(out_pd["index_mat"].values, axis=0)

    signal_og = copy.deepcopy(np.concatenate(pd_struct[signal_field].values, axis=0))
    lowpass_signal = uniform_filter1d(signal_og, size = 4000, axis = 0)
    signal = gaussian_filter1d(signal_og, sigma = sigma, axis = 0)


    for nn in range(signal.shape[1]):
        base_signal = np.histogram(signal_og[:,nn], 100)
        base_signal = base_signal[1][np.argmax(base_signal[0])]
        base_signal = base_signal + lowpass_signal[:,nn] - np.min(lowpass_signal[:,nn]) 

        clean_signal = signal[:,nn]-base_signal
        clean_signal = clean_signal/np.max(clean_signal,axis = 0)
        clean_signal[clean_signal<0] = 0
        MAD = np.median(abs(clean_signal - np.median(clean_signal)), axis=0)
        clean_signal[clean_signal<MAD_th*MAD] = 0

        signal[:,nn] = clean_signal

    out_pd['clean_traces'] = [signal[index_mat[:,0]==out_pd["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd.shape[0])]

    return out_pd

data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'
params = {
    'sigma': 5,
    'MAD_th': 5
}

for mouse in mice_list:
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = gu.load_files(file_path,'*'+file_name,verbose=True,struct_type="pickle")
    fnames = list(mouse_dict.keys())
    for fname in fnames:
        pd_struct= copy.deepcopy(mouse_dict[fname])
        pd_struct = add_dir_mat_field(pd_struct)
        pd_struct = preprocess_traces(pd_struct, signal_field, **params)
        mouse_dict[fname] = pd_struct

    with open(os.path.join(file_path,file_name), 'wb') as f:
        pickle.dump(mouse_dict, f)
