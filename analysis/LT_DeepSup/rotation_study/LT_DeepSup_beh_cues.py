import sys, os, copy, pickle, timeit
from os import listdir


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neural_manifold.general_utils as gu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import seaborn as sns
import umap
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from datetime import datetime


def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))



mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
sup_mice_list = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deep_mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']

base_dir = '/home/julio/Documents/SP_project/LT_DeepSup/data/'

columns_to_drop = ['date','denoised_traces', '*spikes']
columns_to_rename = {'Fs':'sf','pos':'position', 'vel':'speed', 'index_mat': 'trial_idx_mat'}


#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#

in_cue_speed_list = [];
out_cue_speed_list = [];
mouse_name_list = [];
mouse_layer_list = [];
session_case_list = [];

for mouse in mice_list:

    print(f"Working on mouse: {mouse}")
    load_dir = os.path.join(base_dir, mouse)
    animal = gu.load_files(load_dir, '*_PyalData_struct.mat', struct_type = "PyalData")
    session_names = list(animal.keys())
    cues_files = [f for f in os.listdir(load_dir) if 'cues_info.csv' in f]
    for case in ['lt','rot']:
        fname = [fname for fname in session_names if case in fname][0]

        session = copy.deepcopy(animal[fname])

        for column in columns_to_drop:
            if column in session .columns: session .drop(columns=[column], inplace=True)
        for old, new in columns_to_rename.items():
            if old in session .columns: session .rename(columns={old:new}, inplace=True)

        gu.add_trial_id_mat_field(session)
        gu.add_mov_direction_mat_field(session)
        gu.add_trial_type_mat_field(session)

        position = get_signal(session , 'position')
        min_pos = np.percentile(position[:,0], 1)
        max_pos = np.percentile(position[:,0], 99)

        speed = get_signal(session , 'speed')
        
        cues_file = [fname for fname in cues_files if case in fname][0]
        cues_info = pd.read_csv(os.path.join(load_dir, cues_file))

        #no distinction between direction of movement
        st_cue = cues_info['x_start_cm'][0]
        en_cue = cues_info['x_end_cm'][0]

        st_fake_cue = max_pos - en_cue + min_pos;
        en_fake_cue = max_pos - st_cue + min_pos;

        in_cue = np.where(np.logical_and(position[:,0]>st_cue, position[:,0]<en_cue))
        out_cue = np.where(np.logical_and(position[:,0]>st_fake_cue, position[:,0]<en_fake_cue))

        in_cue_speed_list.append(np.mean(speed[in_cue]))
        out_cue_speed_list.append(np.mean(speed[out_cue]))
        if mouse in deep_mice_list:
            mouse_layer_list.append('deep')
        elif mouse in sup_mice_list:
            mouse_layer_list.append('sup')

        mouse_name_list.append(mouse)
        session_case_list.append(case)

pd_cue_beh = pd.DataFrame(data={'mouse': mouse_name_list,
                     'case': session_case_list,
                     'in_cue_speed': in_cue_speed_list,
                     'out_cue_speed': out_cue_speed_list,
                     'layer': mouse_layer_list})    

pd_cue_beh_lt = pd_cue_beh[pd_cue_beh['case']=='lt']
pd_cue_beh_lt['angle'] = [161,158,154,150,129,0,0,0,9,np.nan,12,23,1,12]
pd_cue_beh_lt['cue_ratio'] = pd_cue_beh_lt['in_cue_speed']/pd_cue_beh_lt['out_cue_speed']
sns.scatterplot(data=pd_cue_beh_lt, x='cue_ratio', y='angle', hue='layer')