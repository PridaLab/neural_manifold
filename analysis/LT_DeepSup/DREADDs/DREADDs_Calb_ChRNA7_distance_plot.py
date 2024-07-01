import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neural_manifold.general_utils as gu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import seaborn as sns


def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

palette= ['#666666ff', '#aa0007ff']

#CALB
mice_list = ['CalbCharly2', 'CalbCharly11_concat', 'CalbV23', 'DD2']
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'

case_list = list()
mouse_list = list()
remap_dist_list = list()

for mouse in mice_list:
    for case in ['veh','CNO']:
        data_dir = os.path.join(base_dir,'distance', mouse+'_'+case)
        rot_error_dict = load_pickle(data_dir, f"{mouse}_{case}_distance_dict.pkl")
        case_list.append(case)
        mouse_list.append(mouse)
        remap_dist_list.append(rot_error_dict['umap']['remap_dist'])

calb_pd = pd.DataFrame(data={'remap_dist': remap_dist_list,
                            'case': case_list,
                            'mouse': mouse_list})
calb_pd["strain"] = "calb"
#ChRNA7
mice_list = ['ChRNA7Charly1', 'ChRNA7Charly2']
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/ChRNA7'

case_list = list()
mouse_list = list()
remap_dist_list = list()

for mouse in mice_list:
    for case in ['veh','CNO']:
        data_dir = os.path.join(base_dir,'distance', mouse+'_'+case)
        rot_error_dict = load_pickle(data_dir, f"{mouse}_{case}_distance_dict.pkl")
        case_list.append(case)
        mouse_list.append(mouse)
        remap_dist_list.append(rot_error_dict['umap']['remap_dist'])

chrna7_pd = pd.DataFrame(data={'remap_dist': remap_dist_list,
                            'case': case_list,
                            'mouse': mouse_list})
chrna7_pd["strain"] = "chrna7"


#join 

distances_pd = pd.concat([calb_pd, chrna7_pd], ignore_index=True)
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='case', y='remap_dist',data=distances_pd, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='case', y='remap_dist', data=distances_pd, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='case', y= 'remap_dist', data=distances_pd, units = 'mouse', ax = ax, estimator = None, color = ".7", markers = True)
ax.set_ylabel('Remap distance')
plt.tight_layout()



from scipy import stats
veh_dist = distances_pd.loc[distances_pd['case']=='veh']['remap_dist']
CNO_dist = distances_pd.loc[distances_pd['case']=='CNO']['remap_dist']

veh_dist_norm = stats.shapiro(veh_dist)
CNO_dist_norm = stats.shapiro(CNO_dist)


if veh_dist_norm.pvalue<=0.05 or CNO_dist_norm.pvalue<=0.05:
    print('veh_dist vs CNO_dist:',stats.ks_2samp(veh_dist, CNO_dist))
else:
    print('veh_dist vs CNO_dist:', stats.ttest_rel(veh_dist, CNO_dist))

