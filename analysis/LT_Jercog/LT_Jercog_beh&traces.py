import umap
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import neural_manifold.general_utils as gu
import copy
import matplotlib.pyplot as plt
import os, pickle
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks

#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT BEHAVIOUR                             |#
#|________________________________________________________________________|#
mice_list = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/'

save_dir = '/media/julio/DATOS/spatial_navigation/paper/Fig1/behaviour/'

num_correct = np.zeros((4, len(mice_list)))
num_fail = np.zeros((4, len(mice_list)))
for midx, mouse in enumerate(mice_list):
    print('')
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = gu.load_files(file_path,'*_LT_PyalData_struct.mat',verbose=True,struct_type="PyalData")
    fnames = list(mouse_dict.keys())
    for fidx, fname in enumerate(fnames):
        pd_struct= copy.deepcopy(mouse_dict[fname])
        correct_trials = gu.select_trials(pd_struct,"dir == ['L','R']")
        fail_trials = gu.select_trials(pd_struct,"dir != ['L','R','N', 'FSR','FSL']")

        num_correct[fidx, midx] = correct_trials.shape[0]
        num_fail[fidx, midx] = fail_trials.shape[0]

x = np.tile(np.array([1,2,2.5,4,7]).reshape(-1,1), (1,len(mice_list)))
perf_idx = 1 - (num_fail/(num_correct+num_fail))

fperf_idx = np.zeros((5, 5))*np.nan
fperf_idx[:4,:2] = perf_idx[:, :2]
fperf_idx[[0,1,2,4],2:] = perf_idx[:, 2:]

plt.figure()
ax = plt.subplot(111)
ax.scatter(x.reshape(-1,1), fperf_idx.reshape(-1,1))
for idx in range(2):
    plt.plot(x[:,idx], fperf_idx[:,idx], color=[.5,.5,.5])

for idx in range(2,5):
    plt.plot(x[[0,1,2,4],idx], fperf_idx[[0,1,2,4],idx], color=[.5,.5,.5])    

m = np.nanmean(fperf_idx, axis=1)
sd = np.nanstd(fperf_idx, axis=1)
ax.plot(x[:,0], m, color='b')
ax.fill_between(x[:,0], m-sd, m+sd, alpha = 0.3, color='b')

ax.set_ylim([0.5, 1.025])
ax.set_xlabel('Day')
ax.set_ylabel('Performance index')
plt.savefig(os.path.join(save_dir,'behavior_curve.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'behavior_curve.svg'), dpi = 400,bbox_inches="tight",transparent=True)

#__________________________________________________________________________
#|                                                                        |#
#|                               PLOT TRACES                              |#
#|________________________________________________________________________|#

from scipy.ndimage import gaussian_filter1d
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/processed_data/'
mouse = 'M2019'
file_name =  mouse+'_df_dict.pkl'
file_path = os.path.join(data_dir, mouse)
mouse_dict = gu.load_files(file_path,'*'+file_name,verbose=True,struct_type="pickle")
fnames = list(mouse_dict.keys())
pd_struct = mouse_dict[fnames[1]]

rawProb = copy.deepcopy(np.concatenate(pd_struct['rawProb'].values, axis=0))
cleanProb = copy.deepcopy(np.concatenate(pd_struct['clean_traces'].values, axis=0))


neu = [6,2,3,4,5,1,8,9,10,13,15,16,17,18,19,20]
dur = 60*20
idx_st = 0
idx_en = idx_st+dur
time = np.arange(dur)/20
plt.figure()
ax = plt.subplot(111)
for n in range(len(neu)):
    ax.plot(time, rawProb[idx_st:idx_en,neu[n]]-n)
ax.set_title(fnames[1]+ ' rawProb')
plt.savefig(os.path.join(save_dir,'traces_rawProb.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'traces_rawProb.svg'), dpi = 400,bbox_inches="tight",transparent=True)
dur = 60*20
idx_st = 0
idx_en = idx_st+dur
time = np.arange(dur)/20
plt.figure()
ax = plt.subplot(111)
for n in range(len(neu)):
    ax.plot(time, cleanProb[idx_st:idx_en,neu[n]]-n)
plt.savefig(os.path.join(save_dir,'traces_cleanProb.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'traces_cleanProb.svg'), dpi = 400,bbox_inches="tight",transparent=True)
ax.set_title(fnames[1]+ ' clean_traces')

#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT NUM CELLS                             |#
#|________________________________________________________________________|#
mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/'
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/behavior/'

num_cells = np.zeros((4, len(mice_list)))
for midx, mouse in enumerate(mice_list):
    print('')
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = gu.load_files(file_path,'*_LT_PyalData_struct.mat',verbose=True,struct_type="PyalData")
    fnames = list(mouse_dict.keys())
    for fidx, fname in enumerate(fnames):
        print(f"Working on session: {fname} ({f_idx+1}/{len(fnames)})")
        pd_struct= copy.deepcopy(mouse_dict[fname])
        num_cells[fidx, midx] = pd_struct['rawProb'][0].shape[1]


x = np.tile(np.array([1,2,2.5,4,7]).reshape(-1,1), (1,len(mice_list)))
fnum_cells = np.zeros((5, 6))*np.nan
fnum_cells[:4,:3] = num_cells[:, :3]
fnum_cells[[0,1,2,4],3:] = num_cells[:, 3:]


plt.figure()
ax = plt.subplot(111)

m = np.nanmean(fnum_cells, axis=1)
sd = np.nanstd(fnum_cells, axis=1)
ax.bar(x[:,0], m, yerr=sd, width = 0.4, color='gray')
ax.scatter(x.reshape(-1,1), fnum_cells.reshape(-1,1))
for idx in range(3):
    plt.plot(x[:,idx], fnum_cells[:,idx], color=[.5,.5,.5])

for idx in range(3,6):
    plt.plot(x[[0,1,2,4],idx], fnum_cells[[0,1,2,4],idx], color=[.5,.5,.5])    

plt.savefig(os.path.join(save_dir,'num_cells.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'num_cells.svg'), dpi = 400,bbox_inches="tight",transparent=True)


mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/'

save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/behavior/'

vel_trial = np.zeros((4, len(mice_list)))
for midx, mouse in enumerate(mice_list):
    print('')
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = gu.load_files(file_path,'*_LT_PyalData_struct.mat',verbose=True,struct_type="PyalData")
    fnames = list(mouse_dict.keys())
    for fidx, fname in enumerate(fnames):
        pd_struct= copy.deepcopy(mouse_dict[fname])
        correct_trials = gu.select_trials(pd_struct,"dir == ['L','R']")
        vel = np.concatenate(correct_trials['vel'].values, axis = 0)

        vel_trial[fidx, midx] = np.max(vel)

x = np.tile(np.array([1,2,2.5,4,7]).reshape(-1,1), (1,len(mice_list)))
fvel_trial = np.zeros((5, 6))*np.nan
fvel_trial[:4,:3] = vel_trial[:, :3]
fvel_trial[[0,1,2,4],3:] = vel_trial[:, 3:]

plt.figure()
ax = plt.subplot(111)
ax.scatter(x.reshape(-1,1), fvel_trial.reshape(-1,1))
for idx in range(3):
    plt.plot(x[:,idx], fvel_trial[:,idx], color=[.5,.5,.5])

for idx in range(3,6):
    plt.plot(x[[0,1,2,4],idx], fvel_trial[[0,1,2,4],idx], color=[.5,.5,.5])    

m = np.nanmean(fvel_trial, axis=1)
sd = np.nanstd(fvel_trial, axis=1)
ax.plot(x[:,0], m, color='b')
ax.fill_between(x[:,0], m-sd, m+sd, alpha = 0.3, color='b')