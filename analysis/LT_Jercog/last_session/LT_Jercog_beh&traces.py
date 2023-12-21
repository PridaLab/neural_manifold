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
data_dir = '/home/julio/Documents/SP_project/LT_Jercog/data/'

save_dir = '/home/julio/Documents/SP_project/jercog_learning/behaviour/'

num_correct = np.zeros((4, len(mice_list)))
num_fail = np.zeros((4, len(mice_list)))
for midx, mouse in enumerate(mice_list):
    print('')
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = gu.load_files(file_path,'*_LT_PyalData_struct.mat',verbose=True,struct_type="PyalData")
    fnames = list(mouse_dict.keys())
    fnames.sort()
    for fidx, fname in enumerate(fnames):
        pd_struct= copy.deepcopy(mouse_dict[fname])
        correct_trials = gu.select_trials(pd_struct,"dir == ['L','R']")
        fail_trials = gu.select_trials(pd_struct,"dir != ['L','R','N']")

        num_correct[fidx, midx] = correct_trials.shape[0]
        num_fail[fidx, midx] = fail_trials.shape[0]

x = np.tile(np.array([1,2,2.5,4,7]).reshape(-1,1), (1,len(mice_list)))
perf_idx = 1 - (num_fail/(num_correct+num_fail))

fperf_idx = np.zeros((5, 5))*np.nan
fperf_idx[[0,1,2,4],:2] = perf_idx[:, :2]
fperf_idx[:4,2:] = perf_idx[:, 2:]

plt.figure()
ax = plt.subplot(111)
ax.scatter(x.reshape(-1,1), fperf_idx.reshape(-1,1))
for idx in range(2):
    plt.plot(x[[0,1,2,4],idx], fperf_idx[[0,1,2,4],idx], color=[.5,.5,.5])

for idx in range(2,5):
    plt.plot(x[[0,1,2,3],idx], fperf_idx[[0,1,2,3],idx], color=[.5,.5,.5])    

m = np.nanmean(fperf_idx, axis=1)
sd = np.nanstd(fperf_idx, axis=1)
ax.plot(x[:,0], m, color='b')
ax.fill_between(x[:,0], m-sd, m+sd, alpha = 0.3, color='b')

ax.set_ylim([0.425, 1.025])
ax.set_xlabel('Day')
ax.set_ylabel('Performance index')
plt.savefig(os.path.join(save_dir,'behavior_curve.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'behavior_curve.svg'), dpi = 400,bbox_inches="tight",transparent=True)


mouse = 'M2021'
file_path = os.path.join(data_dir, mouse)
mouse_dict = gu.load_files(file_path,'*_LT_PyalData_struct.mat',verbose=True,struct_type="PyalData")
fnames = list(mouse_dict.keys())
fnames.sort()

nl_num_correct = list()
nl_num_fail = list()
for fidx, fname in enumerate(fnames):
    pd_struct= copy.deepcopy(mouse_dict[fname])
    correct_trials = gu.select_trials(pd_struct,"dir == ['L','R']")
    fail_trials = gu.select_trials(pd_struct,"dir != ['L','R','N']")
    nl_num_correct.append(correct_trials.shape[0])
    nl_num_fail.append(fail_trials.shape[0])


nl_num_correct = np.array(nl_num_correct)
nl_num_fail = np.array(nl_num_fail)
nl_perf_idx = 1 - (nl_num_fail/(nl_num_correct+nl_num_fail))

nl_x = [1,2,2.5,7]

plt.figure()
ax = plt.subplot(111)
ax.scatter(x.reshape(-1,1), fperf_idx.reshape(-1,1))
for idx in range(2):
    ax.plot(x[[0,1,2,4],idx], fperf_idx[[0,1,2,4],idx], color=[.5,.5,.5])

for idx in range(2,5):
    ax.plot(x[[0,1,2,3],idx], fperf_idx[[0,1,2,3],idx], color=[.5,.5,.5])    

m = np.nanmean(fperf_idx, axis=1)
sd = np.nanstd(fperf_idx, axis=1)
ax.plot(x[:,0], m, color='b')
ax.fill_between(x[:,0], m-sd, m+sd, alpha = 0.3, color='b')

ax.plot(nl_x, nl_perf_idx, color = 'r')
ax.set_ylim([0.425, 1.025])
ax.set_xlabel('Day')
ax.set_ylabel('Performance index')


#__________________________________________________________________________
#|                                                                        |#
#|                               PLOT TRACES                              |#
#|________________________________________________________________________|#

from scipy.ndimage import gaussian_filter1d
data_dir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
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
#|                            PLOT ALL TRACES                             |#
#|________________________________________________________________________|#

from scipy.ndimage import gaussian_filter1d
data_dir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
mouse = 'M2019'
file_name =  mouse+'_df_dict.pkl'
file_path = os.path.join(data_dir, mouse)
pd_struct = gu.load_files(file_path,'*'+file_name,verbose=True,struct_type="pickle")


# rawProb = copy.deepcopy(np.concatenate(pd_struct['rawProb'].values, axis=0))
cleanProb = copy.deepcopy(np.concatenate(pd_struct['clean_traces'].values, axis=0))
vel = copy.deepcopy(np.concatenate(pd_struct['vel'].values, axis=0))
pos = copy.deepcopy(np.concatenate(pd_struct['pos'].values, axis=0))

dur = 300*20
idx_st = 60*20
idx_en = idx_st+dur
time = np.arange(dur)/20
plt.figure(figsize=(5,10))
ax = plt.subplot(111)
ax.plot(time, 0.1*vel[idx_st:idx_en])
for n in range(150):
    ax.plot(time, cleanProb[idx_st:idx_en,n]-n-1)
plt.savefig(os.path.join(save_dir,'traces_cleanProb.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'traces_cleanProb.svg'), dpi = 400,bbox_inches="tight",transparent=True)

dur = 300*20
idx_st = 60*20
idx_en = idx_st+dur
time = np.arange(dur)/20
plt.figure(figsize=(5,10))
ax = plt.subplot(111)
ax.plot(time, 0.1*pos[idx_st:idx_en,0])
ax.plot(time, 0.1*vel[idx_st:idx_en])
for n in range(150):
    ax.plot(time, cleanProb[idx_st:idx_en,n]-n-1)
plt.savefig(os.path.join(save_dir,'traces_cleanProb.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'traces_cleanProb.svg'), dpi = 400,bbox_inches="tight",transparent=True)


#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT ALL TRACES                             |#
#|________________________________________________________________________|#

from scipy.ndimage import gaussian_filter1d
data_dir = '/home/julio/Documents/SP_project/LT_Jercog/data/'
mouse = 'M2019'
file_path = os.path.join(data_dir, mouse)
mouse_dict = gu.load_files(file_path,'*_LT_PyalData_struct.mat',verbose=True,struct_type="PyalData")
fileNames = list(mouse_dict.keys())
sessionName = fileNames[0]
for fileName in fileNames[1:]:
    if int(fileName[6:14])>int(sessionName[6:14]):
        sessionName = fileName
pd_struct = mouse_dict[sessionName]
# rawProb = copy.deepcopy(np.concatenate(pd_struct['rawProb'].values, axis=0))
cleanProb = copy.deepcopy(np.concatenate(pd_struct['rawProb'].values, axis=0))
vel = copy.deepcopy(np.concatenate(pd_struct['vel'].values, axis=0))
pos = copy.deepcopy(np.concatenate(pd_struct['pos'].values, axis=0))


dur = 300*20
idx_st = 60*20
idx_en = idx_st+dur
time = np.arange(dur)/20

plt.figure(figsize=(5,10))
ax = plt.subplot(111)
ax.plot(time, 0.1*pos[idx_st:idx_en,0])
ax.plot(time, 0.1*vel[idx_st:idx_en])
for n in range(150):
    ax.plot(time, cleanProb[idx_st:idx_en,n]-5*n-10)
plt.savefig(os.path.join(save_dir,'traces_rawProb.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'traces_rawProb.svg'), dpi = 400,bbox_inches="tight",transparent=True)



#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT NUM CELLS                             |#
#|________________________________________________________________________|#
mice_list = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']

data_dir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
num_cells = list()
for mouse in mice_list:
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_struct = gu.load_files(file_path,'*'+file_name,verbose=True,struct_type="pickle")
    num_cells.append(pd_struct['clean_traces'][0].shape[1])

plt.figure()
ax = plt.subplot(111)
ax.bar(mice_list, num_cells)
plt.savefig(os.path.join(save_dir,'num_cells_last_day.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'num_cells_last_day.svg'), dpi = 400,bbox_inches="tight",transparent=True)

#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT NUM CELLS                             |#
#|________________________________________________________________________|#
miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
data_dir = '/home/julio/Documents/SP_project/LT_Jercog/data/'
save_dir = '/home/julio/Documents/SP_project/Fig1/behaviour/'

numCellsList = list()
mouseList = list()
sessionList = list()

for mouse in miceList:
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = gu.load_files(file_path,'*_LT_PyalData_struct.mat',verbose=True,struct_type="PyalData")
    fileNames = list(mouse_dict.keys())
    sessionNum = list()
    for fileName in fileNames:
        sessionNum.append(int(fileName[6:14]+fileName[15:21]))
    sessionOrder = np.argsort(sessionNum)

    for fidx, fileName in enumerate(fileNames):
        print(f"Working on session: {fileName} ({fidx+1}/{len(fileNames)}: {np.where(sessionOrder==fidx)[0][0]})")
        pd_struct= copy.deepcopy(mouse_dict[fileName])
        numCellsList.append(pd_struct['rawProb'][0].shape[1])
        mouseList.append(mouse)
        sessionList.append(np.where(sessionOrder==fidx)[0][0])

pdCells = pd.DataFrame(data={'mouse': mouseList,
                     'numCells': numCellsList,
                     'session': sessionList})    


fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='mouse', y='numCells', data=pdCells, hue=sessionList,
            linewidth = 1, width= .5, ax = ax)

plt.savefig(os.path.join(save_dir,'num_cells_all.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'num_cells_all.svg'), dpi = 400,bbox_inches="tight",transparent=True)

#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT NUM CELLS                             |#
#|________________________________________________________________________|#
mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
data_dir = '/home/julio/Documents/SP_project/LT_Jercog/data/'
save_dir = '/home/julio/Documents/SP_project/Fig1/behaviour/'

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
data_dir = '/home/julio/Documents/SP_project/LT_Jercog/data/'

save_dir = '/home/julio/Documents/SP_project/Fig1/behaviour/'

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