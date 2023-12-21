import seaborn as sns
import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import neural_manifold.general_utils as gu
from sklearn.metrics import pairwise_distances
import networkx as nx
from structure_index import compute_structure_index, draw_graph
from datetime import datetime
from scipy import stats

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data


def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx


control_mice_list = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
nl_mice_list = ['M2021','M2022']


#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT BEHAVIOUR                             |#
#|________________________________________________________________________|#

mice_list = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
data_dir = '/home/julio/Documents/SP_project/LT_Jercog/data/'

save_dir = '/home/julio/Documents/SP_project/jercog_learning/no_learning_mice/behaviour/'

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

mouse = 'M2022'
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
nl_perf_idx2 = 1 - (nl_num_fail/(nl_num_correct+nl_num_fail))
nl_x = [1,2,2.5,7]


plt.figure()
ax = plt.subplot(111)
m = np.nanmean(fperf_idx, axis=1)
sd = np.nanstd(fperf_idx, axis=1)
ax.plot(x[:,0], m, color='k', linestyle = '--')
ax.fill_between(x[:,0], m-sd, m+sd, alpha = 0.3, color='k')
ax.plot(nl_x, nl_perf_idx, color = 'r')
ax.plot(nl_x, nl_perf_idx2, color = 'r')
ax.set_ylim([0.425, 1.025])
ax.set_xlabel('Day')
ax.set_ylabel('Performance index')
plt.savefig(os.path.join(save_dir,f'behaviour_nl.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'behaviour_nl.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)

#__________________________________________________________________________
#|                                                                        |#
#|                                PLOT DIM                                |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/jercog_learning/dimensionality/inner_dim/'
inner_dim = load_pickle(data_dir, 'inner_dim_dict.pkl')

data_dir = '/home/julio/Documents/SP_project/jercog_learning/dimensionality/'
iso_dim = load_pickle(data_dir, 'isomap_dim_dict.pkl')
pca_dim = load_pickle(data_dir, 'pca_dim_dict.pkl')
umap_dim = load_pickle(data_dir, 'umap_dim_dict.pkl')

dim_list = list()
mouse_list = list()
session_list = list()
real_session_list = list()
for mouse in control_mice_list:
    fileNames = list(inner_dim[mouse].keys())
    og_date = datetime.strptime(fileNames[0][:8], "%Y%m%d")
    for idx, fileName in enumerate(fileNames):
        dim_list.append(inner_dim[mouse][fileName]['abidsDim'])
        dim_list.append(inner_dim[mouse][fileName]['tleDim'])
        dim_list.append(inner_dim[mouse][fileName]['momDim'])
        dim_list.append(umap_dim[mouse][fileName]['trustDim'])
        dim_list.append(umap_dim[mouse][fileName]['contDim'])

        dim_list.append(iso_dim[mouse][fileName]['resVarDim'])
        dim_list.append(iso_dim[mouse][fileName]['recErrorDim'])

        dim_list.append(pca_dim[mouse][fileName]['var80Dim'])
        dim_list.append(pca_dim[mouse][fileName]['kneeDim'])

        mouse_list = mouse_list + [mouse]*9
        session_list = session_list + [idx]*9

        new_date = datetime.strptime(fileName[:8], "%Y%m%d")
        days_diff = (new_date-og_date).days
        if idx==2:
            days_diff += 0.5
        real_session_list = real_session_list + [days_diff]*9

method_list = ['abids', 'tle', 'mom', 'umap_trust', 'umap_cont', 'iso_res_var', 
                'iso_rec_error', 'pca_80', 'pca_knee']*int(len(mouse_list)/9)

palette= ["#b3b3b3ff", "#808080ff", "#4d4d4dff", "#000000ff"]
pd_dim = pd.DataFrame(data={'mouse': mouse_list,
                     'dim': dim_list,
                     'method': method_list,
                     'session': session_list,
                     'real_day': real_session_list})    

for method in pd_dim['method'].unique().tolist():
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    b = sns.barplot(x='session', y='dim', data=pd_dim[pd_dim['method']==method],
                palette = palette, linewidth = 1, width= .5, ax = ax)
    sns.swarmplot(x='session', y='dim', data=pd_dim[pd_dim['method']==method],
                color = 'gray', edgecolor = 'gray', ax = ax)
    sns.lineplot(x = 'session', y= 'dim', data=pd_dim[pd_dim['method']==method], units = 'mouse',
                ax = ax, estimator = None, color = ".7", markers = True)
    if 'pca' not in method:
        ax.set_ylim([0,5.2])
    else:
        ax.set_ylim([0,92])
    ax.set_title(method)
    plt.savefig(os.path.join(data_dir,f'dim_{method}_barplot_session.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(data_dir,f'dim_{method}_barplot_session.png'), dpi = 400,bbox_inches="tight")


pd_abids = pd_dim[pd_dim['method']=='abids']
for session in pd_abids['session'].unique().tolist():
    shaphiroResults  = stats.shapiro(pd_abids[pd_abids['session']==session]['dim'].values)
    print(f"{session}: {shaphiroResults.pvalue}")

for session1 in pd_abids['session'].unique().tolist():
    session1_array = pd_abids[pd_abids['session']==session1]['dim'].values
    for session2 in pd_abids['session'].unique().tolist():
        if session1 == session2:
            continue;
        session2_array = pd_abids[pd_abids['session']==session2]['dim'].values
        # print(f"{session1} vs {session2}:",stats.ks_2samp(session1_array, session2_array))
        print(f"{session1} vs {session2}:",stats.ttest_rel(session1_array, session2_array))

from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=pd_abids, res_var='dim', anova_model='dim~C(session)+C(mouse)+C(session):C(mouse)')
res.anova_summary


#__________________________________________________________________________
#|                                                                        |#
#|                           PLOT SI BOXPLOTS                             |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/jercog_learning/SI'
sI_dict = load_pickle(data_dir, 'sI_clean_dict.pkl')

save_dir =  '/home/julio/Documents/SP_project/jercog_learning/no_learning_mice/SI'

palette= {
    'og': ["#b3b3b3ff", "#808080ff", "#4d4d4dff", "#000000ff"],
    'umap': ["#c7ff99ff", "#8eff33ff", "#5bcc00ff", "#2c6600ff"],
    'isomap': ["#ffd599ff", "#ffa933ff", "#cc7600ff", "#663c00ff"],
    'pca': ["#ff9999ff", "#ff3333ff", "#cc0000ff", "#660000ff"]
}

for featureName in ['pos','dir','(pos_dir)', 'vel', 'time']:
    SI_list = list()
    mouse_list = list()
    dim_list = list()
    mouse_list = list()
    session_list = list()
    real_session_list = list()
    signal_list = list()
    nn_list = list()
    for mouse in control_mice_list:
        fileNames = list(sI_dict[mouse].keys())
        og_date = datetime.strptime(fileNames[0][:8], "%Y%m%d")
        for idx, fileName in enumerate(fileNames):

            for nnIdx, nnVal in enumerate(sI_dict[mouse][fileName]['clean_traces'][featureName]['neighList']):
                SI_list.append(sI_dict[mouse][fileName]['clean_traces'][featureName]['sI'][nnIdx])
                SI_list.append(sI_dict[mouse][fileName]['umap'][featureName]['sI'][nnIdx])
                SI_list.append(sI_dict[mouse][fileName]['isomap'][featureName]['sI'][nnIdx])
                SI_list.append(sI_dict[mouse][fileName]['pca'][featureName]['sI'][nnIdx])

                mouse_list = mouse_list + [mouse]*4
                session_list = session_list + [idx]*4
                new_date = datetime.strptime(fileName[:8], "%Y%m%d")
                days_diff = (new_date-og_date).days
                if idx==2:
                    days_diff += 0.5
                real_session_list = real_session_list + [days_diff]*4
                signal_list += ['og','umap', 'isomap','pca']
                nn_list += [nnVal]*4

    pd_SI_control = pd.DataFrame(data={'mouse': mouse_list,
                                    'SI': SI_list,
                                    'session': session_list,
                                    'real_day': real_session_list,
                                    'signal': signal_list,
                                    'nn': nn_list})

    SI_list = list()
    mouse_list = list()
    dim_list = list()
    mouse_list = list()
    session_list = list()
    real_session_list = list()
    signal_list = list()
    nn_list = list()
    for mouse in nl_mice_list:
        fileNames = list(sI_dict[mouse].keys())
        og_date = datetime.strptime(fileNames[0][:8], "%Y%m%d")
        for idx, fileName in enumerate(fileNames):
            if mouse == 'M2022' and fileName == '20150301_154837':
                continue;
            for nnIdx, nnVal in enumerate(sI_dict[mouse][fileName]['clean_traces'][featureName]['neighList']):
                SI_list.append(sI_dict[mouse][fileName]['clean_traces'][featureName]['sI'][nnIdx])
                SI_list.append(sI_dict[mouse][fileName]['umap'][featureName]['sI'][nnIdx])
                SI_list.append(sI_dict[mouse][fileName]['isomap'][featureName]['sI'][nnIdx])
                SI_list.append(sI_dict[mouse][fileName]['pca'][featureName]['sI'][nnIdx])

                mouse_list = mouse_list + [mouse]*4
                session_list = session_list + [idx]*4
                new_date = datetime.strptime(fileName[:8], "%Y%m%d")
                days_diff = (new_date-og_date).days
                if idx==2:
                    days_diff += 0.5
                real_session_list = real_session_list + [days_diff]*4
                signal_list += ['og','umap', 'isomap','pca']
                nn_list += [nnVal]*4

    pd_SI_nl = pd.DataFrame(data={'mouse': mouse_list,
                                    'SI': SI_list,
                                    'session': session_list,
                                    'real_day': real_session_list,
                                    'signal': signal_list,
                                    'nn': nn_list})

    for nn in pd_SI_nl['nn'].unique().tolist():
        pd_SI_nl_nn = pd_SI_nl[pd_SI_nl['nn']==nn]
        pd_SI_control_nn = pd_SI_control[pd_SI_control['nn']==nn]


        for signal in pd_SI_nl_nn['signal'].unique().tolist():
            pd_SI_nl_signal = pd_SI_nl_nn[pd_SI_nl_nn['signal']==signal]
            pd_SI_control_signal = pd_SI_control_nn[pd_SI_control_nn['signal']==signal]


            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            b = sns.boxplot(x='session', y='SI', data=pd_SI_nl_signal,
                        palette = palette[signal], linewidth = 1, width= .5, ax = ax)

            sns.swarmplot(x='session', y='SI', data=pd_SI_nl_signal, 
                        color = 'red', edgecolor = 'red', ax = ax)

            sns.lineplot(x = 'session', y= 'SI', data = pd_SI_nl_signal, units = 'mouse',
                        ax = ax, estimator = None, color = [0.7,0.1,0.1], markers = True)

            sns.lineplot(x = 'session', y= 'SI', data = pd_SI_control_signal,
                        ax = ax,  color = "k", linestyle = '--')


            b.set_xlabel(" ",fontsize=15)
            b.set_ylabel(f"sI {featureName} {signal} {nn}",fontsize=15)
            b.spines['top'].set_visible(False)
            b.spines['right'].set_visible(False)
            b.tick_params(labelsize=12)
            # ax.set_ylim([0.45, 1.05])
            # b.set_yticks([0, .25, .5, .75, 1])
            plt.tight_layout()
            plt.suptitle(featureName)
            plt.savefig(os.path.join(save_dir,f'SI_{featureName}_{signal}_{nn}_no_learning.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'SI_{featureName}_{signal}_{nn}_no_learning.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)



#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/jercog_learning/decoders'
dec_R2s = load_pickle(data_dir, 'dec_R2s_dict.pkl')

save_dir = '/home/julio/Documents/SP_project/jercog_learning/no_learning_mice/decoders'
palette= {
    'og': ["#b3b3b3ff", "#808080ff", "#4d4d4dff", "#000000ff"],
    'umap': ["#c7ff99ff", "#8eff33ff", "#5bcc00ff", "#2c6600ff"],
    'isomap': ["#ffd599ff", "#ffa933ff", "#cc7600ff", "#663c00ff"],
    'pca': ["#ff9999ff", "#ff3333ff", "#cc0000ff", "#660000ff"]
}


label_list = ['posx', 'posy','vel', 'index_mat', 'dir_mat']
space_list = ['base_signal', 'pca', 'isomap', 'umap']
decoder_list = ['wf','wc','xgb','svr']


for label_idx, label_name in enumerate(label_list):
    R2s_list = list()
    mouse_list = list()
    signal_list = list()
    session_list = list()
    real_session_list = list()
    dec_list = list()
    for dec_idx, dec_name in enumerate(decoder_list):
        for mouse in control_mice_list:
            dec_mouse = copy.deepcopy(dec_R2s[mouse])
            fileNames = list(dec_R2s[mouse].keys())
            og_date = datetime.strptime(fileNames[0][:8], "%Y%m%d")
            for idx, fileName in enumerate(fileNames):
                for signal in space_list:
                    R2s_list.append(np.nanmean(dec_mouse[fileName][signal][dec_name][:,label_idx,0], axis=0))
                    mouse_list.append(mouse)
                    if signal == 'base_signal':
                        signal_list.append('og')
                    else:
                        signal_list.append(signal)
                session_list = session_list + [idx]*4
                new_date = datetime.strptime(fileName[:8], "%Y%m%d")
                days_diff = (new_date-og_date).days
                if idx==2:
                    days_diff += 0.5
                real_session_list += [days_diff]*4
                dec_list += [dec_name]*4

    pd_dec_control = pd.DataFrame(data={'mouse': mouse_list,
                                'R2s': R2s_list,
                                'decoder': dec_list,
                                'session': session_list,
                                'real_day': real_session_list,
                                'signal': signal_list})


    R2s_list = list()
    mouse_list = list()
    signal_list = list()
    session_list = list()
    real_session_list = list()
    dec_list = list()
    for dec_idx, dec_name in enumerate(decoder_list):
        for mouse in nl_mice_list:
            dec_mouse = copy.deepcopy(dec_R2s[mouse])
            fileNames = list(dec_R2s[mouse].keys())
            og_date = datetime.strptime(fileNames[0][:8], "%Y%m%d")
            for idx, fileName in enumerate(fileNames):
                if mouse == 'M2022' and fileName == '20150301_154837':
                    continue;
                for signal in space_list:
                    R2s_list.append(np.nanmean(dec_mouse[fileName][signal][dec_name][:,label_idx,0], axis=0))
                    mouse_list.append(mouse)
                    if signal == 'base_signal':
                        signal_list.append('og')
                    else:
                        signal_list.append(signal)
                session_list = session_list + [idx]*4
                new_date = datetime.strptime(fileName[:8], "%Y%m%d")
                days_diff = (new_date-og_date).days
                if idx==2:
                    days_diff += 0.5
                real_session_list += [days_diff]*4
                dec_list += [dec_name]*4

    pd_dec_nl = pd.DataFrame(data={'mouse': mouse_list,
                                'R2s': R2s_list,
                                'decoder': dec_list,
                                'session': session_list,
                                'real_day': real_session_list,
                                'signal': signal_list})

    for signal in pd_dec_nl['signal'].unique().tolist():
        pd_dec_nl_signal = pd_dec_nl[pd_dec_nl['signal']==signal]
        pd_dec_control_signal = pd_dec_control[pd_dec_control['signal']==signal]
        fig, ax = plt.subplots(1,4,figsize=(15,5))
        for dec_idx, dec_name in enumerate(decoder_list):
            pd_dec_nl_dec = pd_dec_nl_signal[pd_dec_nl_signal['decoder']==dec_name]
            pd_dec_control_dec = pd_dec_control_signal[pd_dec_control_signal['decoder']==dec_name]

            b = sns.barplot(x='session', y='R2s', data=pd_dec_nl_dec,
                    palette = palette[signal], linewidth = 1, width= .5, ax = ax[dec_idx])
            sns.swarmplot(x='session', y='R2s', data=pd_dec_nl_dec, 
                        color = 'red', edgecolor = 'red', ax = ax[dec_idx])
            sns.lineplot(x = 'session', y= 'R2s', data = pd_dec_nl_dec, units = 'mouse',
                        ax = ax[dec_idx], estimator = None, color = "r", markers = True)

            sns.lineplot(x = 'session', y= 'R2s', data = pd_dec_control_dec,
                        ax = ax[dec_idx],  color = "k", linestyle = '--')

            ax[dec_idx].set_ylabel(f'R2s {label_name}')
            ax[dec_idx].set_title(f'{dec_name} {signal}')

        fig.suptitle(label_name)
        plt.savefig(os.path.join(save_dir,f'dec_{label_name}_{signal}_test_no_learning.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(save_dir,f'dec_{label_name}_{signal}_test_no_learning.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)


#__________________________________________________________________________
#|                                                                        |#
#|                              ECCENTRICITY                              |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/jercog_learning/eccentricity'
save_dir = '/home/julio/Documents/SP_project/jercog_learning/no_learning_mice/eccentricity'

# eccen_dict = load_pickle(data_dir, 'ellipse_fit_dict.pkl')

eccen_list = list()
mouse_list = list()
session_list = list()
real_session_list = list()
for mouse in control_mice_list:
    eccen_dict = load_pickle(data_dir, f'{mouse}_ellipse_fit_dict.pkl')
    fileNames = list(eccen_dict.keys())
    og_date = datetime.strptime(fileNames[0][:8], "%Y%m%d")
    for idx, fileName in enumerate(fileNames):
        eccen_list.append(eccen_dict[fileName]['eccentricity'])
        mouse_list.append(mouse)
        session_list.append(idx)
        new_date = datetime.strptime(fileName[:8], "%Y%m%d")
        days_diff = (new_date-og_date).days
        if idx==2:
            days_diff += 0.5
        real_session_list.append(days_diff)

pd_eccen_control = pd.DataFrame(data={'mouse': mouse_list,
                            'eccentricity': eccen_list,
                            'session': session_list,
                            'real_day': real_session_list})


eccen_list = list()
mouse_list = list()
session_list = list()
real_session_list = list()
for mouse in nl_mice_list:
    eccen_dict = load_pickle(data_dir, f'{mouse}_ellipse_fit_dict.pkl')
    fileNames = list(eccen_dict.keys())
    og_date = datetime.strptime(fileNames[0][:8], "%Y%m%d")
    for idx, fileName in enumerate(fileNames):
        if mouse == 'M2022' and fileName == '20150301_154837':
            continue;
        eccen_list.append(eccen_dict[fileName]['eccentricity'])
        mouse_list.append(mouse)
        session_list.append(idx)
        new_date = datetime.strptime(fileName[:8], "%Y%m%d")
        days_diff = (new_date-og_date).days
        if idx==2:
            days_diff += 0.5
        real_session_list.append(days_diff)

pd_eccen_nl = pd.DataFrame(data={'mouse': mouse_list,
                            'eccentricity': eccen_list,
                            'session': session_list,
                            'real_day': real_session_list})

palette = ["#c7ff99ff", "#8eff33ff", "#5bcc00ff", "#2c6600ff"]
fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='session', y='eccentricity', data=pd_eccen_nl,
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='session', y='eccentricity', data=pd_eccen_nl,
            color = 'red', edgecolor = 'red', ax = ax)
sns.lineplot(x = 'session', y= 'eccentricity', data=pd_eccen_nl, units = 'mouse',
            ax = ax, estimator = None, color = "r", markers = True)

sns.lineplot(x = 'session', y= 'eccentricity', data = pd_eccen_control,
            ax = ax,  color = "k", linestyle = '--')
plt.savefig(os.path.join(save_dir,f'jercog_learning_eccentricity_no_learning.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'jercog_learning_eccentricity_no_learning.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)




#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

from sklearn.metrics import median_absolute_error

data_dir = '/home/julio/Documents/SP_project/jercog_learning/decoders'
dec_R2s = load_pickle(data_dir, 'dec_R2s_dict.pkl')
dec_pred = load_pickle(data_dir, 'dec_pred_dict.pkl')

data_dir = '/home/julio/Documents/SP_project/jercog_learning/processed_data/'
save_dir = '/home/julio/Documents/SP_project/jercog_learning/no_learning_mice/decoders/fail_trials/'

for mouse in control_mice_list:
    print(f'\n{mouse}: ')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(data_dir, mouse)
    pdMouseAll = gu.load_files(filePath,'*'+fileName,verbose=True,struct_type="pickle")
    fileNames = list(pdMouseAll.keys())
    fileNames.sort()

    for fileName in fileNames:


        dec_pred_session = dec_pred[mouse][fileName]
        pd_mouse = pdMouseAll[fileName]

        pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
        dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
        trial_mat = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))
        vel =  copy.deepcopy(np.concatenate(pd_mouse['vel'].values, axis=0))

        points_of_interest = np.where(dir_mat>2)[0]
        trials_of_interest = np.unique(trial_mat[points_of_interest]).astype(int)
        trials_of_interest_types = [pd_mouse[pd_mouse['trial_id']==x]['dir'].values[0] for x in trials_of_interest]

        label_idx = 0; label_name = 'posx'
        decoder_idx = 2; decoder_name = 'xgb'
        preds = dec_pred_session[label_idx][decoder_idx]
        min_pos, max_pos = 0, np.max(pos[:,0])

        #comparar level of error entre todo vs fails
        split_error_og = dec_R2s[mouse][fileName]['base_signal']['xgb'][:,0,0]
        split_error_umap = dec_R2s[mouse][fileName]['umap']['xgb'][:,0,0]

        #comparar std en pred (estabilidad en pred) entre fail across split vs correct trials
        fail_pred_og = [[] for x in range(10)]
        fail_pred_umap = [[] for x in range(10)]
        fail_real = [[] for x in range(10)]

        for idx, trial_num in enumerate(trials_of_interest):
            trial_type = trials_of_interest_types[idx]
            indexes_trial = np.where(trial_mat==trial_num)[0]
            test_split = np.where(np.all(preds[:,indexes_trial,0]==1,axis=1))[0]
            for sp in test_split:
                fail_real[sp].append(preds[0,indexes_trial,1])
                fail_pred_umap[sp].append(preds[sp][indexes_trial][:,2])
                fail_pred_og[sp].append(preds[sp][indexes_trial][:,3])


        fail_real = [np.hstack(x) for x in fail_real]        
        fail_pred_og = [np.hstack(x) for x in fail_pred_og]
        fail_pred_umap = [np.hstack(x) for x in fail_pred_umap]

        fail_error = np.zeros((10,2))
        for sp in range(10):
            fail_error[sp,0] = median_absolute_error(fail_real[sp],fail_pred_og[sp])
            fail_error[sp,1] = median_absolute_error(fail_real[sp],fail_pred_umap[sp])


        for idx, trial_num in enumerate(trials_of_interest):
            trial_type = trials_of_interest_types[idx]
            indexes_trial = np.where(trial_mat==trial_num)[0]
            if idx+1 < len(trials_of_interest):
                if 'S' in trials_of_interest_types[idx+1]:
                    indexes_trial_2 = np.where(trial_mat==trials_of_interest[idx+1])[0]
                    indexes_trial = np.concatenate((indexes_trial,indexes_trial_2))
                    trial_type = trial_type + ' - ' + trials_of_interest_types[idx+1]

            test_split = np.where(np.all(preds[:,indexes_trial,0]==1,axis=1))[0]
            #expand indexes 1 second back and forward
            pre_indexes = -np.arange(10)[::-1]+indexes_trial[0]-1
            post_indexes = np.arange(10)+indexes_trial[-1]+1
            indexes_trial = np.concatenate((pre_indexes, indexes_trial, post_indexes))
            indexes_trial = indexes_trial[indexes_trial<pos.shape[0]]
            #find in which splits they were use for test
            real_signal = preds[0,indexes_trial,1]
            real_vel = max_pos*vel[indexes_trial]/np.max(vel)
            pred_signal_og = preds[test_split][:,indexes_trial][:,:,2].T
            pred_signal_umap = preds[test_split][:,indexes_trial][:,:,3].T
            x = np.arange(real_signal.shape[0])/20

            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(x, real_signal,'b')
            ax.plot(x,real_vel,'b--')

            m = np.nanmean(pred_signal_og,axis=1)
            sd = np.nanstd(pred_signal_og,axis=1)
            ax.plot(x, m, 'grey')
            ax.fill_between(x, m-sd,m+sd,alpha=0.3,color='grey')

            m = np.nanmean(pred_signal_umap,axis=1)
            sd = np.nanstd(pred_signal_umap,axis=1)
            ax.plot(x, m, 'g')
            ax.fill_between(x, m-sd,m+sd,alpha=0.3,color='g')
            ax.set_ylabel('pos cm')
            ax.set_xlabel('time (s)')
            ax.set_ylim([min_pos, max_pos])
            ax.set_title(f'{mouse} {fileName} {trial_num} {trial_type}')
            plt.savefig(os.path.join(save_dir,f'{mouse}_{fileName}_{trial_num}_fail_trial.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_{fileName}_{trial_num}_fail_trial.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

#MANUAL CURATION
#M2022 FILENAME[-1] (20150306_104914) trial_index 74
#M2023 FILENAME[-1] (20150306_112849) trial_index 36
#M2025 FILENAME[-1] (20150303_091715) trial_index 76

save_dir = '/home/julio/Documents/SP_project/jercog_learning/no_learning_mice/decoders'
palette= {
    'og': ["#b3b3b3ff", "#808080ff", "#4d4d4dff", "#000000ff"],
    'umap': ["#c7ff99ff", "#8eff33ff", "#5bcc00ff", "#2c6600ff"],
    'isomap': ["#ffd599ff", "#ffa933ff", "#cc7600ff", "#663c00ff"],
    'pca': ["#ff9999ff", "#ff3333ff", "#cc0000ff", "#660000ff"]
}


label_list = ['posx', 'posy','vel', 'index_mat', 'dir_mat']
space_list = ['base_signal', 'pca', 'isomap', 'umap']
decoder_list = ['wf','wc','xgb','svr']


for label_idx, label_name in enumerate(label_list):
    R2s_list = list()
    mouse_list = list()
    signal_list = list()
    session_list = list()
    real_session_list = list()
    dec_list = list()
    for dec_idx, dec_name in enumerate(decoder_list):
        for mouse in control_mice_list:
            dec_mouse = copy.deepcopy(dec_R2s[mouse])
            fileNames = list(dec_R2s[mouse].keys())
            og_date = datetime.strptime(fileNames[0][:8], "%Y%m%d")
            for idx, fileName in enumerate(fileNames):
                for signal in space_list:
                    R2s_list.append(np.nanmean(dec_mouse[fileName][signal][dec_name][:,label_idx,0], axis=0))
                    mouse_list.append(mouse)
                    if signal == 'base_signal':
                        signal_list.append('og')
                    else:
                        signal_list.append(signal)
                session_list = session_list + [idx]*4
                new_date = datetime.strptime(fileName[:8], "%Y%m%d")
                days_diff = (new_date-og_date).days
                if idx==2:
                    days_diff += 0.5
                real_session_list += [days_diff]*4
                dec_list += [dec_name]*4



#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/jercog_learning/processed_data/'
save_dir = '/home/julio/Documents/SP_project/jercog_learning/no_learning_mice/decoders/fail_trials/'

for mouse in control_mice_list:
    print(f'\n{mouse}: ')
    mouse_dict_name =  mouse+'_df_dict.pkl'
    filePath = os.path.join(data_dir, mouse)
    pdMouseAll = gu.load_files(filePath,'*'+mouse_dict_name,verbose=True,struct_type="pickle")
    fileNames = list(pdMouseAll.keys())
    fileNames.sort()

    for session_name in session_names:


        dec_pred_session = dec_pred[mouse][session_name]
        pd_mouse = pdMouseAll[session_name]

        pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
        dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
        trial_mat = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))
        vel =  copy.deepcopy(np.concatenate(pd_mouse['vel'].values, axis=0))
        umap_emb =  copy.deepcopy(np.concatenate(pd_mouse['umap'].values, axis=0))





#__________________________________________________________________________
#|                                                                        |#
#|                                EMBEDDINGS                              |#
#|________________________________________________________________________|#
data_dir = '/home/julio/Documents/SP_project/jercog_learning/processed_data/'
save_dir = '/home/julio/Documents/SP_project/jercog_learning/no_learning_mice/emb_examples/'
from sklearn.decomposition import PCA

def filter_noisy_outliers_mild(data, D=None,k=20):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, k)
    sum(noiseIdx)
    return noiseIdx

for mouse in miceList:
    print(f"\nWorking on mouse {mouse}: ")
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,mouse+'_df_dict.pkl')
    session_names = list(pd_mouse.keys())
    session_names.sort()
    fig = plt.figure(figsize=(15,10))
    for idx, session in enumerate(session_names):

        pos = copy.deepcopy(np.concatenate(pd_mouse[session]['pos'].values, axis=0))
        dir_mat = copy.deepcopy(np.concatenate(pd_mouse[session]['dir_mat'].values, axis=0))
        umap_emb =  copy.deepcopy(np.concatenate(pd_mouse[session]['umap'].values, axis=0))
        time_mat = np.arange(pos.shape[0])/20


        D = pairwise_distances(umap_emb)
        noiseIdx = filter_noisy_outliers_mild(umap_emb,D,k=20)
        umap_emb = umap_emb[~noiseIdx,:]
        pos = pos[~noiseIdx,:]
        dir_mat = dir_mat[~noiseIdx]
        time_mat = time_mat[~noiseIdx]


        dir_color = np.zeros((dir_mat.shape[0],3))
        for point in range(dir_mat.shape[0]):
            if dir_mat[point]==0:
                dir_color[point] = [14/255,14/255,143/255]
            elif dir_mat[point]==1:
                dir_color[point] = [12/255,136/255,249/255]
            else:
                dir_color[point] = [17/255,219/255,224/255]

        modelPCA = PCA(2)
        modelPCA.fit(umap_emb)
        umap_emb_2d = modelPCA.transform(umap_emb)

        ax = plt.subplot(3,4,idx+1)
        ax.scatter(*umap_emb_2d[:,:2].T, color = dir_color,s = 10)
        # ax.view_init(angles_values[mouse][session][0], angles_values[mouse][session][1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(3,4,idx+5)
        ax.scatter(*umap_emb_2d[:,:2].T, c = pos[:,0], cmap = 'magma',s = 10, vmin= 0, vmax = 120)
        # ax.view_init(angles_values[mouse][session][0], angles_values[mouse][session][1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(3,4,idx+9)
        ax.scatter(*umap_emb_2d[:,:2].T,  c = time_mat, cmap = 'YlGn_r',s = 10)
        # ax.view_init(angles_values[mouse][session][0], angles_values[mouse][session][1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_embedding_plots_2D.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_embedding_plots_2D.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)