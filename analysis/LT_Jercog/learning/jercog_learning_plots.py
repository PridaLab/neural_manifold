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
miceList = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']

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
for mouse in miceList:
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

data_dir = '/home/julio/Documents/SP_project/jercog_learning/SI/'
sI_dict = load_pickle(data_dir, 'sI_clean_dict.pkl')

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
    for mouse in miceList:
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

    pd_SI = pd.DataFrame(data={'mouse': mouse_list,
                                    'SI': SI_list,
                                    'session': session_list,
                                    'real_day': real_session_list,
                                    'signal': signal_list,
                                    'nn': nn_list})

    for nn in pd_SI['nn'].unique().tolist():
        pd_SI_nn = pd_SI[pd_SI['nn']==nn]
        for signal in pd_SI_nn['signal'].unique().tolist():
            pd_SI_signal = pd_SI_nn[pd_SI_nn['signal']==signal]
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            b = sns.boxplot(x='session', y='SI', data=pd_SI_signal,
                        palette = palette[signal], linewidth = 1, width= .5, ax = ax)

            sns.swarmplot(x='session', y='SI', data=pd_SI_signal, 
                        color = 'gray', edgecolor = 'gray', ax = ax)

            sns.lineplot(x = 'session', y= 'SI', data = pd_SI_signal, units = 'mouse',
                        ax = ax, estimator = None, color = ".7", markers = True)

            b.set_xlabel(" ",fontsize=15)
            b.set_ylabel(f"sI {featureName} {signal} {nn}",fontsize=15)
            b.spines['top'].set_visible(False)
            b.spines['right'].set_visible(False)
            b.tick_params(labelsize=12)
            # ax.set_ylim([0.45, 1.05])
            # b.set_yticks([0, .25, .5, .75, 1])
            plt.tight_layout()
            plt.suptitle(featureName)
            plt.savefig(os.path.join(data_dir,f'SI_{featureName}_{signal}_{nn}.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(data_dir,f'SI_{featureName}_{signal}_{nn}.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)


#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/jercog_learning/decoders'
dec_R2s = load_pickle(data_dir, 'dec_R2s_dict.pkl')

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
        for mouse in miceList:
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

    pd_dec = pd.DataFrame(data={'mouse': mouse_list,
                                'R2s': R2s_list,
                                'decoder': dec_list,
                                'session': session_list,
                                'real_day': real_session_list,
                                'signal': signal_list})

    for signal in pd_dec['signal'].unique().tolist():
        pd_dec_signal = pd_dec[pd_dec['signal']==signal]
        fig, ax = plt.subplots(1,4,figsize=(15,5))
        for dec_idx, dec_name in enumerate(decoder_list):
            pd_dec_dec = pd_dec_signal[pd_dec_signal['decoder']==dec_name]
            b = sns.barplot(x='session', y='R2s', data=pd_dec_dec,
                    palette = palette[signal], linewidth = 1, width= .5, ax = ax[dec_idx])
            sns.swarmplot(x='session', y='R2s', data=pd_dec_dec, 
                        color = 'gray', edgecolor = 'gray', ax = ax[dec_idx])

            sns.lineplot(x = 'session', y= 'R2s', data = pd_dec_dec, units = 'mouse',
                        ax = ax[dec_idx], estimator = None, color = ".7", markers = True)

            ax[dec_idx].set_ylabel(f'R2s {label_name}')
            ax[dec_idx].set_title(f'{dec_name} {signal}')

        fig.suptitle(label_name)
        plt.savefig(os.path.join(data_dir,f'dec_{label_name}_{signal}_test.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(data_dir,f'dec_{label_name}_{signal}_test.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)


#__________________________________________________________________________
#|                                                                        |#
#|                              ECCENTRICITY                              |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/jercog_learning/eccentricity'
# eccen_dict = load_pickle(data_dir, 'ellipse_fit_dict.pkl')

eccen_list = list()
mouse_list = list()
session_list = list()
real_session_list = list()
for mouse in miceList:
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

pd_eccen = pd.DataFrame(data={'mouse': mouse_list,
                            'eccentricity': eccen_list,
                            'session': session_list,
                            'real_day': real_session_list})

palette = ["#c7ff99ff", "#8eff33ff", "#5bcc00ff", "#2c6600ff"]
fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='session', y='eccentricity', data=pd_eccen,
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='session', y='eccentricity', data=pd_eccen,
            color = 'gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x = 'session', y= 'eccentricity', data=pd_eccen, units = 'mouse',
            ax = ax, estimator = None, color = ".7", markers = True)
plt.savefig(os.path.join(data_dir,f'jercog_learning_eccentricity.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,f'jercog_learning_eccentricity.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)

for session in pd_eccen['session'].unique().tolist():
    shaphiroResults  = stats.shapiro(pd_eccen[pd_eccen['session']==session]['eccentricity'].values)
    print(f"{session}: {shaphiroResults.pvalue}")

for session1 in pd_eccen['session'].unique().tolist():
    session1_array = pd_eccen[pd_eccen['session']==session1]['eccentricity'].values
    for session2 in pd_eccen['session'].unique().tolist():
        if session1 == session2:
            continue;
        session2_array = pd_eccen[pd_eccen['session']==session2]['eccentricity'].values
        # print(f"{session1} vs {session2}:",stats.wilcoxon(session1_array, session2_array))
        print(f"{session1} vs {session2}:",stats.ttest_rel(session1_array, session2_array))


#__________________________________________________________________________
#|                                                                        |#
#|                                EMBEDDINGS                              |#
#|________________________________________________________________________|#
data_dir = '/home/julio/Documents/SP_project/jercog_learning/processed_data/'
save_dir = '/home/julio/Documents/SP_project/jercog_learning/emb_examples/'
def filter_noisy_outliers_mild(data, D=None,k=20):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, k)
    sum(noiseIdx)
    return noiseIdx

from sklearn.decomposition import PCA

angles_values = {
    'M2019': {},
    'M2024': {
        '20150228_153237': [-44, -56],
        '20150301_124134': [88,-21],
        '20150301_173817': [93,-10],
        '20150303_082851': [90,-7]
    },
    'M2025': {
        '20150228_160317': [-44, -56],
        '20150301_130341': [88,-21],
        '20150301_181458': [93,-10],
        '20150303_091715': [90,-7]
    },
    'M2026': {
        '20150228_162802': [140, -40],
        '20150301_132603': [85, -95],
        '20150301_185053': [45, -45],
        '20150303_095846': [75, 30]
    }

}

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
        noiseIdx = filter_noisy_outliers_mild(umap_emb,D,k=10)
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


        ax = plt.subplot(3,4,idx+1, projection='3d')
        ax.scatter(*umap_emb[:,:3].T, color = dir_color,s = 10)
        # ax.view_init(angles_values[mouse][session][0], angles_values[mouse][session][1])
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(3,4,idx+5, projection='3d')
        ax.scatter(*umap_emb[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10, vmin= 0, vmax = 120)
        # ax.view_init(angles_values[mouse][session][0], angles_values[mouse][session][1])
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(3,4,idx+9, projection='3d')
        ax.scatter(*umap_emb[:,:3].T,  c = time_mat, cmap = 'YlGn_r',s = 10)
        # ax.view_init(angles_values[mouse][session][0], angles_values[mouse][session][1])
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_embedding_plots.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_embedding_plots.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)



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
        noiseIdx = filter_noisy_outliers_mild(umap_emb,D,k=10)
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


#__________________________________________________________________________
#|                                                                        |#
#|                          MUTUAL INFORMATION                            |#
#|________________________________________________________________________|#

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
save_dir = '/home/julio/Documents/SP_project/jercog_learning/mutual_information/'
mi_scores_dict = load_pickle(save_dir, 'mi_scores_dict.pkl')

mi_scores = list()
day_list = list()
mouse_list = list()
for mouse in miceList:
    session_list = list(mi_scores_dict[mouse].keys())
    session_list.sort()
    for idx, session in enumerate(session_list):
        mi_scores.append(mi_scores_dict[mouse][session]['mi_scores'])
        num_cells = mi_scores_dict[mouse][session]['mi_scores'].shape[1]
        day_list.append((np.zeros((num_cells,))+idx).astype(int))
        mouse_list.append([mouse]*num_cells)


mi_scores = np.hstack(mi_scores)
day_list = np.hstack(day_list)
mouse_list = sum(mouse_list, []) #[cell for day in mouse_list cell in day]
pd_mi_scores = pd.DataFrame(data={'posx': mi_scores[0,:],
                     'dir': mi_scores[2,:],
                     'vel': mi_scores[3,:],
                     'time': mi_scores[4,:],
                     'session': day_list,
                     'mouse': mouse_list}) 

plt.figure(figsize=(8,10))
for idx, label_name in enumerate(['posx', 'dir', 'vel', 'time']):
    ax = plt.subplot(4,1,idx+1)
    sns.kdeplot(pd_mi_scores, x=label_name, hue='session',ax = ax, fill=True)
    ax.set_xlim([-0.05, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'mutual_info_kde.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_kde.png'), dpi = 400,bbox_inches="tight")

plt.figure(figsize=(8,10))
for idx, label_name in enumerate(['posx', 'dir', 'vel', 'time']):
    ax = plt.subplot(4,1,idx+1)
    sns.histplot(pd_mi_scores, x=label_name, hue='session',ax = ax, stat = 'probability', kde=True)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([0,0.03])
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'mutual_info_dist.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_dist.png'), dpi = 400,bbox_inches="tight")


plt.figure(figsize=(8,8))
sns.kdeplot(pd_mi_scores, x='posx', y='dir',fill=True, alpha=0.7, label = 'posx-dir')
sns.kdeplot(pd_mi_scores, x='posx', y='time',fill=True, alpha=0.7, label = 'posx-time')
sns.kdeplot(pd_mi_scores, x='posx', y='vel',fill=True, alpha=0.7, label = 'posx-vel')
plt.ylim([-0.1,0.85])
plt.xlim([-0.1,0.85])
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes.png'), dpi = 400,bbox_inches="tight")


pd_mi_scores2 = pd.DataFrame(data={'MI': mi_scores[(0,2,3,4),:].reshape(-1,1)[:,0],
                     'label': ['posx']*mi_scores.shape[1] + ['dir']*mi_scores.shape[1] + ['vel']*mi_scores.shape[1] +['time']*mi_scores.shape[1],
                     'session': list(day_list)*4,
                     'mouse': mouse_list*4}) 
plt.figure()
sns.violinplot(data=pd_mi_scores2, x="label", y="MI", hue = 'session', inner="quart", cut=0, linewidth=1, linecolor="k")
plt.savefig(os.path.join(save_dir,'mutual_info_violinplots.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_violinplots.png'), dpi = 400,bbox_inches="tight")




kwargs = {'alpha':0.3, 'cut': 0, 'cbar': False, 'fill': True}

plt.figure(figsize=(14,8))
ax = plt.subplot(1,3,1)
sns.kdeplot(pd_mi_scores, x='posx', y='dir', hue='session', ax=ax, **kwargs)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,1])
ax.set_xlim([0,1])
ax.set_aspect('equal')
ax = plt.subplot(1,3,2)
sns.kdeplot(pd_mi_scores, x='posx', y='time', hue='session', ax=ax, **kwargs)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,1])
ax.set_xlim([0,1])
ax.set_aspect('equal')
ax = plt.subplot(1,3,3)
sns.kdeplot(pd_mi_scores, x='posx', y='vel', hue='session', ax=ax, **kwargs)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,1])
ax.set_xlim([0,1])
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes_separated.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes_separated.png'), dpi = 400,bbox_inches="tight")


plt.figure(figsize=(14,8))
ax = plt.subplot(1,3,1)
sns.scatterplot(pd_mi_scores, x='posx', y='dir', hue='session', ax=ax)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.82])
ax.set_xlim([0,0.82])
ax.set_aspect('equal')
ax = plt.subplot(1,3,2)
sns.scatterplot(pd_mi_scores, x='posx', y='time', hue='session', ax=ax)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.82])
ax.set_xlim([0,0.82])
ax.set_aspect('equal')
ax = plt.subplot(1,3,3)
sns.scatterplot(pd_mi_scores, x='posx', y='vel', hue='session', ax=ax)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.82])
ax.set_xlim([0,0.82])
ax.set_aspect('equal')
plt.tight_layout()



kwargs = { 'cut': 0, 'cbar': False, 'fill': True}
plt.figure(figsize=(14,8))
for session in range(4):
    ax = plt.subplot(3,4,1+session)
    sns.kdeplot(pd_mi_scores[pd_mi_scores['session']==session], x='posx', y='dir', ax=ax, **kwargs)
    ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
    ax.set_ylim([0,0.7])
    ax.set_xlim([0,0.7])
    ax.set_aspect('equal')
    ax = plt.subplot(3,4,5+session)
    sns.kdeplot(pd_mi_scores[pd_mi_scores['session']==session], x='posx', y='time', ax=ax, **kwargs)
    ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
    ax.set_ylim([0,0.7])
    ax.set_xlim([0,0.7])
    ax.set_aspect('equal')
    ax = plt.subplot(3,4,9+session)
    sns.kdeplot(pd_mi_scores[pd_mi_scores['session']==session], x='posx', y='vel', ax=ax, **kwargs)
    ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
    ax.set_ylim([0,0.7])
    ax.set_xlim([0,0.7])
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes_separated.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes_separated.png'), dpi = 400,bbox_inches="tight")
