import seaborn as sns
import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from scipy import stats
import networkx as nx
from structure_index import compute_structure_index, draw_graph
from sklearn.metrics import pairwise_distances

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx

from sklearn.metrics import pairwise_distances


def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data


def meshgrid2(arrs):
    #arrs: tuple with np.arange of shape of all dimensions
    lens = list(map(len, arrs))
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz*=s
    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)
    return tuple(ans)

def personalize_ax(ax, ax_view = None):
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if ax_view:
        ax.view_init(ax_view[0], ax_view[1])


supMice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']

#__________________________________________________________________________
#|                                                                        |#
#|                                PLOT DIM                                |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir =  '/home/julio/Documents/SP_project/Fig2/dimensionality/'


innerDim = load_pickle(os.path.join(dataDir, 'inner_dim'), 'inner_dim_dict.pkl')
fig, ax = plt.subplots(1, 1, figsize=(10,6))
dimList = list()
mouseList = list()
methodList = list()
layerList = list()
for mouse in miceList:
    dimList.append(innerDim[mouse]['momDim'])
    dimList.append(innerDim[mouse]['abidsDim'])
    dimList.append(innerDim[mouse]['tleDim'])
    if mouse in deepMice:
        layerList = layerList + ['deep']*3
    elif mouse in supMice:
        layerList = layerList + ['sup']*3
    mouseList = mouseList + [mouse]*3
    methodList = methodList+['mom', 'abids', 'tle']

palette= ["#ccccccff", "#808080ff", "#4d4d4dff"]
dimPD = pd.DataFrame(data={'mouse': mouseList,
                     'dim': dimList,
                     'method': methodList,
                     'layer': layerList})    

b = sns.barplot(x='layer', y='dim', data=dimPD, hue='method',
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_title(case)
ax.set_ylim([0,4.5])
plt.tight_layout()
plt.savefig(os.path.join(dataDir,'inner_dim.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'inner_dim.png'), dpi = 400,bbox_inches="tight")


umapDim = load_pickle(dataDir, 'umap_dim_dict.pkl')
isoDim = load_pickle(dataDir, 'isomap_dim_dict.pkl')


fig, ax = plt.subplots(1, 1, figsize=(10,6))
dimList = list()
mouseList = list()
methodList = list()
layerList = list()
for mouse in miceList:
    dimList.append(umapDim[mouse]['trustDim'])
    dimList.append(umapDim[mouse]['contDim'])
    dimList.append(isoDim[mouse]['resVarDim'])
    dimList.append(isoDim[mouse]['recErrorDim'])

    if mouse in deepMice:
        layerList = layerList + ['deep']*4
    elif mouse in supMice:
        layerList = layerList + ['sup']*4
    mouseList = mouseList + [mouse]*4
    methodList = methodList+['umap_trust', 'umap_cont', 
                        'iso_res_var', 'iso_rec_error']

palette= ["#5ff444ff", "#5fc010ff", "#e09e38ff", "#e08b10ff"]
dimPD = pd.DataFrame(data={'mouse': mouseList,
                     'dim': dimList,
                     'method': methodList,
                     'layer': layerList})    

b = sns.barplot(x='layer', y='dim', data=dimPD, hue='method',
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_title(case)
ax.set_ylim([0,7.5])
plt.tight_layout()
plt.savefig(os.path.join(dataDir,'umap_iso_dim.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'umap_iso_dim.png'), dpi = 400,bbox_inches="tight")

pcaDim = load_pickle(dataDir, 'pca_dim_dict.pkl')
fig, ax = plt.subplots(1, 1, figsize=(10,6))
dimList = list()
mouseList = list()
methodList = list()
layerList = list()
for mouse in miceList:
    dimList.append(pcaDim[mouse]['var80Dim'])
    dimList.append(pcaDim[mouse]['kneeDim'])
    if mouse in deepMice:
        layerList = layerList + ['deep']*2
    elif mouse in supMice:
        layerList = layerList + ['sup']*2
    mouseList = mouseList + [mouse]*2
    methodList = methodList+['pca_80', 'pca_knee']

palette= ["#e03d27ff", "#e01110ff"]
dimPD = pd.DataFrame(data={'mouse': mouseList,
                     'dim': dimList,
                     'method': methodList,
                     'layer': layerList})    

b = sns.barplot(x='layer', y='dim', data=dimPD, hue='method',
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_title(case)
ax.set_ylim([0,90])

plt.tight_layout()
plt.savefig(os.path.join(dataDir,'pca_dim.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'pca_dim.png'), dpi = 400,bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|                        PLOT SI DIM RED BOXPLOTS                        |#
#|________________________________________________________________________|#
#og clean study
dataDir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(dataDir, 'sI_clean_dict.pkl')
miceList = list(sIDict.keys())
for featureName in ['pos','dir','(pos_dir)', 'vel', 'time']: #,'session']:
    SIList = list()
    mouseList = list()
    layerList = list()
    for mouse in miceList:
        SIList.append(sIDict[mouse]['clean_traces'][featureName]['sI'])
        SIList.append(sIDict[mouse]['umap'][featureName]['sI'])
        # sSIList.append(np.percentile(sIDict[mouse]['clean_traces'][featureName]['ssI'], 99))
        # sSIList.append(np.percentile(sIDict[mouse]['umap'][featureName]['ssI'], 99))
        # sSIList.append(np.percentile(sIDict[mouse]['isomap'][featureName]['ssI'], 99))
        # sSIList.append(np.percentile(sIDict[mouse]['pca'][featureName]['ssI'], 99))

        mouseList = mouseList + [mouse]*2
        if mouse in deepMice:
            layerList = layerList + ['deep']*2
        elif mouse in supMice:
            layerList = layerList + ['sup']*2

    feature_list = ['og','umap']*len(miceList)
    SIPD = pd.DataFrame(data={'mouse': mouseList,
                                     'SI': SIList,
                                     'layer': layerList,
                                     'method': feature_list})

    palette= ['#8a8a8aff', '#5fc010ff', '#e08b10ff', '#e01110ff']
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    b = sns.boxplot(x='method', y='SI', data=SIPD, hue = 'layer',
                palette = palette, linewidth = 1, width= .5, ax = ax)

    sns.swarmplot(x='method', y='SI', data=SIPD, hue= 'layer',
                palette = 'dark:gray', edgecolor = 'gray', ax = ax)

    # for idx in range(4):
    #     x_space = [-.25+idx, 0.25+idx]
    #     m = np.mean(sSI_array[:,idx])
    #     sd = np.std(sSI_array[:,idx])
    #     ax.plot(x_space, [m,m], linestyle='--', color=palette[idx])
    #     ax.fill_between(x_space, m-sd, m+sd, color=palette[idx], alpha = 0.3)

    b.set_xlabel(" ",fontsize=15)
    b.set_ylabel("sI position",fontsize=15)
    b.spines['top'].set_visible(False)
    b.spines['right'].set_visible(False)
    b.tick_params(labelsize=12)
    if 'vel' not in featureName:
        b.set_yticks([0.4,0.6, 0.8, 1.0])
        b.set_ylim([.35, 1.05])
    else:
        b.set_yticks([0,0.2,0.4,0.6, 0.8, 1.0])
        b.set_ylim([-.05, 1.05])
    plt.tight_layout()
    plt.suptitle(featureName)
    plt.savefig(os.path.join(dataDir,f'SI_{featureName}.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(dataDir,f'SI_{featureName}.png'), dpi = 400,bbox_inches="tight")


#plot for all perc
dataDir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(dataDir, 'sI_perc_dict.pkl')
miceList = list(sIDict.keys())
for idx in range(6):
    fig, ax = plt.subplots(1, 5, figsize=(18,6))
    for feat_idx, featureName in enumerate(['pos','dir','(pos,dir)', 'vel', 'globalTime']): #,'session']:
        SIList = list()
        mouseList = list()
        layerList = list()
        for mouse in miceList:
            SIList.append(sIDict[mouse]['clean_traces']['results'][featureName]['sI'][idx])
            SIList.append(sIDict[mouse]['umap']['results'][featureName]['sI'][idx])
            mouseList = mouseList + [mouse]*2
            if mouse in deepMice:
                layerList = layerList + ['deep']*2
            elif mouse in supMice:
                layerList = layerList + ['sup']*2

        feature_list = ['og','umap']*len(miceList)
        SIPD = pd.DataFrame(data={'mouse': mouseList,
                                         'SI': SIList,
                                         'layer': layerList,
                                         'method': feature_list})

        palette= ['#8a8a8aff', '#5fc010ff', '#e08b10ff', '#e01110ff']
        b = sns.boxplot(x='method', y='SI', data=SIPD, hue = 'layer',
                    palette = palette, linewidth = 1, width= .5, ax = ax[feat_idx])

        sns.swarmplot(x='method', y='SI', data=SIPD, hue= 'layer',
                    palette = 'dark:gray', edgecolor = 'gray', ax = ax[feat_idx])

        # for idx in range(4):
        #     x_space = [-.25+idx, 0.25+idx]
        #     m = np.mean(sSI_array[:,idx])
        #     sd = np.std(sSI_array[:,idx])
        #     ax.plot(x_space, [m,m], linestyle='--', color=palette[idx])
        #     ax.fill_between(x_space, m-sd, m+sd, color=palette[idx], alpha = 0.3)

        b.set_xlabel(" ",fontsize=15)
        b.set_ylabel(f"sI {featureName} - NN perc: {100*sIDict[mouse]['clean_traces']['results'][featureName]['nnPercList'][idx]}",fontsize=15)
        b.spines['top'].set_visible(False)
        b.spines['right'].set_visible(False)
        b.tick_params(labelsize=12)
        b.set_yticks([0,0.2,0.4,0.6, 0.8, 1.0])
        b.set_ylim([-.05, 1.05])
        ax[feat_idx].set_title(featureName)

    plt.suptitle(f"NN perc: {100*sIDict[mouse]['clean_traces']['results'][featureName]['nnPercList'][idx]}")
    plt.tight_layout()
    plt.savefig(os.path.join(dataDir,f"SI_perc_{100*sIDict[mouse]['clean_traces']['results'][featureName]['nnPercList'][idx]}_all_feat.svg"), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(dataDir,f"SI_perc_{100*sIDict[mouse]['clean_traces']['results'][featureName]['nnPercList'][idx]}_all_feat.png"), dpi = 400,bbox_inches="tight")

#plot for all abs
dataDir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(dataDir, 'sI_abs_dict.pkl')
miceList = list(sIDict.keys())
for idx in range(6):
    fig, ax = plt.subplots(1, 5, figsize=(18,6))
    for feat_idx, featureName in enumerate(['pos','dir','(pos,dir)', 'vel', 'globalTime']): #,'session']:
        SIList = list()
        mouseList = list()
        layerList = list()
        for mouse in miceList:
            SIList.append(sIDict[mouse]['clean_traces']['results'][featureName]['sI'][idx])
            SIList.append(sIDict[mouse]['umap']['results'][featureName]['sI'][idx])
            mouseList = mouseList + [mouse]*2
            if mouse in deepMice:
                layerList = layerList + ['deep']*2
            elif mouse in supMice:
                layerList = layerList + ['sup']*2

        feature_list = ['og','umap']*len(miceList)
        SIPD = pd.DataFrame(data={'mouse': mouseList,
                                         'SI': SIList,
                                         'layer': layerList,
                                         'method': feature_list})

        palette= ['#8a8a8aff', '#5fc010ff', '#e08b10ff', '#e01110ff']
        b = sns.boxplot(x='method', y='SI', data=SIPD, hue = 'layer',
                    palette = palette, linewidth = 1, width= .5, ax = ax[feat_idx])

        sns.swarmplot(x='method', y='SI', data=SIPD, hue= 'layer',
                    palette = 'dark:gray', edgecolor = 'gray', ax = ax[feat_idx])

        # for idx in range(4):
        #     x_space = [-.25+idx, 0.25+idx]
        #     m = np.mean(sSI_array[:,idx])
        #     sd = np.std(sSI_array[:,idx])
        #     ax.plot(x_space, [m,m], linestyle='--', color=palette[idx])
        #     ax.fill_between(x_space, m-sd, m+sd, color=palette[idx], alpha = 0.3)

        b.set_xlabel(" ",fontsize=15)
        b.set_ylabel(f"sI {featureName} - NN: {sIDict[mouse]['clean_traces']['results'][featureName]['nnList'][idx]}",fontsize=15)
        b.spines['top'].set_visible(False)
        b.spines['right'].set_visible(False)
        b.tick_params(labelsize=12)
        b.set_yticks([0,0.2,0.4,0.6, 0.8, 1.0])
        b.set_ylim([-.05, 1.05])
        ax[feat_idx].set_title(featureName)

    plt.suptitle(f"NN abs: {sIDict[mouse]['clean_traces']['results'][featureName]['nnList'][idx]}")
    plt.tight_layout()
    plt.savefig(os.path.join(dataDir,f"SI_abs_{sIDict[mouse]['clean_traces']['results'][featureName]['nnList'][idx]}_all_feat.svg"), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(dataDir,f"SI_abs_{sIDict[mouse]['clean_traces']['results'][featureName]['nnList'][idx]}_all_feat.png"), dpi = 400,bbox_inches="tight")

#plot ON 0.5 PERC
dataDir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(dataDir, 'sI_perc_dict.pkl')
miceList = list(sIDict.keys())
percIdx = 2
for featureName in ['pos','dir','(pos,dir)', 'vel', 'globalTime', 'trial', 'trialTime']: #,'session']:
    perc = 100*sIDict['GC2']['clean_traces']['results'][featureName]['nnPercList'][percIdx]
    SIList = list()
    mouseList = list()
    layerList = list()
    for mouse in miceList:
        SIList.append(sIDict[mouse]['clean_traces']['results'][featureName]['sI'][percIdx])
        SIList.append(sIDict[mouse]['umap']['results'][featureName]['sI'][percIdx])
        # sSIList.append(np.percentile(sIDict[mouse]['clean_traces'][featureName]['ssI'], 99))
        # sSIList.append(np.percentile(sIDict[mouse]['umap'][featureName]['ssI'], 99))
        # sSIList.append(np.percentile(sIDict[mouse]['isomap'][featureName]['ssI'], 99))
        # sSIList.append(np.percentile(sIDict[mouse]['pca'][featureName]['ssI'], 99))

        mouseList = mouseList + [mouse]*2
        if mouse in deepMice:
            layerList = layerList + ['deep']*2
        elif mouse in supMice:
            layerList = layerList + ['sup']*2

    feature_list = ['og','umap']*len(miceList)
    SIPD = pd.DataFrame(data={'mouse': mouseList,
                                     'SI': SIList,
                                     'layer': layerList,
                                     'method': feature_list})

    palette= ['#8a8a8aff', '#5fc010ff', '#e08b10ff', '#e01110ff']
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    b = sns.boxplot(x='method', y='SI', data=SIPD, hue = 'layer',
                palette = palette, linewidth = 1, width= .5, ax = ax)

    sns.swarmplot(x='method', y='SI', data=SIPD, hue= 'layer',
                palette = 'dark:gray', edgecolor = 'gray', ax = ax)

    # for idx in range(4):
    #     x_space = [-.25+idx, 0.25+idx]
    #     m = np.mean(sSI_array[:,idx])
    #     sd = np.std(sSI_array[:,idx])
    #     ax.plot(x_space, [m,m], linestyle='--', color=palette[idx])
    #     ax.fill_between(x_space, m-sd, m+sd, color=palette[idx], alpha = 0.3)

    b.set_xlabel(" ",fontsize=15)
    b.set_ylabel(f"sI {featureName} - NN perc: {perc}",fontsize=15)
    b.spines['top'].set_visible(False)
    b.spines['right'].set_visible(False)
    b.tick_params(labelsize=12)
    b.set_yticks([0,0.2,0.4,0.6, 0.8, 1.0])
    b.set_ylim([-.05, 1.05])
    plt.tight_layout()
    plt.suptitle(featureName)
    plt.savefig(os.path.join(dataDir,f'SI_perc_{perc}_{featureName}.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(dataDir,f'SI_perc_{perc}_{featureName}.png'), dpi = 400,bbox_inches="tight")


import scipy.stats as stats

dataDir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(dataDir, 'sI_perc_dict.pkl')
miceList = list(sIDict.keys())

percIdx = 2
for featureName in ['pos','dir','(pos,dir)', 'vel', 'globalTime']:

    SI_og_deep = []
    SI_umap_deep = []
    SI_og_sup = []
    SI_umap_sup = []
    for mouse in miceList:
        if mouse in deepMice:
            SI_og_deep.append(sIDict[mouse]['clean_traces']['results'][featureName]['sI'][percIdx])
            SI_umap_deep.append(sIDict[mouse]['umap']['results'][featureName]['sI'][percIdx])
        elif mouse in supMice:
            SI_og_sup.append(sIDict[mouse]['clean_traces']['results'][featureName]['sI'][percIdx])
            SI_umap_sup.append(sIDict[mouse]['umap']['results'][featureName]['sI'][percIdx])
        else:
            print(f"{mouse} not in deep nor sup lists")

    print(f'\n--------------{featureName}--------------')
    print('Og: Deep vs Sup', stats.ttest_ind(SI_og_deep, SI_og_sup))
    print('Umap: Deep vs Sup', stats.ttest_ind(SI_umap_deep, SI_umap_sup))



#__________________________________________________________________________
#|                                                                        |#
#|                             ECCENTRICITY                               |#
#|________________________________________________________________________|#
from scipy.stats import shapiro

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ7', 'GC7','ChZ8','CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']

dataDir =  '/home/julio/Documents/SP_project/Fig2/eccentricity/'
ellipseDict = load_pickle(dataDir, 'ellipse_fit_dict.pkl')

eccenList = list()
mouseList = list()
layerList = list()
for mouse in miceList:
    # eccenList.append(ellipseDict[mouse]['eccentricity'])
    posLength = ellipseDict[mouse]['posLength']
    dirLength = ellipseDict[mouse]['dirLength']
    eccenList.append(100*(posLength-dirLength)/(posLength))

    mouseList.append(mouse)
    if mouse in deepMice:
        layerList.append('deep')
    elif mouse in supMice:
        layerList.append('sup')

eccenPD = pd.DataFrame(data={'mouse': mouseList,
                     'eccentricity': eccenList,
                     'layer': layerList})    
palette= ["#cc9900ff", "#9900ffff"]

fig, ax = plt.subplots(1, 1, figsize=(6,6))

b = sns.boxplot(x='layer', y='eccentricity', data=eccenPD,
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='layer', y='eccentricity', data=eccenPD,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)

deepEccent = [eccenList[idx] for idx in range(len(eccenList)) if mouseList[idx] in deepMice]
supEccent = [eccenList[idx] for idx in range(len(eccenList)) if mouseList[idx] in supMice]
deepShapiro = shapiro(deepEccent)
supShapiro = shapiro(supEccent)

if deepShapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
    print('eccentricity:',stats.ks_2samp(deepEccent, supEccent))
    ax.set_title(f'ks_2samp: {stats.ks_2samp(deepEccent, supEccent).pvalue:.4f}')
else:
    print('eccentricity:', stats.ttest_ind(deepEccent, supEccent))
    ax.set_title(f'ttest: {stats.ttest_ind(deepEccent, supEccent).pvalue:.4f}')

ax.set_ylim([-12, 70])
plt.savefig(os.path.join(dataDir,'DeepSup_eccentricity.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'DeepSup_eccentricity.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig2/decoders'
dec_R2s = load_pickle(data_dir, 'dec_R2s_dict.pkl')

palette= ['grey', '#5bb95bff', '#ff8a17ff', '#249aefff']
label_list = ['posx', 'posy','vel', 'index_mat', 'dir_mat']
signal_list = ['base_signal', 'pca', 'isomap', 'umap']
decoder_list = ['wf','wc','xgb','svr']
miceList = list(dec_R2s.keys())

for label_idx, label_name in enumerate(label_list):
    fig, ax = plt.subplots(1,4,figsize=(15,5))
    for dec_idx, dec_name in enumerate(decoder_list):
        R2s_list = list()
        pd_mice_list = list()
        pd_sig_list = list()
        pd_layer_list = list()
        for mouse in miceList:
            for sig in signal_list:
                R2s_list.append(np.mean(dec_R2s[mouse][sig][dec_name][:,label_idx,0], axis=0))
                pd_mice_list.append(mouse)
                pd_sig_list.append(sig)
                if mouse in deepMice:
                    pd_layer_list.append('deep')
                elif mouse in supMice:
                    pd_layer_list.append('sup')

        pd_R2s = pd.DataFrame(data={'mouse': pd_mice_list,
                                     'R2s': R2s_list,
                                     'signal': pd_sig_list,
                                     'layer': pd_layer_list})

        b = sns.barplot(x='signal', y='R2s', data=pd_R2s, hue='layer',
                palette = palette, linewidth = 1, width= .5, ax = ax[dec_idx])
        sns.swarmplot(x='signal', y='R2s', data=pd_R2s, hue='layer',
                    palette = 'dark:gray', edgecolor = 'gray', ax = ax[dec_idx])

        # sns.lineplot(x = 'signal', y= 'R2s', data = pd_R2s, units = 'mouse',
        #             ax = ax[dec_idx], estimator = None, color = ".7", markers = True)

        ax[dec_idx].set_ylabel(f'R2s {label_name}')
        ax[dec_idx].set_title(dec_name)

    fig.suptitle(label_name)
    plt.savefig(os.path.join(data_dir,f'dec_{label_name}_test.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(data_dir,f'dec_{label_name}_test.png'), dpi = 400,bbox_inches="tight")




from sklearn.metrics import median_absolute_error

data_dir = '/home/julio/Documents/SP_project/Fig2/decoders'
dec_R2s = load_pickle(data_dir, 'dec_R2s_dict.pkl')
dec_pred = load_pickle(data_dir, 'dec_pred_dict.pkl')
miceList = list(dec_R2s.keys())
izq = ['GC2', 'GC3', 'CZ3']
label_name = 'posx'
label_idx = 0
dec_name = 'xgb'
dec_idx = 2
miceList = list(dec_R2s.keys())

n_bins = 12

error_bin_sup = np.zeros((len(supMice),n_bins,2))
error_bin_deep = np.zeros((len(deepMice),n_bins,2))
count_bin_sup = np.zeros((len(supMice),n_bins))
count_bin_deep = np.zeros((len(deepMice),n_bins))
count_bin_pred_sup = np.zeros((len(supMice),n_bins,2))
count_bin_pred_deep = np.zeros((len(deepMice),n_bins,2))


for mouse in miceList:

    test = dec_pred[mouse][label_idx][dec_idx][:,:,0].reshape(-1,1) == 1
    ground_truth = dec_pred[mouse][label_idx][dec_idx][:,:,1].reshape(-1,1)[test]
    og_pred = dec_pred[mouse][label_idx][dec_idx][:,:,2].reshape(-1,1)[test]
    umap_pred = dec_pred[mouse][label_idx][dec_idx][:,:,-1].reshape(-1,1)[test]


    min_x = np.min(ground_truth)
    max_x = np.max(ground_truth)
    ndims = 1
    grid_edges = list()
    steps = (max_x - min_x)/n_bins
    edges = np.linspace(min_x, max_x, n_bins+1).reshape(-1,1)
    grid_edges = np.concatenate((edges[:-1], edges[1:]), axis = 1)

    #Generate the grid containing the indices of the points of the label and 
    #the coordinates as the mid point of edges

    grid = np.empty([grid_edges.shape[0]], object)
    mesh = meshgrid2(tuple([np.arange(s) for s in grid.shape]))
    meshIdx = np.vstack([col.ravel() for col in mesh]).T
    grid = grid.ravel()
    grid_pred = copy.deepcopy(grid)
    grid_pred_umap = copy.deepcopy(grid)
    for elem, idx in enumerate(meshIdx):
        logic = np.zeros(ground_truth.shape[0])
        min_edge = grid_edges[idx,0]
        max_edge = grid_edges[idx,1]
        logic = 1*np.logical_and(ground_truth>=min_edge,ground_truth<=max_edge)
        grid[elem] = list(np.where(logic == meshIdx.shape[1])[0])

        logic = 1*np.logical_and(og_pred>=min_edge,og_pred<=max_edge)
        grid_pred[elem] =  list(np.where(logic == meshIdx.shape[1])[0])

        logic = 1*np.logical_and(umap_pred>=min_edge,umap_pred<=max_edge)
        grid_pred_umap[elem] =  list(np.where(logic == meshIdx.shape[1])[0])

    for b in range(len(grid)):
        if mouse in deepMice:
            deep_idx = [x for x in range(len(deepMice)) if deepMice[x] == mouse][0]
            error_bin_deep[deep_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_deep[deep_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_deep[deep_idx,b] = len(grid[b])
            count_bin_pred_deep[deep_idx,b,0] = len(grid_pred[b])
            count_bin_pred_deep[deep_idx,b,1] = len(grid_pred_umap[b])
        else:
            sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
            error_bin_sup[sup_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_sup[sup_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_sup[sup_idx,b] = len(grid[b])
            count_bin_pred_sup[sup_idx,b,0] = len(grid_pred[b])
            count_bin_pred_sup[sup_idx,b,1] = len(grid_pred_umap[b])


for mouse in ['GC2', 'GC3', 'CZ3']:
    if mouse in deepMice:
        deep_idx = [x for x in range(len(deepMice)) if deepMice[x] == mouse][0]
        error_bin_deep[deep_idx,:,0] = error_bin_deep[deep_idx,::-1,0]
        error_bin_deep[deep_idx,:,1] = error_bin_deep[deep_idx,::-1,1]
        count_bin_deep[deep_idx,:] = count_bin_deep[deep_idx,::-1]
        count_bin_pred_deep[deep_idx,:,0] = count_bin_pred_deep[deep_idx,::-1,0]
        count_bin_pred_deep[deep_idx,:,1] = count_bin_pred_deep[deep_idx,::-1,0]



sup_color = '#9900ffff'
deep_color = '#cc9900ff'

bin_space = np.linspace(0,120,n_bins)

plt.figure()
ax = plt.subplot(2,1,1)
m = np.nanmean(error_bin_deep[:,:,0],axis=0)
sd = np.nanstd(error_bin_deep[:,:,0],axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
m = np.nanmean(error_bin_sup[:,:,0],axis=0)
sd = np.nanstd(error_bin_sup[:,:,0],axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Original space')
ax.set_ylabel('mean abs error posx')
ax.legend()

ax = plt.subplot(2,1,2)
m = np.nanmean(error_bin_deep[:,:,1],axis=0)
sd = np.nanstd(error_bin_deep[:,:,1],axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
m = np.nanmean(error_bin_sup[:,:,1],axis=0)
sd = np.nanstd(error_bin_sup[:,:,1],axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Umap space')
ax.set_ylabel('mean abs error posx')
ax.legend()

plt.savefig(os.path.join(data_dir,'DeepSup_error_by_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'DeepSup_error_by_pos.png'), dpi = 400,bbox_inches="tight")



plt.figure()
ax = plt.subplot(2,1,1)
vals = (count_bin_pred_deep[:,:,0] - count_bin_deep)/(count_bin_deep)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
vals = (count_bin_pred_sup[:,:,0] - count_bin_sup)/(count_bin_sup)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Original space')
ax.set_ylabel('norm change in prediction')
ax.legend()

ax = plt.subplot(2,1,2)
vals = (count_bin_pred_deep[:,:,1] - count_bin_deep)/(count_bin_deep)
vals = (count_bin_pred_deep[:,:,1] - count_bin_deep)/(count_bin_deep)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
vals = (count_bin_pred_sup[:,:,1] - count_bin_sup)/(count_bin_sup)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Umap space')
ax.set_ylabel('norm change in prediction')
ax.legend()
plt.savefig(os.path.join(data_dir,'DeepSup_pred_pos_by_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'DeepSup_pred_pos_by_pos.png'), dpi = 400,bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|                           MANIFOLD ACTIVITY                            |#
#|________________________________________________________________________|#

dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2/manifold_cells'

mouseExamples = {
    'deep': 'GC2',
    'sup': 'CZ3',
}

examplesValues = {
    'supIter': 2,
    'supAngle': [-95,-165],
    'supLims': [0.4, 0.65],
    'deepAngle': [25, -40],
    'deepIter': 1,
    'deepLims': [0.45, 0.55]
}
for case, mouse in mouseExamples.items():

    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis = 0))
    emb = copy.deepcopy(np.concatenate(pdMouse['umap'].values, axis = 0))

    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')
    neuPDF = pcDict['neu_pdf']
    # neuPDF = gaussian_filter1d(neuPDF, sigma = 2, axis = 0)
    normNeuPDF = np.zeros(neuPDF.shape)
    nCells = neuPDF.shape[1]
    for d in range(neuPDF.shape[2]):
        for c in range(nCells):
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.max(neuPDF[:,c,d])

    orderLeft =  np.argsort(np.argmax(normNeuPDF[:,:,0], axis=0))
    orderRight =  np.argsort(np.argmax(normNeuPDF[:,:,1], axis=0))
    meanNormNeuPDF = np.nanmean(normNeuPDF, axis=1)
    mapAxis = pcDict['mapAxis']
    manifoldSignal = np.zeros((emb.shape[0]))*np.nan
    for p in range(emb.shape[0]):
        try:
            x = np.where(mapAxis[0]<=pos[p,0])[0][-1]
        except: 
            x = 0
        dire = direction[p]
        if dire==0:
            manifoldSignal[p] = np.nan
        else:
            manifoldSignal[p] = meanNormNeuPDF[x,dire-1]

    for it in range(examplesValues[case+'Iter']):
        D= pairwise_distances(emb)
        noiseIdx = filter_noisy_outliers(emb,D)
        emb = emb[~noiseIdx,:]
        pos = pos[~noiseIdx,:]
        direction = direction[~noiseIdx]
        manifoldSignal = manifoldSignal[~noiseIdx]

    deleteNaN = np.where(np.isnan(manifoldSignal))
    manifoldSignal = np.delete(manifoldSignal, deleteNaN)
    emb = np.delete(emb, deleteNaN, axis=0)
    pos = np.delete(pos, deleteNaN, axis=0)
    direction = np.delete(direction, deleteNaN, axis=0)
    # manifoldSignal = gaussian_filter1d(manifoldSignal, sigma = 5, axis = 0)

    fig = plt.figure(figsize=(8,12))
    ax = plt.subplot(2,2,1)
    ax.matshow(normNeuPDF[:,orderLeft,0].T, aspect = 'auto')
    histSignalLeft = nCells - 0.5*nCells*(meanNormNeuPDF[:,0]/np.max(meanNormNeuPDF[:,0]))
    ax.plot(histSignalLeft, color = 'white', linewidth = 5)
    ax.set_title('izq')
    ax.set_ylabel('cell number')
    ax.set_xlabel('pos-x')

    ax = plt.subplot(2,2,2)
    ax.matshow(normNeuPDF[:,orderRight,1].T, aspect = 'auto')
    histSignalRight = nCells - 0.5*nCells*(meanNormNeuPDF[:,1]/np.max(meanNormNeuPDF[:,1]))
    ax.plot(histSignalRight, color = 'white', linewidth = 3)
    ax.set_title('dcha')
    ax.set_ylabel('cell number')
    ax.set_xlabel('pos-x')

    ax = plt.subplot(2,2,3, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0], s =10, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    personalize_ax(ax, examplesValues[case+'Angle'])

    ax = plt.subplot(2,2,4, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal, s = 10, vmin=examplesValues[case+'Lims'][0], vmax=examplesValues[case+'Lims'][1])
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    personalize_ax(ax, examplesValues[case+'Angle'])

    plt.tight_layout()
    fig.suptitle(mouse)
    plt.savefig(os.path.join(saveDir,f'example{case}_{mouse}_manifoldCells.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'example{case}_{mouse}_manifoldCells.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                           MANIFOLD ACTIVITY                            |#
#|________________________________________________________________________|#


supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ7', 'ChZ8']
izq = ['GC2', 'GC3', 'CZ3']

dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2/manifold_cells'

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1', 'ChZ7', 'ChZ8']


fig = plt.figure(figsize=(8,12))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

deepMat = np.zeros((20,len(deepMice)))*np.nan
deepMapAxis =  np.zeros((20,len(deepMice)))*np.nan
supMat = np.zeros((20,len(supMice)))*np.nan
supMapAxis =  np.zeros((20,len(supMice)))*np.nan

deepCount = 0
supCount = 0
for mouse in miceList:
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    # fnames = list(pdMouse.keys())
    # fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    # fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    # pdMouse= copy.deepcopy(pdMouse[fnamePre])
    signal = copy.deepcopy(np.concatenate(pdMouse['clean_traces'].values, axis = 0))
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis = 0))
    emb = copy.deepcopy(np.concatenate(pdMouse['umap'].values, axis = 0))

    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')#[fnamePre]
    neuPDF = pcDict['neu_pdf']
    mapAxis = pcDict['mapAxis']
    normNeuPDF = np.zeros(neuPDF.shape)
    nCells = neuPDF.shape[1]
    for d in range(neuPDF.shape[2]):
        for c in range(nCells):
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.sum(neuPDF[:,c,d])
    if mouse in izq:
        normNeuPDF = np.flipud(normNeuPDF)[:,[1,0]]
    meanNormNeuPDF = np.nanmean(np.nanmean(normNeuPDF,axis=2),axis=1)
        #np.squeeze(normNeuPDF.reshape(normNeuPDF.shape[0],-1,1)),axis=1)

    if mouse in deepMice:
        ax1.plot(mapAxis[0][:,0],meanNormNeuPDF, label=mouse)
        deepMat[:meanNormNeuPDF.shape[0], deepCount] = meanNormNeuPDF
        deepMapAxis[:meanNormNeuPDF.shape[0], deepCount] = mapAxis[0][:,0]
        deepCount+=1
    elif mouse in supMice:
        ax2.plot(mapAxis[0][:,0],meanNormNeuPDF, label=mouse)
        supMat[:meanNormNeuPDF.shape[0], supCount] = meanNormNeuPDF
        supMapAxis[:meanNormNeuPDF.shape[0], supCount] = mapAxis[0][:,0]
        supCount+=1

ax1.set_xlabel('pos-x')
ax1.set_ylabel('histogram')
ax2.set_xlabel('pos-x')
ax2.set_ylabel('histogram')
ax1.set_title('Deep')
ax2.set_title('Sup')
plt.tight_layout()

fig = plt.figure(figsize=(8,12))
ax = plt.subplot(1,1,1)
m = np.nanmean(deepMat[:12,:],axis=1)
sd = np.nanstd(deepMat[:12,:],axis=1)
ax.plot(np.nanmean(deepMapAxis[:12],axis=1),m, label = 'deep')
ax.fill_between(np.nanmean(deepMapAxis[:12],axis=1), m-sd, m+sd, alpha = 0.3)

m = np.nanmean(supMat[:12,:],axis=1)
sd = np.nanstd(supMat[:12,:],axis=1)
ax.plot(np.nanmean(supMapAxis[:12],axis=1),m, label = 'sup')
ax.fill_between(np.nanmean(supMapAxis[:12],axis=1), m-sd, m+sd, alpha = 0.3)
ax.legend()

plt.savefig(os.path.join(saveDir,'activity_pdf_manifold_v2.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,'activity_pdf_manifold_v2.png'), dpi = 400,bbox_inches="tight")
#__________________________________________________________________________
#|                                                                        |#
#|                       MANIFOLD ACTIVITY BY DIR                         |#
#|________________________________________________________________________|#

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ7', 'ChZ8']
izq = ['GC2', 'GC3', 'CZ3']

dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2/manifold_cells'

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1', 'ChZ7', 'ChZ8']
# miceList = ['GC5_nvista', 'TGrin1', 'ChZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
# miceList = ['GC2', 'GC3', 'CZ3']
deepMat = np.zeros((20,len(deepMice),2))*np.nan
deepMapAxis =  np.zeros((20,len(deepMice)))*np.nan
supMat = np.zeros((20,len(supMice),2))*np.nan
supMapAxis =  np.zeros((20,len(supMice)))*np.nan

deepCount = 0
supCount = 0
for mouse in miceList:
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    # fnames = list(pdMouse.keys())
    # fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    # fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    # pdMouse= copy.deepcopy(pdMouse[fnamePre])
    signal = copy.deepcopy(np.concatenate(pdMouse['clean_traces'].values, axis = 0))
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis = 0))
    emb = copy.deepcopy(np.concatenate(pdMouse['umap'].values, axis = 0))
    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')#[fnamePre]
    neuPDF = pcDict['neu_pdf']
    mapAxis = pcDict['mapAxis']
    normNeuPDF = np.zeros(neuPDF.shape)
    nCells = neuPDF.shape[1]
    for d in range(neuPDF.shape[2]):
        for c in range(nCells):
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.sum(neuPDF[:,c,d])
    meanNormNeuPDF = np.nanmean(normNeuPDF,axis=1)
    if mouse in izq:
        meanNormNeuPDF = np.flipud(meanNormNeuPDF)[:,[1,0]]
    if mouse in deepMice:
        deepMat[:meanNormNeuPDF.shape[0], deepCount,0] = meanNormNeuPDF[:,0]#/np.sum(meanNormNeuPDF[:,0])
        deepMat[:meanNormNeuPDF.shape[0], deepCount,1] = meanNormNeuPDF[:,1]#/np.sum(meanNormNeuPDF[:,1])

        deepMapAxis[:meanNormNeuPDF.shape[0], deepCount] = mapAxis[0][:,0]
        deepCount+=1
    elif mouse in supMice:
        supMat[:meanNormNeuPDF.shape[0], supCount,0] = meanNormNeuPDF[:,0]#/np.sum(meanNormNeuPDF[:,0])
        supMat[:meanNormNeuPDF.shape[0], supCount,1] = meanNormNeuPDF[:,1]#/np.sum(meanNormNeuPDF[:,1])
        supMapAxis[:meanNormNeuPDF.shape[0], supCount] = mapAxis[0][:,0]
        supCount+=1


fig = plt.figure(figsize=(8,12))
ax = plt.subplot(2,1,1)
m = np.nanmean(deepMat[:12,:,0],axis=1)
sd = np.nanstd(deepMat[:12,:,0],axis=1)
ax.plot(np.nanmean(deepMapAxis[:12],axis=1),m, label = 'deep')
ax.fill_between(np.nanmean(deepMapAxis[:12],axis=1), m-sd, m+sd, alpha = 0.3)

m = np.nanmean(supMat[:12,:,0],axis=1)
sd = np.nanstd(supMat[:12,:,0],axis=1)
ax.plot(np.nanmean(supMapAxis[:12],axis=1),m, label = 'sup')
ax.fill_between(np.nanmean(supMapAxis[:12],axis=1), m-sd, m+sd, alpha = 0.3)
ax.legend()
ax.set_title('desde')
# ax.set_ylim([0.18, 0.68])
ax = plt.subplot(2,1,2)
m = np.nanmean(deepMat[:12,:,1],axis=1)
sd = np.nanstd(deepMat[:12,:,1],axis=1)
ax.plot(np.nanmean(deepMapAxis[:12],axis=1),m, label = 'deep')
ax.fill_between(np.nanmean(deepMapAxis[:12],axis=1), m-sd, m+sd, alpha = 0.3)

m = np.nanmean(supMat[:12,:,1],axis=1)
sd = np.nanstd(supMat[:12,:,1],axis=1)
ax.plot(np.nanmean(supMapAxis[:12],axis=1),m, label = 'sup')
ax.fill_between(np.nanmean(supMapAxis[:12],axis=1), m-sd, m+sd, alpha = 0.3)
ax.legend()
ax.set_title('hacia')
# ax.set_ylim([0.18, 0.68])
plt.savefig(os.path.join(saveDir,'activity_pdf_manifold_dir_v2.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,'activity_pdf_manifold_dir_v2.png'), dpi = 400,bbox_inches="tight")

print('Left:', stats.ttest_ind(np.nanmean(deepMat[6:9,:,0],axis=0), np.nanmean(supMat[6:9,:,0],axis=0), equal_var=True))
print('Right:', stats.ttest_ind(np.nanmean(deepMat[6:9,:,1],axis=0), np.nanmean(supMat[6:9,:,1],axis=0), equal_var=True))

#__________________________________________________________________________
#|                                                                        |#
#|                           UMAP/ISOMAP/PCA EMB                          |#
#|________________________________________________________________________|#


dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/emb_example/'

mouseExamples = {
    'deep': 'GC2',
    'sup': 'CZ3',
}
examplesValues = {
    'supIterIsomap': 2,
    'supAngleIsomap': [75,-95],
    'deepAngleIsomap': [69, 78],
    'deepIterIsomap': 1,
    
    'supIterPCA': 2,
    'supAnglePCA': [76,-50],
    'deepAnglePCA': [42, 90],
    'deepIterPCA': 1,

    'supIterUmap': 2,
    'supAngleUmap': [70,-130],
    'deepAngleUmap': [51, -68],
    'deepIterUmap': 1,
}

for case, mouse in mouseExamples.items():
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    for emb_method in ['Isomap', 'PCA']:
        pos = copy.deepcopy(np.concatenate(pdMouse ['pos'].values, axis=0))
        dir_mat = copy.deepcopy(np.concatenate(pdMouse ['dir_mat'].values, axis=0))
        time = np.arange(pos.shape[0])
        emb = copy.deepcopy(np.concatenate(pdMouse [emb_method.lower()].values, axis=0))

        for it in range(examplesValues[case+'Iter'+emb_method]):
            D= pairwise_distances(emb)
            noiseIdx = filter_noisy_outliers(emb,D)
            emb = emb[~noiseIdx,:]
            pos = pos[~noiseIdx,:]
            dir_mat = dir_mat[~noiseIdx]
            time = time[~noiseIdx]

        dir_color = np.zeros((dir_mat.shape[0],3))
        for point in range(dir_mat.shape[0]):
            if dir_mat[point]==0:
                dir_color[point] = [14/255,14/255,143/255]
            elif dir_mat[point]==1:
                dir_color[point] = [12/255,136/255,249/255]
            else:
                dir_color[point] = [17/255,219/255,224/255]

        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(1,1,1, projection = '3d')
        b = ax.scatter(*emb[0,:3].T, c = pos[0,0], cmap = 'magma',s = 10)
        personalize_ax(ax, examplesValues[case+'Angle'+emb_method])
        plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_{emb_method}_emb_empty.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_{emb_method}_emb_empty.png'), dpi = 400,bbox_inches="tight")

        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(1,1,1, projection = '3d')
        b = ax.scatter(*emb[:,:3].T, color = dir_color,s = 10)
        personalize_ax(ax, examplesValues[case+'Angle'+emb_method])
        plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_{emb_method}_emb_dir.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_{emb_method}_emb_dir.png'), dpi = 400,bbox_inches="tight")

        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(1,1,1, projection = '3d')
        b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 10, cmap = 'inferno', vmin= 0, vmax = 70)
        personalize_ax(ax, examplesValues[case+'Angle'+emb_method])
        plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_{emb_method}_emb_pos.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_{emb_method}_emb_pos.png'), dpi = 400,bbox_inches="tight")

        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(1,1,1, projection = '3d')
        b = ax.scatter(*emb[:,:3].T, c = time[:],s = 10, cmap = 'YlGn_r', vmax = np.percentile(time, 95))
        personalize_ax(ax, examplesValues[case+'Angle'+emb_method])
        plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_{emb_method}_emb_time.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_{emb_method}_emb_time.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                      MANIFOLD ACTIVITY UMAP EMB                        |#
#|________________________________________________________________________|#


dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2/manifold_cells'

mouseExamples = {
    'deep': 'GC2',
    'sup': 'CZ3',
}

examplesValues = {
    'supIter': 2,
    'supAngle': [-95,-165],
    'deepAngle': [25, -40],
    'deepIter': 1,
}
for case, mouse in mouseExamples.items():

    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis = 0))
    emb = copy.deepcopy(np.concatenate(pdMouse['umap'].values, axis = 0))

    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')
    neuPDF = pcDict['neu_pdf']
    # neuPDF = gaussian_filter1d(neuPDF, sigma = 2, axis = 0)
    normNeuPDF = np.zeros(neuPDF.shape)
    nCells = neuPDF.shape[1]
    for d in range(neuPDF.shape[2]):
        for c in range(nCells):
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.sum(neuPDF[:,c,d])

    orderLeft =  np.argsort(np.argmax(normNeuPDF[:,:,0], axis=0))
    orderRight =  np.argsort(np.argmax(normNeuPDF[:,:,1], axis=0))
    meanNormNeuPDF = np.nanmean(normNeuPDF, axis=1)
    mapAxis = pcDict['mapAxis']
    manifoldSignal = np.zeros((emb.shape[0]))*np.nan
    for p in range(emb.shape[0]):
        try:
            x = np.where(mapAxis[0]<=pos[p,0])[0][-1]
        except: 
            x = 0
        dire = direction[p]
        if dire==0:
            manifoldSignal[p] = np.nan
        else:
            manifoldSignal[p] = meanNormNeuPDF[x,dire-1]

    for it in range(examplesValues[case+'Iter']):
        D= pairwise_distances(emb)
        noiseIdx = filter_noisy_outliers(emb,D)
        emb = emb[~noiseIdx,:]
        manifoldSignal = manifoldSignal[~noiseIdx]

    deleteNaN = np.where(np.isnan(manifoldSignal))
    manifoldSignal = np.delete(manifoldSignal, deleteNaN)
    emb = np.delete(emb, deleteNaN, axis=0)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 20)
    personalize_ax(ax, examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_manifoldCells_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_manifoldCells_emb.png'), dpi = 400,bbox_inches="tight")

dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2/manifold_cells'

mouseExamples = {
    'deep': 'GC2',
    'sup': 'CZ3',
}

examplesValues = {
    'supIter': 2,
    'supAngle': [-95,-165],
    'deepAngle': [25, -40],
    'deepIter': 1,
}

for case, mouse in mouseExamples.items():

    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis = 0))
    emb = copy.deepcopy(np.concatenate(pdMouse['umap'].values, axis = 0))
    signal = copy.deepcopy(np.concatenate(pdMouse['clean_traces'].values, axis = 0))

    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')
    neuPDF = pcDict['neu_pdf']
    # neuPDF = gaussian_filter1d(neuPDF, sigma = 2, axis = 0)
    normNeuPDF = np.zeros(neuPDF.shape)
    normSignal = np.zeros(signal.shape)
    nCells = neuPDF.shape[1]
    for d in range(neuPDF.shape[2]):
        for c in range(nCells):
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.sum(neuPDF[:,c,d])
            normSignal[:,c] = signal[:,c]/np.sum(signal[:,c])
    orderLeft =  np.argsort(np.argmax(normNeuPDF[:,:,0], axis=0))
    orderRight =  np.argsort(np.argmax(normNeuPDF[:,:,1], axis=0))
    meanNormNeuPDF = np.nanmean(normNeuPDF, axis=1)
    mapAxis = pcDict['mapAxis']
    manifoldSignal = np.zeros((emb.shape[0]))*np.nan
    for p in range(emb.shape[0]):
        manifoldSignal[p] = np.nanmean(normSignal[p,:])
        # try:
        #     x = np.where(mapAxis[0]<=pos[p,0])[0][-1]
        # except: 
        #     x = 0
        # dire = direction[p]
        # if dire==0:
        #     manifoldSignal[p] = np.nan
        # else:
        #     manifoldSignal[p] = meanNormNeuPDF[x,dire-1]
    for it in range(examplesValues[case+'Iter']):
        D= pairwise_distances(emb)
        noiseIdx = filter_noisy_outliers(emb,D)
        emb = emb[~noiseIdx,:]
        manifoldSignal = manifoldSignal[~noiseIdx]


    deleteNaN = np.where(np.isnan(manifoldSignal))
    manifoldSignal = np.delete(manifoldSignal, deleteNaN)
    emb = np.delete(emb, deleteNaN, axis=0)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 20, vmin=0.0001, vmax = 0.00025)#, vmin=0.04, vmax=0.08)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    personalize_ax(ax, examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_manifoldCells_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_manifoldCells_emb.png'), dpi = 400,bbox_inches="tight")

