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

supMice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']

thy1Mice = ['GC2','GC3','GC5_nvista', 'TGrin1','GC7']
chrna7Mice = ['ChZ4','ChZ7', 'ChZ8']

save_dir = '/home/julio/Documents/DeepSup_project/thy1_vs_chrna7/'
#__________________________________________________________________________
#|                                                                        |#
#|                                PLOT DIM                                |#
#|________________________________________________________________________|#
data_dir =  '/home/julio/Documents/SP_project/Fig2/dimensionality/'
innerDim = load_pickle(os.path.join(data_dir, 'inner_dim'), 'inner_dim_dict.pkl')
miceList = list(innerDim.keys())
dimList = list()
mouseList = list()
methodList = list()
layerList = list()
for mouse in miceList:
    dimList.append(innerDim[mouse]['momDim'])
    dimList.append(innerDim[mouse]['abidsDim'])
    dimList.append(innerDim[mouse]['tleDim'])
    if mouse in thy1Mice:
        layerList += ['thy1']*3
    elif mouse in supMice:
        layerList += ['sup']*3
    elif mouse in chrna7Mice:
        layerList += ['chrna7']*3
    mouseList = mouseList + [mouse]*3
    methodList = methodList+['mom', 'abids', 'tle']

palette= ["#ccccccff", "#808080ff", "#4d4d4dff"]
dimPD = pd.DataFrame(data={'mouse': mouseList,
                     'dim': dimList,
                     'method': methodList,
                     'layer': layerList})    
fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='layer', y='dim', data=dimPD, hue='method',
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylim([0,4.5])
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'dimensionality','inner_dim.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dimensionality','inner_dim.png'), dpi = 400,bbox_inches="tight")


umapDim = load_pickle(data_dir, 'umap_dim_dict.pkl')
isoDim = load_pickle(data_dir, 'isomap_dim_dict.pkl')

dimList = list()
mouseList = list()
methodList = list()
layerList = list()
for mouse in miceList:
    dimList.append(umapDim[mouse]['trustDim'])
    dimList.append(umapDim[mouse]['contDim'])
    dimList.append(isoDim[mouse]['resVarDim'])
    dimList.append(isoDim[mouse]['recErrorDim'])

    if mouse in thy1Mice:
        layerList += ['thy1']*4
    elif mouse in supMice:
        layerList += ['sup']*4
    elif mouse in chrna7Mice:
        layerList += ['chrna7']*4
    mouseList = mouseList + [mouse]*4
    methodList = methodList+['umap_trust', 'umap_cont', 
                        'iso_res_var', 'iso_rec_error']

palette= ["#5ff444ff", "#5fc010ff", "#e09e38ff", "#e08b10ff"]

dimPD = pd.DataFrame(data={'mouse': mouseList,
                     'dim': dimList,
                     'method': methodList,
                     'layer': layerList})    

fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='layer', y='dim', data=dimPD, hue='method',
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylim([0,7.5])
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'dimensionality','umap_iso_dim.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dimensionality','umap_iso_dim.png'), dpi = 400,bbox_inches="tight")

pcaDim = load_pickle(data_dir, 'pca_dim_dict.pkl')
dimList = list()
mouseList = list()
methodList = list()
layerList = list()
for mouse in miceList:
    dimList.append(pcaDim[mouse]['var80Dim'])
    dimList.append(pcaDim[mouse]['kneeDim'])
    if mouse in thy1Mice:
        layerList += ['thy1']*2
    elif mouse in supMice:
        layerList += ['sup']*2
    elif mouse in chrna7Mice:
        layerList += ['chrna7']*2
    mouseList = mouseList + [mouse]*2
    methodList = methodList+['pca_80', 'pca_knee']

palette= ["#e03d27ff", "#e01110ff"]
dimPD = pd.DataFrame(data={'mouse': mouseList,
                     'dim': dimList,
                     'method': methodList,
                     'layer': layerList})    
fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='layer', y='dim', data=dimPD, hue='method',
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylim([0,100])
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'dimensionality','pca_dim.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dimensionality','pca_dim.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                        PLOT SI DIM RED BOXPLOTS                        |#
#|________________________________________________________________________|#
#og clean study
data_dir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(data_dir, 'sI_clean_dict.pkl')
miceList = list(sIDict.keys())
for featureName in ['pos','dir','(pos_dir)', 'vel', 'time']: #,'session']:
    SIList = list()
    mouseList = list()
    layerList = list()
    for mouse in miceList:
        SIList.append(sIDict[mouse]['clean_traces'][featureName]['sI'])
        SIList.append(sIDict[mouse]['umap'][featureName]['sI'])
        mouseList = mouseList + [mouse]*2
        if mouse in thy1Mice:
            layerList += ['thy1']*2
        elif mouse in supMice:
            layerList += ['sup']*2
        elif mouse in chrna7Mice:
            layerList += ['chrna7']*2

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
    plt.savefig(os.path.join(save_dir,'SI',f'SI_{featureName}.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,'SI',f'SI_{featureName}.png'), dpi = 400,bbox_inches="tight")

#plot for all perc
data_dir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(data_dir, 'sI_perc_dict.pkl')
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
    plt.savefig(os.path.join(save_dir,'SI',f"SI_perc_{100*sIDict[mouse]['clean_traces']['results'][featureName]['nnPercList'][idx]}_all_feat.svg"), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,'SI',f"SI_perc_{100*sIDict[mouse]['clean_traces']['results'][featureName]['nnPercList'][idx]}_all_feat.png"), dpi = 400,bbox_inches="tight")

#plot for all abs
data_dir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(data_dir, 'sI_abs_dict.pkl')
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
    plt.savefig(os.path.join(save_dir,'SI',f"SI_abs_{sIDict[mouse]['clean_traces']['results'][featureName]['nnList'][idx]}_all_feat.svg"), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,'SI',f"SI_abs_{sIDict[mouse]['clean_traces']['results'][featureName]['nnList'][idx]}_all_feat.png"), dpi = 400,bbox_inches="tight")

#plot ON 0.5 PERC
data_dir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(data_dir, 'sI_perc_dict.pkl')
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
        if mouse in thy1Mice:
            layerList += ['thy1']*2
        elif mouse in supMice:
            layerList += ['sup']*2
        elif mouse in chrna7Mice:
            layerList += ['chrna7']*2

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
    plt.savefig(os.path.join(save_dir,'SI',f'SI_perc_{perc}_{featureName}.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,'SI',f'SI_perc_{perc}_{featureName}.png'), dpi = 400,bbox_inches="tight")


import scipy.stats as stats

data_dir = '/home/julio/Documents/SP_project/Fig2/SI/'
sIDict = load_pickle(data_dir, 'sI_perc_dict.pkl')
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

data_dir =  '/home/julio/Documents/SP_project/Fig2/eccentricity/'
ellipseDict = load_pickle(data_dir, 'ellipse_fit_dict.pkl')
miceList = list(ellipseDict.keys())
eccenList = list()
mouseList = list()
layerList = list()
for mouse in miceList:
    # eccenList.append(ellipseDict[mouse]['eccentricity'])
    posLength = ellipseDict[mouse]['posLength']
    dirLength = ellipseDict[mouse]['dirLength']
    eccenList.append(100*(posLength-dirLength)/(posLength))

    mouseList.append(mouse)
    if mouse in thy1Mice:
        layerList += ['thy1']
    elif mouse in supMice:
        layerList += ['sup']
    elif mouse in chrna7Mice:
        layerList += ['chrna7']

eccenPD = pd.DataFrame(data={'mouse': mouseList,
                     'eccentricity': eccenList,
                     'layer': layerList})   


fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='layer', y='eccentricity', data=eccenPD,
            linewidth = 1, width= .5, ax = ax, order= ['thy1', 'chrna7', 'sup'])
sns.swarmplot(x='layer', y='eccentricity', data=eccenPD,
        edgecolor = 'gray', ax = ax,order= ['thy1', 'chrna7', 'sup'])
ax.set_ylim([-12, 70])
plt.savefig(os.path.join(save_dir,'eccentricity','DeepSup_eccentricity.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'eccentricity','DeepSup_eccentricity.png'), dpi = 400,bbox_inches="tight")


thy1Eccent = [eccenList[idx] for idx in range(len(eccenList)) if mouseList[idx] in thy1Mice]
chrna7Eccent = [eccenList[idx] for idx in range(len(eccenList)) if mouseList[idx] in chrna7Mice]
supEccent = [eccenList[idx] for idx in range(len(eccenList)) if mouseList[idx] in supMice]

thy1Shapiro = stats.shapiro(thy1Eccent)
chrna7Shapiro = stats.shapiro(chrna7Eccent)
supShapiro = stats.shapiro(supEccent)

if thy1Shapiro.pvalue<=0.05 or chrna7Shapiro.pvalue<=0.05:
    print('Thy1 vs ChRNA7 eccentricity:',stats.ks_2samp(thy1Shapiro, chrna7Shapiro))
else:
    print('Thy1 vs ChRNA7 eccentricity:', stats.ttest_ind(thy1Shapiro, chrna7Shapiro))

if thy1Shapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
    print('Thy1 vs Sup eccentricity:',stats.ks_2samp(thy1Shapiro, supShapiro))
else:
    print('Thy1 vs Sup eccentricity:', stats.ttest_ind(thy1Shapiro, supShapiro))

if supShapiro.pvalue<=0.05 or chrna7Shapiro.pvalue<=0.05:
    print('Sup vs ChRNA7 eccentricity:',stats.ks_2samp(supShapiro, chrna7Shapiro))
else:
    print('Sup vs ChRNA7 eccentricity:', stats.ttest_ind(supShapiro, chrna7Shapiro))


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
                if mouse in thy1Mice:
                    pd_layer_list += ['thy1']
                elif mouse in supMice:
                    pd_layer_list += ['sup']
                elif mouse in chrna7Mice:
                    pd_layer_list += ['chrna7']

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
    plt.savefig(os.path.join(save_dir,'decoders',f'dec_{label_name}_test.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,'decoders',f'dec_{label_name}_test.png'), dpi = 400,bbox_inches="tight")

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
n_bins = 20

error_bin_thy1 = np.zeros((len(thy1Mice),n_bins,2))
count_bin_thy1 = np.zeros((len(thy1Mice),n_bins))
count_bin_pred_thy1 = np.zeros((len(thy1Mice),n_bins,2))

error_bin_chrna7 = np.zeros((len(chrna7Mice),n_bins,2))
count_bin_chrna7 = np.zeros((len(chrna7Mice),n_bins))
count_bin_pred_chrna7 = np.zeros((len(chrna7Mice),n_bins,2))

error_bin_sup = np.zeros((len(supMice),n_bins,2))
count_bin_sup = np.zeros((len(supMice),n_bins))
count_bin_pred_sup = np.zeros((len(supMice),n_bins,2))

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

    if mouse in thy1Mice:
        thy1_idx = [x for x in range(len(thy1Mice)) if thy1Mice[x] == mouse][0]
        for b in range(len(grid)):
            error_bin_thy1[thy1_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_thy1[thy1_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_thy1[thy1_idx,b] = len(grid[b])
            count_bin_pred_thy1[thy1_idx,b,0] = len(grid_pred[b])
            count_bin_pred_thy1[thy1_idx,b,1] = len(grid_pred_umap[b])
    elif mouse in chrna7Mice:
        chrna7_idx = [x for x in range(len(chrna7Mice)) if chrna7Mice[x] == mouse][0]
        for b in range(len(grid)):
            error_bin_chrna7[chrna7_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_chrna7[chrna7_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_chrna7[chrna7_idx,b] = len(grid[b])
            count_bin_pred_chrna7[chrna7_idx,b,0] = len(grid_pred[b])
            count_bin_pred_chrna7[chrna7_idx,b,1] = len(grid_pred_umap[b])
    else:
        sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
        for b in range(len(grid)):
            error_bin_sup[sup_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_sup[sup_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_sup[sup_idx,b] = len(grid[b])
            count_bin_pred_sup[sup_idx,b,0] = len(grid_pred[b])
            count_bin_pred_sup[sup_idx,b,1] = len(grid_pred_umap[b])

for mouse in ['GC2', 'GC3', 'CZ3']:
    if mouse in thy1Mice:
        thy1_idx = [x for x in range(len(thy1Mice)) if thy1Mice[x] == mouse][0]
        error_bin_thy1[thy1_idx,:,0] = error_bin_thy1[thy1_idx,::-1,0]
        error_bin_thy1[thy1_idx,:,1] = error_bin_thy1[thy1_idx,::-1,1]
        count_bin_thy1[thy1_idx,:] = count_bin_thy1[thy1_idx,::-1]
        count_bin_pred_thy1[thy1_idx,:,0] = count_bin_pred_thy1[thy1_idx,::-1,0]
        count_bin_pred_thy1[thy1_idx,:,1] = count_bin_pred_thy1[thy1_idx,::-1,0]
    elif mouse in supMice:
        sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
        error_bin_sup[sup_idx,:,0] = error_bin_sup[sup_idx,::-1,0]
        error_bin_sup[sup_idx,:,1] = error_bin_sup[sup_idx,::-1,1]
        count_bin_sup[sup_idx,:] = count_bin_sup[sup_idx,::-1]
        count_bin_pred_sup[sup_idx,:,0] = count_bin_pred_sup[sup_idx,::-1,0]
        count_bin_pred_sup[sup_idx,:,1] = count_bin_pred_sup[sup_idx,::-1,0]


sup_color = '#9900ffff'
thy1_color = '#cc9900ff'
chrna7_color = '#5fc010ff'
bin_space = np.linspace(0,75,n_bins)

plt.figure()
ax = plt.subplot(2,1,1)
m = np.nanmean(error_bin_thy1[:,:,0],axis=0)
sd = np.nanstd(error_bin_thy1[:,:,0],axis=0)
ax .plot(bin_space, m, color=thy1_color, linewidth= 2,label='thy1')
ax .fill_between(bin_space, m-sd, m+sd,color=thy1_color,alpha = 0.3)
m = np.nanmean(error_bin_chrna7[:,:,0],axis=0)
sd = np.nanstd(error_bin_chrna7[:,:,0],axis=0)
ax .plot(bin_space, m, color=chrna7_color, linewidth= 2,label='chrna7')
ax .fill_between(bin_space, m-sd, m+sd,color=chrna7_color,alpha = 0.3)
m = np.nanmean(error_bin_sup[:,:,0],axis=0)
sd = np.nanstd(error_bin_sup[:,:,0],axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Original space')
ax.set_ylabel('mean abs error posx')
ax.legend()

ax = plt.subplot(2,1,2)
m = np.nanmean(error_bin_thy1[:,:,1],axis=0)
sd = np.nanstd(error_bin_thy1[:,:,1],axis=0)
ax .plot(bin_space, m, color=thy1_color, linewidth= 2,label='thy1')
ax .fill_between(bin_space, m-sd, m+sd,color=thy1_color,alpha = 0.3)
m = np.nanmean(error_bin_chrna7[:,:,1],axis=0)
sd = np.nanstd(error_bin_chrna7[:,:,1],axis=0)
ax .plot(bin_space, m, color=chrna7_color, linewidth= 2,label='chrna7')
ax .fill_between(bin_space, m-sd, m+sd,color=chrna7_color,alpha = 0.3)
m = np.nanmean(error_bin_sup[:,:,1],axis=0)
sd = np.nanstd(error_bin_sup[:,:,1],axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Umap space')
ax.set_ylabel('mean abs error posx')
ax.legend()
plt.savefig(os.path.join(save_dir,'decoders','DeepSup_error_by_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'decoders','DeepSup_error_by_pos.png'), dpi = 400,bbox_inches="tight")



thy1R2s = np.mean(error_bin_thy1[:,12:14,1],axis=1)
chrna7R2s = np.mean(error_bin_chrna7[:,12:14,1],axis=1)
supR2s = np.mean(error_bin_sup[:,12:14,1],axis=1)


thy1R2s = error_bin_thy1[:,8,1]
chrna7R2s = error_bin_chrna7[:,8,1]
supR2s = error_bin_sup[:,8,1]


thy1Shapiro = stats.shapiro(thy1R2s)
chrna7Shapiro = stats.shapiro(chrna7R2s)
supShapiro = stats.shapiro(supR2s)

if thy1Shapiro.pvalue<=0.05 or chrna7Shapiro.pvalue<=0.05:
    print('Thy1 vs ChRNA7 R2s:',stats.ks_2samp(thy1Shapiro, chrna7Shapiro))
else:
    print('Thy1 vs ChRNA7 R2s:', stats.ttest_ind(thy1Shapiro, chrna7Shapiro))

if thy1Shapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
    print('Thy1 vs Sup R2s:',stats.ks_2samp(thy1Shapiro, supShapiro))
else:
    print('Thy1 vs Sup R2s:', stats.ttest_ind(thy1Shapiro, supShapiro))

if supShapiro.pvalue<=0.05 or chrna7Shapiro.pvalue<=0.05:
    print('Sup vs ChRNA7 R2:',stats.ks_2samp(supShapiro, chrna7Shapiro))
else:
    print('Sup vs ChRNA7 R2s:', stats.ttest_ind(supShapiro, chrna7Shapiro))

plt.figure()
ax = plt.subplot(2,1,1)
vals = (count_bin_pred_thy1[:,:,0] - count_bin_thy1)/(count_bin_thy1)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=thy1_color, linewidth= 2,label='thy1')
ax .fill_between(bin_space, m-sd, m+sd,color=thy1_color,alpha = 0.3)
vals = (count_bin_pred_chrna7[:,:,0] - count_bin_chrna7)/(count_bin_chrna7)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=chrna7_color, linewidth= 2,label='chrna7')
ax .fill_between(bin_space, m-sd, m+sd,color=chrna7_color,alpha = 0.3)
vals = (count_bin_pred_sup[:,:,0] - count_bin_sup)/(count_bin_sup)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Original space')
ax.set_ylabel('norm change in prediction')
ax.legend()

ax = plt.subplot(2,1,2)
vals = (count_bin_pred_thy1[:,:,1] - count_bin_thy1)/(count_bin_thy1)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=thy1_color, linewidth= 2,label='thy1')
ax .fill_between(bin_space, m-sd, m+sd,color=thy1_color,alpha = 0.3)
vals = (count_bin_pred_chrna7[:,:,1] - count_bin_chrna7)/(count_bin_chrna7)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=chrna7_color, linewidth= 2,label='chrna7')
ax .fill_between(bin_space, m-sd, m+sd,color=chrna7_color,alpha = 0.3)
vals = (count_bin_pred_sup[:,:,1] - count_bin_sup)/(count_bin_sup)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Umap space')
ax.set_ylabel('norm change in prediction')
ax.legend()
plt.savefig(os.path.join(save_dir,'decoders','DeepSup_pred_by_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'decoders','DeepSup_pred_by_pos.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                                 UMAP EMB                               |#
#|________________________________________________________________________|#

def personalize_ax(ax, ax_view = None):
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if ax_view:
        ax.view_init(ax_view[0], ax_view[1])


data_dir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/emb_example/'

mouseExamples = {
    'deep': 'GC2',
    'sup': 'CZ3',
}
examplesValues = {
    'supIter': 2,
    'supAngle': [70,-130],
    #'deepAngle': [25, -40],
    'deepAngle': [51, -68],
    'deepIter': 1,
}

for case, mouse in mouseExamples.items():
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(data_dir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    pos = copy.deepcopy(np.concatenate(pdMouse ['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse ['dir_mat'].values, axis=0))
    time = np.arange(pos.shape[0])
    emb = copy.deepcopy(np.concatenate(pdMouse ['umap'].values, axis=0))

    for it in range(examplesValues[case+'Iter']):
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
    b = ax.scatter(*emb[:,:3].T, color = dir_color,s = 20)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    personalize_ax(ax,examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umap_emb_dir.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umap_emb_dir.png'), dpi = 400,bbox_inches="tight")

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 20, cmap = 'inferno', vmin= 0, vmax = 70)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    personalize_ax(ax,examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umap_emb_pos.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umap_emb_pos.png'), dpi = 400,bbox_inches="tight")

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = time[:],s = 20, cmap = 'YlGn_r', vmax = np.percentile(time, 95))
    personalize_ax(ax,examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umap_emb_time.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umap_emb_time.png'), dpi = 400,bbox_inches="tight")
