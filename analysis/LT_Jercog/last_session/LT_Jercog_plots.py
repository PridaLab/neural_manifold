import seaborn as sns
import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import neural_manifold.general_utils as gu

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data
import networkx as nx
from structure_index import compute_structure_index, draw_graph

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx

from sklearn.metrics import pairwise_distances
#__________________________________________________________________________
#|                                                                        |#
#|                           PLOT SI OG BOXPLOTS                          |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig1/SI'
sI_dict = load_pickle(data_dir, 'sI_clean_dict.pkl')
miceList = list(sI_dict.keys())
SI_list = list()
sSI_list = list()
mouse_list = list()
for mouse in miceList:
    SI_list.append(sI_dict[mouse]['clean_traces']['pos']['sI'])
    SI_list.append(sI_dict[mouse]['clean_traces']['dir']['sI'])
    SI_list.append(sI_dict[mouse]['clean_traces']['(pos_dir)']['sI'])
    SI_list.append(sI_dict[mouse]['clean_traces']['vel']['sI'])
    SI_list.append(sI_dict[mouse]['clean_traces']['time']['sI'])

    sSI_list.append(np.percentile(sI_dict[mouse]['clean_traces']['pos']['ssI'], 99))
    sSI_list.append(np.percentile(sI_dict[mouse]['clean_traces']['dir']['ssI'], 99))
    sSI_list.append(np.percentile(sI_dict[mouse]['clean_traces']['(pos_dir)']['ssI'], 99))
    sSI_list.append(np.percentile(sI_dict[mouse]['clean_traces']['vel']['ssI'], 99))
    sSI_list.append(np.percentile(sI_dict[mouse]['clean_traces']['time']['ssI'], 99))

    mouse_list = mouse_list + [mouse]*5
feature_list = ['pos', 'dir', '(pos_dir)', 'vel', 'time']*len(miceList)
sSI_array = np.array(sSI_list).reshape(-1,5)

pd_SI = pd.DataFrame(data={'mouse': mouse_list,
                                 'SI': SI_list,
                                 'feature': feature_list})

palette= ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='feature', y='SI', data=pd_SI,
            palette = palette, linewidth = 1, width= .5, ax = ax)

sns.swarmplot(x='feature', y='SI', data=pd_SI, 
            color = 'gray', edgecolor = 'gray', ax = ax)

# sns.lineplot(x = 'feature', y= 'SI', data = pd_SI, units = 'mouse',
#             ax = ax, estimator = None, color = ".7", markers = True)

for idx in range(5):
    x_space = [-.25+idx, 0.25+idx]
    m = np.mean(sSI_array[:,idx])
    sd = np.std(sSI_array[:,idx])
    ax.plot(x_space, [m,m], linestyle='--', color=palette[idx])
    ax.fill_between(x_space, m-sd, m+sd, color=palette[idx], alpha = 0.3)

b.set_xlabel(" ",fontsize=15)
b.set_ylabel("SI",fontsize=15)
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)
b.tick_params(labelsize=12)
b.set_yticks([0, .25, .5, .75, 1])
b.set_ylim([-0.05, 1.1])
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'SI_og.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'SI_og.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT SI OG GRAPHS                           |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig1/SI'
sI_dict = load_pickle(data_dir, 'sI_clean_dict.pkl')

eg = 'M2019'
f, ax = plt.subplots(1, 3, figsize=(18,5))
overlapMat = sI_dict[eg]['clean_traces']['pos']['overlapMat']
overlapMat[np.isnan(overlapMat)] = 0
binLabel = sI_dict[eg]['clean_traces']['pos']['binLabel']
draw_graph(overlapMat, ax[0],node_cmap=plt.cm.inferno,edge_cmap=plt.cm.Greys, scale_edges=5,layout_type  = nx.spring_layout, 
                        arrow_size=10,node_names = np.round(binLabel[1][:,0,1],2),node_size=400);
ax[0].set_xlim(1.2*np.array(ax[0].get_xlim()));
ax[0].set_ylim(1.2*np.array(ax[0].get_ylim()));
ax[0].set_title('Directed graph', size=16);
ax[0].text(0.98, 0.05, f"SI pos: {sI_dict[eg]['clean_traces']['pos']['sI']:.2f}", horizontalalignment='right',
    verticalalignment='bottom', transform=ax[0].transAxes,fontsize=25)


overlapMat = sI_dict[eg]['clean_traces']['dir']['overlapMat']
overlapMat[np.isnan(overlapMat)] = 0
binLabel = sI_dict[eg]['clean_traces']['dir']['binLabel']
draw_graph(overlapMat, ax[1],node_cmap=plt.cm.Accent,edge_cmap=plt.cm.Greys, scale_edges=10,
                        layout_type  = nx.spring_layout, 
                        arrow_size=10,node_names = np.round(binLabel[1][:,0,1],2),node_size=400);
ax[1].set_xlim(1.2*np.array(ax[1].get_xlim()));
ax[1].set_ylim(1.2*np.array(ax[1].get_ylim()));
ax[1].set_title('Directed graph', size=16);
ax[1].text(0.98, 0.05, f"SI dir: {sI_dict[eg]['clean_traces']['dir']['sI']:.2f}", horizontalalignment='right',
    verticalalignment='bottom', transform=ax[1].transAxes,fontsize=25)

overlapMat = sI_dict[eg]['clean_traces']['(pos_dir)']['overlapMat']
overlapMat[np.isnan(overlapMat)] = 0
# overlapMat[overlapMat<np.percentile(overlapMat,70)] = 0
binLabel = sI_dict[eg]['clean_traces']['(pos_dir)']['binLabel']
# idx = np.unique(binLabel[0])[:-1].astype(int)
validBins = [0,1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,27,28,29]
binLabel = binLabel[1][validBins,:,:]
# new_order = [0,1,3,4,6, 8,10,12,14,16,19,18,22,23,21,20,17,15,13,11,9,7,5,2]
# new_order = [0,1,2,4,6, 8,10,12,14,16,19,18,21,20,17,15,13,11,9,7,5,3]
# new_order = [0,1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,27,28,29]

new_order = [0,1,3,5,7,9,11,13,15,17,20,19,21,18,16,14,12,10,8,6,4,2]
overlapMat = overlapMat[new_order, :]
overlapMat = overlapMat[:, new_order]
binLabel = binLabel[new_order,:,:]
draw_graph(overlapMat, ax[2],node_cmap=plt.cm.inferno,edge_cmap=plt.cm.Greys, scale_edges=5,layout_type  = nx.circular_layout, 
                        arrow_size=10, node_names = np.round(binLabel[:,0,1],2),edge_vmin=0.1, node_size=400);
ax[2].set_xlim(1.2*np.array(ax[2].get_xlim()));
ax[2].set_ylim(1.2*np.array(ax[2].get_ylim()));
ax[2].set_title('Directed graph', size=16);
ax[2].text(0.98, 0.05, f"SI posdir: {sI_dict[eg]['clean_traces']['(pos_dir)']['sI']:.2f}", horizontalalignment='right',
    verticalalignment='bottom', transform=ax[2].transAxes,fontsize=25)
plt.tight_layout()
plt.savefig(os.path.join(data_dir,f'{eg}_cleanSI_og_graph1.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,f'{eg}_cleanSI_og_graph1.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                                PLOT DIM                                |#
#|________________________________________________________________________|#

miceList = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']

data_dir = '/home/julio/Documents/SP_project/Fig1/dimensionality/inner_dim/'
inner_dim = load_pickle(data_dir, 'inner_dim_dict.pkl')

data_dir = '/home/julio/Documents/SP_project/Fig1/dimensionality/'
iso_dim = load_pickle(data_dir, 'isomap_dim_dict.pkl')
pca_dim = load_pickle(data_dir, 'pca_dim_dict.pkl')
umap_dim = load_pickle(data_dir, 'umap_dim_dict.pkl')

dim_list = list()
mouse_list = list()
for mouse in miceList:
    dim_list.append(inner_dim[mouse]['abidsDim'])

    dim_list.append(umap_dim[mouse]['trustDim'])
    dim_list.append(umap_dim[mouse]['contDim'])

    dim_list.append(iso_dim[mouse]['resVarDim'])
    dim_list.append(iso_dim[mouse]['recErrorDim'])

    dim_list.append(pca_dim[mouse]['var80Dim'])
    dim_list.append(pca_dim[mouse]['kneeDim'])

    mouse_list = mouse_list + [mouse]*7

method_list = ['abids', 'umap_trust', 'umap_cont', 'iso_res_var', 
                'iso_rec_error', 'pca_80', 'pca_knee']*len(miceList)

palette= ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]
pd_dim = pd.DataFrame(data={'mouse': mouse_list,
                     'dim': dim_list,
                     'method': method_list})    

fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='method', y='dim', data=pd_dim,
            palette = palette, linewidth = 1, width= .5, ax = ax)

sns.swarmplot(x='method', y='dim', data=pd_dim, 
            color = 'gray', edgecolor = 'gray', ax = ax)

sns.lineplot(x = 'method', y= 'dim', data = pd_dim, units = 'mouse',
            ax = ax, estimator = None, color = ".7", markers = True)

b.tick_params(labelsize=12)
b.set_ylim([0, 120])
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'dim_barplot_all.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'dim_barplot_all.png'), dpi = 400,bbox_inches="tight")

fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='method', y='dim', data=pd_dim,
            palette = palette, linewidth = 1, width= .5, ax = ax)

sns.swarmplot(x='method', y='dim', data=pd_dim, 
            color = 'gray', edgecolor = 'gray', ax = ax)

sns.lineplot(x = 'method', y= 'dim', data = pd_dim, units = 'mouse',
            ax = ax, estimator = None, color = ".7", markers = True)

b.tick_params(labelsize=12)
b.set_ylim([0, 5])
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'dim_barplot_zoom.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'dim_barplot_zoom.png'), dpi = 400,bbox_inches="tight")


f, (ax_top, ax_bottom) = plt.subplots(figsize=(10,6),ncols=1, nrows=2, sharex=False, gridspec_kw={'hspace':0.05})
b = sns.barplot(x='method', y='dim', data=pd_dim,
            palette = palette, linewidth = 1, width= .5, ax = ax_top)
sns.swarmplot(x='method', y='dim', data=pd_dim, 
            color = 'gray', edgecolor = 'gray', ax = ax_top)
sns.lineplot(x = 'method', y= 'dim', data = pd_dim, units = 'mouse',
            ax = ax_top, estimator = None, color = ".7", markers = True)

b = sns.barplot(x='method', y='dim', data=pd_dim,
            palette = palette, linewidth = 1, width= .5, ax = ax_bottom)
sns.swarmplot(x='method', y='dim', data=pd_dim, 
            color = 'gray', edgecolor = 'gray', ax = ax_bottom)
b = sns.barplot(x='method', y='dim', data=pd_dim,
            palette = palette, linewidth = 1, width= .5, ax = ax_bottom)
sns.lineplot(x = 'method', y= 'dim', data = pd_dim, units = 'mouse',
            ax = ax_bottom, estimator = None, color = ".7", markers = True)
ax_top.set_ylim(bottom=10)   # those limits are fake
ax_bottom.set_ylim(0,5)

sns.despine(ax=ax_bottom)
sns.despine(ax=ax_top, bottom=True)

ax = ax_top
d = .01  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax_top.set_xticks([])
ax_top.set_ylabel([])
ax2 = ax_bottom
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

#remove one of the legend
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'dim_barplot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'dim_barplot.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                        PLOT SI DIM RED BOXPLOTS                        |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig1/SI/preNYC'
sI_dict = load_pickle(data_dir, 'sI_clean_dict.pkl')

miceList = list(sI_dict.keys())

for featureName in ['pos','dir','(pos_dir)', 'vel', 'time']:
    SI_list = list()
    sSI_list = list()
    mouse_list = list()
    for mouse in miceList:
        SI_list.append(sI_dict[mouse]['clean_traces'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['umap'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['isomap'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['pca'][featureName]['sI'])

        sSI_list.append(np.percentile(sI_dict[mouse]['clean_traces'][featureName]['ssI'], 99))
        sSI_list.append(np.percentile(sI_dict[mouse]['umap'][featureName]['ssI'], 99))
        sSI_list.append(np.percentile(sI_dict[mouse]['isomap'][featureName]['ssI'], 99))
        sSI_list.append(np.percentile(sI_dict[mouse]['pca'][featureName]['ssI'], 99))

        mouse_list = mouse_list + [mouse]*4

    feature_list = ['og','umap', 'isomap','pca']*len(miceList)
    sSI_array = np.array(sSI_list).reshape(-1,4)

    pd_SI = pd.DataFrame(data={'mouse': mouse_list,
                                     'SI': SI_list,
                                     'method': feature_list})

    palette= ['#8a8a8aff', '#5fc010ff', '#e08b10ff', '#e01110ff']
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    b = sns.boxplot(x='method', y='SI', data=pd_SI,
                palette = palette, linewidth = 1, width= .5, ax = ax)

    sns.swarmplot(x='method', y='SI', data=pd_SI, 
                color = 'gray', edgecolor = 'gray', ax = ax)

    sns.lineplot(x = 'method', y= 'SI', data = pd_SI, units = 'mouse',
                ax = ax, estimator = None, color = ".7", markers = True)

    # for idx in range(4):
    #     x_space = [-.25+idx, 0.25+idx]
    #     m = np.mean(sSI_array[:,idx])
    #     sd = np.std(sSI_array[:,idx])
    #     ax.plot(x_space, [m,m], linestyle='--', color=palette[idx])
    #     ax.fill_between(x_space, m-sd, m+sd, color=palette[idx], alpha = 0.3)

    b.set_xlabel(" ",fontsize=15)
    b.set_ylabel(f"sI {featureName}",fontsize=15)
    b.spines['top'].set_visible(False)
    b.spines['right'].set_visible(False)
    b.tick_params(labelsize=12)
    # ax.set_ylim([0.45, 1.05])
    # b.set_yticks([0, .25, .5, .75, 1])
    plt.tight_layout()
    plt.suptitle(featureName)
    plt.savefig(os.path.join(data_dir,f'SI_{featureName}_no_sSI.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(data_dir,f'SI_{featureName}_no_sSI.png'), dpi = 400,bbox_inches="tight")


import scipy.stats as stats

for featureName in ['pos','dir','(pos_dir)', 'vel', 'time']:




from statsmodels.stats.anova import AnovaRM
for featureName in ['pos','dir','(pos_dir)', 'vel', 'time']:
    SI_list = list()
    sSI_list = list()
    mouse_list = list()
    for mouse in miceList:
        SI_list.append(sI_dict[mouse]['clean_traces'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['umap'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['isomap'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['pca'][featureName]['sI'])
        mouse_list = mouse_list + [mouse]*4

    feature_list = ['og','umap', 'isomap','pca']*len(miceList)
    pd_SI = pd.DataFrame(data={'mouse': mouse_list,
                                 'SI': SI_list,
                                 'method': feature_list})
    print(f'-----------{featureName}-----------')
    print(AnovaRM(data=pd_SI, depvar='SI', 
                  subject='mouse', within=['method']).fit()) 
    SI_og = []
    SI_umap = []
    SI_isomap = []
    SI_pca = []
    for mouse in miceList:
        SI_og.append(sI_dict[mouse]['clean_traces'][featureName]['sI'])
        SI_umap.append(sI_dict[mouse]['umap'][featureName]['sI'])
        SI_isomap.append(sI_dict[mouse]['isomap'][featureName]['sI'])
        SI_pca.append(sI_dict[mouse]['pca'][featureName]['sI'])

    print('Og - Umap', stats.ttest_rel(SI_og, SI_umap))
    print('Og - Isomap', stats.ttest_rel(SI_og, SI_isomap))
    print('Og - PCA', stats.ttest_rel(SI_og, SI_pca))
    print('Umap - Isomap', stats.ttest_rel(SI_umap, SI_isomap))
    print('Umap - PCA', stats.ttest_rel(SI_umap, SI_pca))
    print('Isomap - PCA', stats.ttest_rel(SI_isomap, SI_pca))




SI_list = list()
sSI_list = list()
mouse_list = list()
feature_list = list()
emb_list = list()
for featureName in ['pos','dir','(pos_dir)', 'vel', 'time']:
    for mouse in miceList:
        SI_list.append(sI_dict[mouse]['clean_traces'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['umap'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['isomap'][featureName]['sI'])
        SI_list.append(sI_dict[mouse]['pca'][featureName]['sI'])




        sSI_list.append(np.percentile(sI_dict[mouse]['clean_traces'][featureName]['ssI'], 99))
        sSI_list.append(np.percentile(sI_dict[mouse]['umap'][featureName]['ssI'], 99))
        sSI_list.append(np.percentile(sI_dict[mouse]['isomap'][featureName]['ssI'], 99))
        sSI_list.append(np.percentile(sI_dict[mouse]['pca'][featureName]['ssI'], 99))



        mouse_list = mouse_list + [mouse]*4
        feature_list = feature_list + [featureName]*4
        emb_list = emb_list + ['og','umap', 'isomap','pca']

sSI_array = np.array(sSI_list).reshape(-1,4)

pd_SI = pd.DataFrame(data={'mouse': mouse_list,
                                 'SI': SI_list,
                                 'method': emb_list,
                                 'feature': feature_list})

palette= ['#8a8a8aff', '#5fc010ff', '#e08b10ff', '#e01110ff']
fig, ax = plt.subplots(1, 1, figsize=(14,6))
b = sns.boxplot(x='feature', y='SI', data=pd_SI, hue='method',
            palette = palette, linewidth = 1, width= .6, ax = ax)
plt.savefig(os.path.join(data_dir,f'SI_all_features.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,f'SI_all_features.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig1/decoders'
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
        pd_miceList = list()
        pd_sig_list = list()
        for mouse in miceList:
            for sig in signal_list:
                R2s_list.append(np.mean(dec_R2s[mouse][sig][dec_name][:,label_idx,0], axis=0))
                pd_miceList.append(mouse)
                pd_sig_list.append(sig)

        pd_R2s = pd.DataFrame(data={'mouse': pd_miceList,
                                     'R2s': R2s_list,
                                     'signal': pd_sig_list})

        b = sns.barplot(x='signal', y='R2s', data=pd_R2s,
                palette = palette, linewidth = 1, width= .5, ax = ax[dec_idx])
        sns.swarmplot(x='signal', y='R2s', data=pd_R2s, 
                    color = 'gray', edgecolor = 'gray', ax = ax[dec_idx])

        sns.lineplot(x = 'signal', y= 'R2s', data = pd_R2s, units = 'mouse',
                    ax = ax[dec_idx], estimator = None, color = ".7", markers = True)

        ax[dec_idx].set_ylabel(f'R2s {label_name}')
        ax[dec_idx].set_title(dec_name)

    fig.suptitle(label_name)
    plt.savefig(os.path.join(data_dir,f'dec_{label_name}_test.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(data_dir,f'dec_{label_name}_test.png'), dpi = 400,bbox_inches="tight")


dec_name = 'xgb'
label_name = 'posx'
label_idx = 0
R2s_list = list()
pd_miceList = list()
pd_sig_list = list()
for mouse in miceList:
    for sig in signal_list:
        R2s_list.append(np.mean(dec_R2s[mouse][sig][dec_name][:,label_idx,0], axis=0))
        pd_miceList.append(mouse)
        pd_sig_list.append(sig)

pd_R2s = pd.DataFrame(data={'mouse': pd_miceList,
                             'R2s': R2s_list,
                             'signal': pd_sig_list})
print(f'-----------{dec_name} {label_idx}-----------')
print(AnovaRM(data=pd_R2s, depvar='R2s', 
              subject='mouse', within=['signal']).fit()) 

R2s_og = []
R2s_umap = []
R2s_isomap = []
R2s_pca = []
for mouse in miceList:
    R2s_og.append(np.mean(dec_R2s[mouse]['base_signal'][dec_name][:,label_idx,0], axis=0))
    R2s_umap.append(np.mean(dec_R2s[mouse]['umap'][dec_name][:,label_idx,0], axis=0))
    R2s_isomap.append(np.mean(dec_R2s[mouse]['isomap'][dec_name][:,label_idx,0], axis=0))
    R2s_pca.append(np.mean(dec_R2s[mouse]['pca'][dec_name][:,label_idx,0], axis=0))

print('Og - Umap', stats.ttest_rel(R2s_og, R2s_umap))
print('Og - Isomap', stats.ttest_rel(R2s_og, R2s_isomap))
print('Og - PCA', stats.ttest_rel(R2s_og, R2s_pca))
print('Umap - Isomap', stats.ttest_rel(R2s_umap, R2s_isomap))
print('Umap - PCA', stats.ttest_rel(R2s_umap, R2s_pca))
print('Isomap - PCA', stats.ttest_rel(R2s_isomap, R2s_pca))


for label_idx, label_name in enumerate(label_list):
    fig, ax = plt.subplots(1,4,figsize=(6,6))
    for dec_idx, dec_name in enumerate(decoder_list):
        R2s_list = list()
        pd_miceList = list()
        pd_sig_list = list()
        for mouse in miceList:
            for sig in signal_list:
                R2s_list.append(np.mean(dec_R2s[mouse][sig][dec_name][:,label_idx,1], axis=0))
                pd_miceList.append(mouse)
                pd_sig_list.append(sig)

        pd_R2s = pd.DataFrame(data={'mouse': pd_miceList,
                                     'R2s': R2s_list,
                                     'signal': pd_sig_list})

        b = sns.barplot(x='signal', y='R2s', data=pd_R2s,
                palette = palette, linewidth = 1, width= .5, ax = ax[dec_idx])
        sns.swarmplot(x='signal', y='R2s', data=pd_R2s, 
                    color = 'gray', edgecolor = 'gray', ax = ax[dec_idx])

        # sns.lineplot(x = 'signal', y= 'R2s', data = pd_R2s, units = 'mouse',
        #             ax = ax[dec_idx], estimator = None, color = ".7", markers = True)

        ax[dec_idx].set_ylabel(f'R2s {label_name}')
        ax[dec_idx].set_title(dec_name)

    fig.suptitle(label_name)
    plt.savefig(os.path.join(data_dir,f'dec_{label_name}_train.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(data_dir,f'dec_{label_name}_train.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                             CROSS DECODERS                             |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig1/decoders'
dec_R2s = load_pickle(data_dir, 'cross_dec_R2s_dict.pkl')

palette= ['#5bb95bff', '#ff8a17ff', '#249aefff']
label_list = ['posx', 'posy','vel', 'index_mat', 'dir_mat']
signal_list = ['pca', 'isomap', 'umap']
decoder_list = ['wf','wc','xgb','svr']
miceList = list(dec_R2s.keys())

for label_idx, label_name in enumerate(label_list):
    fig, ax = plt.subplots(1,4,figsize=(15,5))
    for dec_idx, dec_name in enumerate(decoder_list):
        R2s_list = list()
        pd_miceList = list()
        pd_sig_list = list()
        pd_state_list = list()
        for mouse in miceList:
            for sig in signal_list:
                R2s_list.append(dec_R2s[mouse][sig][dec_name][label_idx,0])
                pd_miceList.append(mouse)
                pd_sig_list.append(sig)
                pd_state_list.append('pre')
                R2s_list.append(dec_R2s[mouse][sig][dec_name][label_idx,1])
                pd_miceList.append(mouse)
                pd_sig_list.append(sig)            
                pd_state_list.append('aligned')

        pd_R2s = pd.DataFrame(data={'mouse': pd_miceList,
                                     'R2s': R2s_list,
                                     'state': pd_state_list,
                                     'signal': pd_sig_list})

        b = sns.barplot(x='state', y='R2s', data=pd_R2s, hue='signal',
                palette = palette, linewidth = 1, width= .5, ax = ax[dec_idx])
        sns.stripplot(x='state', y='R2s', data=pd_R2s, hue='signal', dodge=True,
                    palette='dark:gray', edgecolor = 'gray', ax = ax[dec_idx])

        ax[dec_idx].set_ylabel(f'R2s {label_name}')
        ax[dec_idx].set_title(dec_name)
        fig.suptitle(label_name)

        plt.savefig(os.path.join(data_dir,f'cross_dec_{label_name}_test.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(data_dir,f'cross_dec_{label_name}_test.png'), dpi = 400,bbox_inches="tight")


data_dir = '/home/julio/Documents/SP_project/Fig1/decoders'
dec_R2s = load_pickle(data_dir, 'cross_dec_R2s_dict.pkl')

palette= ['#5bb95bff', '#ff8a17ff', '#249aefff']
label_list = ['posx', 'posy','vel', 'index_mat', 'dir_mat']
signal_list = ['pca', 'isomap', 'umap']
decoder_list = ['wf','wc','xgb','svr']
miceList = list(dec_R2s.keys())

dec_name = 'xgb'
label_name = 'posx'
label_idx = 0

R2s_list = list()
pd_miceList = list()
pd_sig_list = list()
pd_state_list = list()
for mouse in miceList:
    for sig in signal_list:
        R2s_list.append(dec_R2s[mouse][sig][dec_name][label_idx,0])
        pd_miceList.append(mouse)
        pd_sig_list.append(sig)
        pd_state_list.append('pre')
        R2s_list.append(dec_R2s[mouse][sig][dec_name][label_idx,1])
        pd_miceList.append(mouse)
        pd_sig_list.append(sig)            
        pd_state_list.append('aligned')

pd_R2s = pd.DataFrame(data={'mouse': pd_miceList,
                             'R2s': R2s_list,
                             'state': pd_state_list,
                             'signal': pd_sig_list})

print(f'-----------{dec_name} {label_idx}-----------')
print(AnovaRM(data=pd_R2s, depvar='R2s', 
              subject='mouse', within=['signal', 'state']).fit()) 




R2s_umap_og = []
R2s_isomap_og = []
R2s_pca_og = []

R2s_umap_align = []
R2s_isomap_align = []
R2s_pca_align = []


for mouse in miceList:
    R2s_umap_og.append(dec_R2s[mouse]['umap'][dec_name][label_idx,0])
    R2s_umap_align.append(dec_R2s[mouse]['umap'][dec_name][label_idx,1])

    R2s_isomap_og.append(dec_R2s[mouse]['isomap'][dec_name][label_idx,0])
    R2s_isomap_align.append(dec_R2s[mouse]['isomap'][dec_name][label_idx,1])


    R2s_pca_og.append(dec_R2s[mouse]['pca'][dec_name][label_idx,0])
    R2s_pca_align.append(dec_R2s[mouse]['pca'][dec_name][label_idx,1])


print('Umap_og - Isomap_og', stats.ttest_rel(R2s_umap_og, R2s_isomap_og))
print('Umap_og - PCA_og', stats.ttest_rel(R2s_umap_og, R2s_pca_og))
print('Isomap_og - PCA_og', stats.ttest_rel(R2s_isomap_og, R2s_pca_og))
print('--')
print('Umap_align - Isomap_align', stats.ttest_rel(R2s_umap_align, R2s_isomap_align))
print('Umap_align - PCA_align', stats.ttest_rel(R2s_umap_align, R2s_pca_align))
print('Isomap_align - PCA_align', stats.ttest_rel(R2s_isomap_align, R2s_pca_align))
print('--')
print('Umap_og - Umap_align', stats.ttest_rel(R2s_umap_og, R2s_umap_align))
print('Umap_og - Isomap_align', stats.ttest_rel(R2s_umap_og, R2s_isomap_align))
print('Umap_og - PCA_align', stats.ttest_rel(R2s_umap_og, R2s_pca_align))
print('--')
print('Isomap_og - Umap_align', stats.ttest_rel(R2s_isomap_og, R2s_umap_align))
print('Isomap_og - Isomap_align', stats.ttest_rel(R2s_isomap_og, R2s_isomap_align))
print('Isomap_og - PCA_align', stats.ttest_rel(R2s_isomap_og, R2s_pca_align))
print('--')
print('PCA_og - Umap_align', stats.ttest_rel(R2s_pca_og, R2s_umap_align))
print('PCA_og - Isomap_align', stats.ttest_rel(R2s_pca_og, R2s_isomap_align))
print('PCA_og - PCA_align', stats.ttest_rel(R2s_pca_og, R2s_pca_align))





#__________________________________________________________________________
#|                                                                        |#
#|                             NOISE DECODERS                             |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig1/decoders'
dec_R2s = load_pickle(data_dir, 'noise_dec_R2s_dict.pkl')
dec_SNR = load_pickle(data_dir, 'noise_dec_SNR.pkl')

palette= ['grey', '#5bb95bff', '#ff8a17ff', '#249aefff']
label_list = ['posx', 'posy','vel', 'index_mat', 'dir_mat']
signal_list = ['base_signal','pca', 'isomap', 'umap']
decoder_list = ['wf','wc','xgb','svr']
miceList = list(dec_R2s.keys())
noise_list = [0, 0.001, 0.01, 0.05, 0.1]

for label_idx, label_name in enumerate(label_list):
    fig, ax = plt.subplots(1,4,figsize=(15,5))
    for dec_idx, dec_name in enumerate(decoder_list):
        R2s_list = list()
        pd_miceList = list()
        pd_sig_list = list()
        pd_noise_list = list()
        for mouse in miceList:
            for sig in signal_list:
                temp = np.mean(dec_R2s[mouse][sig][dec_name][:,:-1,label_idx,0], axis=0)
                
                for ii in range(temp.shape[0]):
                    R2s_list.append(temp[ii])
                    pd_miceList.append(mouse)
                    pd_sig_list.append(sig)
                    pd_noise_list.append(noise_list[ii])


        pd_R2s = pd.DataFrame(data={'mouse': pd_miceList,
                                     'R2s': R2s_list,
                                     'noise': pd_noise_list,
                                     'signal': pd_sig_list})

        b = sns.barplot(x='noise', y='R2s', data=pd_R2s, hue='signal',
                palette = palette, linewidth = 1, width= .5, ax = ax[dec_idx])
        sns.stripplot(x='noise', y='R2s', data=pd_R2s, hue='signal', dodge=True,
                    palette='dark:gray', edgecolor = 'gray', ax = ax[dec_idx])

        ax[dec_idx].set_ylabel(f'R2s {label_name}')
        ax[dec_idx].set_title(dec_name)
        fig.suptitle(label_name)
        plt.savefig(os.path.join(data_dir,f'noise_dec_{label_name}_test_bar.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(data_dir,f'noise_dec_{label_name}_test_bar.png'), dpi = 400,bbox_inches="tight")

for label_idx, label_name in enumerate(label_list):
    fig, ax = plt.subplots(1,4,figsize=(15,5))
    for dec_idx, dec_name in enumerate(decoder_list):
        R2s_list = list()
        pd_miceList = list()
        pd_sig_list = list()
        pd_noise_list = list()
        for mouse in miceList:
            for sig in signal_list:
                temp = np.mean(dec_R2s[mouse][sig][dec_name][:,:-1,label_idx,0], axis=0)
                
                for ii in range(temp.shape[0]):
                    R2s_list.append(temp[ii])
                    pd_miceList.append(mouse)
                    pd_sig_list.append(sig)
                    pd_noise_list.append(noise_list[ii])

        pd_R2s = pd.DataFrame(data={'mouse': pd_miceList,
                                     'R2s': R2s_list,
                                     'noise': pd_noise_list,
                                     'signal': pd_sig_list})

        sns.lineplot(data=pd_R2s, x="noise", y="R2s",hue='signal',palette = palette, ax = ax[dec_idx])
        ax[dec_idx].set_ylabel(f'R2s {label_name}')
        ax[dec_idx].set_title(dec_name)
        fig.suptitle(label_name)
        plt.savefig(os.path.join(data_dir,f'noise_dec_{label_name}_test_line.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(data_dir,f'noise_dec_{label_name}_test_line.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

bin_width = 5
data_dir = '/home/julio/Documents/SP_project/Fig1/decoders'
dec_pred = load_pickle(data_dir, 'dec_pred_dict.pkl')

palette= ['grey', '#5bb95bff', '#ff8a17ff', '#249aefff']
label_list = ['posx', 'posy','vel', 'index_mat', 'dir_mat']
signal_list = ['base_signal', 'pca', 'isomap', 'umap']
decoder_list = ['wf','wc','xgb','svr']
miceList = list(dec_pred.keys())

min_pos = np.inf
max_pos = -np.inf
for mouse in miceList:
    min_pos = np.min([min_pos, np.min(dec_pred[mouse][0][0,:,1])])
    max_pos = np.max([max_pos, np.max(dec_pred[mouse][0][0,:,1])])


mapAxis = np.arange(min_pos,max_pos, bin_width)
error_bin = np.zeros((mapAxis.shape[0],10,4))
for signal in range(4):
    mapAxis_count = np.zeros(mapAxis.shape)
    mapAxis_pred_count = np.zeros(mapAxis.shape)
    for kfold in range(10):
        real = dec_pred[mouse][0][kfold,:,1]
        pred = dec_pred[mouse][0][kfold,:,signal+2]

        test_real = real[dec_pred[mouse][0][kfold,:,0]==1]
        test_pred = pred[dec_pred[mouse][0][kfold,:,0]==1]

        for sample in range(test_real.shape[0]):
            temp_entry = np.where(test_real[sample]>=mapAxis)
            if np.any(temp_entry):
                temp_entry = temp_entry[0][-1]
            else:
                temp_entry = 0
        
            error_temp = np.abs(test_real[sample] - test_pred[sample])
            mapAxis_count[temp_entry] += 1
            error_bin[temp_entry, kfold, signal] += (1/mapAxis_count[temp_entry])* \
                                       (error_temp-error_bin[temp_entry, kfold, signal]) #online average

            temp_entry = np.where(test_pred[sample]>=mapAxis)
            if np.any(temp_entry):
                temp_entry = temp_entry[0][-1]
            else:
                temp_entry = 0
            mapAxis_pred_count[temp_entry] += 1

    plt.figure()
    plt.plot(mapAxis, mapAxis_count, label='real pdf')
    plt.plot(mapAxis, mapAxis_pred_count, label='pred pdf')
    plt.legend()

    plt.figure()
    m = np.mean(error_bin[:,:, signal], axis = 1)
    sd = np.std(error_bin[:,:, signal], axis = 1)

    plt.plot(mapAxis, m, label='error')
    plt.fill_between(mapAxis, m-sd, m+sd, alpha = .3)


#__________________________________________________________________________
#|                                                                        |#
#|                                 UMAP EMB                               |#
#|________________________________________________________________________|#

miceList = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig1/dimensionality/emb_example/umap/'

for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")


    pos = copy.deepcopy(np.concatenate(pdMouse ['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse ['dir_mat'].values, axis=0))
    vectorial_feature = np.concatenate((pos[:,0].reshape(-1,1),dir_mat),axis=1)
    time = np.arange(pos.shape[0])
    umap_emb = copy.deepcopy(np.concatenate(pdMouse ['umap'].values, axis=0))


    D= pairwise_distances(umap_emb)
    noiseIdx = filter_noisy_outliers(umap_emb,D)
    umap_emb = umap_emb[~noiseIdx,:]
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
    b = ax.scatter(*umap_emb[0,:3].T, c = pos[0,0], cmap = 'magma',s = 10)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xlim([-5,5])
    ax.set_ylim([-4,4])
    ax.set_zlim([-4, 4])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_empty.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_empty.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*umap_emb[:,:3].T, color = dir_color,s = 20)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_dir.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_dir.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*umap_emb[:,:2].T, color = dir_color,s = 20)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_dir_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_dir_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*umap_emb[:,(0,2)].T, color = dir_color,s = 20)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_dir_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_dir_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*umap_emb[:,:3].T, c = pos[:,0],s = 20, cmap = 'inferno', vmin= 0, vmax = 120)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_pos.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_pos.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*umap_emb[:,:2].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_pos_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_pos_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*umap_emb[:,(0,2)].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_pos_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_pos_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*umap_emb[:,:3].T, c = time,s = 20, cmap = 'YlGn_r', vmax = 13000)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_time.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_time.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*umap_emb[:,:2].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_time_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_time_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*umap_emb[:,(0,2)].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_time_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_umap_emb_time_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

#_________________________________________________________________________
#|                                                                        |#
#|                               ISOMAP EMB                               |#
#|________________________________________________________________________|#

miceList = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig1/dimensionality/emb_example/isomap/'

for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")


    pos = copy.deepcopy(np.concatenate(pdMouse ['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse ['dir_mat'].values, axis=0))
    vectorial_feature = np.concatenate((pos[:,0].reshape(-1,1),dir_mat),axis=1)
    time = np.arange(pos.shape[0])
    isomap_emb = copy.deepcopy(np.concatenate(pdMouse ['isomap'].values, axis=0))

    D= pairwise_distances(isomap_emb)
    noiseIdx = filter_noisy_outliers(isomap_emb,D)
    isomap_emb = isomap_emb[~noiseIdx,:]
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
    b = ax.scatter(*isomap_emb[0,:3].T, c = pos[0,0], cmap = 'magma',s = 10)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xlim([-5,5])
    ax.set_ylim([-4,4])
    ax.set_zlim([-4, 4])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_empty.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_empty.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*isomap_emb[:,:3].T, color = dir_color,s = 20)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_dir.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_dir.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*isomap_emb[:,:2].T, color = dir_color,s = 20)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_dir_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_dir_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*isomap_emb[:,(0,2)].T, color = dir_color,s = 20)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_dir_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_dir_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*isomap_emb[:,:3].T, c = pos[:,0],s = 20, cmap = 'inferno', vmin= 0, vmax = 120)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_pos.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_pos.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*isomap_emb[:,:2].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_pos_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_pos_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*isomap_emb[:,(0,2)].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_pos_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_pos_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*isomap_emb[:,:3].T, c = time,s = 20, cmap = 'YlGn_r', vmax = 13000)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_time.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_time.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*isomap_emb[:,:2].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_time_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_time_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*isomap_emb[:,(0,2)].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_time_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_isomap_emb_time_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)


#__________________________________________________________________________
#|                                                                        |#
#|                                 PCA EMB                                |#
#|________________________________________________________________________|#

miceList = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig1/dimensionality/emb_example/pca/'

for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")


    pos = copy.deepcopy(np.concatenate(pdMouse ['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse ['dir_mat'].values, axis=0))
    vectorial_feature = np.concatenate((pos[:,0].reshape(-1,1),dir_mat),axis=1)
    time = np.arange(pos.shape[0])
    pca_emb = copy.deepcopy(np.concatenate(pdMouse ['pca'].values, axis=0))

    D= pairwise_distances(pca_emb)
    noiseIdx = filter_noisy_outliers(pca_emb,D)
    pca_emb = pca_emb[~noiseIdx,:]
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
    b = ax.scatter(*pca_emb[0,:3].T, c = pos[0,0], cmap = 'magma',s = 10)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xlim([-5,5])
    ax.set_ylim([-4,4])
    ax.set_zlim([-4, 4])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_empty.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_empty.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*pca_emb[:,:3].T, color = dir_color,s = 20)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_dir.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_dir.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*pca_emb[:,:2].T, color = dir_color,s = 20)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_dir_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_dir_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*pca_emb[:,(0,2)].T, color = dir_color,s = 20)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_dir_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_dir_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*pca_emb[:,:3].T, c = pos[:,0],s = 20, cmap = 'inferno', vmin= 0, vmax = 120)
    # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_pos.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_pos.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*pca_emb[:,:2].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_pos_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_pos_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*pca_emb[:,(0,2)].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_pos_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_pos_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection = '3d')
    b = ax.scatter(*pca_emb[:,:3].T, c = time,s = 20, cmap = 'YlGn_r', vmax = 13000)
    ax.view_init(45, -45)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_time.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_time.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*pca_emb[:,:2].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_time_xy.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_time_xy.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    b = ax.scatter(*pca_emb[:,(0,2)].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_time_xz.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_pca_emb_time_xz.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

#__________________________________________________________________________
#|                                                                        |#
#|                              BETTI NUMBERS                             |#
#|________________________________________________________________________|#

dataDir = '/home/julio/Documents/SP_project/Fig1/betti_numbers'
miceList = ['M2019','M2023','M2024', 'M2025', 'M2026']
hList = list()
lifeTimeList = list()
mouseList = list()
for mouse in miceList:
    mouseDict = load_pickle(dataDir, mouse+'_betti_dict.pkl')
    h1Diagrams = np.array(mouseDict['diagrams'][1])
    h1Length = np.sort(np.diff(h1Diagrams, axis=1)[:,0])
    lifeTimeList.append(h1Length[-1]/h1Length[-2])
    hList.append(1)
    hList.append(2)
    mouseList.append(mouse)

pdBetti = pd.DataFrame(data={'mouse': mouseList,
                     'lifeTime': lifeTimeList,
                     'betti': hList})    

fig, ax = plt.subplots(1, 1, figsize=(6,10))
b = sns.boxplot(x='betti', y='lifeTime', data=pdBetti,
            linewidth = 1, width= .5, ax = ax)

sns.swarmplot(x='betti', y='lifeTime', data=pdBetti,
            color = 'gray', edgecolor = 'gray', ax = ax)
ax.plot([-.25,.25], [1,1], linestyle='--', color='black')

plt.savefig(os.path.join(dataDir,'lifetime_betti.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'lifetime_betti.png'), dpi = 400,bbox_inches="tight")

dataDir = '/home/julio/Documents/SP_project/Fig1/betti_numbers'
miceList = ['M2019','M2023','M2024', 'M2025', 'M2026']
hList = list()
lifeTimeList = list()
mouseList = list()
for mouse in miceList:
    mouseDict = load_pickle(dataDir, mouse+'_betti_dict.pkl')
    confInterval = mouseDict['confInterval']
    h1Diagrams = np.array(mouseDict['diagrams'][1])
    h1Length = np.sort(np.diff(h1Diagrams, axis=1)[:,0])
    lifeTimeList.append(h1Length[-1]/np.max([confInterval[1],h1Length[-2]]))
    hList.append(1)

    h2Diagrams = np.array(mouseDict['diagrams'][2])
    h2Length = np.sort(np.diff(h2Diagrams, axis=1)[:,0])
    lifeTimeList.append(h2Length[-1]/np.max([confInterval[2], h2Length[-2]]))
    hList.append(2)

    mouseList.append(mouse)
    mouseList.append(mouse)

pdBetti = pd.DataFrame(data={'mouse': mouseList,
                     'lifeTime': lifeTimeList,
                     'betti': hList})    

fig, ax = plt.subplots(1, 1, figsize=(6,10))
b = sns.boxplot(x='betti', y='lifeTime', data=pdBetti,
            linewidth = 1, width= .5, ax = ax)

sns.swarmplot(x='betti', y='lifeTime', data=pdBetti,
            color = 'gray', edgecolor = 'gray', ax = ax)
ax.plot([-.25,.25], [1,1], linestyle='--', color='black')
ax.plot([.75,1.25], [1,1], linestyle='--', color='black')
plt.savefig(os.path.join(dataDir,'lifetime_betti_v2.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'lifetime_betti_v2.png'), dpi = 400,bbox_inches="tight")
