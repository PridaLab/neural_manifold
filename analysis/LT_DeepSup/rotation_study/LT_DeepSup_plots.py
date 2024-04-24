import seaborn as sns
import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.metrics import pairwise_distances
import networkx as nx
from structure_index import compute_structure_index, draw_graph

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 5)
    sum(noiseIdx)
    return noiseIdx

def personalize_ax(ax, ax_view = None):
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if ax_view:
        ax.view_init(ax_view[0], ax_view[1])


supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']

#__________________________________________________________________________
#|                                                                        |#
#|                                PLOT DIM                                |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/dimensionality/'


innerDim = load_pickle(os.path.join(dataDir, 'inner_dim'), 'inner_dim_dict.pkl')
fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, case in enumerate(['pre','rot','both']):
    dimList = list()
    mouseList = list()
    methodList = list()
    layerList = list()
    for mouse in miceList:
        dimList.append(innerDim[mouse][case]['momDim'])
        dimList.append(innerDim[mouse][case]['abidsDim'])
        dimList.append(innerDim[mouse][case]['tleDim'])
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
                palette = palette, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
            palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])
    ax[idx].set_title(case)
    ax[idx].set_ylim([0,4.5])
plt.tight_layout()
plt.savefig(os.path.join(dataDir,'inner_dim_all.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'inner_dim_all.png'), dpi = 400,bbox_inches="tight")


umapDim = load_pickle(dataDir, 'umap_dim_dict.pkl')
isoDim = load_pickle(dataDir, 'isomap_dim_dict.pkl')


fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, case in enumerate(['pre','rot','both']):
    dimList = list()
    mouseList = list()
    methodList = list()
    layerList = list()
    for mouse in miceList:
        dimList.append(umapDim[mouse][case]['trustDim'])
        dimList.append(umapDim[mouse][case]['contDim'])
        dimList.append(isoDim[mouse][case]['resVarDim'])
        dimList.append(isoDim[mouse][case]['recErrorDim'])

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
                palette = palette, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
            palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])
    ax[idx].set_title(case)
    ax[idx].set_ylim([0,7.5])

plt.tight_layout()
plt.savefig(os.path.join(dataDir,'umap_iso_dim_all.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'umap_iso_dim_all.png'), dpi = 400,bbox_inches="tight")

pcaDim = load_pickle(dataDir, 'pca_dim_dict.pkl')
fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, case in enumerate(['pre','rot','both']):
    dimList = list()
    mouseList = list()
    methodList = list()
    layerList = list()
    for mouse in miceList:
        dimList.append(pcaDim[mouse][case]['var80Dim'])
        dimList.append(pcaDim[mouse][case]['kneeDim'])
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
                palette = palette, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dim', data=dimPD,  hue='method',
            palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])
    ax[idx].set_title(case)
    ax[idx].set_ylim([0,90])

plt.tight_layout()
plt.savefig(os.path.join(dataDir,'pca_dim_all.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'pca_dim_all.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                        PLOT SI DIM RED BOXPLOTS                        |#
#|________________________________________________________________________|#

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/SI/'
sIDict = load_pickle(dataDir, 'sI_clean_dict.pkl')
miceList = list(sIDict.keys())

case = 'pre'
for featureName in ['pos','dir','(pos_dir)', 'vel', 'trial','time']: #,'session']:
    SIList = list()
    mouseList = list()
    layerList = list()
    for mouse in miceList:
        SIList.append(sIDict[mouse]['clean_traces'][case][featureName]['sI'])
        SIList.append(sIDict[mouse]['umap'][case][featureName]['sI'])
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
    b = sns.boxplot(x='layer', y='SI', data=SIPD, hue = 'method',
                palette = palette, linewidth = 1, width= .5, ax = ax)

    sns.swarmplot(x='layer', y='SI', data=SIPD, hue= 'method',
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
    b.set_yticks([0,0.2,0.4,0.6])
    b.set_ylim([-.05, 0.65])
    plt.tight_layout()
    plt.suptitle(featureName)
    plt.savefig(os.path.join(dataDir,f'SI_{featureName}_{case}Session.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(dataDir,f'SI_{featureName}_{case}Session.png'), dpi = 400,bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|                          PLOT ROTATION ERRORS                          |#
#|________________________________________________________________________|#

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'
rot_error_dict = load_pickle(dataDir, 'rot_error_dict.pkl')
miceList = list(rot_error_dict.keys())

#PLOT LINES
for embName in ['pca', 'isomap', 'umap']:
    rotErrorSup = np.zeros((100, len(supMice)))
    rotErrorDeep = np.zeros((100, len(deepMice)))
    supIdx = 0
    deepIdx = 0
    angleDeg = rot_error_dict[miceList[0]][embName]['angles']
    for mouse in miceList:
        normError = rot_error_dict[mouse][embName]['normError']
        if mouse in deepMice:
            rotErrorDeep[:,deepIdx] = normError
            deepIdx += 1
        elif mouse in supMice:
            rotErrorSup[:,supIdx] = normError
            supIdx += 1
    plt.figure()
    ax = plt.subplot(111)
    m = np.mean(rotErrorDeep,axis=1)
    sd = np.std(rotErrorDeep,axis=1)
    ax.plot(angleDeg, m, color = '#32E653',label = 'deep')
    ax.fill_between(angleDeg, m-sd, m+sd, color = '#32E653', alpha = 0.3)
    m = np.mean(rotErrorSup,axis=1)
    sd = np.std(rotErrorSup,axis=1)
    ax.plot(angleDeg, m, color = '#E632C5', label = 'sup')
    ax.fill_between(angleDeg, m-sd, m+sd, color = '#E632C5', alpha = 0.3)
    ax.set_xlabel('Angle of rotation (º)')
    ax.set_ylabel('Aligment Error')
    ax.set_title(embName)
    ax.legend()
    plt.savefig(os.path.join(dataDir,f'DeepSup_{embName}_rotation_error.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(dataDir,f'DeepSup_{embName}_rotation_error.svg'), dpi = 400,bbox_inches="tight",transparent=True)


#PLOT BOXPLOTS
rotAngleList = list()
layerList = list()
embList = list()
mouseList = list()
for mouse in miceList:
    if mouse == 'CGrin1': continue;
    for embName in['pca', 'isomap', 'umap']:
        rotAngleList.append(rot_error_dict[mouse][embName]['rotAngle'])
        embList.append(embName)
        if mouse in deepMice:
            layerList.append('deep')
        elif mouse in supMice:
            layerList.append('sup')
        mouseList.append(mouse)
anglePD = pd.DataFrame(data={'angle': rotAngleList,
                            'emb': embList,
                            'layer': layerList,
                            'mouse': mouseList})

palette= ["#32e653", "#E632C5"]
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='emb', y='angle', hue='layer', data=anglePD, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='emb', y='angle', hue = 'layer', data=anglePD, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylabel('Angle Rotation')
plt.tight_layout()

plt.savefig(os.path.join(dataDir,f'DeepSup_rotation_boxplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(dataDir,f'DeepSup_rotation_boxplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)

from scipy.stats import shapiro
from scipy import stats

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'
rot_error_dict = load_pickle(dataDir, 'rot_error_dict.pkl')
miceList = list(rot_error_dict.keys())
for emb in ['umap','isomap','pca']:
    deepAngle = anglePD[anglePD['emb']==emb][anglePD['layer']=='deep']['angle']
    supAngle = anglePD[anglePD['emb']==emb][anglePD['layer']=='sup']['angle']
    deepShapiro = shapiro(deepAngle)
    supShapiro = shapiro(supAngle)

    if deepShapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
        print(f'{emb} Angle:',stats.ks_2samp(deepAngle, supAngle))
    else:
        print(f'{emb} Angle:', stats.ttest_ind(deepAngle, supAngle))

#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT PLACE CELLS                           |#
#|________________________________________________________________________|#


dcha = ['CGrin1', 'ChZ4','CZ6','CZ8','CZ9','GC5_nvista', 'TGrin1']
izq = ['CZ3','GC2','GC3']


# plt.figure()
deepPDF = list()
supPDF = list()
axisPDF = np.linspace(0,1,20)
for mouse in list(pcManifoldDict.keys()):
    mneuPDF = pcManifoldDict[mouse]['mneuPDF']
    mapAxis = np.squeeze(np.array(pcManifoldDict[mouse]['mapAxis']))

    if mouse in izq:
        mneuPDF = np.flipud(mneuPDF)

    mapAxis = (mapAxis-np.min(mapAxis))/(np.max(mapAxis)-np.min(mapAxis))
    newneuPDF = np.interp(axisPDF, mapAxis, mneuPDF)
    if mouse in deepMice:
        # deepPDF.append(newneuPDF)
        deepPDF.append(mneuPDF[:20])

        # plt.plot(mneuPDF, label = mouse)
    else:
        # supPDF.append(newneuPDF)
        supPDF.append(mneuPDF[:20])
# plt.legend()
deepPDF = np.array(deepPDF).T
supPDF = np.array(supPDF).T

plt.figure()
ax = plt.subplot(111)
m = np.mean(deepPDF,axis=1)
sd = np.std(deepPDF,axis=1)/np.sqrt(deepPDF.shape[1])
ax.plot(axisPDF, m, color = '#32E653',label = 'deep')
ax.fill_between(axisPDF, m-sd, m+sd, color = '#32E653', alpha = 0.3)
m = np.mean(supPDF,axis=1)
sd = np.std(supPDF,axis=1)/np.sqrt(supPDF.shape[1])
ax.plot(axisPDF, m, color = '#E632C5', label = 'sup')
ax.fill_between(axisPDF, m-sd, m+sd, color = '#E632C5', alpha = 0.3)
ax.set_xlabel('x pos')
ax.set_ylabel('mean neu PDF')
ax.legend()
plt.savefig(os.path.join(saveDir,f'DeepSup_neuPDF_emb.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(saveDir,f'DeepSup_neuPDF_emb.svg'), dpi = 400,bbox_inches="tight",transparent=True)

from scipy import stats

for idx in range(10,1,-1):
    print(stats.ttest_ind(np.percentile(deepPDF,95, axis=0)/np.percentile(deepPDF,5, axis=0), np.percentile(supPDF,95, axis=0)/np.percentile(supPDF,5, axis=0), equal_var=True))


deepPDF = (deepPDF - np.min(deepPDF, axis=0))/(np.max(deepPDF, axis=0)-np.min(deepPDF,axis=0))
supPDF = (supPDF - np.min(supPDF, axis=0))/(np.max(supPDF, axis=0)-np.min(supPDF,axis=0))



    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(1,2,1, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 30, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(130,110,90)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax = plt.subplot(1,2,2, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = time,s = 30)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(130,110,90)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


#__________________________________________________________________________
#|                                                                        |#
#|                                 UMAP EMB                               |#
#|________________________________________________________________________|#

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/dimensionality/emb_example/'

mouseExamples = {
    'deep': 'GC2',
    'sup': 'CZ8',
}

examplesValues = {
    'supIter': 1,
    'supAngle': [43,-108], #[-133,-57]
    'supLims': [0.4, 0.65],
    'deepAngle': [-133,125],#[55,65], #[16,-20]
    'deepIter': 1,
    'deepLims': [0.45, 0.55]
}


for case, mouse in mouseExamples.items():
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    animalDict = load_pickle(filePath,fileName)
    fnames = list(animalDict.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]

    animalPre= copy.deepcopy(animalDict[fnamePre])
    animalRot= copy.deepcopy(animalDict[fnameRot])

    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
    embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))
    timePre = np.arange(posPre.shape[0])

    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))
    timeRot = np.arange(posRot.shape[0])+ posPre.shape[0]

    for it in range(examplesValues[case+'Iter']):
        DPre= pairwise_distances(embPre)
        noiseIdx = filter_noisy_outliers(embPre,DPre)
        embPre = embPre[~noiseIdx,:]
        posPre = posPre[~noiseIdx,:]
        dirMatPre = dirMatPre[~noiseIdx]
        timePre = timePre[~noiseIdx]

        DRot= pairwise_distances(embRot)
        noiseIdx = filter_noisy_outliers(embRot,DRot)
        embRot = embRot[~noiseIdx,:]
        posRot = posRot[~noiseIdx,:]
        dirMatRot = dirMatRot[~noiseIdx]
        timeRot = timeRot[~noiseIdx]

    dirColorPre = np.zeros((dirMatPre.shape[0],3))
    for point in range(dirMatPre.shape[0]):
        if dirMatPre[point]==0:
            dirColorPre[point] = [14/255,14/255,143/255]
        elif dirMatPre[point]==1:
            dirColorPre[point] = [12/255,136/255,249/255]
        else:
            dirColorPre[point] = [17/255,219/255,224/255]

    dirColorRot = np.zeros((dirMatRot.shape[0],3))
    for point in range(dirMatRot.shape[0]):
        if dirMatRot[point]==0:
            dirColorRot[point] = [14/255,14/255,143/255]
        elif dirMatRot[point]==1:
            dirColorRot[point] = [12/255,136/255,249/255]
        else:
            dirColorRot[point] = [17/255,219/255,224/255]

    fig = plt.figure(figsize=((6,6)))
    ax = plt.subplot(1,1,1, projection = '3d')
    ax.scatter(*embPre[0,:3].T, c = posPre[0,0], cmap = 'inferno',s = 10)
    personalize_ax(ax, examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_empty.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_empty.png'), dpi = 400,bbox_inches="tight")

    fig = plt.figure(figsize=((6,6)))
    ax = plt.subplot(1,1,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color = 'b',s = 10)
    ax.scatter(*embRot[:,:3].T, color = 'r',s = 10)
    personalize_ax(ax, examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_preRot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_preRot.png'), dpi = 400,bbox_inches="tight")

    fig = plt.figure(figsize=((6,6)))
    ax = plt.subplot(1,1,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap = 'inferno',s = 10)
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap = 'inferno',s = 10)
    personalize_ax(ax, examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_pos.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_pos.png'), dpi = 400,bbox_inches="tight")

    fig = plt.figure(figsize=((6,6)))
    ax = plt.subplot(1,1,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color = dirColorPre, s = 10)
    ax.scatter(*embRot[:,:3].T, color = dirColorRot, s = 10)
    personalize_ax(ax, examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_dir.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_dir.png'), dpi = 400,bbox_inches="tight")

    fig = plt.figure(figsize=((6,6)))
    ax = plt.subplot(1,1,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = timePre, cmap = 'YlGn_r',s = 10)
    ax.scatter(*embRot[:,:3].T, c = timeRot, cmap = 'YlGn_r',s = 10)
    personalize_ax(ax, examplesValues[case+'Angle'])
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_time.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{case}_{mouse}_umapEmb_time.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                         PLACE CELLS ON UMAP EMB                        |#
#|________________________________________________________________________|#

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/manifold_cells/'
pcManifoldDict = load_pickle(dataDir,'manifold_pc_dict.pkl')

mouse = 'CZ8'
pos = pcManifoldDict[mouse]['pos']
emb = pcManifoldDict[mouse]['emb']
manifoldSignal = pcManifoldDict[mouse]['manifoldSignal']

fig = plt.figure(figsize=(15,5))
ax = plt.subplot(1,2,1, projection = '3d')
b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 30, cmap = 'inferno')
cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
# ax.view_init(130,110,90)
ax.view_init(60, -45)
personalize_ax(ax,[60-45])

ax = plt.subplot(1,2,2, projection = '3d')
b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 30)
cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
personalize_ax(ax,[60-45])
plt.savefig(os.path.join(saveDir,mouse+'_PDF_emb.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(saveDir,mouse+'_PDF_emb.png'), dpi = 400,bbox_inches="tight")


dcha = ['CGrin1', 'ChZ4','CZ6','CZ8','CZ9','GC5_nvista', 'TGrin1']
izq = ['CZ3','GC2','GC3']

deepPDF = list()
for mouse in deepMice:
    dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/manifold_cells/'
    pcManifoldDict = load_pickle(dataDir,'manifold_pc_dict.pkl')
    pos = pcManifoldDict[mouse]['pos']
    emb = pcManifoldDict[mouse]['emb']
    manifoldSignal = pcManifoldDict[mouse]['manifoldSignal']
    neuPDF = pcManifoldDict[mouse]['neuPDF']
    mapAxis = pcManifoldDict[mouse]['mapAxis']
    normNeuPDF = np.zeros((neuPDF.shape[0],neuPDF.shape[1]))
    for c in range(neuPDF.shape[1]):
        normNeuPDF[:,c] = neuPDF[:,c,0]/np.max(neuPDF[:,c,0])

    if mouse in izq:
        normNeuPDF = np.flipud(normNeuPDF)
    deepPDF.append(normNeuPDF[:22,:])
deepPDF = np.concatenate(deepPDF, axis=1)


supPDF = list()
for mouse in supMice:
    dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/manifold_cells/'
    pcManifoldDict = load_pickle(dataDir,'manifold_pc_dict.pkl')
    pos = pcManifoldDict[mouse]['pos']
    emb = pcManifoldDict[mouse]['emb']
    manifoldSignal = pcManifoldDict[mouse]['manifoldSignal']
    neuPDF = pcManifoldDict[mouse]['neuPDF']
    mapAxis = pcManifoldDict[mouse]['mapAxis']
    normNeuPDF = np.zeros((neuPDF.shape[0],neuPDF.shape[1]))
    for c in range(neuPDF.shape[1]):
        normNeuPDF[:,c] = neuPDF[:,c,0]/np.max(neuPDF[:,c,0])

    if mouse in izq:
        normNeuPDF = np.flipud(normNeuPDF)
    supPDF.append(normNeuPDF[:21,:])
supPDF = np.concatenate(supPDF, axis=1)


fig, ax = plt.subplots(1, 2, figsize=(10,6))
ax[0].matshow(deepPDF[:, np.argsort(np.argmax(deepPDF, axis=0))].T, aspect='auto')
ax[1].matshow(supPDF[:, np.argsort(np.argmax(supPDF, axis=0))].T, aspect='auto')


order = np.argsort(np.argmax(normNeuPDF, axis=0))
plt.figure(); 

plt.matshow(normNeuPDF[:, order].T)

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/'
pcDict = load_pickle(dataDir, mouse+'_pc_dict.pkl')
fnames = list(pcDict.keys())
fnamePre = [fname for fname in fnames if 'rot' in fname][0]


neuPDF = pcDict[fnamePre]['neu_pdf']
mapAxis = pcDict[fnamePre]['mapAxis']

normNeuPDF = np.zeros((neuPDF.shape[0],neuPDF.shape[1]))
for c in range(neuPDF.shape[1]):
    normNeuPDF[:,c] = neuPDF[:,c,0]/np.max(neuPDF[:,c,0])

order = np.argsort(np.argmax(normNeuPDF, axis=0))
fig, ax = plt.subplots(1, 2, figsize=(10,6))

ax[0].matshow(deepPDF[:, np.argsort(np.argmax(normNeuPDF, axis=0))].T)


#__________________________________________________________________________
#|                                                                        |#
#|                        FUNCTIONAL CLASSIFICATION                        |#
#|________________________________________________________________________|#
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

percList = list()
typeList = list()
layerList = list()
for mouse in miceList:
    fileName =  mouse+'_cellType.npy'
    filePath = os.path.join(dataDir, mouse)
    cellType = np.load(os.path.join(filePath, fileName))
    numCells =cellType.shape[0]

    staticCells = np.where(cellType==0)[0].shape[0]
    percList.append(staticCells/numCells)
    rotCells = np.where(np.logical_and(cellType<4,cellType>0))[0].shape[0]
    percList.append(rotCells/numCells)
    remapCells = np.where(cellType==4)[0].shape[0]
    percList.append(remapCells/numCells)
    naCells = np.where(cellType==5)[0].shape[0]
    percList.append(naCells/numCells)


    typeList += ['static', 'rot','remap', 'N/A']
    if mouse in deepMice:
        layerList += ['deep']*4
    elif mouse in supMice:
        layerList += ['sup']*4

cellsPD = pd.DataFrame(data={'typeList': typeList,
                            'layerList': layerList,
                            'percList': percList})

#PLOT BOXPLOTS
palette= ['#F28286','#F0CC6B','#856BA8','#BDBDBD']
fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.boxplot(x='layerList', y='percList', hue='typeList', data=cellsPD, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='layerList', y='percList', hue = 'typeList', data=cellsPD, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylabel('Cell Perc')
ax.set_ylim([-0.02,0.72])
plt.tight_layout()
plt.savefig(os.path.join(dataDir,'funtionalCellType.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'funtionalCellType.png'), dpi = 400,bbox_inches="tight")




from statsmodels.formula.api import ols
import statsmodels.api as sm
#perform two-way ANOVA
model = ols('percList ~ C(typeList) + C(layerList) + C(typeList):C(layerList)', data=cellsPD).fit()
sm.stats.anova_lm(model, typ=2)

for cell_type in ['static', 'rot','remap', 'N/A']:

    deep_perc = cellsPD.loc[cellsPD['layerList']=='deep'].loc[cellsPD['typeList']==cell_type]['percList'].to_list()
    sup_perc = cellsPD.loc[cellsPD['layerList']=='sup'].loc[cellsPD['typeList']==cell_type]['percList'].to_list()

    if shapiro(deep_perc).pvalue<=0.05 or shapiro(sup_perc).pvalue<=0.05:
        print(f'{cell_type} perc:',stats.ks_2samp(deep_perc, sup_perc))
    else:
        print(f'{cell_type} perc:', stats.ttest_ind(deep_perc, sup_perc))
#__________________________________________________________________________
#|                                                                        |#
#|                  DELETE FUNCTIONAL CLASSIFICATION DEEP                 |#
#|________________________________________________________________________|#
#####################
##### ROT CELLS #####
#####################
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
deepMice = ['GC2','GC3','GC5_nvista','TGrin1', 'ChZ4']
fig, ax = plt.subplots(2, 2, figsize=(10,10))
for mouse in deepMice:
    fileName =  mouse+'_remRotCells.pkl'
    filePath = os.path.join(dataDir, mouse)
    remRotDict = load_pickle(filePath, fileName)

    rotAngle = np.abs(remRotDict['rotAngle'])*180/np.pi
    x = np.arange(rotAngle.shape[1])/rotAngle.shape[1]
    m = np.mean(rotAngle, axis=0)
    sd = np.std(rotAngle, axis=0)/np.sqrt(rotAngle.shape[0])
    ax[0,0].plot(x,m, label = mouse)
    ax[0,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,0].set_ylim([0, 180])

    rotSI = np.mean(remRotDict['SIVal'], axis=0)
    x = np.arange(rotSI.shape[1])/rotSI.shape[1]
    m = np.mean(rotSI, axis=0)
    sd = np.std(rotSI, axis=0)/np.sqrt(rotSI.shape[0])
    ax[0,1].plot(x,m, label = mouse)
    ax[0,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,1].set_ylim([0, 1])

    remapDist = remRotDict['remapDist']
    remapDist = remapDist/np.nanmax(np.nanmean(remapDist,axis=0))
    x = np.arange(remapDist.shape[1])/remapDist.shape[1]
    m = np.mean(remapDist, axis=0)
    sd = np.std(remapDist, axis=0)/np.sqrt(remapDist.shape[0])
    ax[1,0].plot(x,m, label = mouse)
    ax[1,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,0].set_ylim([0, 1.1])

    entang = np.mean(remRotDict['entang'], axis=0)
    x = np.arange(entang.shape[1])/entang.shape[1]
    m = np.mean(entang, axis=0)
    sd = np.std(entang, axis=0)/np.sqrt(entang.shape[0])
    ax[1,1].plot(x,m, label = mouse)
    ax[1,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,1].set_ylim([0, 1])

ax[0,0].set_ylabel('Angle of rotation (º)')
ax[0,0].set_xlabel('Percentage of Rotation Cells Removed')
ax[0,0].legend()
ax[0,1].set_ylabel('Structure Index pos')
ax[0,1].set_xlabel('Percentage of Rotation Cells Removed')
ax[0,1].legend()
ax[1,0].set_ylabel('Norm Remap Dist')
ax[1,0].set_xlabel('Percentage of Rotation Cells Removed')
ax[1,0].legend()
ax[1,1].set_ylabel('Entanglement')
ax[1,1].set_xlabel('Percentage of Rotation Cells Removed')
ax[1,1].legend()
plt.savefig(os.path.join(dataDir,'removeRotValues_normRotcells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'removeRotValues_normRotcells.png'), dpi = 400,bbox_inches="tight")

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
deepMice = ['GC2','GC3','GC5_nvista','TGrin1', 'ChZ4']
fig, ax = plt.subplots(2, 2, figsize=(10,10))
for mouse in deepMice:
    fileName =  mouse+'_remRotCells.pkl'
    filePath = os.path.join(dataDir, mouse)
    remRotDict = load_pickle(filePath, fileName)
    numCells = len(remRotDict['newOrderList'][0])

    rotAngle = np.abs(remRotDict['rotAngle'])*180/np.pi
    x = np.arange(rotAngle.shape[1])/numCells
    m = np.mean(rotAngle, axis=0)
    sd = np.std(rotAngle, axis=0)/np.sqrt(rotAngle.shape[0])
    ax[0,0].plot(x,m, label = mouse)
    ax[0,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,0].set_ylim([0, 180])
    rotSI = np.mean(remRotDict['SIVal'], axis=0)
    x = np.arange(rotSI.shape[1])/numCells
    m = np.mean(rotSI, axis=0)
    sd = np.std(rotSI, axis=0)/np.sqrt(rotSI.shape[0])
    ax[0,1].plot(x,m, label = mouse)
    ax[0,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,1].set_ylim([0, 1])

    remapDist = remRotDict['remapDist']
    x = np.arange(remapDist.shape[1])/numCells
    m = np.mean(remapDist, axis=0)
    sd = np.std(remapDist, axis=0)/np.sqrt(remapDist.shape[0])
    ax[1,0].plot(x,m, label = mouse)
    ax[1,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,0].set_ylim([0.5, 1.7])

    entang = np.mean(remRotDict['entang'], axis=0)
    x = np.arange(entang.shape[1])/numCells
    m = np.mean(entang, axis=0)
    sd = np.std(entang, axis=0)/np.sqrt(entang.shape[0])
    ax[1,1].plot(x,m, label = mouse)
    ax[1,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,1].set_ylim([0, 0.13])

ax[0,0].set_ylabel('Angle of rotation (º)')
ax[0,0].set_xlabel('Number of Rotation Cells Removed')
ax[0,0].legend()
ax[0,1].set_ylabel('Structure Index pos')
ax[0,1].set_xlabel('Number of Rotation Cells Removed')
ax[0,1].legend()
ax[1,0].set_ylabel('Remap Dist')
ax[1,0].set_xlabel('Number of Rotation Cells Removed')
ax[1,0].legend()
ax[1,1].set_ylabel('Entanglement')
ax[1,1].set_xlabel('Number of Rotation Cells Removed')
ax[1,1].legend()
plt.savefig(os.path.join(dataDir,'removeRotValues_normAllcells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'removeRotValues_normAllcells.png'), dpi = 400,bbox_inches="tight")


#######################
##### REMAP CELLS #####
#######################
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
deepMice = ['GC2','GC3','GC5_nvista','TGrin1', 'ChZ4']
fig, ax = plt.subplots(2, 2, figsize=(10,10))
for mouse in deepMice:
    fileName =  mouse+'_remRemapCells.pkl'
    filePath = os.path.join(dataDir, mouse)
    remRotDict = load_pickle(filePath, fileName)

    rotAngle = np.abs(remRotDict['rotAngle'])*180/np.pi
    x = np.arange(rotAngle.shape[1])/rotAngle.shape[1]
    m = np.nanmean(rotAngle, axis=0)
    sd = np.nanstd(rotAngle, axis=0)/np.sqrt(rotAngle.shape[0])
    ax[0,0].plot(x,m, label = mouse)
    ax[0,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,0].set_ylim([0, 180])

    rotSI = np.mean(remRotDict['SIVal'], axis=0)
    x = np.arange(rotSI.shape[1])/rotSI.shape[1]
    m = np.nanmean(rotSI, axis=0)
    sd = np.nanstd(rotSI, axis=0)/np.sqrt(rotSI.shape[0])
    ax[0,1].plot(x,m, label = mouse)
    ax[0,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,1].set_ylim([0, 1])

    remapDist = remRotDict['remapDist']
    remapDist = remapDist/np.nanmax(np.nanmean(remapDist,axis=0))
    x = np.arange(remapDist.shape[1])/remapDist.shape[1]
    m = np.nanmean(remapDist, axis=0)
    sd = np.nanstd(remapDist, axis=0)/np.sqrt(remapDist.shape[0])
    ax[1,0].plot(x,m, label = mouse)
    ax[1,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,0].set_ylim([0, 1.1])

    entang = np.mean(remRotDict['entang'], axis=0)
    x = np.arange(entang.shape[1])/entang.shape[1]
    m = np.nanmean(entang, axis=0)
    sd = np.nanstd(entang, axis=0)/np.sqrt(entang.shape[0])
    ax[1,1].plot(x,m, label = mouse)
    ax[1,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,1].set_ylim([0, 1])

ax[0,0].set_ylabel('Angle of rotation (º)')
ax[0,0].set_xlabel('Percentage of Remap Cells Removed')
ax[0,0].legend()
ax[0,1].set_ylabel('Structure Index pos')
ax[0,1].set_xlabel('Percentage of Remap Cells Removed')
ax[0,1].legend()
ax[1,0].set_ylabel('Norm Remap Dist')
ax[1,0].set_xlabel('Percentage of Remap Cells Removed')
ax[1,0].legend()
ax[1,1].set_ylabel('Entanglement')
ax[1,1].set_xlabel('Percentage of Remap Cells Removed')
ax[1,1].legend()
plt.savefig(os.path.join(dataDir,'removeRemapValues_normRemapcells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'removeRemapValues_normRemapcells.png'), dpi = 400,bbox_inches="tight")

dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
deepMice = ['GC2','GC3','GC5_nvista','TGrin1', 'ChZ4']
fig, ax = plt.subplots(2, 2, figsize=(10,10))
for mouse in deepMice:
    fileName =  mouse+'_remRemapCells.pkl'
    filePath = os.path.join(dataDir, mouse)
    remRotDict = load_pickle(filePath, fileName)
    numCells = len(remRotDict['newOrderList'][0])

    rotAngle = np.abs(remRotDict['rotAngle'])*180/np.pi
    x = np.arange(rotAngle.shape[1])/numCells
    m = np.mean(rotAngle, axis=0)
    sd = np.std(rotAngle, axis=0)/np.sqrt(rotAngle.shape[0])
    ax[0,0].plot(x,m, label = mouse)
    ax[0,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,0].set_ylim([0, 180])

    rotSI = np.mean(remRotDict['SIVal'], axis=0)
    x = np.arange(rotSI.shape[1])/numCells
    m = np.mean(rotSI, axis=0)
    sd = np.std(rotSI, axis=0)/np.sqrt(rotSI.shape[0])
    ax[0,1].plot(x,m, label = mouse)
    ax[0,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,1].set_ylim([0, 1])

    remapDist = remRotDict['remapDist']
    remapDist = remapDist/np.nanmax(np.nanmean(remapDist,axis=0))
    x = np.arange(remapDist.shape[1])/numCells
    m = np.nanmean(remapDist, axis=0)
    sd = np.nanstd(remapDist, axis=0)/np.sqrt(remapDist.shape[0])
    ax[1,0].plot(x,m, label = mouse)
    ax[1,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,0].set_ylim([0.5, 1.7])

    entang = np.mean(remRotDict['entang'], axis=0)
    x = np.arange(entang.shape[1])/numCells
    m = np.mean(entang, axis=0)
    sd = np.std(entang, axis=0)/np.sqrt(entang.shape[0])
    ax[1,1].plot(x,m, label = mouse)
    ax[1,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,1].set_ylim([0, 0.13])

ax[0,0].set_ylabel('Angle of rotation (º)')
ax[0,0].set_xlabel('Number of Remap Cells Removed')
ax[0,0].legend()
ax[0,1].set_ylabel('Structure Index pos')
ax[0,1].set_xlabel('Number of Remap Cells Removed')
ax[0,1].legend()
ax[1,0].set_ylabel('Remap Dist')
ax[1,0].set_xlabel('Number of Remap Cells Removed')
ax[1,0].legend()
ax[1,1].set_ylabel('Entanglement')
ax[1,1].set_xlabel('Number of Remap Cells Removed')
ax[1,1].legend()
plt.savefig(os.path.join(dataDir,'removeRemapValues_normAllcells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'removeRemapValues_normAllcells.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                  DELETE FUNCTIONAL CLASSIFICATION SUP                  |#
#|________________________________________________________________________|#

########################
#####  REMAP CELLS #####
########################
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
fig, ax = plt.subplots(2, 2, figsize=(10,10))
for mouse in supMice:
    fileName =  mouse+'_remRemapCells.pkl'
    filePath = os.path.join(dataDir, mouse)
    remRotDict = load_pickle(filePath, fileName)
    nCells = np.where(np.sum(remRotDict['rotAngle'],axis=0)==0)[0][0]

    rotAngle = np.abs(remRotDict['rotAngle'][:,:nCells])*180/np.pi
    x = np.arange(nCells)/nCells
    m = np.nanmean(rotAngle, axis=0)
    sd = np.nanstd(rotAngle, axis=0)/np.sqrt(rotAngle.shape[0])
    ax[0,0].plot(x,m, label = mouse)
    ax[0,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,0].set_ylim([0, 180])

    rotSI = np.mean(remRotDict['SIVal'], axis=0)[:,:nCells]
    x = np.arange(nCells)/nCells
    m = np.nanmean(rotSI, axis=0)
    sd = np.nanstd(rotSI, axis=0)/np.sqrt(rotSI.shape[0])
    ax[0,1].plot(x,m, label = mouse)
    ax[0,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,1].set_ylim([0, 1])

    remapDist = remRotDict['remapDist'][:,:nCells]
    remapDist = remapDist/np.nanmax(np.nanmean(remapDist,axis=0))
    x = np.arange(nCells)/nCells
    m = np.nanmean(remapDist, axis=0)
    sd = np.nanstd(remapDist, axis=0)/np.sqrt(remapDist.shape[0])
    ax[1,0].plot(x,m, label = mouse)
    ax[1,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,0].set_ylim([0, 1.1])

    entang = np.mean(remRotDict['entang'], axis=0)[:,:nCells]
    x = np.arange(nCells)/nCells
    m = np.nanmean(entang, axis=0)
    sd = np.nanstd(entang, axis=0)/np.sqrt(entang.shape[0])
    ax[1,1].plot(x,m, label = mouse)
    ax[1,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,1].set_ylim([0, 1])

ax[0,0].set_ylabel('Angle of rotation (º)')
ax[0,0].set_xlabel('Percentage of Remap Cells Removed')
ax[0,0].legend()
ax[0,1].set_ylabel('Structure Index pos')
ax[0,1].set_xlabel('Percentage of Remap Cells Removed')
ax[0,1].legend()
ax[1,0].set_ylabel('Norm Remap Dist')
ax[1,0].set_xlabel('Percentage of Remap Cells Removed')
ax[1,0].legend()
ax[1,1].set_ylabel('Entanglement')
ax[1,1].set_xlabel('Percentage of Remap Cells Removed')
ax[1,1].legend()
plt.savefig(os.path.join(dataDir,'removeRemapValues_Sup_normRemapcells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'removeRemapValues_Sup_normRemapcells.png'), dpi = 400,bbox_inches="tight")

#######################
#####  ALLO CELLS #####
#######################
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
fig, ax = plt.subplots(2, 2, figsize=(10,10))
for mouse in supMice:
    fileName =  mouse+'_remAlloCells.pkl'
    filePath = os.path.join(dataDir, mouse)
    remRotDict = load_pickle(filePath, fileName)
    # numCells = len(remRotDict['newOrderList'][0])

    rotAngle = np.abs(remRotDict['rotAngle'])*180/np.pi
    x = np.arange(rotAngle.shape[1])/rotAngle.shape[1]
    m = np.mean(rotAngle, axis=0)
    sd = np.std(rotAngle, axis=0)/np.sqrt(rotAngle.shape[0])
    ax[0,0].plot(x,m, label = mouse)
    ax[0,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,0].set_ylim([0, 180])

    rotSI = np.mean(remRotDict['SIVal'], axis=0)
    x = np.arange(rotSI.shape[1])/rotSI.shape[1]
    m = np.mean(rotSI, axis=0)
    sd = np.std(rotSI, axis=0)/np.sqrt(rotSI.shape[0])
    ax[0,1].plot(x,m, label = mouse)
    ax[0,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[0,1].set_ylim([0, 1])

    remapDist = remRotDict['remapDist']
    remapDist = remapDist/np.nanmax(np.nanmean(remapDist,axis=0))
    x = np.arange(remapDist.shape[1])/remapDist.shape[1]
    m = np.nanmean(remapDist, axis=0)
    sd = np.nanstd(remapDist, axis=0)/np.sqrt(remapDist.shape[0])
    ax[1,0].plot(x,m, label = mouse)
    ax[1,0].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,0].set_ylim([0, 1.1])

    entang = np.mean(remRotDict['entang'], axis=0)
    x = np.arange(entang.shape[1])/entang.shape[1]
    m = np.mean(entang, axis=0)
    sd = np.std(entang, axis=0)/np.sqrt(entang.shape[0])
    ax[1,1].plot(x,m, label = mouse)
    ax[1,1].fill_between(x, m-sd, m+sd,alpha = 0.3)
    ax[1,1].set_ylim([0, 1])

ax[0,0].set_ylabel('Angle of rotation (º)')
ax[0,0].set_xlabel('Number of Allo Cells Removed')
ax[0,0].legend()
ax[0,1].set_ylabel('Structure Index pos')
ax[0,1].set_xlabel('Number of Allo Cells Removed')
ax[0,1].legend()
ax[1,0].set_ylabel('Allo Dist')
ax[1,0].set_xlabel('Number of Allo Cells Removed')
ax[1,0].legend()
ax[1,1].set_ylabel('Entanglement')
ax[1,1].set_xlabel('Number of Allo Cells Removed')
ax[1,1].legend()
plt.savefig(os.path.join(dataDir,'removeAlloValues_normAllocells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'removeAlloValues_normAllocells.png'), dpi = 400,bbox_inches="tight")
#__________________________________________________________________________
#|                                                                        |#
#|         EXAMPLE EMBEDDING DELETE FUNCTIONAL CLASSIFICATION  DEEP       |#
#|________________________________________________________________________|#
def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx


def personalize_ax(ax, ax_view = None):
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if ax_view:
        ax.view_init(ax_view[0], ax_view[1])


#####################
##### ROT CELLS #####
#####################
dataDirPC = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
dataDirPD = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

deepMice = ['GC2','GC3','GC5_nvista','TGrin1', 'ChZ4']
for mouse in deepMice:
    fileNamePC =  mouse+'_remRotCells.pkl'
    filePathPC = os.path.join(dataDirPC, mouse)
    remRotDict = load_pickle(filePathPC, fileNamePC)

    fileNamePD =  mouse+'_df_dict.pkl'
    filePathPD = os.path.join(dataDirPD, mouse)
    animal = load_pickle(filePathPD,fileNamePD)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])


    a = np.argsort(np.abs(remRotDict['rotAngle'].flatten('F')))
    index = np.unravel_index(a, remRotDict['rotAngle'].shape, 'F')

    for x in range(10):

        posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
        dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
        embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

        posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
        dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
        embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

        embPre0 = remRotDict['embPreSave'][(index[0][x],index[1][x])]
        embRot0 = remRotDict['embRotSave'][(index[0][x],index[1][x])]


        DPre = pairwise_distances(embPre0)
        noiseIdxPre = filter_noisy_outliers(embPre0,DPre)
        embPre = embPre[~noiseIdxPre,:]
        embPre0 = embPre0[~noiseIdxPre,:]
        posPre = posPre[~noiseIdxPre,:]
        dirMatPre = dirMatPre[~noiseIdxPre]

        DRot = pairwise_distances(embRot0)
        noiseIdxRot = filter_noisy_outliers(embRot0,DRot)
        # noiseIdxRot[dirMatRot==0] = True
        embRot0 = embRot0[~noiseIdxRot,:]
        embRot = embRot[~noiseIdxRot,:]
        posRot = posRot[~noiseIdxRot,:]
        dirMatRot = dirMatRot[~noiseIdxRot]


        dirColorPre = np.zeros((dirMatPre.shape[0],3))
        for point in range(dirMatPre.shape[0]):
            if dirMatPre[point]==0:
                dirColorPre[point] = [14/255,14/255,143/255]
            elif dirMatPre[point]==1:
                dirColorPre[point] = [12/255,136/255,249/255]
            else:
                dirColorPre[point] = [17/255,219/255,224/255]
        dirColorRot = np.zeros((dirMatRot.shape[0],3))
        for point in range(dirMatRot.shape[0]):
            if dirMatRot[point]==0:
                dirColorRot[point] = [14/255,14/255,143/255]
            elif dirMatRot[point]==1:
                dirColorRot[point] = [12/255,136/255,249/255]
            else:
                dirColorRot[point] = [17/255,219/255,224/255]


        fig = plt.figure(figsize=((6,6)))
        ax = plt.subplot(2,3,1, projection = '3d')
        ax.scatter(*embPre[:,:3].T, c = 'b',s = 30)
        ax.scatter(*embRot[:,:3].T, c = 'r',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,2, projection = '3d')
        ax.scatter(*embPre[:,:3].T, color = dirColorPre,s = 30)
        ax.scatter(*embRot[:,:3].T, color = dirColorRot,s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,3, projection = '3d')
        ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap='inferno',s = 30)
        ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap='inferno',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(2,3,4, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, color = 'b',s = 30)
        ax.scatter(*embRot0[:,:3].T, color = 'r',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,5, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, color = dirColorPre,s = 30)
        ax.scatter(*embRot0[:,:3].T, color = dirColorRot,s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,6, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, c = posPre[:,0], cmap='inferno',s = 30)
        ax.scatter(*embRot0[:,:3].T, c = posRot[:,0], cmap='inferno',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        fig.suptitle(mouse+'-'+str(x))
        plt.tight_layout()


exampleDict = {
    'GC2':8,
    'TGrin1': 6
}

mouse = 'TGrin1'
fileNamePC =  mouse+'_remRotCells.pkl'
filePathPC = os.path.join(dataDirPC, mouse)
remRotDict = load_pickle(filePathPC, fileNamePC)

fileNamePD =  mouse+'_df_dict.pkl'
filePathPD = os.path.join(dataDirPD, mouse)
animal = load_pickle(filePathPD,fileNamePD)
fnames = list(animal.keys())
fnamePre = [fname for fname in fnames if 'lt' in fname][0]
fnameRot = [fname for fname in fnames if 'rot' in fname][0]
animalPre= copy.deepcopy(animal[fnamePre])
animalRot= copy.deepcopy(animal[fnameRot])

posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

a = np.argsort(np.abs(remRotDict['rotAngle'].flatten('F')))
index = np.unravel_index(a, remRotDict['rotAngle'].shape, 'F')

embPre0 = remRotDict['embPreSave'][(index[0][exampleDict[mouse]],index[1][exampleDict[mouse]])]
embRot0 = remRotDict['embRotSave'][(index[0][exampleDict[mouse]],index[1][exampleDict[mouse]])]

DPre = pairwise_distances(embPre0)
noiseIdxPre = filter_noisy_outliers(embPre0,DPre)
embPre = embPre[~noiseIdxPre,:]
embPre0 = embPre0[~noiseIdxPre,:]
posPre = posPre[~noiseIdxPre,:]
dirMatPre = dirMatPre[~noiseIdxPre]

DRot = pairwise_distances(embRot0)
noiseIdxRot = filter_noisy_outliers(embRot0,DRot)
embRot0 = embRot0[~noiseIdxRot,:]
embRot = embRot[~noiseIdxRot,:]
posRot = posRot[~noiseIdxRot,:]
dirMatRot = dirMatRot[~noiseIdxRot]

dirColorPre = np.zeros((dirMatPre.shape[0],3))
for point in range(dirMatPre.shape[0]):
    if dirMatPre[point]==0:
        dirColorPre[point] = [14/255,14/255,143/255]
    elif dirMatPre[point]==1:
        dirColorPre[point] = [12/255,136/255,249/255]
    else:
        dirColorPre[point] = [17/255,219/255,224/255]
dirColorRot = np.zeros((dirMatRot.shape[0],3))
for point in range(dirMatRot.shape[0]):
    if dirMatRot[point]==0:
        dirColorRot[point] = [14/255,14/255,143/255]
    elif dirMatRot[point]==1:
        dirColorRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRot[point] = [17/255,219/255,224/255]

fig = plt.figure(figsize=((13,9)))
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = 'b',s = 10)
ax.scatter(*embRot[:,:3].T, c = 'r',s = 10)
personalize_ax(ax)
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*embPre[:,:3].T, color = dirColorPre,s = 10)
ax.scatter(*embRot[:,:3].T, color = dirColorRot,s = 10)
personalize_ax(ax)
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap='inferno',s = 10)
ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap='inferno',s = 10)
personalize_ax(ax)
ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*embPre0[:,:3].T, color = 'b',s = 10)
ax.scatter(*embRot0[:,:3].T, color = 'r',s = 10)
personalize_ax(ax)
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*embPre0[:,:3].T, color = dirColorPre,s = 10)
ax.scatter(*embRot0[:,:3].T, color = dirColorRot,s = 10)
personalize_ax(ax)
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*embPre0[:,:3].T, c = posPre[:,0], cmap='inferno',s = 10)
ax.scatter(*embRot0[:,:3].T, c = posRot[:,0], cmap='inferno',s = 10)
personalize_ax(ax)
fig.suptitle(mouse+'-'+str(exampleDict[mouse]))
plt.tight_layout()
plt.savefig(os.path.join(dataDir,mouse+'_remRotCells_emb.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,mouse+'_remRotCells_emb.png'), dpi = 400,bbox_inches="tight")


#######################
##### REMAP CELLS #####
#######################
dataDirPC = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
dataDirPD = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

deepMice = ['GC2','GC3','GC5_nvista','TGrin1', 'ChZ4']
for mouse in deepMice:
    fileNamePC =  mouse+'_remRemapCells.pkl'
    filePathPC = os.path.join(dataDirPC, mouse)
    remRemapDict = load_pickle(filePathPC, fileNamePC)

    fileNamePD =  mouse+'_df_dict.pkl'
    filePathPD = os.path.join(dataDirPD, mouse)
    animal = load_pickle(filePathPD,fileNamePD)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
    embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

    a = np.argsort(np.abs(remRemapDict['remapDist'].flatten('F')))
    index = np.unravel_index(a, remRemapDict['remapDist'].shape, 'F')

    for x in range(10):
        posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
        dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
        embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))
        posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
        dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
        embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

        embPre0 = remRemapDict['embPreSave'][(index[0][x],index[1][x])]
        embRot0 = remRemapDict['embRotSave'][(index[0][x],index[1][x])]

        DPre = pairwise_distances(embPre0)
        noiseIdxPre = filter_noisy_outliers(embPre0,DPre)
        embPre = embPre[~noiseIdxPre,:]
        embPre0 = embPre0[~noiseIdxPre,:]
        posPre = posPre[~noiseIdxPre,:]
        dirMatPre = dirMatPre[~noiseIdxPre]

        DRot = pairwise_distances(embRot0)
        noiseIdxRot = filter_noisy_outliers(embRot0,DRot)
        embRot0 = embRot0[~noiseIdxRot,:]
        embRot = embRot[~noiseIdxRot,:]
        posRot = posRot[~noiseIdxRot,:]
        dirMatRot = dirMatRot[~noiseIdxRot]


        dirColorPre = np.zeros((dirMatPre.shape[0],3))
        for point in range(dirMatPre.shape[0]):
            if dirMatPre[point]==0:
                dirColorPre[point] = [14/255,14/255,143/255]
            elif dirMatPre[point]==1:
                dirColorPre[point] = [12/255,136/255,249/255]
            else:
                dirColorPre[point] = [17/255,219/255,224/255]
        dirColorRot = np.zeros((dirMatRot.shape[0],3))
        for point in range(dirMatRot.shape[0]):
            if dirMatRot[point]==0:
                dirColorRot[point] = [14/255,14/255,143/255]
            elif dirMatRot[point]==1:
                dirColorRot[point] = [12/255,136/255,249/255]
            else:
                dirColorRot[point] = [17/255,219/255,224/255]


        fig = plt.figure(figsize=((6,6)))
        ax = plt.subplot(2,3,1, projection = '3d')
        ax.scatter(*embPre[:,:3].T, c = 'b',s = 30)
        ax.scatter(*embRot[:,:3].T, c = 'r',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,2, projection = '3d')
        ax.scatter(*embPre[:,:3].T, color = dirColorPre,s = 30)
        ax.scatter(*embRot[:,:3].T, color = dirColorRot,s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,3, projection = '3d')
        ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap='inferno',s = 30)
        ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap='inferno',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(2,3,4, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, color = 'b',s = 30)
        ax.scatter(*embRot0[:,:3].T, color = 'r',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,5, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, color = dirColorPre,s = 30)
        ax.scatter(*embRot0[:,:3].T, color = dirColorRot,s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,6, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, c = posPre[:,0], cmap='inferno',s = 30)
        ax.scatter(*embRot0[:,:3].T, c = posRot[:,0], cmap='inferno',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        fig.suptitle(mouse+'-'+str(x))
        plt.tight_layout()

mouse = 'GC2'
fileNamePC =  mouse+'_remRemapCells.pkl'
filePathPC = os.path.join(dataDirPC, mouse)
remRemapDict = load_pickle(filePathPC, fileNamePC)

fileNamePD =  mouse+'_df_dict.pkl'
filePathPD = os.path.join(dataDirPD, mouse)
animal = load_pickle(filePathPD,fileNamePD)
fnames = list(animal.keys())
fnamePre = [fname for fname in fnames if 'lt' in fname][0]
fnameRot = [fname for fname in fnames if 'rot' in fname][0]
animalPre= copy.deepcopy(animal[fnamePre])
animalRot= copy.deepcopy(animal[fnameRot])

posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

a = np.argsort(np.abs(remRemapDict['remapDist'].flatten('F')))
index = np.unravel_index(a, remRemapDict['remapDist'].shape, 'F')


embPre0 = remRemapDict['embPreSave'][(index[0][8],index[1][8])]
embRot0 = remRemapDict['embRotSave'][(index[0][8],index[1][8])]

DPre = pairwise_distances(embPre0)
noiseIdxPre = filter_noisy_outliers(embPre0,DPre)
embPre = embPre[~noiseIdxPre,:]
embPre0 = embPre0[~noiseIdxPre,:]
posPre = posPre[~noiseIdxPre,:]
dirMatPre = dirMatPre[~noiseIdxPre]

DRot = pairwise_distances(embRot0)
noiseIdxRot = filter_noisy_outliers(embRot0,DRot)
embRot0 = embRot0[~noiseIdxRot,:]
embRot = embRot[~noiseIdxRot,:]
posRot = posRot[~noiseIdxRot,:]
dirMatRot = dirMatRot[~noiseIdxRot]


dirColorPre = np.zeros((dirMatPre.shape[0],3))
for point in range(dirMatPre.shape[0]):
    if dirMatPre[point]==0:
        dirColorPre[point] = [14/255,14/255,143/255]
    elif dirMatPre[point]==1:
        dirColorPre[point] = [12/255,136/255,249/255]
    else:
        dirColorPre[point] = [17/255,219/255,224/255]
dirColorRot = np.zeros((dirMatRot.shape[0],3))
for point in range(dirMatRot.shape[0]):
    if dirMatRot[point]==0:
        dirColorRot[point] = [14/255,14/255,143/255]
    elif dirMatRot[point]==1:
        dirColorRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRot[point] = [17/255,219/255,224/255]


fig = plt.figure(figsize=((13,9)))
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = 'b',s = 30)
ax.scatter(*embRot[:,:3].T, c = 'r',s = 30)
personalize_ax(ax, [142,92])
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*np.concatenate((embPre, embRot),axis=0)[:,:3].T, c = np.concatenate((dirColorPre, dirColorRot),axis=0), cmap='inferno',s = 30)
personalize_ax(ax, [142,92])
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*np.concatenate((embPre, embRot),axis=0)[:,:3].T, c = np.concatenate((posPre, posRot),axis=0)[:,0], cmap='inferno',s = 30)
personalize_ax(ax, [142,92])
ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*embPre0[:,:3].T, color = 'b',s = 30)
ax.scatter(*embRot0[:,:3].T, color = 'r',s = 30)
personalize_ax(ax, [-167, 62])
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*np.concatenate((embPre0, embRot0),axis=0)[:,:3].T, c = np.concatenate((dirColorPre, dirColorRot),axis=0), cmap='inferno',s = 30)
personalize_ax(ax, [-167, 62])
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*np.concatenate((embPre0, embRot0),axis=0)[:,:3].T, c = np.concatenate((posPre, posRot),axis=0)[:,0], cmap='inferno',s = 30)
personalize_ax(ax, [-167, 62])
fig.suptitle(mouse+'-'+str(8))
plt.tight_layout()
plt.savefig(os.path.join(dataDir,mouse+'_remRemapCells_emb.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,mouse+'_remRemapCells_emb.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|         EXAMPLE EMBEDDING DELETE FUNCTIONAL CLASSIFICATION  SUP        |#
#|________________________________________________________________________|#
def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx

def personalize_ax(ax, ax_view = None):
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if ax_view:
        ax.view_init(ax_view[0], ax_view[1])

######################
##### ALLO CELLS #####
######################
dataDirPC = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
dataDirPD = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
for mouse in supMice:
    fileNamePC =  mouse+'_remAlloCells.pkl'
    filePathPC = os.path.join(dataDirPC, mouse)
    remRotDict = load_pickle(filePathPC, fileNamePC)

    fileNamePD =  mouse+'_df_dict.pkl'
    filePathPD = os.path.join(dataDirPD, mouse)
    animal = load_pickle(filePathPD,fileNamePD)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    a = np.argsort(np.abs(remRotDict['rotAngle'].flatten('F')))[::-1]
    index = np.unravel_index(a, remRotDict['rotAngle'].shape, 'F')

    for x in range(10):
        posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
        dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
        embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

        posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
        dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
        embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

        embPre0 = remRotDict['embPreSave'][(index[0][x],index[1][x])]
        embRot0 = remRotDict['embRotSave'][(index[0][x],index[1][x])]


        DPre = pairwise_distances(embPre0)
        noiseIdxPre = filter_noisy_outliers(embPre0,DPre)
        embPre = embPre[~noiseIdxPre,:]
        embPre0 = embPre0[~noiseIdxPre,:]
        posPre = posPre[~noiseIdxPre,:]
        dirMatPre = dirMatPre[~noiseIdxPre]

        DRot = pairwise_distances(embRot0)
        noiseIdxRot = filter_noisy_outliers(embRot0,DRot)
        # noiseIdxRot[dirMatRot==0] = True
        embRot0 = embRot0[~noiseIdxRot,:]
        embRot = embRot[~noiseIdxRot,:]
        posRot = posRot[~noiseIdxRot,:]
        dirMatRot = dirMatRot[~noiseIdxRot]


        dirColorPre = np.zeros((dirMatPre.shape[0],3))
        for point in range(dirMatPre.shape[0]):
            if dirMatPre[point]==0:
                dirColorPre[point] = [14/255,14/255,143/255]
            elif dirMatPre[point]==1:
                dirColorPre[point] = [12/255,136/255,249/255]
            else:
                dirColorPre[point] = [17/255,219/255,224/255]
        dirColorRot = np.zeros((dirMatRot.shape[0],3))
        for point in range(dirMatRot.shape[0]):
            if dirMatRot[point]==0:
                dirColorRot[point] = [14/255,14/255,143/255]
            elif dirMatRot[point]==1:
                dirColorRot[point] = [12/255,136/255,249/255]
            else:
                dirColorRot[point] = [17/255,219/255,224/255]


        fig = plt.figure(figsize=((6,6)))
        ax = plt.subplot(2,3,1, projection = '3d')
        ax.scatter(*embPre[:,:3].T, c = 'b',s=10)
        ax.scatter(*embRot[:,:3].T, c = 'r',s=10)
        personalize_ax(ax)
        ax = plt.subplot(2,3,2, projection = '3d')
        ax.scatter(*embPre[:,:3].T, color = dirColorPre,s=10)
        ax.scatter(*embRot[:,:3].T, color = dirColorRot,s=10)
        personalize_ax(ax)
        ax = plt.subplot(2,3,3, projection = '3d')
        ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap='inferno',s=10)
        ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap='inferno',s=10)
        personalize_ax(ax)

        ax = plt.subplot(2,3,4, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, color = 'b',s=10)
        ax.scatter(*embRot0[:,:3].T, color = 'r',s=10)
        personalize_ax(ax)
        ax = plt.subplot(2,3,5, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, color = dirColorPre,s=10)
        ax.scatter(*embRot0[:,:3].T, color = dirColorRot,s=10)
        personalize_ax(ax)
        ax = plt.subplot(2,3,6, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, c = posPre[:,0], cmap='inferno',s=10)
        ax.scatter(*embRot0[:,:3].T, c = posRot[:,0], cmap='inferno',s=10)
        personalize_ax(ax)
        fig.suptitle(mouse+'-'+str(x))
        plt.tight_layout()


exampleDict = {
    'CZ3':5,
    'CZ6': 6
}

mouse = 'CZ6'
fileNamePC =  mouse+'_remAlloCells.pkl'
filePathPC = os.path.join(dataDirPC, mouse)
remRotDict = load_pickle(filePathPC, fileNamePC)

fileNamePD =  mouse+'_df_dict.pkl'
filePathPD = os.path.join(dataDirPD, mouse)
animal = load_pickle(filePathPD,fileNamePD)
fnames = list(animal.keys())
fnamePre = [fname for fname in fnames if 'lt' in fname][0]
fnameRot = [fname for fname in fnames if 'rot' in fname][0]
animalPre= copy.deepcopy(animal[fnamePre])
animalRot= copy.deepcopy(animal[fnameRot])

posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

a = np.argsort(np.abs(remRotDict['rotAngle'].flatten('F')))[::-1]
index = np.unravel_index(a, remRotDict['rotAngle'].shape, 'F')

embPre0 = remRotDict['embPreSave'][(index[0][exampleDict[mouse]],index[1][exampleDict[mouse]])]
embRot0 = remRotDict['embRotSave'][(index[0][exampleDict[mouse]],index[1][exampleDict[mouse]])]

DPre = pairwise_distances(embPre0)
noiseIdxPre = filter_noisy_outliers(embPre0,DPre)
embPre = embPre[~noiseIdxPre,:]
embPre0 = embPre0[~noiseIdxPre,:]
posPre = posPre[~noiseIdxPre,:]
dirMatPre = dirMatPre[~noiseIdxPre]

DRot = pairwise_distances(embRot0)
noiseIdxRot = filter_noisy_outliers(embRot0,DRot)
embRot0 = embRot0[~noiseIdxRot,:]
embRot = embRot[~noiseIdxRot,:]
posRot = posRot[~noiseIdxRot,:]
dirMatRot = dirMatRot[~noiseIdxRot]

dirColorPre = np.zeros((dirMatPre.shape[0],3))
for point in range(dirMatPre.shape[0]):
    if dirMatPre[point]==0:
        dirColorPre[point] = [14/255,14/255,143/255]
    elif dirMatPre[point]==1:
        dirColorPre[point] = [12/255,136/255,249/255]
    else:
        dirColorPre[point] = [17/255,219/255,224/255]
dirColorRot = np.zeros((dirMatRot.shape[0],3))
for point in range(dirMatRot.shape[0]):
    if dirMatRot[point]==0:
        dirColorRot[point] = [14/255,14/255,143/255]
    elif dirMatRot[point]==1:
        dirColorRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRot[point] = [17/255,219/255,224/255]

fig = plt.figure(figsize=((13,9)))
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = 'b',s = 10)
ax.scatter(*embRot[:,:3].T, c = 'r',s = 10)
personalize_ax(ax, [20,98])
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*embPre[:,:3].T, color = dirColorPre,s = 10)
ax.scatter(*embRot[:,:3].T, color = dirColorRot,s = 10)
personalize_ax(ax, [20,98])
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap='inferno',s = 10)
ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap='inferno',s = 10)
personalize_ax(ax, [20,98])
ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*embPre0[:,:3].T, color = 'b',s = 10)
ax.scatter(*embRot0[:,:3].T, color = 'r',s = 10)
personalize_ax(ax, [-3,-57])
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*embPre0[:,:3].T, color = dirColorPre,s = 10)
ax.scatter(*embRot0[:,:3].T, color = dirColorRot,s = 10)
personalize_ax(ax, [-3,-57])
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*embPre0[:,:3].T, c = posPre[:,0], cmap='inferno',s = 10)
ax.scatter(*embRot0[:,:3].T, c = posRot[:,0], cmap='inferno',s = 10)
personalize_ax(ax, [-3,-57])
fig.suptitle(mouse)
plt.tight_layout()
plt.savefig(os.path.join(dataDir,mouse+'_remAloCells_emb.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,mouse+'_remAloCells_emb.png'), dpi = 400,bbox_inches="tight")


#######################
##### REMAP CELLS #####
#######################
dataDirPC = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
dataDirPD = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
for mouse in supMice:
    fileNamePC =  mouse+'_remRemapCells.pkl'
    filePathPC = os.path.join(dataDirPC, mouse)
    remRemapDict = load_pickle(filePathPC, fileNamePC)

    fileNamePD =  mouse+'_df_dict.pkl'
    filePathPD = os.path.join(dataDirPD, mouse)
    animal = load_pickle(filePathPD,fileNamePD)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
    dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
    embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

    posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

    a = np.argsort(np.abs(remRemapDict['remapDist'].flatten('F')))
    index = np.unravel_index(a, remRemapDict['remapDist'].shape, 'F')

    for x in range(10):
        posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
        dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
        embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))
        posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
        dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
        embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

        embPre0 = remRemapDict['embPreSave'][(index[0][x],index[1][x])]
        embRot0 = remRemapDict['embRotSave'][(index[0][x],index[1][x])]

        DPre = pairwise_distances(embPre0)
        noiseIdxPre = filter_noisy_outliers(embPre0,DPre)
        embPre = embPre[~noiseIdxPre,:]
        embPre0 = embPre0[~noiseIdxPre,:]
        posPre = posPre[~noiseIdxPre,:]
        dirMatPre = dirMatPre[~noiseIdxPre]

        DRot = pairwise_distances(embRot0)
        noiseIdxRot = filter_noisy_outliers(embRot0,DRot)
        embRot0 = embRot0[~noiseIdxRot,:]
        embRot = embRot[~noiseIdxRot,:]
        posRot = posRot[~noiseIdxRot,:]
        dirMatRot = dirMatRot[~noiseIdxRot]


        dirColorPre = np.zeros((dirMatPre.shape[0],3))
        for point in range(dirMatPre.shape[0]):
            if dirMatPre[point]==0:
                dirColorPre[point] = [14/255,14/255,143/255]
            elif dirMatPre[point]==1:
                dirColorPre[point] = [12/255,136/255,249/255]
            else:
                dirColorPre[point] = [17/255,219/255,224/255]
        dirColorRot = np.zeros((dirMatRot.shape[0],3))
        for point in range(dirMatRot.shape[0]):
            if dirMatRot[point]==0:
                dirColorRot[point] = [14/255,14/255,143/255]
            elif dirMatRot[point]==1:
                dirColorRot[point] = [12/255,136/255,249/255]
            else:
                dirColorRot[point] = [17/255,219/255,224/255]


        fig = plt.figure(figsize=((6,6)))
        ax = plt.subplot(2,3,1, projection = '3d')
        ax.scatter(*embPre[:,:3].T, c = 'b',s = 30)
        ax.scatter(*embRot[:,:3].T, c = 'r',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,2, projection = '3d')
        ax.scatter(*embPre[:,:3].T, color = dirColorPre,s = 30)
        ax.scatter(*embRot[:,:3].T, color = dirColorRot,s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,3, projection = '3d')
        ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap='inferno',s = 30)
        ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap='inferno',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(2,3,4, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, color = 'b',s = 30)
        ax.scatter(*embRot0[:,:3].T, color = 'r',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,5, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, color = dirColorPre,s = 30)
        ax.scatter(*embRot0[:,:3].T, color = dirColorRot,s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,6, projection = '3d')
        ax.scatter(*embPre0[:,:3].T, c = posPre[:,0], cmap='inferno',s = 30)
        ax.scatter(*embRot0[:,:3].T, c = posRot[:,0], cmap='inferno',s = 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        fig.suptitle(mouse+'-'+str(x))
        plt.tight_layout()

mouse = 'CZ8'
fileNamePC =  mouse+'_remRemapCells.pkl'
filePathPC = os.path.join(dataDirPC, mouse)
remRemapDict = load_pickle(filePathPC, fileNamePC)

fileNamePD =  mouse+'_df_dict.pkl'
filePathPD = os.path.join(dataDirPD, mouse)
animal = load_pickle(filePathPD,fileNamePD)
fnames = list(animal.keys())
fnamePre = [fname for fname in fnames if 'lt' in fname][0]
fnameRot = [fname for fname in fnames if 'rot' in fname][0]
animalPre= copy.deepcopy(animal[fnamePre])
animalRot= copy.deepcopy(animal[fnameRot])

posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
dirMatPre = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis=0))
embPre = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis=0))

posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

a = np.argsort(np.abs(remRemapDict['remapDist'][:,:45].flatten('F')))
index = np.unravel_index(a, remRemapDict['remapDist'][:,:45].shape, 'F')

# a = np.argsort(np.abs(remRemapDict['remapDist'][:,:67].flatten('F')))
# index = np.unravel_index(a, (100,67), 'F')

embPre0 = remRemapDict['embPreSave'][(index[0][2],index[1][2])]
embRot0 = remRemapDict['embRotSave'][(index[0][2],index[1][2])]

DPre = pairwise_distances(embPre0)
noiseIdxPre = filter_noisy_outliers(embPre0,DPre)
embPre = embPre[~noiseIdxPre,:]
embPre0 = embPre0[~noiseIdxPre,:]
posPre = posPre[~noiseIdxPre,:]
dirMatPre = dirMatPre[~noiseIdxPre]

DRot = pairwise_distances(embRot0)
noiseIdxRot = filter_noisy_outliers(embRot0,DRot)
embRot0 = embRot0[~noiseIdxRot,:]
embRot = embRot[~noiseIdxRot,:]
posRot = posRot[~noiseIdxRot,:]
dirMatRot = dirMatRot[~noiseIdxRot]


dirColorPre = np.zeros((dirMatPre.shape[0],3))
for point in range(dirMatPre.shape[0]):
    if dirMatPre[point]==0:
        dirColorPre[point] = [14/255,14/255,143/255]
    elif dirMatPre[point]==1:
        dirColorPre[point] = [12/255,136/255,249/255]
    else:
        dirColorPre[point] = [17/255,219/255,224/255]
dirColorRot = np.zeros((dirMatRot.shape[0],3))
for point in range(dirMatRot.shape[0]):
    if dirMatRot[point]==0:
        dirColorRot[point] = [14/255,14/255,143/255]
    elif dirMatRot[point]==1:
        dirColorRot[point] = [12/255,136/255,249/255]
    else:
        dirColorRot[point] = [17/255,219/255,224/255]


fig = plt.figure(figsize=((13,9)))
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = 'b',s = 30)
ax.scatter(*embRot[:,:3].T, c = 'r',s = 30)
personalize_ax(ax, [-150,132])
ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*np.concatenate((embPre, embRot),axis=0)[:,:3].T, c = np.concatenate((dirColorPre, dirColorRot),axis=0), cmap='inferno',s = 30)
personalize_ax(ax, [-150,132])
ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*np.concatenate((embPre, embRot),axis=0)[:,:3].T, c = np.concatenate((posPre, posRot),axis=0)[:,0], cmap='inferno',s = 30)
personalize_ax(ax, [-150,132])
ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*embPre0[:,:3].T, color = 'b',s = 30)
ax.scatter(*embRot0[:,:3].T, color = 'r',s = 30)
personalize_ax(ax, [1,-70])
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*np.concatenate((embPre0, embRot0),axis=0)[:,:3].T, c = np.concatenate((dirColorPre, dirColorRot),axis=0), cmap='inferno',s = 30)
personalize_ax(ax, [1,-70])
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*np.concatenate((embPre0, embRot0),axis=0)[:,:3].T, c = np.concatenate((posPre, posRot),axis=0)[:,0], cmap='inferno',s = 30)
personalize_ax(ax, [1,-70])
fig.suptitle(mouse+'-'+str(8))
plt.tight_layout()
plt.savefig(os.path.join(dataDir,mouse+'_remRemapCells_emb.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,mouse+'_remRemapCells_emb.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                                 UMAP EMB                               |#
#|________________________________________________________________________|#

mouse = 'GC2'
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
pcdataDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/manifold_cells/'
pcManifoldDict = load_pickle(pcdataDir, 'manifold_pc_dict.pkl')

print(f"Working on mouse {mouse}: ")
fileName =  mouse+'_df_dict.pkl'
filePath = os.path.join(dataDir, mouse)
animal = load_pickle(filePath,fileName)
fnames = list(animal.keys())
fnamePre = [fname for fname in fnames if 'rot' in fname][0]
pdMouse= copy.deepcopy(animal[fnamePre])

pos = copy.deepcopy(np.concatenate(pdMouse ['pos'].values, axis=0))
dirMat = copy.deepcopy(np.concatenate(pdMouse ['dir_mat'].values, axis=0))
signal = copy.deepcopy(np.concatenate(pdMouse ['clean_traces'].values, axis=0))

mapAxis = np.squeeze(np.array(pcManifoldDict[mouse]['mapAxis']))
neuPDF = pcManifoldDict[mouse]['neuPDF']
neuPDFNorm = np.zeros((neuPDF.shape[0],neuPDF.shape[1]))
for c in range(neuPDF.shape[1]):
    neuPDFNorm[:,c] = neuPDF[:,c,0]/np.max(neuPDF[:,c,0])
if mouse in izq:
    mneuPDF = np.flipud(mneuPDF)


deepPDF = list()
for mouse in deepMice:
    dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/manifold_cells/'
    pcManifoldDict = load_pickle(dataDir,'manifold_pc_dict.pkl')
    pos = pcManifoldDict[mouse]['pos']
    emb = pcManifoldDict[mouse]['emb']
    manifoldSignal = pcManifoldDict[mouse]['manifoldSignal']
    neuPDF = pcManifoldDict[mouse]['neuPDF']
    mapAxis = pcManifoldDict[mouse]['mapAxis']
    normNeuPDF = np.zeros((neuPDF.shape[0],neuPDF.shape[1]))
    for c in range(neuPDF.shape[1]):
        normNeuPDF[:,c] = neuPDF[:,c,0]/np.max(neuPDF[:,c,0])

    if mouse in izq:
        normNeuPDF = np.flipud(normNeuPDF)
    order = np.argsort(np.argmax(normNeuPDF, axis=0))
    plt.figure(); plt.matshow(normNeuPDF[:, order].T, aspect= 'auto'); plt.title(mouse)



    deepPDF.append(normNeuPDF[:22,:])
deepPDF = np.concatenate(deepPDF, axis=1)

supPDF = list()
for mouse in supMice:
    dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/manifold_cells/'
    pcManifoldDict = load_pickle(dataDir,'manifold_pc_dict.pkl')
    pos = pcManifoldDict[mouse]['pos']
    emb = pcManifoldDict[mouse]['emb']
    manifoldSignal = pcManifoldDict[mouse]['manifoldSignal']
    neuPDF = pcManifoldDict[mouse]['neuPDF']
    mapAxis = pcManifoldDict[mouse]['mapAxis']
    normNeuPDF = np.zeros((neuPDF.shape[0],neuPDF.shape[1]))
    for c in range(neuPDF.shape[1]):
        normNeuPDF[:,c] = neuPDF[:,c,0]/np.max(neuPDF[:,c,0])

    if mouse in izq:
        normNeuPDF = np.flipud(normNeuPDF)
    order = np.argsort(np.argmax(normNeuPDF, axis=0))
    plt.figure(); plt.matshow(normNeuPDF[:, order].T, aspect= 'auto'); plt.title(mouse)



fig, ax = plt.subplots(1, 2, figsize=(10,6))
ax[0].matshow(deepPDF[:, np.argsort(np.argmax(deepPDF, axis=0))].T, aspect='auto')
ax[1].matshow(supPDF[:, np.argsort(np.argmax(supPDF, axis=0))].T, aspect='auto')



#__________________________________________________________________________
#|                                                                        |#
#|                             ECCENTRICITY                               |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/ellipse/'
ellipseDict = load_pickle(dataDir, 'ellipse_fit_dict.pkl')

eccenList = list()
mouseList = list()
layerList = list()
for mouse in miceList:
    fnames = list(ellipseDict[mouse].keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    eccenList.append(ellipseDict[mouse][fnamePre]['eccentricity'])
    mouseList.append(mouse)
    if mouse in deepMice:
        layerList.append('deep')
    elif mouse in supMice:
        layerList.append('sup')

fig, ax = plt.subplots(1, 1, figsize=(6,6))
palette= ["#32e653", "#E632C5"]
eccenPD = pd.DataFrame(data={'mouse': mouseList,
                     'eccentricity': eccenList,
                     'layer': layerList})    

b = sns.barplot(x='layer', y='eccentricity', data=eccenPD,
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='layer', y='eccentricity', data=eccenPD,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)

print('eccentricity:', stats.ttest_ind(eccenList[:5], eccenList[5:], equal_var=True))
plt.savefig(os.path.join(dataDir,'DeepSup_eccentricity.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'DeepSup_eccentricity.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                               REMAP DIST                               |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ12', 'CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'
remapDist_dict = load_pickle(dataDir, 'remap_distance_dict.pkl')
fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, emb in enumerate(['umap','isomap','pca']):
    remapDist = list()
    for mouse in miceList:
        remapDist.append(remapDist_dict[mouse][emb]['remapDist'])
    print(emb, ':', stats.ttest_ind(remapDist[:6], remapDist[6:], equal_var=True))
    pd_dist = pd.DataFrame(data={'layer': ['deep']*6 + ['sup']*5,
                                     'dist': remapDist})
    palette= ["#1EE153", "#E11EAC"]
    b = sns.barplot(x='layer', y='dist', data=pd_dist,
                palette = palette, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dist', data=pd_dist,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])
    ax[idx].set_title(emb)
    ax[idx].set_ylim([0,1.6])
plt.tight_layout()

plt.savefig(os.path.join(dataDir,'remapDist.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(dataDir,'remapDist.png'), dpi = 400,bbox_inches="tight")


from scipy.stats import shapiro
from scipy import stats

for emb in ['umap','isomap','pca']:
    deepDist = []
    supDist = []
    for mouse in miceList:
        if mouse in deepMice:
            deepDist.append(remapDist_dict[mouse][emb]['remapDist'])
        if mouse in supMice:
            supDist.append(remapDist_dict[mouse][emb]['remapDist'])

    deepShapiro = shapiro(deepDist)
    supShapiro = shapiro(supDist)

    if deepShapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
        print(f'{emb} Distance:',ks_2samp(deepDist, supDist))
    else:
        print(f'{emb} Distance:', stats.ttest_ind(deepDist, supDist))



#__________________________________________________________________________
#|                                                                        |#
#|                      PLOT FUNCTIONAL CELLS EXAMPLES                    |#
#|________________________________________________________________________|#
from matplotlib.gridspec import GridSpec
def color_title_helper(fig,labels, colors, ax, y = 1.013, precision = 10**-3):
    "Creates a centered title with multiple colors. Don't change axes limits afterwards." 

    transform = ax.transAxes # use axes coords
    # initial params
    xT = 0# where the text ends in x-axis coords
    shift = -0.1 # where the text starts
    
    # for text objects
    text = dict()
    while (np.abs(shift - (2-xT)) > precision) and (shift <= xT) :         
        x_pos = shift 
        for label, col in zip(labels, colors):
            try:
                text[label].remove()
            except KeyError:
                pass
            if 'Cell' in label:
                fontsize = 10
            else:
                fontsize = 8
            text[label] = fig.text(x_pos, y, label, 
                        transform = transform, 
                        ha = 'left',
                        color = col,
                        fontsize = fontsize)
            
            x_pos = text[label].get_window_extent()\
                   .transformed(transform.inverted()).x1 + 0.15
            
        xT = x_pos # where all text ends
        shift += precision/2 # increase for next iteration
        if x_pos > 2: # guardrail 
            break



mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir = '/home/julio/Documents/SP_project/LT_DeepSup'


cell_type_keys = {
    'allo': 0,
    'rot': 1,
    'remap': 4
}
cells_per_row = 3
cells_per_column = 4
cells_per_slide = cells_per_row*cells_per_column

for mouse in mice_list:
    cell_type = np.load(os.path.join(data_dir, 'functional_cells', mouse, mouse+'_cellType.npy'))
    mouse_pc = load_pickle(os.path.join(data_dir, 'place_cells'), mouse+'_pc_dict.pkl')
    fnames = list(mouse_pc.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]
    for type_name in ['allo','rot','remap']:
        idx_cells = np.where(cell_type==cell_type_keys[type_name])[0]
        num_cells = len(idx_cells)
        num_slides = np.ceil(num_cells/cells_per_slide).astype(int)
        for curr_slide in range(num_slides):
            #prepare figure
            fig = plt.figure(figsize=(20,6))
            gs = GridSpec(2*cells_per_column-1,9*cells_per_row, figure = fig)
            ax = list()
            for idx in range(cells_per_slide):
                ax.append(list([0,0]))
                row_idx = idx//cells_per_row
                col_idx = idx%cells_per_row
                ax[idx][0] = fig.add_subplot(gs[2*row_idx,9*col_idx:9*col_idx+3]) #scatter pre
                ax[idx][1] = fig.add_subplot(gs[2*row_idx,9*col_idx+4:9*col_idx+7]) #scatter rot


            global_cell = curr_slide*cells_per_slide
            for idx in range(cells_per_slide):
                curr_cell = idx+global_cell
                if curr_cell>=num_cells:
                    continue
                neu_pdf_p = mouse_pc[fname_pre]['neu_pdf'][:,idx_cells[curr_cell]]
                neu_pdf_r = mouse_pc[fname_rot]['neu_pdf'][:,idx_cells[curr_cell]]
                maxVal = np.percentile(np.concatenate((neu_pdf_p.flatten(),neu_pdf_r.flatten())),95)

                p = ax[idx][0].matshow(neu_pdf_p.T, vmin = 0, vmax = maxVal, aspect = 'auto')
                ax[idx][0].set_yticks([])
                ax[idx][0].set_xticks([])

                p = ax[idx][1].matshow(neu_pdf_r.T, vmin = 0, vmax = maxVal, aspect = 'auto')
                ax[idx][1].set_yticks([])
                ax[idx][1].set_xticks([])
                fig.text(0.9, 1.1, f'{mouse} cell {idx_cells[curr_cell]}', 
                        transform = ax[idx][0].transAxes, ha = 'left')

            fig.suptitle(f'{mouse} {type_name} {curr_slide+1}/{num_slides}')
            plt.savefig(os.path.join(data_dir, 'functional_cells','heat_map_plots',f'{mouse}_{type_name}_{curr_slide+1}.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(data_dir, 'functional_cells','heat_map_plots',f'{mouse}_{type_name}_{curr_slide+1}.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)