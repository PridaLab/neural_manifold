import ripserplusplus as rpp_py
from persim import plot_diagrams
from matplotlib import gridspec
from numpy.random import choice
import numpy as np
from sklearn.metrics import pairwise_distances
import umap
import os, pickle, copy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from tqdm.auto import tqdm

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def save_pickle(path, name, data):
    if ('.pkl' not in name) and ('.pickle' not in name):
        name += '.pkl'
    save_pickle = open(os.path.join(path, name), "wb")
    pickle.dump(data, save_pickle)
    save_pickle.close()
    return True

def filter_noisy_outliers(data, D=None, distTh = 5, noiseTh = 25):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,distTh), axis=1)
    noiseIdx = np.where(nnDist < np.percentile(nnDist, noiseTh))[0]
    signalIdx = np.where(nnDist >= np.percentile(nnDist, noiseTh))[0]
    return noiseIdx, signalIdx

def topological_denoising(data, numSample=None, numIters=100, inds=[],  sig=None, w=None, c=None, metric='euclidean'):
    n = np.float64(data.shape[0])
    d = data.shape[1]
    if len(inds)==0:
        inds = np.unique(np.floor(np.arange(0,n-1, n/numSample)).astype(int))
    else:
        numSample = len(inds)
    S = data[inds, :] 
    if not sig:
        sig = np.sqrt(np.var(S))
    if not c:
        c = 0.05*max(pdist(S, metric = metric)) 
    if not w:
        w = 0.3

    dF1 = np.zeros((len(inds), d), float)
    dF2 = np.zeros((len(inds), d), float)

    for i in range(numSample):
        dF1[i, :] = np.dot((data - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], data, metric = metric), 2) / (2*sig)).T)[:, 0]
        dF2[i, :] = np.dot((S - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], S, metric = metric), 2) / (2*sig)).T)[:, 0]

    dF = 1/sig*(1/n * dF1 - (w / numSample) * dF2)
    M = dF.max()
    for k in range(numIters):
        S += c*dF/M
        for i in range(numSample):
            dF1[i, :] = np.dot((data - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], data, metric = metric), 2) / (2*sig)).T)[:, 0]
            dF2[i, :] = np.dot((S - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], S, metric = metric), 2) / (2*sig)).T)[:, 0]
        dF = 1/sig*(1/n * dF1 - (w / numSample) * dF2)
    data_denoised = S
    return data_denoised, inds

def plot_betti_bars(diagrams, max_dist=1e3, conf_interval = None):
    col_list = ['r', 'g', 'm', 'c']
    diagrams[0][~np.isfinite(diagrams[0])] = max_dist
    max_len = [np.max(h) for h in diagrams]
    max_len = np.max(max_len)
    # Plot the 30 longest barcodes only
    to_plot = []
    for curr_h in diagrams:
         bar_lens = curr_h[:,1] - curr_h[:,0]
         plot_h = curr_h[(-bar_lens).argsort()[:30]]
         to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(len(diagrams), 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            if conf_interval:
                ax.plot([interval[0], interval[0]+conf_interval[curr_betti]], [i,i], linewidth = 5, color =[.8,.8,.8])

            ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
                lw=1.5)

        ax.set_ylabel('H' + str(curr_betti))
        ax.set_xlim([-0.1, max_len+0.1])
        # ax.set_xticks([0, xlim])
        ax.set_ylim([-1, len(curr_bar)])
    return fig

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
saveDir = '/home/julio/Documents/SP_project/Fig1/betti_numbers/'
try:
    os.mkdir(saveDir)
except:
    pass
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'

distTh = 5
noiseTh = 20
numSamples = 500
numIters = 150
dim = 3
numNeigh = 120
minDist = 0.1
percCell = 20
percTime = 0.1
fluoTh = 0.1
signalName = 'clean_traces'

for mouse in miceList:
    #load data
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdAnimal = load_pickle(filePath,fileName)

    #signal
    signal = np.concatenate(pdAnimal[signalName].values, axis = 0)
    pos = np.concatenate(pdAnimal['pos'].values, axis=0)
    dir_mat = np.concatenate(pdAnimal['dir_mat'].values, axis=0)
    index_mat = np.concatenate(pdAnimal['index_mat'].values, axis=0)

    #delete cells that are almost inactive
    delCells = np.mean(signal,axis=0)<np.percentile(np.mean(signal,axis=0), percCell)
    signal = np.delete(signal, np.where(delCells)[0],1)
    #delete timestamps where almost all cells are inactive
    delTime = np.sum(signal>fluoTh, axis=1)<percTime*signal.shape[1]
    signal = np.delete(signal, np.where(delTime)[0],0)
    pos = np.delete(pos, np.where(delTime)[0],0)
    dir_mat = np.delete(dir_mat, np.where(delTime)[0],0)
    index_mat = np.delete(index_mat, np.where(delTime)[0],0)

    #fit umap
    print("\tFitting umap model...", sep= '', end = '')
    model_umap = umap.UMAP(n_neighbors=numNeigh, n_components =dim, min_dist=minDist)
    model_umap.fit(signal)
    emb_umap = model_umap.transform(signal)
    print("\b\b\b: Done")

    #filter outliers
    print("\tFiltering noisy outliers...", sep= '', end = '')
    D = pairwise_distances(signal)
    noiseIdx, signalIdx = filter_noisy_outliers(signal, D=D, distTh=distTh, noiseTh=noiseTh)
    cleanSignal = signal[signalIdx,:]
    cleanEmb = emb_umap[signalIdx,:]
    cleanPos = pos[signalIdx,:]
    print("\b\b\b: Done")

    #topological denoising
    print("\tPerforming topological denoising...", sep= '', end = '')
    downSignal, downIdx = topological_denoising(cleanSignal, numSample=numSamples, numIters=numIters)
    downEmb, downIdx2 = topological_denoising(cleanEmb, numSample=numSamples, numIters=numIters)
    print("\b\b\b: Done")

    #plot embeddings
    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(2,3,1, projection = '3d')
    b = ax.scatter(*emb_umap[:,:3].T, c = dir_mat, cmap = 'Accent',s = 10, vmin= 0, vmax = 8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Umap direction')

    ax = plt.subplot(2,3,4, projection = '3d')
    b = ax.scatter(*emb_umap[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Umap pos')

    ax = plt.subplot(2,3,2, projection = '3d')
    b = ax.scatter(*emb_umap[:,:3].T, color = 'blue',s = 5)
    b = ax.scatter(*emb_umap[noiseIdx,:3].T, color = 'red',s = 5)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.legend(['Good', 'Outliers'])
    ax.set_title('Umap outliers')

    ax = plt.subplot(2,3,5, projection = '3d')
    b = ax.scatter(*cleanEmb[:,:3].T, c = cleanPos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Umap pos w/o outliers')

    ax = plt.subplot(2,3,3, projection = '3d')
    b = ax.scatter(*cleanEmb[downIdx,:3].T,c = cleanPos[downIdx,0], cmap = 'magma', s = 10)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Umap pos after topological denoising on signal')

    ax = plt.subplot(2,3,6, projection = '3d')
    b = ax.scatter(*downEmb[:,:3].T,c = cleanPos[downIdx2,0], cmap = 'magma', s = 10)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Umap pos after topological denoising on emb')
    fig.suptitle(f"{mouse}: distTh={distTh} | noiseTh={noiseTh} | numSamples={numSamples}")
    plt.savefig(os.path.join(saveDir, f'{mouse}_cloudDenoising.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    #compute betti numbers
    print("\tComputing Betti Numbers...", sep= '', end = '')
    a = rpp_py.run(f"--dim 2 --format point-cloud --threshold {int(np.nanmax(D))}",downSignal)
    print("\b\b\b: Done")

    diagrams = list()
    diagrams.append(np.zeros((a[0].shape[0],2)))
    for b in range(diagrams[0].shape[0]):
        diagrams[0][b][0] = a[0][b][0]
        diagrams[0][b][1] = a[0][b][1]
    diagrams[0][-1,1] = np.nanmax(D)

    diagrams.append(np.zeros((a[1].shape[0],2)))
    for b in range(diagrams[1].shape[0]):
        diagrams[1][b][0] = a[1][b][0]
        diagrams[1][b][1] = a[1][b][1]

    diagrams.append(np.zeros((a[2].shape[0],2)))
    for b in range(diagrams[2].shape[0]):
        diagrams[2][b][0] = a[2][b][0]
        diagrams[2][b][1] = a[2][b][1]

    #plot birth/death scatter
    fig = plt.figure()
    plot_diagrams(diagrams, show=True)
    plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_scatter.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_scatter.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    #plot betti bars
    fig = plot_betti_bars(diagrams, max_dist=np.nanmax(D))
    plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_bars.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_bars.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    #save
    params = {
        'distTh':distTh,
        'noiseTh':noiseTh,
        'numSample':numSamples,
        'numIters': numIters,
        'dim': dim,
        'numNeigh': numNeigh,
        'minDist': minDist,
        'signalName': signalName,
        'percCell': percCell,
        'percTime': percTime,
        'fluoTh': fluoTh
    }

    bettiDict = {
        'mouse': mouse,
        'dataDir': dataDir,
        'saveDir': saveDir,
        'pdAnimal': pdAnimal,
        'D': D,
        'delCells': delCells,
        'delTime': delTime,
        'signalIdx': signalIdx,
        'downIdx': downIdx,
        'downSignal': downSignal,
        'diagrams': diagrams,
        'params': params
    }

    save_pickle(saveDir, mouse+'_betti_dict.pkl', bettiDict)


for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_betti_dict.pkl'
    bettiDict = load_pickle(saveDir,fileName)

    signal = bettiDict['downSignal']

    minShift = 1
    maxShift = signal.shape[0]-1
    shiftDiagrams = list()
    denseShiftDiagrams = list()
    bar=tqdm(total=1000, desc='Computing Betti on Shuffling')
    for iter in range(1000):
        shiftSignal = copy.deepcopy(signal)
        timeShift = np.random.randint(minShift, maxShift,signal.shape[1])
        for cell, shift in enumerate(timeShift):
            shiftSignal[:-shift,cell] = copy.deepcopy(signal[shift:,cell])
            shiftSignal[-shift:,cell] = copy.deepcopy(signal[:shift,cell])
        #compute betti numbers
        downD = pairwise_distances(shiftSignal)
        trilD = downD[np.tril_indices_from(downD, k=-1)]
        nEdges = int(downD.shape[0]*(downD.shape[0]-1)/2)
        a = rpp_py.run(f"--dim 2 --format point-cloud --threshold {int(np.ceil(np.nanmax(downD)))}",shiftSignal)

        computedDistances = dict()
        if iter == 0:
            #betti 0
            shiftDiagrams.append(np.zeros((a[0].shape[0],2)))
            denseShiftDiagrams.append(np.zeros((a[0].shape[0],2)))
            for b in range(shiftDiagrams[0].shape[0]):
                shiftDiagrams[0][b][0] = a[0][b][0]
                shiftDiagrams[0][b][1] = a[0][b][1]
            #density
            bar_lens = shiftDiagrams[0][:,1] - shiftDiagrams[0][:,0]
            longest_bar = shiftDiagrams[0][(-bar_lens).argsort()[0]]
            denseShiftDiagrams[0][b][0] = sum(trilD <= longest_bar[0])/nEdges
            denseShiftDiagrams[0][b][1] = sum(trilD <= longest_bar[1])/nEdges

            #betti 1
            shiftDiagrams.append(np.zeros((a[1].shape[0],2)))
            denseShiftDiagrams.append(np.zeros((a[1].shape[0],2)))
            for b in range(shiftDiagrams[1].shape[0]):
                shiftDiagrams[1][b][0] = a[1][b][0]
                shiftDiagrams[1][b][1] = a[1][b][1]
            #density
            bar_lens = shiftDiagrams[1][:,1] - shiftDiagrams[1][:,0]
            longest_bar = shiftDiagrams[1][(-bar_lens).argsort()[0]]
            denseShiftDiagrams[1][b][0] = sum(trilD <= longest_bar[0])/nEdges
            denseShiftDiagrams[1][b][1] = sum(trilD <= longest_bar[1])/nEdges

            #betti 2
            shiftDiagrams.append(np.zeros((a[2].shape[0],2)))
            denseShiftDiagrams.append(np.zeros((a[2].shape[0],2)))
            for b in range(shiftDiagrams[2].shape[0]):
                shiftDiagrams[2][b][0] = a[2][b][0]
                shiftDiagrams[2][b][1] = a[2][b][1]

            #density
            bar_lens = shiftDiagrams[2][:,1] - shiftDiagrams[2][:,0]
            longest_bar = shiftDiagrams[2][(-bar_lens).argsort()[0]]
            denseShiftDiagrams[2][b][0] = sum(trilD <= longest_bar[0])/nEdges
            denseShiftDiagrams[2][b][1] = sum(trilD <= longest_bar[1])/nEdges


        else:
            #betti 0 
            st = shiftDiagrams[0].shape[0]
            shiftDiagrams[0] = np.concatenate((shiftDiagrams[0],np.zeros((a[0].shape[0],2))))
            denseShiftDiagrams[0] = np.concatenate((denseShiftDiagrams[0],np.zeros((a[0].shape[0],2))))
            for b in range(a[0].shape[0]):
                shiftDiagrams[0][b+st][0] = a[0][b][0]
                shiftDiagrams[0][b+st][1] = a[0][b][1]
            #density
            bar_lens = shiftDiagrams[0][st:,1] - shiftDiagrams[0][st:,0]
            longest_bar = shiftDiagrams[0][st+(-bar_lens).argsort()[0]]
            denseShiftDiagrams[0][b+st][0] = sum(trilD <= longest_bar[0])/nEdges
            denseShiftDiagrams[0][b+st][1] = sum(trilD <= longest_bar[1])/nEdges

            #betti 1
            st = shiftDiagrams[1].shape[0]
            shiftDiagrams[1] = np.concatenate((shiftDiagrams[1],np.zeros((a[1].shape[0],2))))
            denseShiftDiagrams[1] = np.concatenate((denseShiftDiagrams[1],np.zeros((a[1].shape[0],2))))

            for b in range(a[1].shape[0]):
                shiftDiagrams[1][b+st][0] = a[1][b][0]
                shiftDiagrams[1][b+st][1] = a[1][b][1]
            #density
            bar_lens = shiftDiagrams[1][st:,1] - shiftDiagrams[1][st:,0]
            longest_bar = shiftDiagrams[1][st+(-bar_lens).argsort()[0]]
            denseShiftDiagrams[1][b+st][0] = sum(trilD <= longest_bar[0])/nEdges
            denseShiftDiagrams[1][b+st][1] = sum(trilD <= longest_bar[1])/nEdges


            #betti 2
            st = shiftDiagrams[2].shape[0]
            shiftDiagrams[2] = np.concatenate((shiftDiagrams[2],np.zeros((a[2].shape[0],2))))
            denseShiftDiagrams[2] = np.concatenate((denseShiftDiagrams[2],np.zeros((a[2].shape[0],2))))

            for b in range(a[2].shape[0]):
                shiftDiagrams[2][b+st][0] = a[2][b][0]
                shiftDiagrams[2][b+st][1] = a[2][b][1]
            #density
            bar_lens = shiftDiagrams[2][st:,1] - shiftDiagrams[2][st:,0]
            longest_bar = shiftDiagrams[0][st+(-bar_lens).argsort()[0]]
            denseShiftDiagrams[2][b+st][0] = sum(trilD <= longest_bar[0])/nEdges
            denseShiftDiagrams[2][b+st][1] = sum(trilD <= longest_bar[1])/nEdges
        bar.update(1)  
    bar.close()

    confInterval = list()
    confInterval.append(np.max(np.diff(shiftDiagrams[0],axis=1)))
    confInterval.append(np.max(np.diff(shiftDiagrams[1],axis=1)))
    confInterval.append(np.max(np.diff(shiftDiagrams[2],axis=1)))

    bettiDict['confInterval'] = confInterval
    bettiDict['shiftDiagrams'] = shiftDiagrams


    denseConfInterval = list()
    denseConfInterval.append(np.max(np.diff(denseShiftDiagrams[0],axis=1)))
    denseConfInterval.append(np.max(np.diff(denseShiftDiagrams[1],axis=1)))
    denseConfInterval.append(np.max(np.diff(denseShiftDiagrams[2],axis=1)))  

    bettiDict['denseConfInterval'] = denseConfInterval
    bettiDict['denseShiftDiagrams'] = denseShiftDiagrams
    save_pickle(saveDir, mouse+'_betti_dict.pkl', bettiDict)

    fig = plot_betti_bars(bettiDict['diagrams'], conf_interval = confInterval)
    plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_bars_conf.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_bars_conf.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    fig = plot_betti_bars(bettiDict['denseDiagrams'], conf_interval = denseConfInterval)
    plt.savefig(os.path.join(saveDir, f'{mouse}_dense_diagrams_bars_conf.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir, f'{mouse}_dense_diagrams_bars_conf.svg'), dpi = 400,bbox_inches="tight")

    plt.close(fig)




for mouse in miceList:
    mouseDict = load_pickle(saveDir, mouse+'_betti_dict.pkl')

    diagrams = mouseDict['diagrams']
    denseDiagrams = copy.deepcopy(diagrams)
    D = mouseDict['D']
    downD = D[mouseDict['downIdx']][:,mouseDict['downIdx']]
    trilD = downD[np.tril_indices_from(downD, k=-1)]

    nEdges = int(downD.shape[0]*(downD.shape[0]-1)/2)
    computedDistances = dict()

    for betti_num in range(len(diagrams)):
        for bar_num in range(diagrams[betti_num].shape[0]):
            st = diagrams[betti_num][bar_num][0]
            en = diagrams[betti_num][bar_num][1]

            if st in computedDistances.keys():
                denseDiagrams[betti_num][bar_num][0] = computedDistances[st]
            else:
                denseSt = sum(trilD <= st)/nEdges
                denseDiagrams[betti_num][bar_num][0] = denseSt
                computedDistances[st] = denseSt

            if en in computedDistances.keys():
                denseDiagrams[betti_num][bar_num][1] = computedDistances[en]
            else:
                denseEn =  sum(trilD <= en)/nEdges
                denseDiagrams[betti_num][bar_num][1] = denseEn
                computedDistances[en] = denseEn

    mouseDict['denseDiagrams'] = denseDiagrams
    save_pickle(saveDir, mouse+'_betti_dict.pkl', mouseDict)

for mouse in miceList:
    mouseDict = load_pickle(saveDir, mouse+'_betti_dict.pkl')
    fig = plot_betti_bars(mouseDict['diagrams'], conf_interval = mouseDict['confInterval'])
    # plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_bars_conf.png'), dpi = 400,bbox_inches="tight")
    # plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_bars_conf.svg'), dpi = 400,bbox_inches="tight")
    plt.suptitle(mouse)

    fig = plot_betti_bars(mouseDict['denseDiagrams'], conf_interval = mouseDict['denseConfInterval'])
    # plt.savefig(os.path.join(saveDir, f'{mouse}_dense_diagrams_bars_conf.png'), dpi = 400,bbox_inches="tight")
    # plt.savefig(os.path.join(saveDir, f'{mouse}_dense_diagrams_bars_conf.svg'), dpi = 400,bbox_inches="tight")
    plt.suptitle(mouse)


'''
for mouse in miceList:
    mouseDict = load_pickle(saveDir, mouse+'_betti_dict.pkl')
    fig = plot_betti_bars(mouseDict['diagrams'], conf_interval = mouseDict['confInterval'])
    shiftDiagrams = mouseDict['shiftDiagrams']
    confInterval = list()
    confInterval.append(np.percentile(np.diff(shiftDiagrams[0],axis=1),95))
    confInterval.append(np.percentile(np.diff(shiftDiagrams[1],axis=1),95))
    confInterval.append(np.percentile(np.diff(shiftDiagrams[2],axis=1),95))
    plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_bars_conf_95.png'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir, f'{mouse}_diagrams_bars_conf_95.svg'), dpi = 400,bbox_inches="tight")
    plt.close(fig)
'''
#__________________________________________________________________________
#|                                                                        |#
#|                             plot life time                            |#
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





dataDir = '/home/julio/Documents/SP_project/Fig1/betti_numbers'
miceList = ['M2019','M2023','M2024', 'M2025', 'M2026']
hList = list()
lifeTimeList = list()
mouseList = list()
for mouse in miceList:
    mouseDict = load_pickle(dataDir, mouse+'_betti_dict.pkl')
    h1Diagrams = np.array(mouseDict['denseDiagrams'][1])
    h1Length = np.sort(np.diff(h1Diagrams, axis=1)[:,0])
    lifeTimeList.append(h1Length[-1]/h1Length[-2])
    hList.append(1)
    mouseList.append(mouse)

    h2Diagrams = np.array(mouseDict['denseDiagrams'][2])
    h2Length = np.sort(np.diff(h2Diagrams, axis=1)[:,0])
    lifeTimeList.append(h2Length[-1]/h2Length[-2])
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