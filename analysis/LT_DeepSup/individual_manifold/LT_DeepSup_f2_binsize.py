def preprocess_traces(signal, sigma = 5, sig_up = 4, sig_down = 12, peak_th=0.1):

    lowpassSignal = uniform_filter1d(signal, size = 4000, axis = 0)
    filt_signal = gaussian_filter1d(signal, sigma = sigma, axis = 0)

    for nn in range(filt_signal.shape[1]):
        baseSignal = np.histogram(signal[:,nn], 100)
        baseSignal = baseSignal[1][np.argmax(baseSignal[0])]
        baseSignal = baseSignal + lowpassSignal[:,nn] - np.min(lowpassSignal[:,nn]) 

        cleanSignal = filt_signal[:,nn]-baseSignal
        cleanSignal = cleanSignal/np.max(cleanSignal,axis = 0)
        cleanSignal[cleanSignal<0] = 0
        filt_signal[:,nn] = cleanSignal

    biSignal = np.zeros(filt_signal.shape)

    gaus = lambda x,sig,amp,vo: amp*np.exp(-(((x)**2)/(2*sig**2)))+vo;
    x = np.arange(-int(5*sig_down), int(5*sig_down),1);
    upGaus = gaus(x,sig_up, 1, 0); 
    upGaus[int(5*sig_down)+1:] = 0
    downGaus = gaus(x,sig_down, 1, 0); 
    downGaus[:int(5*sig_down)+1] = 0
    finalGaus = downGaus + upGaus;

    for nn in range(filt_signal.shape[1]):
        peakSignal,_ =find_peaks(filt_signal[:,nn],height=peak_th)
        biSignal[peakSignal, nn] = filt_signal[peakSignal, nn]
        if finalGaus.shape[0]<filt_signal.shape[0]:
            biSignal[:, nn] = np.convolve(biSignal[:, nn],finalGaus, 'same')

    return biSignal

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nnDist < np.percentile(nnDist, 20)
    return noiseIdx

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
bin_size_list = [1,2,3,4]

data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
save_dir =  '/home/julio/Documents/SP_project/Fig2/bin_size/'
try:
    os.mkdir(save_dir)
except:
    pass

supMice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']
#__________________________________________________________________________
#|                                                                        |#
#|                               INNER DIM                                |#
#|________________________________________________________________________|#
import skdim
import time

params = {
    'signalName': 'clean_traces',
    'nNeigh': 30,
    'verbose': True
}


id_dict = dict()
params['bin_size_list'] = bin_size_list
for mouse in mice_list:
    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
    id_dict[mouse] = dict()
    for bin_size in bin_size_list:
        print(f"bin-size: {bin_size}")
        og_signal = np.concatenate(pd_mouse['raw_traces'].values, axis = 0)[::bin_size,:]
        signal = preprocess_traces(og_signal, sigma = 6/bin_size, sig_up = 4/bin_size, sig_down = 12/bin_size, peak_th=0.1)
        #compute abids dim
        abidsDim = np.nanmean(dim_red.compute_abids(signal, int(params['nNeigh']/bin_size)))
        print(f"\tABIDS: {abidsDim:.2f}", end='', flush=True)
        time.sleep(.2)

        # corrIntDim = skdim.id.CorrInt(k1 = 5, k2 = int(params['nNeigh']/bin_size)).fit_transform(signal)
        # print(f" | CorrInt: {corrIntDim:.2f}", end='', flush=True)
        # time.sleep(.2)

        # dancoDim = skdim.id.DANCo().fit_transform(signal)
        # print(f" | DANCo: {dancoDim:.2f}", end='', flush=True)
        # time.sleep(.2)

        # essDim = skdim.id.ESS().fit_transform(signal,n_neighbors = int(params['nNeigh']/bin_size))
        # print(f" | ESS: {essDim:.2f}", end='', flush=True)
        # time.sleep(.2)

        # fishersDim = skdim.id.FisherS(conditional_number=5).fit_transform(signal)
        # print(f" | FisherS: {fishersDim:.2f}", end='', flush=True)
        # time.sleep(.2)

        # knnDim = skdim.id.KNN(k=int(params['nNeigh']/bin_size)).fit_transform(signal)
        # print(f" | KNN: {knnDim:.2f}", end='', flush=True)
        # time.sleep(.2)

        # lPCADim = skdim.id.lPCA(ver='broken_stick').fit_transform(signal)
        # print(f" | lPCA: {lPCADim:.2f}", end='', flush=True)
        # time.sleep(.2)

        # madaDim = skdim.id.MADA().fit_transform(signal)
        # print(f" | MADA: {madaDim:.2f}", end='', flush=True)
        # time.sleep(.2)

        # mindDim = skdim.id.MiND_ML(k=int(params['nNeigh']/bin_size)).fit_transform(signal)
        # print(f" | MiND_ML: {mindDim:.2f}", end='', flush=True)
        # time.sleep(.2)

        # mleDim = skdim.id.MLE(K=int(params['nNeigh']/bin_size)).fit_transform(signal)
        # print(f" | MLE: {mleDim:.2f}", end='', flush=True)
        # time.sleep(.2)

        momDim = skdim.id.MOM().fit_transform(signal,n_neighbors = int(params['nNeigh']/bin_size))
        print(f" | MOM: {momDim:.2f}", end='', flush=True)
        time.sleep(.2)

        tleDim = skdim.id.TLE().fit_transform(signal,n_neighbors = int(params['nNeigh']/bin_size))
        print(f" | TLE: {tleDim:.2f}")
        time.sleep(.2)

        #save results
        id_dict[mouse][bin_size] = {
            'abidsDim': abidsDim,
            # 'corrIntDim': corrIntDim,
            # 'dancoDim': dancoDim,
            # 'essDim': essDim,
            # 'fishersDim': fishersDim,
            # 'knnDim': knnDim,
            # 'lPCADim': lPCADim,
            # 'madaDim': madaDim,
            # 'mindDim': mindDim,
            # 'mleDim': mleDim,
            'momDim': momDim,
            'tleDim': tleDim,
            'params': params
        }

        saveFile = open(os.path.join(save_dir, 'inner_dim_dict.pkl'), "wb")
        pickle.dump(id_dict, saveFile)
        saveFile.close()

saveFile = open(os.path.join(save_dir, 'inner_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# create list of strings
paramsList = [ f'{key} : {params[key]}' for key in params]
# write string one by one adding newline
saveParamsFile = open(os.path.join(save_dir, "inner_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()




#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT INNER DIM                             |#
#|________________________________________________________________________|#

id_dict = load_pickle(save_dir, 'inner_dim_dict.pkl')

dimList = list()
mouseList = list()
methodList = list()
layerList = list()
binsizeList = list()
strainList = list()
for mouse in list(id_dict.keys()):
    for bin_size in list(id_dict[mouse].keys()):
        dimList.append(id_dict[mouse][bin_size]['momDim'])
        dimList.append(id_dict[mouse][bin_size]['abidsDim'])
        dimList.append(id_dict[mouse][bin_size]['tleDim'])
        if mouse in deepMice:
            layerList = layerList + ['deep']*3
        elif mouse in supMice:
            layerList = layerList + ['sup']*3

        if ('GC' in mouse) or ('TG' in mouse):
            strainList = strainList + ['Thy1']*3
        elif ('CZ' in mouse) or ('CG' in mouse):
            strainList = strainList + ['Calb']*3
        elif 'Ch' in mouse:
            strainList = strainList + ['ChRNA7']*3


        mouseList = mouseList + [mouse]*3
        methodList = methodList+['mom', 'abids', 'tle']
        binsizeList = binsizeList + [bin_size]*3

pd_dim = pd.DataFrame(data={'mouse': mouseList,
                     'dim': dimList,
                     'method': methodList,
                     'layer': layerList,
                     'bin_size': binsizeList,
                     'strain': strainList})    

fig, ax = plt.subplots(1, 3, figsize=(10,6))

for idx, method in enumerate(['mom', 'abids', 'tle']):
    pd_method = pd_dim[pd_dim['method'] == method]
    b = sns.barplot(x='layer', y='dim', data=pd_method, hue='bin_size',
                linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dim', data=pd_method,  hue='bin_size',
            palette= 'dark:gray',edgecolor = 'gray', ax = ax[idx])
    ax[idx].set_title(method)
plt.tight_layout()

plt.savefig(os.path.join(save_dir,'inner_dim_layer.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'inner_dim_layer.png'), dpi = 400,bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|                                UMAP DIM                                |#
#|________________________________________________________________________|#

params = {
    'maxDim':10,
    'nNeigh': 120,
    'minDist': 0.1,
    'nnDim': 30,
    'signalName': 'clean_traces',
}

params['bin_size_list'] = bin_size_list
umap_dim_dict = dict()
for mouse in mice_list:
    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
    umap_dim_dict[mouse] = dict()
    for bin_size in bin_size_list:
        print(f"bin-size: {bin_size}")
        og_signal = np.concatenate(pd_mouse['raw_traces'].values, axis = 0)[::bin_size,:]
        signal = preprocess_traces(og_signal, sigma = 6/bin_size, sig_up = 4/bin_size, sig_down = 12/bin_size, peak_th=0.1)

        print("Computing rank indices og space...", end = '', sep = '')
        rankIdx = dim_red.validation.compute_rank_indices(signal)
        print("\b\b\b: Done")
        trustNum = np.zeros((params['maxDim'],))
        contNum = np.zeros((params['maxDim'],))
        for dim in range(params['maxDim']):
            emb_space = np.arange(dim+1)
            print(f"Dim: {dim+1} ({dim+1}/{params['maxDim']})")
            model = umap.UMAP(n_neighbors = int(params['nNeigh']/bin_size), n_components =dim+1, min_dist=params['minDist'])
            print("\tFitting model...", sep= '', end = '')
            emb = model.fit_transform(signal)
            print("\b\b\b: Done")
            #1. Compute trustworthiness
            print("\tComputing trustworthiness...", sep= '', end = '')
            temp = dim_red.validation.trustworthiness_vector(signal, emb, int(params['nnDim']/bin_size), indices_source = rankIdx)
            trustNum[dim] = temp[-1]
            print(f"\b\b\b: {trustNum[dim]:.4f}")
            #2. Compute continuity
            print("\tComputing continuity...", sep= '', end = '')
            temp = dim_red.validation.continuity_vector(signal, emb ,int(params['nnDim']/bin_size))
            contNum[dim] = temp[-1]
            print(f"\b\b\b: {contNum[dim]:.4f}")

        dimSpace = np.linspace(1,params['maxDim'], params['maxDim']).astype(int)   

        kl = KneeLocator(dimSpace, trustNum, curve = "concave", direction = "increasing")
        if kl.knee:
            trustDim = kl.knee
            print('Trust final dimension: %d - Final error of %.4f' %(trustDim, 1-trustNum[dim-1]))
        else:
            trustDim = np.nan
        kl = KneeLocator(dimSpace, contNum, curve = "concave", direction = "increasing")
        if kl.knee:
            contDim = kl.knee
            print('Cont final dimension: %d - Final error of %.4f' %(contDim, 1-contNum[dim-1]))
        else:
            contDim = np.nan
        hmeanDim = (2*trustDim*contDim)/(trustDim+contDim)
        print('Hmean final dimension: %d' %(hmeanDim))

        umap_dim_dict[mouse][bin_size] = {
            'trustNum': trustNum,
            'contNum': contNum,
            'trustDim': trustDim,
            'contDim':contDim,
            'hmeanDim': hmeanDim,
            'params': copy.deepcopy(params)
        }
        saveFile = open(os.path.join(save_dir, 'umap_dim_dict.pkl'), "wb")
        pickle.dump(umap_dim_dict, saveFile)
        saveFile.close()

saveFile = open(os.path.join(save_dir, 'umap_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# write string one by one adding newline
paramsList = [ f'{key} : {params[key]}' for key in params]
saveParamsFile = open(os.path.join(save_dir, "umap_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveParamsFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()




#__________________________________________________________________________
#|                                                                        |#
#|                              PLOT UMAP DIM                             |#
#|________________________________________________________________________|#

umap_dict = load_pickle(save_dir, 'umap_dim_dict.pkl')

dimList = list()
mouseList = list()
methodList = list()
layerList = list()
binsizeList = list()
strainList = list()
for mouse in list(umap_dict.keys()):
    for bin_size in list(umap_dict[mouse].keys()):
        dimList.append(umap_dict[mouse][bin_size]['trustDim'])
        dimList.append(umap_dict[mouse][bin_size]['contDim'])
        dimList.append(umap_dict[mouse][bin_size]['hmeanDim'])
        if mouse in deepMice:
            layerList = layerList + ['deep']*3
        elif mouse in supMice:
            layerList = layerList + ['sup']*3

        if 'GC' in mouse:
            strainList = strainList + ['Thy1']*3
        elif 'CZ' in mouse:
            strainList = strainList + ['Calb']*3
        elif 'Ch' in mouse:
            strainList = strainList + ['ChRNA7']*3


        mouseList = mouseList + [mouse]*3
        methodList = methodList+['trust', 'cont', 'hmean']
        binsizeList = binsizeList + [bin_size]*3

pd_dim = pd.DataFrame(data={'mouse': mouseList,
                     'dim': dimList,
                     'method': methodList,
                     'layer': layerList,
                     'bin_size': binsizeList,
                     'strain': strainList})    

fig, ax = plt.subplots(1, 3, figsize=(10,6))

for idx, method in enumerate(['trust', 'cont', 'hmean']):
    pd_method = pd_dim[pd_dim['method'] == method]

    b = sns.barplot(x='strain', y='dim', data=pd_method, hue='bin_size',
                linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='strain', y='dim', data=pd_method,  hue='bin_size',
            palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])
    ax[idx].set_title(method)
plt.tight_layout()

plt.savefig(os.path.join(save_dir,'umap_dim.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'umap_dim.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                          COMPUTE SI NN PERC                            |#
#|________________________________________________________________________|#

si_dict = dict()
featParamsDict = {
    'pos': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    'dir': {'discrete_label':True, 'num_shuffles':1, 'verbose':False},
    '(pos,dir)': {'discrete_label':[False, True], 'num_shuffles':1, 'verbose':False},
}

nn_perc = 0.005

for mouse in mice_list:
    print(f"\nWorking on mouse {mouse}: ")
    si_dict[mouse] = dict()
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
    #keep only right left trials
    pd_mouse = gu.select_trials(pd_mouse,"dir == ['N','L','R']")

    for bin_size in bin_size_list:
        #signals
        og_signal = np.concatenate(pd_mouse['raw_traces'].values, axis = 0)[::bin_size,:]
        signal = preprocess_traces(og_signal, sigma = 6/bin_size, sig_up = 4/bin_size, sig_down = 12/bin_size, peak_th=0.1)
        #umap
        model = umap.UMAP(n_neighbors = int(120/bin_size), n_components =3, min_dist=0.1)
        emb = model.fit_transform(signal)
        #features
        pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))[::bin_size]
        dirMat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))[::bin_size]
        vecFeat = np.concatenate((pos[:,0].reshape(-1,1),dirMat),axis=1)


        D = pairwise_distances(signal)
        noiseIdx = filter_noisy_outliers(signal,D=D)
        csignal = signal[~noiseIdx,:]
        cemb = emb[~noiseIdx,:]

        featDict = {
            'pos': pos[~noiseIdx,0],
            'dir': dirMat[~noiseIdx],
            '(pos,dir)': vecFeat[~noiseIdx,:],
        }

        nn = np.max([np.round(nn_perc*csignal.shape[0]).astype(int),3])

        si_dict[mouse][bin_size] = dict()
        si_dict[mouse][bin_size]['featDict'] = copy.deepcopy(featDict)
        si_dict[mouse][bin_size]['featParamsDict'] = copy.deepcopy(featDict)
        si_dict[mouse][bin_size]['signalDict'] = {
                                                'signal': signal,
                                                'csignal': csignal,
                                                'noiseIdx': noiseIdx,
                                                'umap': emb,
                                                'cumap': cemb
                                                }

        si_dict[mouse][bin_size]['results'] = {
            'csignal': {},
            'cumap': {}
        }
        for feat_name in list(featDict.keys()):
            for idx, signal_name in enumerate(['csignal', 'cumap']):
                if 'csignal' in signal_name:
                    signal_mat = csignal
                elif 'cumap' in signal_name:
                    signal_mat = cemb
                sI, binLabel, overlapMat, ssI = compute_structure_index(signal_mat, featDict[feat_name], 
                                            n_neighbors=nn, **featParamsDict[feat_name])
                print(f'\t{bin_size} | {signal_name} | {feat_name}: {np.nanmean(sI):.2f}')

                si_dict[mouse][bin_size]['results'][signal_name][feat_name] = {
                    'sI': sI,
                    'binLabel': binLabel,
                    'overlapMat': overlapMat,
                    'ssI': ssI,
                    'n_neighbors': nn,
                    'params': copy.deepcopy(featParamsDict[feat_name])
                }

        with open(os.path.join(save_dir,'sI_dict.pkl'), 'wb') as f:
            pickle.dump(si_dict, f)




#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT SI NN PERC                            |#
#|________________________________________________________________________|#

#plot ON 0.5 PERC
si_dict = load_pickle(save_dir, 'sI_dict.pkl')
mice_list = list(si_dict.keys())

perc = 0.05

for feature_name in ['pos','dir','(pos,dir)']:

    SIList = list()
    mouseList = list()
    layerList = list()
    strainList = list()
    signalList = list()
    binsizeList = list()
    for mouse in mice_list:
        for bin_size in list(si_dict[mouse].keys()):

            SIList.append(si_dict[mouse][bin_size]['results']['csignal'][feature_name]['sI'])
            SIList.append(si_dict[mouse][bin_size]['results']['cumap'][feature_name]['sI'])

            mouseList = mouseList + [mouse]*2
            binsizeList += [bin_size]*2

            if mouse in deepMice:
                layerList += ['deep']*2
            elif mouse in supMice:
                layerList += ['sup']*2

            signalList = signalList + ['csignal', 'cumap']
            if 'GC' in mouse:
                strainList += ['Thy1']*2
            elif 'CZ' in mouse:
                strainList += ['Calb']*2
            elif 'Ch' in mouse:
                strainList += ['ChRNA7']*2

    pd_si = pd.DataFrame(data={'mouse': mouseList,
                                     'SI': SIList,
                                     'layer': layerList,
                                     'signal': signalList,
                                     'strain': strainList,
                                     'bin_size': binsizeList})

    fig, ax = plt.subplots(1, 3, figsize=(6,6))
    for idx, strain in enumerate(['Thy1', 'Calb', 'ChRNA7']):
        pd_strain = pd_si[pd_si['strain']==strain]
        b = sns.boxplot(x='signal', y='SI', data=pd_strain, hue = 'bin_size',
                    linewidth = 1, width= .5, ax = ax[idx])

        sns.swarmplot(x='signal', y='SI', data=pd_strain, hue= 'bin_size',
                    palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])

        ax[idx].set_title(f"sI {strain}",fontsize=15)
        b.spines['top'].set_visible(False)
        b.spines['right'].set_visible(False)
        b.tick_params(labelsize=12)
        b.set_yticks([0,0.2,0.4,0.6, 0.8, 1.0])
        b.set_ylim([-.05, 1.05])
        b.set_ylabel(f"SI {feature_name}")
        plt.tight_layout()
        plt.suptitle(feature_name)


    plt.savefig(os.path.join(save_dir,f'SI_perc_{feature_name}.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'SI_perc_{feature_name}.png'), dpi = 400,bbox_inches="tight")

