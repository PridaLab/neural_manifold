##DUALES##
mice_list = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']
base_dir = '/home/julio/Documents/DeepSup_project/DualColor/'
jRGECO_dir = os.path.join(base_dir,'Thy1jRGECO')
RCaMP_dir = os.path.join(base_dir,'ThyCalbRCaMP')

name_list = list()
layer_list = list()
SI_list = list()
session_list = list()
nn_list = list()


for mouse in mice_list:

    print(f"Working on {mouse}:")
    if 'Thy1jRGECO' in mouse:
        load_dir = os.path.join(jRGECO_dir,'rotation',mouse)
    else:
        load_dir = os.path.join(RCaMP_dir,'rotation',mouse)


    rotation_dict = load_pickle(load_dir, mouse+'_rotation_dict.pkl')

    for case in ['deep', 'sup', 'all']:

        embPre = rotation_dict[case][case+'_emb_pre']
        posPre = rotation_dict[case][case+'_pos_pre'][:,0]
        timePre = np.arange(posPre.shape[0])

        embRot = rotation_dict[case][case+'_emb_rot']
        posRot = rotation_dict[case][case+'_pos_rot'][:,0]
        timeRot = np.arange(posRot.shape[0])

        # fig = plt.figure(figsize=((6,6)))
        # ax = plt.subplot(1,2,1, projection = '3d')
        # ax.scatter(*embPre[:,:3].T, c = posPre, cmap = 'inferno',s = 10)
        # ax.scatter(*embRot[:,:3].T, c = posRot, cmap = 'inferno',s = 10)

        # ax = plt.subplot(1,2,2, projection = '3d')
        # ax.scatter(*embPre[:,:3].T, c = timePre, cmap = 'YlGn_r',s = 10)
        # ax.scatter(*embRot[:,:3].T, c = timeRot, cmap = 'YlGn_r',s = 10)


        # plt.suptitle(mouse+'_'+case)


        nn = np.max([int(embPre.shape[0]*perc),20])
        SI, binLabel, overlapMat, ssI = compute_structure_index(embPre, timePre, 
                                                n_neighbors=nn,  n_bins=10, num_shuffles=1)
        SI_list.append(SI)
        name_list.append(mouse)
        layer_list.append(case)
        session_list.append('pre')
        nn_list.append(nn)

        nn = np.max([int(embRot.shape[0]*perc),20])
        SI, binLabel, overlapMat, ssI = compute_structure_index(embRot, timeRot, 
                                                n_neighbors=nn,  n_bins=10, num_shuffles=1)
        SI_list.append(SI)
        name_list.append(mouse)
        layer_list.append(case)
        session_list.append('rot')
        nn_list.append(nn)


pd_si = pd.DataFrame(data={'session': session_list,
                     'SI': SI_list,
                     'mouse': name_list,
                     'layer': layer_list,
                     'nn': nn_list}) 


fig = plt.figure(figsize=(8,8))
ax = plt.subplot(1,2,1)
sns.boxplot(data=pd_si, x='layer', y='SI', ax=ax)

deep_si = pd_si[pd_si['layer']=='deep']['SI'].to_list()
sup_si = pd_si[pd_si['layer']=='sup']['SI'].to_list()
if stats.shapiro(deep_si).pvalue<=0.05 or stats.shapiro(sup_si).pvalue<=0.05:
    ax.set_title(f"SI ks_2samp pvalue= {stats.ks_2samp(deep_si, sup_si)[1]:.4f}")
else:
    ax.set_title(f"SI ttest_ind pvalue: {stats.ttest_ind(deep_si, sup_si)[1]:.4f}")
ax = plt.subplot(1,2,2)
sns.boxplot(data=pd_si, x='layer', y='SI', hue='session', ax=ax)

##DEEP/SUP



dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/emb_example/'
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
supMice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ7', 'ChZ8', 'GC7']

from scipy import stats



for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    emb_method = 'umap'

    pos = copy.deepcopy(np.concatenate(pdMouse ['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse ['dir_mat'].values, axis=0))
    time = copy.deepcopy(np.concatenate(pdMouse['index_mat'].values, axis=0))#np.arange(pos.shape[0])
    emb = copy.deepcopy(np.concatenate(pdMouse [emb_method.lower()].values, axis=0))

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
    ax = plt.subplot(1,3,1, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, color = dir_color,s = 10)
    ax = plt.subplot(1,3,2, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 10, cmap = 'inferno', vmin= 0, vmax = 70)
    ax.set_aspect('equal', adjustable='box')
    ax = plt.subplot(1,3,3, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = time[:],s = 10, cmap = 'YlGn_r', vmax = np.percentile(time, 95))
    ax.set_aspect('equal', adjustable='box')

    plt.suptitle(mouse)



perc = 0.015
min_nn = 50



for perc in [0.005, 0.01, 0.015, 0.02, 0.05]:
    print(f"Working on perc {perc}: ")

    name_list = list()
    layer_list = list()
    SI_list = list()
    strain_list = list()
    nn_list = list()
    for mouse in miceList:
        fileName =  mouse+'_df_dict.pkl'
        filePath = os.path.join(dataDir, mouse)
        pdMouse = load_pickle(filePath,fileName)
        emb_method = 'umap'

        pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
        dir_mat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
        time = copy.deepcopy(np.concatenate(pdMouse['index_mat'].values, axis=0))#np.arange(pos.shape[0])
        emb = copy.deepcopy(np.concatenate(pdMouse[emb_method.lower()].values, axis=0))

        D= pairwise_distances(emb)
        noiseIdx = filter_noisy_outliers(emb,D)
        emb = emb[~noiseIdx,:]
        pos = pos[~noiseIdx,:]
        dir_mat = dir_mat[~noiseIdx]
        time = time[~noiseIdx]


        nn = np.max([int(emb.shape[0]*perc),50])
        SI, binLabel, overlapMat, ssI = compute_structure_index(emb, time, 
                                                n_neighbors=nn,  n_bins=10, num_shuffles=1)
        SI_list.append(SI)
        name_list.append(mouse)
        if mouse in deepMice:
            layer_list.append('deep')
        elif mouse in supMice:
            layer_list.append('sup')

        if ('GC' in mouse) or ('TG' in mouse):
            strain_list.append('thy1')
        elif ('CZ' in mouse) or ('CG' in mouse):
            strain_list.append('Calb')
        elif 'Ch' in mouse:
            strain_list.append('ChRNA7')
        nn_list.append(nn)

    pd_si = pd.DataFrame(data={'strain': strain_list,
                         'SI': SI_list,
                         'mouse': name_list,
                         'layer': layer_list,
                         'nn': nn_list}) 



    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,2,1)
    sns.boxplot(data=pd_si, x='layer', y='SI', ax=ax)


    deep_si = pd_si[pd_si['layer']=='deep']['SI'].to_list()
    sup_si = pd_si[pd_si['layer']=='sup']['SI'].to_list()
    if stats.shapiro(deep_si).pvalue<=0.05 or stats.shapiro(sup_si).pvalue<=0.05:
        ax.set_title(f"SI ks_2samp pvalue= {stats.ks_2samp(deep_si, sup_si)[1]:.4f}")
    else:
        ax.set_title(f"SI ttest_ind pvalue: {stats.ttest_ind(deep_si, sup_si)[1]:.4f}")
    ax = plt.subplot(1,2,2)
    sns.boxplot(data=pd_si, x='strain', y='SI', ax=ax)



    thy1_si = pd_si[pd_si['layer']=='thy1']['SI'].to_list()
    calb_si = pd_si[pd_si['layer']=='calb']['SI'].to_list()
    chrna7_si = pd_si[pd_si['layer']=='ChRNA7']['SI'].to_list()
    text = []
    if stats.shapiro(thy1_si).pvalue<=0.05 or stats.shapiro(calb_si).pvalue<=0.05:
        ax.set_title(f"SI ks_2samp pvalue= {stats.ks_2samp(thy1_si, calb_si)[1]:.4f}")
    else:
        ax.set_title(f"SI ttest_ind pvalue: {stats.ttest_ind(thy1_si, calb_si)[1]:.4f}")
    plt.suptitle(perc)




for nn in [10,25,50,75,100,150,200]:
    print(f"Working on nn {nn}: ")

    name_list = list()
    layer_list = list()
    SI_list = list()
    strain_list = list()
    for mouse in miceList:
        fileName =  mouse+'_df_dict.pkl'
        filePath = os.path.join(dataDir, mouse)
        pdMouse = load_pickle(filePath,fileName)
        emb_method = 'umap'

        pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
        dir_mat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
        time = np.arange(pos.shape[0])
        emb = copy.deepcopy(np.concatenate(pdMouse[emb_method.lower()].values, axis=0))

        D= pairwise_distances(emb)
        noiseIdx = filter_noisy_outliers(emb,D)
        emb = emb[~noiseIdx,:]
        pos = pos[~noiseIdx,:]
        dir_mat = dir_mat[~noiseIdx]
        time = time[~noiseIdx]

        SI, binLabel, overlapMat, ssI = compute_structure_index(emb, time, 
                                                n_neighbors=nn,  n_bins=10, num_shuffles=1)
        SI_list.append(SI)
        name_list.append(mouse)
        if mouse in deepMice:
            layer_list.append('deep')
        elif mouse in supMice:
            layer_list.append('sup')

        if ('GC' in mouse) or ('TG' in mouse):
            strain_list.append('thy1')
        elif ('CZ' in mouse) or ('CG' in mouse):
            strain_list.append('Calb')
        elif 'Ch' in mouse:
            strain_list.append('ChRNA7')


    pd_si = pd.DataFrame(data={'strain': strain_list,
                         'SI': SI_list,
                         'mouse': name_list,
                         'layer': layer_list}) 



    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,2,1)
    sns.boxplot(data=pd_si, x='layer', y='SI', ax=ax)


    deep_si = pd_si[pd_si['layer']=='deep']['SI'].to_list()
    sup_si = pd_si[pd_si['layer']=='sup']['SI'].to_list()
    if stats.shapiro(deep_si).pvalue<=0.05 or stats.shapiro(sup_si).pvalue<=0.05:
        ax.set_title(f"SI ks_2samp pvalue= {stats.ks_2samp(deep_si, sup_si)[1]:.4f}")
    else:
        ax.set_title(f"SI ttest_ind pvalue: {stats.ttest_ind(deep_si, sup_si)[1]:.4f}")
    ax = plt.subplot(1,2,2)
    sns.boxplot(data=pd_si, x='strain', y='SI', ax=ax)
    plt.suptitle(nn)



miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7','ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
saveDir = '/home/julio/Documents/SP_project/Fig2/SI/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
sIDict = dict()
featParamsDict = {
    'pos': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    'dir': {'discrete_label':True, 'num_shuffles':1, 'verbose':False},
    'vel': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    '(pos,dir)': {'discrete_label':[False, True], 'num_shuffles':1, 'verbose':False},
    'globalTime': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    'trial': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False},
    'trialTime': {'discrete_label':False, 'n_bins':10, 'num_shuffles':1, 'verbose':False}
}

nnPercList = [0.01, 0.015, 0.02, 0.05]

for mouse in miceList:
    print(f"\nWorking on mouse {mouse}: ")
    sIDict[mouse] = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    #get data
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
    dirMat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pdMouse['vel'].values, axis=0))
    vecFeat = np.concatenate((pos[:,0].reshape(-1,1),dirMat),axis=1)
    time = np.arange(pos.shape[0])
    trial = copy.deepcopy(np.concatenate(pdMouse['index_mat'].values, axis=0))
    trialTime = copy.deepcopy(np.concatenate(pdMouse['inner_trial_time'].values, axis=0))

    for signalName in ['clean_traces', 'umap','isomap', 'pca']:
        signal = copy.deepcopy(np.concatenate(pdMouse[signalName].values, axis=0))
        if signalName == 'pca' or signalName =='isomap':
            signal = signal[:,:3]
        D = pairwise_distances(signal)
        noiseIdx = filter_noisy_outliers(signal,D=D)
        csignal = signal[~noiseIdx,:]
        featDict = {
            'pos': pos[~noiseIdx,0],
            'dir': dirMat[~noiseIdx],
            'vel': vel[~noiseIdx],
            '(pos,dir)': vecFeat[~noiseIdx,:],
            'globalTime': time[~noiseIdx],
            'trial': trial[~noiseIdx],
            'trialTime': trialTime[~noiseIdx]
        }

        nnList = [np.max([np.round(nnPercList[idx]*csignal.shape[0]).astype(int),50]) for idx in range(len(nnPercList))]
        sIDict[mouse][signalName] = dict()
        sIDict[mouse][signalName]['featDict'] = featDict
        sIDict[mouse][signalName]['featParamsDict'] = featParamsDict
        sIDict[mouse][signalName]['signalDict'] = {
                                                'signal': signal,
                                                'csignal': csignal,
                                                'noiseIdx': noiseIdx
                                                }
        sIDict[mouse][signalName]['results'] = dict()
        for featName in list(featDict.keys()):
            sI = np.zeros((len(nnList),))*np.nan
            binLabel = list()
            overlapMat = list()
            ssI = np.zeros((len(nnList),featParamsDict[featName]['num_shuffles']))

            for nnIdx in range(len(nnList)):
                print(f'\t{signalName} - {featName}: {nnList[nnIdx]} ({nnIdx+1}/{len(nnList)})', end='\r', flush=True)
                nn = nnList[nnIdx]
                sI[nnIdx], tbinLabel, toverlapMat, ssI[nnIdx] = compute_structure_index(csignal, featDict[featName], 
                                            n_neighbors=nn, **featParamsDict[featName])
                binLabel.append(tbinLabel)
                overlapMat.append(toverlapMat)
            print(f'\t{signalName} - {featName}: {np.nanmean(sI)}')
            sIDict[mouse][signalName]['results'][featName] = {
                'sI': sI,
                'binLabel': binLabel,
                'overlapMat': overlapMat,
                'ssI': ssI,
                'nnList': nnList,
                'nnPercList': nnPercList,
                'params': featParamsDict[featName]
            }

        with open(os.path.join(saveDir,'sI_perc_dict_v2.pkl'), 'wb') as f:
            pickle.dump(sIDict, f)


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


