





rawsignalPre = 
rawsignalPre = copy.deepcopy(np.concatenate(animalPre['raw_traces'].values, axis=0))
rawsignalRot = copy.deepcopy(np.concatenate(animalRot['raw_traces'].values, axis=0))
rawBoth = np.concatenate((rawsignalPre, rawsignalRot), axis=0)
degree = np.max(rawBoth, axis=0)

actOrder = np.argsort(degree)

plt.figure(); plt.plot(degree[actOrder])


signalPre = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis=0))[:, actOrder[-80:]]
posPre = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis=0))
signalRot = copy.deepcopy(np.concatenate(animalRot['clean_traces'].values, axis=0))[:, actOrder[-80:]]
posRot = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
#%%all data
index = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalRot.shape[0],1))+1))
concat_signal = np.vstack((signalPre, signalRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
# model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
model.fit(concat_signal)
concat_emb = model.transform(concat_signal)
embPre = concat_emb[index[:,0]==0,:]
embRot = concat_emb[index[:,0]==1,:]




fig = plt.figure()
ax = plt.subplot(1,3,1, projection = '3d')
ax.scatter(*embPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
ax.scatter(*embRot[:,:3].T, color = 'r', s= 30, cmap = 'magma')
ax.set_title('All')
ax = plt.subplot(1,3,2, projection = '3d')
ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s= 30, cmap = 'magma')
ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s= 30, cmap = 'magma')
ax = plt.subplot(1,3,3, projection = '3d')
ax.scatter(*centPre[:,:3].T, color ='b', s= 30, cmap = 'magma')
ax.scatter(*centRot[:,:3].T, color = 'r', s= 30, cmap = 'magma')

plt.suptitle(f"{mouse}")




miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
saveDir = '/home/julio/Documents/SP_project/Fig1/SI/'
dataDir = '/home/julio/Documents/SP_project/Fig1/processed_data/'

sI_dict = load_pickle(saveDir, 'sI_clean_dict.pkl')
for mouse in miceList:
    print(f"\nWorking on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #keep only right left trials
    pdMouse = gu.select_trials(pdMouse,"dir == ['N','L','R']")
    #get data
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pdMouse['vel'].values, axis=0))
    vectorial_feature = np.concatenate((pos[:,0].reshape(-1,1),dir_mat),axis=1)
    time = np.arange(pos.shape[0])

    for signalName in ['clean_traces', 'umap', 'isomap', 'pca']:
        print(f'\t{signalName}: ')
        signal = copy.deepcopy(np.concatenate(pdMouse[signalName].values, axis=0))
        if signalName == 'pca' or signalName =='isomap':
            signal = signal[:,:3]
        D = pairwise_distances(signal)
        noiseIdx = filter_noisy_outliers(signal,D=D)
        csignal = signal[~noiseIdx,:]
        cpos = pos[~noiseIdx,:]
        cdir_mat = dir_mat[~noiseIdx]
        cvel = vel[~noiseIdx]
        cvectorial_feature = vectorial_feature[~noiseIdx,:]
        ctime = time[~noiseIdx]
        sI_dict[mouse][signalName] = dict()

        sI, binLabel, overlapMat, ssI = compute_structure_index(csignal, ctime, 
                                                    n_neighbors=10, num_shuffles=100, verbose=True)
        sI_dict[mouse][signalName]['time'] = {
            'sI': sI,
            'binLabel': binLabel,
            'overlapMat': overlapMat,
            'ssI': ssI
        }

        with open(os.path.join(saveDir,'sI_clean_dict.pkl'), 'wb') as f:
            pickle.dump(sI_dict, f)



 cells_to_keep =[ True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True, False,  True,  True, False,
         True,  True, False,  True,  True,  True,  True,  True, False,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True, False,  True, False,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True, False, False,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True]



false_cells = [14,  17,  20,  26, 105, 107, 118, 119]


def compute_entanglement(points):
    distance_b = pairwise_distances(points)
    model_iso = Isomap(n_neighbors = 10, n_components = 1)
    emb = model_iso.fit_transform(points)
    distance_a = model_iso.dist_matrix_
    entanglement_boundary = np.max(distance_a[1:,0])/np.min(distance_b[1:,0])
    entanglement = np.max((distance_a[1:,0]/distance_b[1:,0]))
    return (entanglement-1)/entanglement_boundary



plt.figure()
for mouse in ['GC2','GC3','GC5_nvista']:

    filePath = os.path.join('/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/', mouse, 'old')
    remRemap = load_pickle(filePath, mouse+'remove_rotCells_dict.pkl')
    angleRot = np.abs(remRemap['rotAngle'])
    m = np.mean(angleRot, axis=0)*180/np.pi
    sd = np.std(angleRot, axis=0)*180/np.pi
    plt.plot(np.arange(m.shape[0]), m,label = mouse)
    plt.fill_between(np.arange(m.shape[0]), m-sd, m+sd, alpha = 0.3)
plt.legend()


plt.figure()
for mouse in ['GC2','GC3','GC5_nvista']:

    filePath = os.path.join('/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/', mouse, 'old')
    remRemap = load_pickle(filePath, mouse+'remove_rotCells_dict.pkl')
    sI = remRemap['SIVal']
    m = np.mean(sI, axis=(0,1))
    sd = np.std(sI, axis=(0,1))
    plt.plot(np.arange(m.shape[0]), m,label = mouse)
    plt.fill_between(np.arange(m.shape[0]), m-sd, m+sd, alpha = 0.3)
plt.legend()


    sI = remRemap['SIVal']

a = load_pickle('/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/GC2/', 'GC2remove_remapCells_dict.pkl')
b = np.abs(a['rotAngle'])

plt.figure()
ax = plt.subplot(111)

m = np.mean(b[:10,:],axis=0)
sd = np.std(b[:10,:],axis=0)
ax.plot(np.arange(m.shape[0]), m, color = '#32E653',label = '10')
ax.fill_between(np.arange(m.shape[0]), m-sd, m+sd, color = '#32E653', alpha = 0.3)

m = np.nanmean(b,axis=0)
sd = np.nanstd(b,axis=0)
ax.plot(np.arange(m.shape[0]), m, color = 'red',label = 'all')
ax.fill_between(np.arange(m.shape[0]), m-sd, m+sd, color = 'red', alpha = 0.3)
ax.legend()


plt.figure()
ax = plt.subplot(111)
m = np.nanmean(b,axis=(0,1))
sd = np.nanstd(b,axis=(0,1))
ax.plot(np.arange(m.shape[0]), m, color = 'red',label = 'all')
ax.fill_between(np.arange(m.shape[0]), m-sd, m+sd, color = 'red', alpha = 0.3)
ax.legend()




####################################################################
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
    dim_list.append(inner_dim[mouse]['corrIntDim'])
    dim_list.append(inner_dim[mouse]['momDim'])
    dim_list.append(inner_dim[mouse]['essDim'])
    dim_list.append(inner_dim[mouse]['tleDim'])


    mouse_list = mouse_list + [mouse]*5

method_list = ['abids', 'CorrInt', 'MOM', 'ESS', 'TLE']*len(miceList)


palette= ["#96A2A5", "#8ECAE6", "#219EBC", "#023047","#FFB703", "#FB8500"]
pd_dim = pd.DataFrame(data={'mouse': mouse_list,
                     'dim': dim_list,
                     'method': method_list})    

fig, ax = plt.subplots(1, 1, figsize=(10,6))
b = sns.barplot(x='method', y='dim', data=pd_dim,
            palette = palette, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='method', y='dim', data=pd_dim, 
            color = 'gray', edgecolor = 'gray', ax = ax)
b.tick_params(labelsize=12)
b.set_ylim([0, 5])
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'inner_dim_barplot_all.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'inner_dim_barplot_all.png'), dpi = 400,bbox_inches="tight")



#######################################################################################################


import scipy.signal as scs
from scipy.ndimage import convolve1d
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

#Adapted from PyalData package (19/10/21) (added variable win_length)
def _norm_gauss_window(bin_size, std, num_std = 5):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_size (float): binning size of the array we want to smooth in same time 
                units as the std
    
    std (float): standard deviation of the window use hw_to_std to calculate 
                std based from half-width (same time units as bin_size)
                
    num_std (int): size of the window to convolve in #of stds

    Returns
    -------
    win (1D np.array): Gaussian kernel with length: num_bins*std/bin_length
                mass normalized to 1
    """
    win_len = int(num_std*std/bin_size)
    if win_len%2==0:
        win_len = win_len+1
    win = scs.gaussian(win_len, std/bin_size)      
    return win / np.sum(win)

#Copied from PyalData package (19/10/21)
def _hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))

#Copied from PyalData package (19/10/21)
def _smooth_data(mat, bin_size=None, std=None, hw=None, win=None, axis=0):
    """
    Smooth a 1D array or every column of a 2D array

    Parameters
    ----------
    mat : 1D or 2D np.array
        vector or matrix whose columns to smooth
        e.g. recorded spikes in a time x neuron array
    bin_size : float
        length of the timesteps in seconds
    std : float (optional)
        standard deviation of the smoothing window
    hw : float (optional)
        half-width of the smoothing window
    win : 1D array-like (optional)
        smoothing window to convolve with

    Returns
    -------
    np.array of the same size as mat
    """
    #assert mat.ndim == 1 or mat.ndim == 2, "mat has to be a 1D or 2D array"
    assert  sum([arg is not None for arg in [win, hw, std]]) == 1, "only give win, hw, or std"
    if win is None:
        assert bin_size is not None, "specify bin_size if not supplying window"
        if std is None:
            std = _hw_to_std(hw)
        win = _norm_gauss_window(bin_size, std)
    return convolve1d(mat, win, axis=axis, output=np.float32, mode='reflect')

def _get_pdf(pos, neu_signal, mapAxis, dim, dir_mat = None):
    """Obtain probability density function of 'pos' along each bin defined by 
    the limits mapAxis as well as the mean signal of 'neu_signal' on those bins.
    If dir_mat is provided, the above is computed for each possible direction.

    Parameters
    ----------
    pos: numpy array (T, space_dims)
        Numpy array with the position
        
    neu_signal: numpy array (T, nn)
        Numpy array containing the neural signal (spikes, rates, or traces).

    mapAxis: list
        List with as many entries as spatial dimensions. Each entry contains 
        the lower limits of the bins one wants to divide that spatial dimension 
        into. 
            e.g.
            If we want to divide our 2D space, then mapAxis will contain 2 
            entries. The first entry will have the lower limits of the x-bins 
            and the second entry those of the y-bins.
    
    dim: integer
        Integer indicating the number of spatial dimenions of the bins. That
        is, 
            dim = len(mapAxis) <= pos.shape[1]
            
    Optional parameters:
    --------------------
    dir_mat: numpy array (T,1)
        Numpy array containing the labels of the direction the animal was 
        moving on each timestamp. For each label, a separate pdf will be 
        computed.
        
    Returns
    -------
    pos_pdf: numpy array (space_dims_bins, 1, dir_labels)
        Numpy array containing the probability the animal is at each bin for
        each of the directions if dir_mat was indicated.
    
    neu_pdf: numpy array (space_dims_bins, nn, dir_labels)
        Numpy array containing the mean neural activity of each neuron at each 
        bin for each of the directions if dir_mat was indicated.
    """
    if isinstance(dir_mat, type(None)):
        dir_mat = np.zeros((pos.shape[0],1))
        
    val_dirs = np.array(np.unique(dir_mat))
    num_dirs = len(val_dirs)
                   
    subAxis_length =  [len(mapsubAxis) for mapsubAxis in mapAxis] #nbins for each axis
    access_neu = np.linspace(0,neu_signal.shape[1]-1,neu_signal.shape[1]).astype(int) #used to later access neu_pdf easily

    neu_pdf = np.zeros(subAxis_length+[neu_signal.shape[1]]+[num_dirs])
    pos_pdf = np.zeros(subAxis_length+[1]+[num_dirs])
    
    for sample in range(pos.shape[0]):
        pos_entry_idx = list()
        for d in range(dim):
            temp_entry = np.where(pos[sample,d]>=mapAxis[d])
            if np.any(temp_entry):
                temp_entry = temp_entry[0][-1]
            else:
                temp_entry = 0
            pos_entry_idx.append(temp_entry)
        
        dir_idx = np.where(dir_mat[sample]==val_dirs)[0][0]
        pos_idxs = tuple(pos_entry_idx + [0]+[dir_idx])
        neu_idxs = tuple(pos_entry_idx + [access_neu]+[dir_idx])
        
        pos_pdf[pos_idxs] += 1
        neu_pdf[neu_idxs] += (1/pos_pdf[pos_idxs])* \
                                   (neu_signal[sample,:]-neu_pdf[neu_idxs]) #online average
    


    pos_pdf = pos_pdf/np.sum(pos_pdf,axis=tuple(range(pos_pdf.ndim - 1))) #normalize probability density function
    
    return pos_pdf, neu_pdf

def _get_edges(pos, limit, dim):
    """Obtain which points of 'pos' are inside the limits of the border defined
    by limit (%).
    
    Parameters
    ----------
    pos: numpy array
        Numpy array with the position
        
    limit: numerical (%)
        Limit (%) used to decided which points are kepts and which ones are 
        discarded. (i.e. if limit=10 and pos=[0,100], only the points inside the
                    range [10,90] will be kepts).
    
    dim: integer
        Dimensionality of the division.    
                
    Returns
    -------
    signal: numpy array
        Concatenated dataframe column into numpy array along axis=0.
    """  
    assert pos.shape[1]>=dim, f"pos has less dimensions ({pos.shape[1]}) " + \
                                f"than the indicated in dim ({dim})"
    norm_limit = limit/100
    minpos = np.min(pos, axis=0)
    maxpos = np.max(pos,axis=0)
    
    trackLength = maxpos - minpos
    trackLimits = np.vstack((minpos + norm_limit*trackLength,
                             maxpos - norm_limit*trackLength))
    
    points_inside_lim = list()
    for d in range(dim):
        points_inside_lim.append(np.vstack((pos[:,d]<trackLimits[0,d],
                                       pos[:,d]>trackLimits[1,d])).T)
        
    points_inside_lim = np.concatenate(points_inside_lim, axis=1)
    points_inside_lim = ~np.any(points_inside_lim ,axis=1)
    
    return points_inside_lim

def get_neu_pdf(pos, neuSignal, vel = None, dim=2, **kwargs):
    #smooth pos signal if applicable

    if kwargs['std_pos']>0:
        pos = _smooth_data(pos, std = kwargs['std_pos'], bin_size = 1/kwargs['sF'])
    if pos.ndim == 1:
        pos = pos.reshape(-1,1)

    #Compute velocity if not provided
    if isinstance(vel, type(None)):
        vel = np.linalg.norm(np.diff(pos_signal, axis= 0), axis=1)*kwargs['sF']
        vel = np.hstack((vel[0], vel))
    vel = vel.reshape(-1)

    #Compute dirSignal if not provided
    if 'dirSignal' in kwargs:
        dirSignal = kwargs["dirSignal"]
    else:
        dirSignal = np.zeros((pos.shape[0],1))

    #Discard edges
    if kwargs['ignoreEdges']>0:
        posBoolean = _get_edges(pos, kwargs['ignoreEdges'], 1)
        pos = pos[posBoolean]
        neuSignal = neuSignal[posBoolean]
        dirSignal = dirSignal[posBoolean]
        vel = vel[posBoolean]

    #compute moving epochs
    moveEpochs = vel>=kwargs['velTh'] 
    #keep only moving epochs
    pos = pos[moveEpochs]
    neuSignal = neuSignal[moveEpochs]
    dirSignal = dirSignal[moveEpochs]
    vel = vel[moveEpochs]
        
    #Create grid along each dimensions
    minPos = np.percentile(pos,1, axis=0) #(pos.shape[1],)
    maxPos = np.percentile(pos,99, axis = 0) #(pos.shape[1],)
    obsLength = maxPos - minPos #(pos.shape[1],)
    if 'binWidth' in kwargs:
        binWidth = kwargs['binWidth']
        if isinstance(binWidth, list):
            binWidth = np.array(binWidth)
        else:
            binWidth = np.repeat(binWidth, dim, axis=0) #(pos.shape[1],)
        nbins = np.ceil(obsLength[:dim]/binWidth).astype(int) #(pos.shape[1],)
        kwargs['nbins'] = nbins
    elif 'binNum' in kwargs:
        nbins = kwargs['binNum']
        if isinstance(nbins, list):
            nbins = np.array(nbins)
        else:
            nbins = np.repeat(nbins, dim, axis=0) #(pos.shape[1],)
        binWidth = np.round(obsLength[:dim]/nbins,4) #(pos.shape[1],)
        kwargs['binWidth'] = binWidth
    mapAxis = list()
    for d in range(dim):
        mapAxis.append(np.linspace(minPos[d], maxPos[d], nbins[d]+1)[:-1].reshape(-1,1)); #(nbins[d],)

    #Compute probability density function
    posPDF, neuPDF = _get_pdf(pos, neuSignal, mapAxis, dim, dirSignal)
    for d in range(dim):
        posPDF = _smooth_data(posPDF, std=kwargs['stdPDF'], bin_size=binWidth[d], axis=d)
        neuPDF = _smooth_data(neuPDF, std=kwargs['stdPDF'], bin_size=binWidth[d], axis=d)
    return posPDF, neuPDF, mapAxis


supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']


dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/manifold_cells/pre/'
placeDir = '/home/julio/Documents/SP_project/LT_DeepSup/place_cells/'
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

nNeigh = 120
dim = 30
minDist = 0.1

for mouse in miceList:
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]

    animalPre= copy.deepcopy(animal[fnamePre])

    signal = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis = 0))
    pos = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(animalPre['dir_mat'].values, axis = 0))

    model = umap.UMAP(n_neighbors=nNeigh, n_components=dim, min_dist=minDist)
    model.fit(signal)
    emb = model.transform(signal)
    # emb = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis = 0))

    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')
    pcPre = copy.deepcopy(pcDict[fnamePre])
    neuPDF = pcPre['neu_pdf']
    normNeuPDF = np.zeros(neuPDF.shape)
    for d in range(neuPDF.shape[2]):
        for c in range(neuPDF.shape[1]):
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.max(neuPDF[:,c,d])
    order =  np.argsort(np.argmax(normNeuPDF[:,:,0], axis=0))


    meanNormNeuPDF = np.nanmean(normNeuPDF, axis=1)
    mapAxis = pcPre['mapAxis']
    manifoldSignal = np.zeros((emb.shape[0]))
    for p in range(emb.shape[0]):
        try:
            x = np.where(mapAxis[0]<=pos[p,0])[0][-1]
        except: 
            x = 0
        dire = direction[p]
        if dire==0:
            manifoldSignal[p] = 0
        else:
            manifoldSignal[p] = meanNormNeuPDF[x,dire-1]

    fig = plt.figure()
    ax = plt.subplot(2,2,1)
    ax.matshow(normNeuPDF[:,order,0].T, aspect = 'auto')
    ax.set_title('izq')
    ax.set_ylabel('cell number')
    ax.set_xlabel('pos-x')
    ax = plt.subplot(2,2,2)
    ax.matshow(normNeuPDF[:,order,1].T, aspect = 'auto')
    ax.set_title('dcha')
    ax.set_ylabel('cell number')
    ax.set_xlabel('pos-x')

    ax = plt.subplot(2,2,3, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 30, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax = plt.subplot(2,2,4, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 30)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.suptitle(mouse)






    emb = copy.deepcopy(np.concatenate(animalPre['umap'].values, axis = 0))
    signal = copy.deepcopy(np.concatenate(animalPre['clean_traces'].values, axis = 0))
    pos = copy.deepcopy(np.concatenate(animalPre['pos'].values, axis = 0))
    vel = copy.deepcopy(np.concatenate(animalPre['vel'].values, axis = 0))

    posPDF, neuPDF, mapAxis = get_neu_pdf(pos,signal, vel = vel, dim= 1, **params)
    neuPDF_norm = np.zeros((neuPDF.shape[0],neuPDF.shape[1]))
    for c in range(neuPDF.shape[1]):
        neuPDF_norm[:,c] = neuPDF[:,c,0]/np.max(neuPDF[:,c,0])

    mneuPDF = np.nanmean(neuPDF_norm, axis=1)

    manifoldSignal = np.zeros((emb.shape[0]))
    for p in range(emb.shape[0]):
        try:
            x = np.where(mapAxis[0]<=pos[p,0])[0][-1]
        except: 
            x = 0
        manifoldSignal[p] = mneuPDF[x]
    manifoldSignal = gaussian_filter1d(manifoldSignal, sigma = 3, axis = 0)
    pcManifoldDict[mouse] = {
        'emb': emb,
        'pos': pos,
        'signal': signal,
        'vel': vel,
        'mapAxis': mapAxis,
        'posPDF': posPDF,
        'neuPDF': neuPDF,
        'mneuPDF': mneuPDF,
        'manifoldSignal': manifoldSignal,
        'fname': fnamePre,
        'params': params
    }

    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(1,2,1, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 30, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    # ax.view_init(130,110,90)
    ax.view_init(-65,-85,90)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax = plt.subplot(1,2,2, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 30)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    # ax.view_init(130,110,90)
    ax.view_init(-65,-85,90)

    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,mouse+'_PDF_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_PDF_emb.png'), dpi = 400,bbox_inches="tight")

    with open(os.path.join(saveDir, "manifold_pc_dict.pkl"), "wb") as file:
        pickle.dump(pcManifoldDict, file, protocol=pickle.HIGHEST_PROTOCOL)


plt.figure()
plt.scatter(*pos.T,c=np.mean(signal,axis=1))


#############################################################################################
import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neural_manifold.general_utils as gu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from neural_manifold import dimensionality_reduction as dim_red
import seaborn as sns
from neural_manifold.dimensionality_reduction import validation as dim_validation 
import umap
from kneed import KneeLocator
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from structure_index import compute_structure_index, draw_graph
from neural_manifold import decoders as dec
from scipy.stats import pearsonr
import random

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nnDist < np.percentile(nnDist, 20)
    sum(noiseIdx)
    return noiseIdx

def get_centroids(input_A, label_A, dir_A = None, ndims = 2, nCentroids = 20):
    input_A = input_A[:,:ndims]
    if label_A.ndim>1:
        label_A = label_A[:,0]
    #compute label max and min to divide into centroids
    labelLimits = np.array([(np.percentile(label_A,5), np.percentile(label_A,95))]).T[:,0] 
    #find centroid size
    centSize = (labelLimits[1] - labelLimits[0]) / (nCentroids)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    centEdges = np.column_stack((np.linspace(labelLimits[0],labelLimits[0]+centSize*(nCentroids),nCentroids),
                                np.linspace(labelLimits[0],labelLimits[0]+centSize*(nCentroids),nCentroids)+centSize))

    if isinstance(dir_A, type(None)) :
        centLabel_A = np.zeros((nCentroids,ndims))
        ncentLabel_A = np.zeros((nCentroids,))
        for c in range(nCentroids):
            points_A = input_A[np.logical_and(label_A >= centEdges[c,0], label_A<centEdges[c,1]),:]
            centLabel_A[c,:] = np.median(points_A, axis=0)
            ncentLabel_A[c] = points_A.shape[0]
    else:
        input_A_left = copy.deepcopy(input_A[dir_A[:,0]==1,:])
        label_A_left = copy.deepcopy(label_A[dir_A[:,0]==1])
        input_A_right = copy.deepcopy(input_A[dir_A[:,0]==2,:])
        label_A_right = copy.deepcopy(label_A[dir_A[:,0]==2])
        
        centLabel_A = np.zeros((2*nCentroids,ndims))
        ncentLabel_A = np.zeros((2*nCentroids,))
        for c in range(nCentroids):
            points_A_left = input_A_left[np.logical_and(label_A_left >= centEdges[c,0], label_A_left<centEdges[c,1]),:]
            centLabel_A[2*c,:] = np.median(points_A_left, axis=0)
            ncentLabel_A[2*c] = points_A_left.shape[0]
            points_A_right = input_A_right[np.logical_and(label_A_right >= centEdges[c,0], label_A_right<centEdges[c,1]),:]
            centLabel_A[2*c+1,:] = np.median(points_A_right, axis=0)
            ncentLabel_A[2*c+1] = points_A_right.shape[0]

    del_cent_nan = np.all(np.isnan(centLabel_A), axis= 1)
    del_cent_num = (ncentLabel_A<20)
    del_cent = del_cent_nan + del_cent_num
    centLabel_A = np.delete(centLabel_A, del_cent, 0)
    return centLabel_A

def _create_save_folders(saveDir, mouse):
    #try creating general folder
    try:
        os.mkdir(saveDir)
    except:
        pass
    #add new folder with mouse name + current date-time
    saveDir = os.path.join(saveDir, mouse)
    #create this new folder
    try:
        os.mkdir(saveDir)
    except:
        pass
    return saveDir

def add_dir_mat_field(pdMouse):
    pdOut = copy.deepcopy(pdMouse)
    if 'dir_mat' not in pdOut.columns:
        pdOut["dir_mat"] = [np.zeros((pdOut["pos"][idx].shape[0],1)).astype(int)+
                            ('L' == pdOut["dir"][idx])+ 2*('R' == pdOut["dir"][idx])+
                            4*('F' in pdOut["dir"][idx]) for idx in pdOut.index]
    return pdOut

def preprocess_traces(pdMouse, signal_field, sigma = 5, sig_up = 4, sig_down = 12, peak_th=0.1):
    pdOut = copy.deepcopy(pdMouse)

    pdOut["index_mat"] = [np.zeros((pdOut[signal_field][idx].shape[0],1))+pdOut["trial_id"][idx] 
                                  for idx in range(pdOut.shape[0])]                     
    indexMat = np.concatenate(pdOut["index_mat"].values, axis=0)

    ogSignal = copy.deepcopy(np.concatenate(pdMouse[signal_field].values, axis=0))
    lowpassSignal = uniform_filter1d(ogSignal, size = 4000, axis = 0)
    signal = gaussian_filter1d(ogSignal, sigma = sigma, axis = 0)

    for nn in range(signal.shape[1]):
        baseSignal = np.histogram(ogSignal[:,nn], 100)
        baseSignal = baseSignal[1][np.argmax(baseSignal[0])]
        baseSignal = baseSignal + lowpassSignal[:,nn] - np.min(lowpassSignal[:,nn]) 

        cleanSignal = signal[:,nn]-baseSignal
        cleanSignal = cleanSignal/np.max(cleanSignal,axis = 0)
        cleanSignal[cleanSignal<0] = 0
        signal[:,nn] = cleanSignal

    biSignal = np.zeros(signal.shape)

    gaus = lambda x,sig,amp,vo: amp*np.exp(-(((x)**2)/(2*sig**2)))+vo;
    x = np.arange(-5*sig_down, 5*sig_down,1);
    upGaus = gaus(x,sig_up, 1, 0); 
    upGaus[5*sig_down+1:] = 0
    downGaus = gaus(x,sig_down, 1, 0); 
    downGaus[:5*sig_down+1] = 0
    finalGaus = downGaus + upGaus;

    for nn in range(signal.shape[1]):
        peakSignal,_ =find_peaks(signal[:,nn],height=peak_th)
        biSignal[peakSignal, nn] = signal[peakSignal, nn]
        if finalGaus.shape[0]<signal.shape[0]:
            biSignal[:, nn] = np.convolve(biSignal[:, nn],finalGaus, 'same')

    pdOut['clean_traces'] = [biSignal[indexMat[:,0]==pdOut["trial_id"][idx] ,:] 
                                                                for idx in range(pdOut.shape[0])]

    return pdOut

#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir = '/home/julio/Documents/SP_project/LT_DeepSup/data/'
saveDir = '/home/julio/Documents/SP_project/Fig2v2/processed_data/'

#%% PARAMS
sigma = 6
upSig = 4
downSig = 12
signalField = 'raw_traces'
peakTh = 0.1
velTh = 3
verbose = True

for mouse in miceList:
    #initialize time
    globalTime = timeit.default_timer()
    #create save folder data by adding time suffix
    mouseDataDir = os.path.join(dataDir, mouse)
    mouseSaveDir = _create_save_folders(saveDir, mouse)
    #check if verbose has to be saved into txt file
    if verbose:
        f = open(os.path.join(mouseSaveDir,mouse + '_logFile.txt'), 'w')
        original = sys.stdout
        sys.stdout = gu.Tee(sys.stdout, f)
    #%% 1.LOAD DATA
    localTime = timeit.default_timer()
    print('\n### 1. LOAD DATA ###')
    print('1 Searching & loading data in directory:\n', mouseDataDir)
    dfMouse = gu.load_files(mouseDataDir, '*_PyalData_struct*.mat', verbose=verbose)
    fileNames = list(dfMouse.keys())
    sessionPre = [fname for fname in fileNames if 'lt' in fname][0]
    print(f"\tSession selected: {sessionPre}")
    if verbose:
        gu.print_time_verbose(localTime, globalTime)
    #%% KEEP ONLY LAST SESSION
    pdMouse = copy.deepcopy(dfMouse[sessionPre])
    print('\n### 2. PROCESS DATA ###')
    print(f'Working on session: {sessionPre}')
    #%% 2. PROCESS DATA
    pdMouse = add_dir_mat_field(pdMouse)
    #2.1 keep only moving epochs
    print(f'2.1 Dividing into moving/still data w/ velTh= {velTh:.2f}.')
    if velTh>0:
        og_dur = np.concatenate(pdMouse["pos"].values, axis=0).shape[0]
        pdMouse, still_pdMouse = gu.keep_only_moving(pdMouse, velTh)
        move_dur = np.concatenate(pdMouse["pos"].values, axis=0).shape[0]
        print(f"\t{sessionPre}: Og={og_dur} ({og_dur/20}s) Move= {move_dur} ({move_dur/20}s)")
    else:
        print('2.1 Keeping all data (not limited to moving periods).')
        still_pdMouse = dict()
    #2.2 compute clean traces
    print(f"2.2 Computing clean-traces from {signalField} with sigma = {sigma}," +
        f" sigma_up = {upSig}, sigma_down = {downSig}", sep='')
    pdMouse = preprocess_traces(pdMouse, signalField, sigma = sigma, sig_up = upSig,
                            sig_down = downSig, peak_th = peakTh)
    if velTh>0:
        still_pdMouse = preprocess_traces(still_pdMouse, signalField, sigma = sigma, sig_up = upSig,
                                sig_down = downSig, peak_th = peakTh)
        save_still = open(os.path.join(mouseSaveDir, mouse+ "_still_df_dict.pkl"), "wb")
        pickle.dump(still_pdMouse, save_still)
        save_still.close()

    save_df = open(os.path.join(mouseSaveDir, mouse+ "_df_dict.pkl"), "wb")
    pickle.dump(pdMouse, save_df)
    save_df.close()
    params = {
        'sigma': sigma,
        'upSig': upSig,
        'downSig': downSig,
        'signalField': signalField,
        'peakTh': peakTh,
        'dataDir': mouseDataDir,
        'saveDir': mouseSaveDir,
        'mouse': mouse
    }
    save_params = open(os.path.join(mouseSaveDir, mouse+ "_params.pkl"), "wb")
    pickle.dump(params, save_params)
    save_params.close()
    # create list of strings
    paramsList = [ f'{key} : {params[key]}' for key in params]
    # write string one by one adding newline
    saveParamsFile = open(os.path.join(mouseSaveDir, mouse+ "_params.txt"), "w")
    with saveParamsFile as saveFile:
        [saveFile.write("%s\n" %st) for st in paramsList]
    saveParamsFile.close()
    sys.stdout = original
    f.close()

#__________________________________________________________________________
#|                                                                        |#
#|                               INNER DIM                                |#
#|________________________________________________________________________|#
import skdim
import time

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'signalName': 'clean_traces',
    'nNeigh': 30,
    'verbose': True
}

saveDir =  '/home/julio/Documents/SP_project/Fig2/dimensionality/inner_dim'
try:
    os.mkdir(saveDir)
except:
    pass
dataDir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'

idDict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    #compute abids dim
    abidsDim = np.nanmean(dim_red.compute_abids(signal, params['nNeigh']))
    print(f"\tABIDS: {abidsDim:.2f}", end='', flush=True)
    time.sleep(.2)

    corrIntDim = skdim.id.CorrInt(k1 = 5, k2 = params['nNeigh']).fit_transform(signal)
    print(f" | CorrInt: {corrIntDim:.2f}", end='', flush=True)
    time.sleep(.2)

    dancoDim = skdim.id.DANCo().fit_transform(signal)
    print(f" | DANCo: {dancoDim:.2f}", end='', flush=True)
    time.sleep(.2)

    essDim = skdim.id.ESS().fit_transform(signal,n_neighbors = params['nNeigh'])
    print(f" | ESS: {essDim:.2f}", end='', flush=True)
    time.sleep(.2)

    fishersDim = skdim.id.FisherS(conditional_number=5).fit_transform(signal)
    print(f" | FisherS: {fishersDim:.2f}", end='', flush=True)
    time.sleep(.2)

    knnDim = skdim.id.KNN(k=params['nNeigh']).fit_transform(signal)
    print(f" | KNN: {knnDim:.2f}", end='', flush=True)
    time.sleep(.2)

    lPCADim = skdim.id.lPCA(ver='broken_stick').fit_transform(signal)
    print(f" | lPCA: {lPCADim:.2f}", end='', flush=True)
    time.sleep(.2)

    madaDim = skdim.id.MADA().fit_transform(signal)
    print(f" | MADA: {madaDim:.2f}", end='', flush=True)
    time.sleep(.2)

    mindDim = skdim.id.MiND_ML(k=params['nNeigh']).fit_transform(signal)
    print(f" | MiND_ML: {mindDim:.2f}", end='', flush=True)
    time.sleep(.2)

    mleDim = skdim.id.MLE(K=params['nNeigh']).fit_transform(signal)
    print(f" | MLE: {mleDim:.2f}", end='', flush=True)
    time.sleep(.2)

    momDim = skdim.id.MOM().fit_transform(signal,n_neighbors = params['nNeigh'])
    print(f" | MOM: {momDim:.2f}", end='', flush=True)
    time.sleep(.2)

    tleDim = skdim.id.TLE().fit_transform(signal,n_neighbors = params['nNeigh'])
    print(f" | TLE: {tleDim:.2f}")
    time.sleep(.2)

    #save results
    idDict[mouse] = {
        'abidsDim': abidsDim,
        'corrIntDim': corrIntDim,
        'dancoDim': dancoDim,
        'essDim': essDim,
        'fishersDim': fishersDim,
        'knnDim': knnDim,
        'lPCADim': lPCADim,
        'madaDim': madaDim,
        'mindDim': mindDim,
        'mleDim': mleDim,
        'momDim': momDim,
        'tleDim': tleDim,
        'params': params
    }

    saveFile = open(os.path.join(saveDir, 'inner_dim_dict.pkl'), "wb")
    pickle.dump(idDict, saveFile)
    saveFile.close()

saveFile = open(os.path.join(saveDir, 'inner_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# create list of strings
paramsList = [ f'{key} : {params[key]}' for key in params]
# write string one by one adding newline
saveParamsFile = open(os.path.join(saveDir, "inner_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()


#__________________________________________________________________________
#|                                                                        |#
#|                                UMAP DIM                                |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

params = {
    'maxDim':10,
    'nNeigh': 120,
    'minDist': 0.1,
    'nnDim': 30,
    'signalName': 'clean_traces',
}

saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
umap_dim_dict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    print("Computing rank indices og space...", end = '', sep = '')
    rankIdx = dim_validation.compute_rank_indices(signal)
    print("\b\b\b: Done")
    trustNum = np.zeros((params['maxDim'],))
    contNum = np.zeros((params['maxDim'],))
    for dim in range(params['maxDim']):
        emb_space = np.arange(dim+1)
        print(f"Dim: {dim+1} ({dim+1}/{params['maxDim']})")
        model = umap.UMAP(n_neighbors = params['nNeigh'], n_components =dim+1, min_dist=params['minDist'])
        print("\tFitting model...", sep= '', end = '')
        emb = model.fit_transform(signal)
        print("\b\b\b: Done")
        #1. Compute trustworthiness
        print("\tComputing trustworthiness...", sep= '', end = '')
        temp = dim_validation.trustworthiness_vector(signal, emb, params['nnDim'], indices_source = rankIdx)
        trustNum[dim] = temp[-1]
        print(f"\b\b\b: {trustNum[dim]:.4f}")
        #2. Compute continuity
        print("\tComputing continuity...", sep= '', end = '')
        temp = dim_validation.continuity_vector(signal, emb ,params['nnDim'])
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

    umap_dim_dict[mouse] = {
        'trustNum': trustNum,
        'contNum': contNum,
        'trustDim': trustDim,
        'contDim':contDim,
        'hmeanDim': hmeanDim,
        'params': params
    }
    saveFile = open(os.path.join(saveDir, 'umap_dim_dict.pkl'), "wb")
    pickle.dump(umap_dim_dict, saveFile)
    saveFile.close()

saveFile = open(os.path.join(saveDir, 'umap_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# write string one by one adding newline
paramsList = [ f'{key} : {params[key]}' for key in params]
saveParamsFile = open(os.path.join(saveDir, "umap_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveParamsFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                               ISOMAP DIM                               |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'maxDim':10,
    'nNeigh': 120,
    'signalName': 'clean_traces',
}
saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
isomap_dim_dict = dict()

#define kernel function for rec error
K = lambda D: -0.5*((np.eye(D.shape[0])-(1/D.shape[0])).dot(np.square(D))).dot(np.eye(D.shape[0])-(1/D.shape[0]))

for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    #initialize isomap object
    isoModel = Isomap(n_neighbors = params['nNeigh'], n_components = params['maxDim'])
    #fit and project data
    print("\tFitting model...", sep= '', end = '')
    emb = isoModel.fit_transform(signal)
    print("\b\b\b: Done")
    resVar = np.zeros((params['maxDim'], ))
    recError = np.zeros((params['maxDim'], ))
    nSamples = signal.shape[0]
    #compute Isomap kernel for input data once before going into the loop
    signalKD = K(isoModel.dist_matrix_)
    for dim in range(1, params['maxDim']+1):
        print(f"Dim {dim+1}/{params['maxDim']}")
        embD = pairwise_distances(emb[:,:dim], metric = 'euclidean') 
        #compute residual variance
        resVar[dim-1] = 1 - pearsonr(np.matrix.flatten(isoModel.dist_matrix_),
                                            np.matrix.flatten(embD.astype('float32')))[0]**2
        #compute residual error
        embKD = K(embD)
        recError[dim-1] = np.linalg.norm((signalKD-embKD)/nSamples, 'fro')

    #find knee in residual variance
    dimSpace = np.linspace(1,params['maxDim'], params['maxDim']).astype(int)

    kl = KneeLocator(dimSpace, resVar, curve = "convex", direction = "decreasing")                                                      
    if kl.knee:
        resVarDim = kl.knee
        print('Final dimension: %d - Final residual variance: %.4f' %(resVarDim, resVar[resVarDim-1]))
    else:
        resVarDim = np.nan
        print('Could estimate final dimension (knee not found). Returning nan.')

    #find knee in reconstruction error      
    kl = KneeLocator(dimSpace, recError, curve = "convex", direction = "decreasing")                                                      
    if kl.knee:
        recErrorDim = kl.knee
        print('Final dimension: %d - Final reconstruction error: %.4f' %(recErrorDim, recError[recErrorDim-1]))
    else:
        recErrorDim = np.nan
        print('Could estimate final dimension (knee not found). Returning nan.')
    isomap_dim_dict[mouse] = {
        'resVar': resVar,
        'resVarDim': resVarDim,
        'recError': recError,
        'recErrorDim': recErrorDim,
        'params': params
    }
    saveFile = open(os.path.join(saveDir, 'isomap_dim_dict.pkl'), "wb")
    pickle.dump(isomap_dim_dict, saveFile)
    saveFile.close()

saveFile = open(os.path.join(saveDir, 'isomap_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# write string one by one adding newline
paramsList = [ f'{key} : {params[key]}' for key in params]
saveParamsFile = open(os.path.join(saveDir, "isomap_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                                 PCA DIM                                |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'signalName': 'clean_traces',
}
saveDir = '/home/julio/Documents/SP_project/Fig2/dimensionality/'
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
pca_dim_dict = dict()
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    #initialize isomap object
    modelPCA = PCA(signal.shape[1])
    #fit data
    modelPCA = modelPCA.fit(signal)
    #dim 80% variance explained
    var80Dim = np.where(np.cumsum(modelPCA.explained_variance_ratio_)>=0.80)[0][0]
    print('80%% Final dimension: %d' %(var80Dim))

    #dim drop in variance explained by knee
    dimSpace = np.linspace(1,signal.shape[1], signal.shape[1]).astype(int)        
    kl = KneeLocator(dimSpace, modelPCA.explained_variance_ratio_, curve = "convex", direction = "decreasing")                                                      
    if kl.knee:
        dim = kl.knee
        print('Knee Final dimension: %d - Final variance: %.4f' %(dim, np.cumsum(modelPCA.explained_variance_ratio_)[dim]))
    else:
        dim = np.nan
        print('Could estimate final dimension (knee not found). Returning nan.')

    pca_dim_dict[mouse] = {
        'explained_variance_ratio_': modelPCA.explained_variance_ratio_,
        'var80Dim': var80Dim,
        'kneeDim': dim,
        'params': params
    }
    saveFile = open(os.path.join(saveDir, 'pca_dim_dict.pkl'), "wb")
    pickle.dump(pca_dim_dict, saveFile)
    saveFile.close()

saveFile = open(os.path.join(saveDir, 'pca_dim_params.pkl'), "wb")
pickle.dump(params, saveFile)
saveFile.close()
# write string one by one adding newline
paramsList = [ f'{key} : {params[key]}' for key in params]
saveParamsFile = open(os.path.join(saveDir, "pca_dim_params.txt"), "w")
with saveParamsFile as saveFile:
    [saveFile.write("%s\n" %st) for st in paramsList]
saveParamsFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                              SAVE DIM RED                              |#
#|________________________________________________________________________|#
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
params = {
    'dim':3,
    'nNeigh': 120,
    'minDist': 0.1,
    'signalName': 'clean_traces',
}
dataDir = '/home/julio/Documents/SP_project/Fig2/processed_data/'
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    dim_red_object = dict()
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    pos = np.concatenate(pdMouse['pos'].values, axis=0)
    dirMat = np.concatenate(pdMouse['dir_mat'].values, axis=0)
    indexMat = np.concatenate(pdMouse['index_mat'].values, axis=0)

    print("\tFitting umap model...", sep= '', end = '')
    modelUmap = umap.UMAP(n_neighbors =params['nNeigh'], n_components =params['dim'], min_dist=params['minDist'])
    modelUmap.fit(signal)
    embUmap = modelUmap.transform(signal)
    print("\b\b\b: Done")
    pdMouse['umap'] = [embUmap[indexMat[:,0]==pdMouse["trial_id"][idx] ,:] 
                                                    for idx in pdMouse.index]
    dim_red_object['umap'] = copy.deepcopy(modelUmap)

    print("\tFitting isomap model...", sep= '', end = '')
    modelIsomap = Isomap(n_neighbors =params['nNeigh'], n_components = signal.shape[1])
    modelIsomap.fit(signal)
    embIsomap = modelIsomap.transform(signal)
    print("\b\b\b: Done")
    pdMouse['isomap'] = [embIsomap[indexMat[:,0]==pdMouse["trial_id"][idx] ,:] 
                                                    for idx in pdMouse.index]
    dim_red_object['isomap'] = copy.deepcopy(modelIsomap)

    print("\tFitting pca model...", sep= '', end = '')
    modelPCA = PCA(signal.shape[1])
    modelPCA.fit(signal)
    embPCA = modelPCA.transform(signal)
    print("\b\b\b: Done")
    pdMouse['pca'] = [embPCA[indexMat[:,0]==pdMouse["trial_id"][idx] ,:] 
                                                    for idx in pdMouse.index]
    dim_red_object['pca'] = copy.deepcopy(modelPCA)
    #%%
    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(2,3,1, projection = '3d')
    b = ax.scatter(*embUmap[:,:3].T, c = dirMat, cmap = 'Accent',s = 10, vmin= 0, vmax = 8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Umap')
    ax = plt.subplot(2,3,4, projection = '3d')
    b = ax.scatter(*embUmap[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)

    ax = plt.subplot(2,3,2, projection = '3d')
    b = ax.scatter(*embIsomap[:,:3].T, c = dirMat, cmap = 'Accent',s = 10, vmin= 0, vmax = 8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('Isomap')
    ax = plt.subplot(2,3,5, projection = '3d')
    b = ax.scatter(*embIsomap[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)

    ax = plt.subplot(2,3,3, projection = '3d')
    b = ax.scatter(*embPCA[:,:3].T, c = dirMat, cmap = 'Accent',s = 10, vmin= 0, vmax = 8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_title('PCA')
    ax = plt.subplot(2,3,6, projection = '3d')
    b = ax.scatter(*embPCA[:,:3].T, c = pos[:,0], cmap = 'magma',s = 10)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    plt.suptitle(f"{mouse}")
    plt.savefig(os.path.join(filePath,f'{mouse}_dim_red.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(filePath, fileName), "wb") as file:
        pickle.dump(pdMouse, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(filePath, mouse+"_dim_red_object.pkl"), "wb") as file:
        pickle.dump(dim_red_object, file, protocol=pickle.HIGHEST_PROTOCOL)

#__________________________________________________________________________
#|                                                                        |#
#|                      COMPUTE ELLIPSE ECCENTRICITY                      |#
#|________________________________________________________________________|#
def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx


miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2/eccentricity/'
ellipseDict = dict()

for mouse in miceList:
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    pdMouse = load_pickle(filePath,fileName)

    pos = np.concatenate(pdMouse['pos'].values, axis = 0)
    dirMat = np.concatenate(pdMouse['dir_mat'].values, axis=0)
    emb = np.concatenate(pdMouse['umap'].values, axis = 0)[:,:3]

    D = pairwise_distances(emb)
    noiseIdx = filter_noisy_outliers(emb,D)
    cemb = emb[~noiseIdx,:]
    cpos = pos[~noiseIdx,:]
    cdirMat = dirMat[~noiseIdx]

    #compute centroids
    cent = get_centroids(cemb, cpos[:,0],cdirMat, ndims = 3, nCentroids=40)  
    modelPCA = PCA(2)
    modelPCA.fit(cent)
    cent2D = modelPCA.transform(cent)
    cent2D = cent2D - np.tile(np.mean(cent2D,axis=0), (cent2D.shape[0],1))

    plt.figure()
    ax = plt.subplot(1,2,1, projection = '3d')
    ax.scatter(*emb[:,:3].T, c = pos[:,0], s=10, cmap = 'magma')

    ########################
    #         PRE          #
    ########################
    # Formulate and solve the least squares problem ||Ax - b ||^2
    X = cent2D[:,0].reshape(-1,1)
    Y = cent2D[:,1].reshape(-1,1)
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(cent2D[:,0])
    x = np.linalg.lstsq(A, b)[0].squeeze()

    xLim = [np.min(cent2D[:,0]), np.max(cent2D[:,0])]
    yLim = [np.min(cent2D[:,1]), np.max(cent2D[:,1])]
    x_coord = np.linspace(xLim[0]-np.abs(xLim[0]*0.1),xLim[1]+np.abs(xLim[1]*0.1),10000)
    y_coord = np.linspace(yLim[0]-np.abs(yLim[0]*0.1),yLim[1]+np.abs(yLim[1]*0.1),10000)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0]*X_coord**2 + x[1]*X_coord*Y_coord + x[2]*Y_coord**2 + x[3]*X_coord + x[4]*Y_coord

    flatX = X_coord.reshape(-1,1)
    flatY = Y_coord.reshape(-1,1)
    flatZ = Z_coord.reshape(-1,1)
    idxValid = np.abs(flatZ-1)
    idxValid = idxValid<np.percentile(idxValid,0.01)
    xValid = flatX[idxValid]
    yValid = flatY[idxValid]

    x0 = (x[1]*x[4] - 2*x[2]*x[3])/(4*x[0]*x[2] - x[1]**2)
    y0 = (x[1]*x[3] - 2*x[0]*x[4])/(4*x[0]*x[2] - x[1]**2)
    center = [x0, y0]

    #Compute Excentricity
    distEllipse = np.sqrt((xValid-center[0])**2 + (yValid-center[1])**2)
    pointLong = [xValid[np.argmax(distEllipse)], yValid[np.argmax(distEllipse)]]
    pointShort = [xValid[np.argmin(distEllipse)], yValid[np.argmin(distEllipse)]]
    longAxis = np.max(distEllipse)
    shortAxis = np.min(distEllipse)

    #Plot
    ax = plt.subplot(1,2,2)
    ax.scatter(*cent2D[:,:2].T, color ='b', s=10)
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    plt.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, color ='m', s=10)
    ax.plot([center[0],pointLong[0]], [center[1], pointLong[1]], color = 'c')
    ax.plot([center[0],pointShort[0]], [center[1], pointShort[1]], color = 'c')
    ax.set_xlim(np.min(cent2D[:,0])-0.2, np.max(cent2D[:,0])+0.2)
    ax.set_ylim(np.min(cent2D[:,1])-0.2, np.max(cent2D[:,1])+0.2)
    ax.set_xlim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_ylim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{longAxis/shortAxis:4f}")
    plt.suptitle(mouse)
    plt.tight_layout()

    ellipseDict[mouse] = {
        'pos':pos,
        'dirMat': dirMat,
        'emb': emb,
        'D': D,
        'noiseIdx': noiseIdx,
        'cpos': cpos,
        'cdirMat': cdirMat,
        'cemb': cemb,
        'cent': cent,
        'cent2D': cent2D,
        'ellipseCoeff': x,
        'xLim': xLim,
        'yLim': yLim,
        'X_coord': X_coord,
        'Y_coord': Y_coord,
        'Z_coord': Z_coord,
        'idxValid':idxValid,
        'xValid': xValid,
        'yValid': yValid,
        'center': center,
        'distEllipse': distEllipse,
        'pointLong': pointLong,
        'pointShort': pointShort,
        'longAxis': longAxis,
        'shortAxis': shortAxis,
        'eccentricity': longAxis/shortAxis
    }

    with open(os.path.join(saveDir,'ellipse_fit_dict.pkl'), 'wb') as f:
        pickle.dump(ellipseDict, f)

    plt.savefig(os.path.join(saveDir,f'{mouse}_ellipse_fit.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(saveDir,f'{mouse}_ellipse_fit.svg'), dpi = 400,bbox_inches="tight",transparent=True)


#__________________________________________________________________________
#|                                                                        |#
#|                             ECCENTRICITY                               |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir =  '/home/julio/Documents/SP_project/Fig2/ellipse/'
ellipseDict = load_pickle(dataDir, 'ellipse_fit_dict.pkl')

eccenList = list()
mouseList = list()
layerList = list()
for mouse in miceList:
    eccenList.append(ellipseDict[mouse]['eccentricity'])
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
#|                              PLACE CELLS                               |#
#|________________________________________________________________________|#
from neural_manifold import place_cells as pc

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

dataDir = '/home/julio/Documents/SP_project/Fig2_edges/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig2_edges/place_cells/'

params = {
    'sF': 20,
    'bin_width': 5,
    'std_pos': 0,
    'std_pdf': 5,
    'method': 'spatial_info',
    'num_shuffles': 100,
    'min_shift': 10,
    'th_metric': 99,
    'ignore_edges':10
    }

for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)

    pdMouse = load_pickle(filePath,fileName)

    neuSignal = np.concatenate(pdMouse['clean_traces'].values, axis=0)
    posSignal = np.concatenate(pdMouse['pos'].values, axis=0)
    velSignal = np.concatenate(pdMouse['vel'].values, axis=0)
    dirSignal = np.concatenate(pdMouse['dir_mat'].values, axis=0)

    to_keep = np.logical_and(dirSignal[:,0]>0,dirSignal[:,0]<=2)
    posSignal = posSignal[to_keep,:] 
    velSignal = velSignal[to_keep] 
    neuSignal = neuSignal[to_keep,:] 
    dirSignal = dirSignal[to_keep,:] 

    mousePC = pc.get_place_cells(posSignal, neuSignal, vel_signal = velSignal, dim = 1,
                          direction_signal = dirSignal, mouse = mouse, save_dir = saveDir, **params)

    print('\tNum place cells:')
    num_cells = neuSignal.shape[1]
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,0]*(mousePC['place_cells_dir'][:,1]==0))
    print(f'\t\t Only left cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,1]*(mousePC['place_cells_dir'][:,0]==0))
    print(f'\t\t Only right cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    num_place_cells = np.sum(mousePC['place_cells_dir'][:,0]*mousePC['place_cells_dir'][:,1])
    print(f'\t\t Both dir cells {num_place_cells}/{num_cells} ({100*num_place_cells/num_cells})')
    with open(os.path.join(saveDir,mouse+'_pc_dict.pkl'), 'wb') as f:
        pickle.dump(mousePC, f)


#__________________________________________________________________________
#|                                                                        |#
#|                           MANIFOLD ACTIVITY                            |#
#|________________________________________________________________________|#

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']

dataDir = '/home/julio/Documents/SP_project/Fig2_edges/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2_edges/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2_edges/manifold_cells'

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

for mouse in miceList:
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    signal = copy.deepcopy(np.concatenate(pdMouse['clean_traces'].values, axis = 0))
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis = 0))
    emb = copy.deepcopy(np.concatenate(pdMouse['umap'].values, axis = 0))

    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')
    neuPDF = pcDict['neu_pdf']
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
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 30, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax = plt.subplot(2,2,4, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 30)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()
    fig.suptitle(mouse)
    plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells.png'), dpi = 400,bbox_inches="tight")

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

    histSignal = np.nanmean(np.squeeze(normNeuPDF.reshape(normNeuPDF.shape[0],-1,1)),axis=1)
    ax = plt.subplot(2,1,2)
    ax.plot(mapAxis[0][:,0], histSignal, linewidth=2)
    ax.set_xlabel('pos-x')
    ax.set_ylabel('histogram')
    plt.tight_layout()
    fig.suptitle(mouse)
    plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells_histogram.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells_histogram.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                           MANIFOLD ACTIVITY                            |#
#|________________________________________________________________________|#

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']

dataDir = '/home/julio/Documents/SP_project/Fig2_edges/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2_edges/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2_edges/manifold_cells'

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

for mouse in miceList:
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)

    signal = copy.deepcopy(np.concatenate(pdMouse['clean_traces'].values, axis = 0))
    pos = copy.deepcopy(np.concatenate(pdMouse['pos'].values, axis = 0))
    direction = copy.deepcopy(np.concatenate(pdMouse['dir_mat'].values, axis = 0))
    emb = copy.deepcopy(np.concatenate(pdMouse['umap'].values, axis = 0))

    #load place cells
    pcDict = load_pickle(placeDir, mouse+'_pc_dict.pkl')
    neuPDF = pcDict['neu_pdf']
    normNeuPDF = np.zeros(neuPDF.shape)
    nCells = neuPDF.shape[1]
    for d in range(neuPDF.shape[2]):
        for c in range(nCells):
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.max(neuPDF[:,c,d])
    orderLeft =  np.argsort(np.argmax(neuPDF[:,:,0], axis=0))
    orderRight =  np.argsort(np.argmax(neuPDF[:,:,1], axis=0))

    
    neuPDF = normNeuPDF
    meanNormNeuPDF = np.nanmean(np.concatenate((neuPDF[:, pcDict['place_cells_dir'][:,0],0],neuPDF[:, pcDict['place_cells_dir'][:,1],1]),axis=1),axis=1)
    mNeuPDFLeft = np.nanmean(neuPDF[:, pcDict['place_cells_dir'][:,0],0], axis=1)
    mNeuPDFRight = np.nanmean(neuPDF[:, pcDict['place_cells_dir'][:,1],1], axis=1)

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
        elif dire==1:
            manifoldSignal[p] = mNeuPDFLeft[x]
        elif dire==2:
            manifoldSignal[p] = mNeuPDFRight[x]

    fig = plt.figure(figsize=(8,12))
    ax = plt.subplot(2,2,1)
    ax.matshow(normNeuPDF[:,orderLeft,0].T, aspect = 'auto')
    histSignalLeft = nCells - 0.5*nCells*(mNeuPDFLeft/np.max(mNeuPDFLeft))
    ax.plot(histSignalLeft, color = 'white', linewidth = 5)
    ax.set_title('izq')
    ax.set_ylabel('cell number')
    ax.set_xlabel('pos-x')

    ax = plt.subplot(2,2,2)
    ax.matshow(normNeuPDF[:,orderRight,1].T, aspect = 'auto')
    histSignalRight = nCells - 0.5*nCells*(mNeuPDFRight/np.max(mNeuPDFRight))
    ax.plot(histSignalRight, color = 'white', linewidth = 3)
    ax.set_title('dcha')
    ax.set_ylabel('cell number')
    ax.set_xlabel('pos-x')

    ax = plt.subplot(2,2,3, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = pos[:,0],s = 30, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax = plt.subplot(2,2,4, projection = '3d')
    b = ax.scatter(*emb[:,:3].T, c = manifoldSignal,s = 30)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()
    fig.suptitle(mouse)
    plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells_onlyPC.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells_onlyPC.png'), dpi = 400,bbox_inches="tight")


    fig = plt.figure(figsize=(8,12))
    ax = plt.subplot(2,2,1)
    ax.matshow(normNeuPDF[:,orderLeft,0].T, aspect = 'auto')
    histSignalLeft = nCells - 0.5*nCells*(mNeuPDFLeft/np.max(mNeuPDFLeft))
    ax.plot(histSignalLeft, color = 'white', linewidth = 5)
    ax.set_title('izq')
    ax.set_ylabel('cell number')
    ax.set_xlabel('pos-x')

    ax = plt.subplot(2,2,2)
    ax.matshow(normNeuPDF[:,orderRight,1].T, aspect = 'auto')
    histSignalRight = nCells - 0.5*nCells*(mNeuPDFRight/np.max(mNeuPDFRight))
    ax.plot(histSignalRight, color = 'white', linewidth = 3)
    ax.set_title('dcha')
    ax.set_ylabel('cell number')
    ax.set_xlabel('pos-x')

    histSignal =meanNormNeuPDF
    ax = plt.subplot(2,1,2)
    ax.plot(mapAxis[0][:,0], histSignal, linewidth=2)
    ax.set_xlabel('pos-x')
    ax.set_ylabel('histogram')
    plt.tight_layout()
    fig.suptitle(mouse)
    plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells_histogram_onlyPC.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_manifoldCells_histogram_onlyPC.png'), dpi = 400,bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|                           MANIFOLD ACTIVITY                            |#
#|________________________________________________________________________|#

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
izq = ['GC2', 'GC3', 'CZ3']

dataDir = '/home/julio/Documents/SP_project/Fig2_edges/processed_data/'
placeDir = '/home/julio/Documents/SP_project/Fig2_edges/place_cells/'
saveDir = '/home/julio/Documents/SP_project/Fig2_edges/manifold_cells'

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
fig = plt.figure(figsize=(8,12))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

deepMat = np.zeros((40,5))*np.nan
deepMapAxis =  np.zeros((40,5))*np.nan
supMat = np.zeros((40,5))*np.nan
supMapAxis =  np.zeros((40,5))*np.nan

deepCount = 0
supCount = 0
for mouse in miceList:
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    
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
            normNeuPDF[:,c,d] = neuPDF[:,c,d]/np.max(neuPDF[:,c,d])
    meanNormNeuPDF = np.nanmean(np.squeeze(normNeuPDF.reshape(normNeuPDF.shape[0],-1,1)),axis=1)
    if mouse in izq:
        meanNormNeuPDF = np.flipud(meanNormNeuPDF)
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
m = np.nanmean(deepMat,axis=1)
sd = np.nanstd(deepMat,axis=1)
ax.plot(np.nanmean(deepMapAxis,axis=1),m, label = 'deep')
ax.fill_between(np.nanmean(deepMapAxis,axis=1), m-sd, m+sd, alpha = 0.3)

m = np.nanmean(supMat,axis=1)
sd = np.nanstd(supMat,axis=1)
ax.plot(np.nanmean(supMapAxis,axis=1),m, label = 'sup')
ax.fill_between(np.nanmean(supMapAxis,axis=1), m-sd, m+sd, alpha = 0.3)
ax.legend()