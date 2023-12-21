import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neural_manifold.general_utils as gu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import seaborn as sns
import umap
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from datetime import datetime

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#

def get_neuronal_fields(trial_data, ref_field=None):
    """
    Identify time-varying fields in the dataset
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    neuronal_fields : list of str
        list of fieldnames that store time-varying signals
    """
    if ref_field is None:
        # look for a spikes field
        ref_field = [col for col in trial_data.columns.values
                     if col.endswith("spikes") or col.endswith("rates")][0]

    # identify candidates based on the first trial
    first_trial = trial_data.iloc[0]
    T = first_trial[ref_field].shape[1]
    neuronal_fields = []
    for col in first_trial.index:
        try:
            if first_trial[col].shape[1] == T:
                neuronal_fields.append(col)
        except:
            pass

    # but check the rest of the trials, too
    ref_lengths = np.array([arr.shape[1] for arr in trial_data[ref_field]])
    for col in neuronal_fields:
        col_lengths = np.array([arr.shape[1] for arr in trial_data[col]])
        assert np.all(col_lengths == ref_lengths), f"not all lengths in {col} match the reference {ref_field}"

    return neuronal_fields

def add_dir_mat_field(pd_struct):
    out_pd = copy.deepcopy(pd_struct)
    if 'dir_mat' not in out_pd.columns:
        out_pd["dir_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            ('L' == out_pd["dir"][idx])+ 2*('R' == out_pd["dir"][idx])+
                            4*('F' in out_pd["dir"][idx]) for idx in out_pd.index]
    return out_pd

def preprocess_traces(pd_struct_p, pd_struct_r, signal_field, sigma = 5,sig_up = 4, sig_down = 12, peak_th=0.1):
    out_pd_p = copy.deepcopy(pd_struct_p)
    out_pd_r = copy.deepcopy(pd_struct_r)

    out_pd_p["index_mat"] = [np.zeros((out_pd_p[signal_field][idx].shape[0],1))+out_pd_p["trial_id"][idx] 
                                  for idx in range(out_pd_p.shape[0])]                     
    index_mat_p = np.concatenate(out_pd_p["index_mat"].values, axis=0)

    out_pd_r["index_mat"] = [np.zeros((out_pd_r[signal_field][idx].shape[0],1))+out_pd_r["trial_id"][idx] 
                                  for idx in range(out_pd_r.shape[0])]
    index_mat_r = np.concatenate(out_pd_r["index_mat"].values, axis=0)

    signal_p_og = copy.deepcopy(np.concatenate(pd_struct_p[signal_field].values, axis=0))
    lowpass_p = uniform_filter1d(signal_p_og, size = 4000, axis = 0)
    signal_p = gaussian_filter1d(signal_p_og, sigma = sigma, axis = 0)

    signal_r_og = copy.deepcopy(np.concatenate(pd_struct_r[signal_field].values, axis=0))
    lowpass_r = uniform_filter1d(signal_r_og, size = 4000, axis = 0)
    signal_r = gaussian_filter1d(signal_r_og, sigma = sigma, axis = 0)

    for nn in range(signal_p.shape[1]):
        base_p = np.histogram(signal_p_og[:,nn], 100)
        base_p = base_p[1][np.argmax(base_p[0])]
        base_p = base_p + lowpass_p[:,nn] - np.min(lowpass_p[:,nn]) 

        base_r = np.histogram(signal_r_og[:,nn], 100)
        base_r = base_r[1][np.argmax(base_r[0])]   
        base_r = base_r + lowpass_r[:,nn] - np.min(lowpass_r[:,nn])   

        concat_signal = np.concatenate((signal_p[:,nn]-base_p, signal_r[:,nn]-base_r))

        concat_signal = concat_signal/np.max(concat_signal,axis = 0)
        concat_signal[concat_signal<0] = 0
        signal_p[:,nn] = concat_signal[:signal_p.shape[0]]
        signal_r[:,nn] = concat_signal[signal_p.shape[0]:]

    bi_signal_p = np.zeros(signal_p.shape)
    bi_signal_r = np.zeros(signal_r.shape)

    gaus = lambda x,sig,amp,vo: amp*np.exp(-(((x)**2)/(2*sig**2)))+vo;

    x = np.arange(-5*sig_down, 5*sig_down,1);
    gaus_up = gaus(x,sig_up, 1, 0); 
    gaus_up[5*sig_down+1:] = 0
    gaus_down = gaus(x,sig_down, 1, 0); 
    gaus_down[:5*sig_down+1] = 0
    gaus_final = gaus_down + gaus_up;

    for nn in range(signal_p.shape[1]):
        peaks_p,_ =find_peaks(signal_p[:,nn],height=peak_th)
        bi_signal_p[peaks_p, nn] = signal_p[peaks_p, nn]
        if gaus_final.shape[0]<signal_p.shape[0]:
            bi_signal_p[:, nn] = np.convolve(bi_signal_p[:, nn],gaus_final, 'same')

        peaks_r,_ =find_peaks(signal_r[:,nn],height=peak_th)
        bi_signal_r[peaks_r, nn] = signal_r[peaks_r, nn]
        if gaus_final.shape[0]<signal_r.shape[0]:
            bi_signal_r[:, nn] = np.convolve(bi_signal_r[:, nn],gaus_final, 'same')


    out_pd_p['clean_traces'] = [bi_signal_p[index_mat_p[:,0]==out_pd_p["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_p.shape[0])]
    out_pd_r['clean_traces'] = [bi_signal_r[index_mat_r[:,0]==out_pd_r["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_r.shape[0])]
    return out_pd_p, out_pd_r

def remove_cells(pd_struct, cells_to_keep):
    out_pd = copy.deepcopy(pd_struct)

    out_pd["index_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            out_pd["trial_id"][idx]
                            for idx in out_pd.index]

    index_mat = np.concatenate(out_pd["index_mat"].values, axis=0)
    neuro_fields = get_neuronal_fields(out_pd, 'raw_traces')
    for field in neuro_fields:
        signal = np.concatenate(out_pd[field].values, axis=0)
        out_pd[field] = [signal[index_mat[:,0]==out_pd["trial_id"][idx]][:, cells_to_keep] 
                                                for idx in range(out_pd.shape[0])]
    return out_pd


miceList = ['CalbCh1_CNO', 'CalbCh2_CNO']
miceList = ['CalbCh1_veh', 'CalbCh1_CNO']
base_load_dir = '/home/julio/Documents/SP_project/LT_DREADD/data/'
base_save_dir = '/home/julio/Documents/SP_project/LT_DREADD/processed_data/other/'
signal_field = 'raw_traces'
vel_th = 6
sigma = 6
sig_up = 4
sig_down = 12
nn_val = 120
dim = 3

for mouse in miceList:
    print(f"Working on mouse: {mouse}")
    load_dir = os.path.join(base_load_dir, mouse)
    save_dir = os.path.join(base_save_dir, mouse)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir_fig = os.path.join(save_dir, 'figures')
    if not os.path.exists(save_dir_fig):
        os.mkdir(save_dir_fig)
    now = datetime.now()
    params = {
        "date": now.strftime("%d/%m/%Y %H:%M:%S"),
        "mouse": mouse,
        "load_dir": load_dir,
        "save_dir": save_dir,
        "signal_field": signal_field,
        "vel_th": vel_th,
        "sigma": sigma,
        "sig_up": sig_up,
        "sig_down": sig_down
    }
    #__________________________________________________________________________
    #|                                                                        |#
    #|                               LOAD DATA                                |#
    #|________________________________________________________________________|#
    animal = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")
    fnames = list(animal.keys())

    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]

    animal_p= copy.deepcopy(animal[fnamePre])
    animal_r= copy.deepcopy(animal[fnameRot])

    #__________________________________________________________________________
    #|                                                                        |#
    #|                          PREPROCESS TRACES                             |#
    #|________________________________________________________________________|#
    animal_p = add_dir_mat_field(animal_p)
    animal_r = add_dir_mat_field(animal_r)

    animal_p = gu.select_trials(animal_p,"dir == ['L','R','N']")
    animal_r = gu.select_trials(animal_r,"dir == ['L','R','N']")
    animal_p, animal_p_still = gu.keep_only_moving(animal_p, vel_th)
    animal_r, animal_r_still = gu.keep_only_moving(animal_r, vel_th)

    animal_p, animal_r = preprocess_traces(animal_p, animal_r, signal_field, sigma=sigma, sig_up = sig_up, sig_down = sig_down)
    animal_p_still, animal_r_still = preprocess_traces(animal_p_still, animal_r_still, signal_field, sigma=sigma, sig_up = sig_up, sig_down = sig_down)


    #__________________________________________________________________________
    #|                                                                        |#
    #|                               REMOVE CELLS                             |#
    #|________________________________________________________________________|#

    signal_p = copy.deepcopy(np.concatenate(animal_p['clean_traces'].values, axis=0))
    pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
    signal_r = copy.deepcopy(np.concatenate(animal_r['clean_traces'].values, axis=0))
    pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
    #%%all data
    index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
    concat_signal = np.vstack((signal_p, signal_r))
    model = umap.UMAP(n_neighbors =120, n_components =dim, min_dist=0.1)
    # model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
    model.fit(concat_signal)
    concat_emb = model.transform(concat_signal)
    emb_p = concat_emb[index[:,0]==0,:]
    emb_r = concat_emb[index[:,0]==1,:]

    #%%
    plt.figure()
    ax = plt.subplot(1,2,1, projection = '3d')
    ax.scatter(*emb_p[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*emb_r[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax.set_title('All')
    ax = plt.subplot(1,2,2, projection = '3d')
    ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
    ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
    plt.suptitle(f"{mouse}: clean_traces - vel: {vel_th} - nn: {nn_val} - dim: {dim}")
    plt.savefig(os.path.join(save_dir_fig,mouse+'_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)

    do_cells = input(f"{signal_p.shape[1]} cells. Do you want to review them?: ([Y]/N)")
    cells_to_keep = np.zeros((signal_p.shape[1],), dtype=bool)*True


    if not any(do_cells) or 'Y' in do_cells:

        signal_p_og = copy.deepcopy(np.concatenate(animal_p[signal_field].values, axis=0))
        signal_r_og = copy.deepcopy(np.concatenate(animal_r[signal_field].values, axis=0))
        lowpass_r = uniform_filter1d(signal_r_og, size = 4000, axis = 0)
        lowpass_p = uniform_filter1d(signal_p_og, size = 4000, axis = 0)
        signal_p = copy.deepcopy(np.concatenate(animal_p['clean_traces'].values, axis=0))
        signal_r = copy.deepcopy(np.concatenate(animal_r['clean_traces'].values, axis=0))
        for n in range(signal_p.shape[1]):
            ylims = [np.min([np.min(signal_p_og[:,n]), np.min(signal_r_og[:,n])]),
                    1.1*np.max([np.max(signal_p_og[:,n]), np.max(signal_r_og[:,n])]) ]

            f = plt.figure()
            ax = plt.subplot(2,2,1)
            ax.plot(signal_p_og[:,n])
            base = np.histogram(signal_p_og[:,n], 100)
            base_p = base[1][np.argmax(base[0])]
            base_p = base_p + lowpass_p[:,n] - np.min(lowpass_p[:,n])   
            ax.plot(base_p, color = 'r')
            ax.set_ylim(ylims)

            ax = plt.subplot(2,2,2)
            ax.plot(signal_r_og[:,n])
            base = np.histogram(signal_r_og[:,n], 100)
            base_r = base[1][np.argmax(base[0])]
            base_r = base_r + lowpass_r[:,n] - np.min(lowpass_r[:,n])   
            ax.plot(base_r, color = 'r')
            ax.set_ylim(ylims)

            ax = plt.subplot(2,2,3)
            ax.plot(signal_p[:,n])
            ax.set_ylim([-0.05, 1.5])

            ax = plt.subplot(2,2,4)
            ax.plot(signal_r[:,n])
            ax.set_ylim([-0.05, 1.5])
            plt.suptitle(f"{n}/{signal_p.shape[1]}")
            a = input()
            cells_to_keep[n] = not any(a)
            plt.close(f)

        print(f"Removing {np.sum(~cells_to_keep)} cells ({100*np.sum(~cells_to_keep)/cells_to_keep.shape[0]:.2f} %)")

        concat_signal = np.vstack((signal_p[:, cells_to_keep], signal_r[:, cells_to_keep]))
        model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
        model.fit(concat_signal)
        concat_emb = model.transform(concat_signal)
        emb_p_clean = concat_emb[index[:,0]==0,:]
        emb_r_clean = concat_emb[index[:,0]==1,:]


        plt.figure()
        ax = plt.subplot(2,2,1, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, color ='b', s= 30, cmap = 'magma')
        ax.scatter(*emb_r[:,:3].T, color = 'r', s= 30, cmap = 'magma')
        ax.set_title('All')
        ax = plt.subplot(2,2,2, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
        ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
        ax = plt.subplot(2,2,3, projection = '3d')
        ax.scatter(*emb_p_clean[:,:3].T, color ='b', s= 30, cmap = 'magma')
        ax.scatter(*emb_r_clean[:,:3].T, color = 'r', s= 30, cmap = 'magma')
        ax.set_title('Clean')
        ax = plt.subplot(2,2,4, projection = '3d')
        ax.scatter(*emb_p_clean[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
        ax.scatter(*emb_r_clean[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
        plt.tight_layout()
        plt.suptitle(f"{mouse}: conv_traces vel: {vel_th}")
        do_remove_cells = input(f"Do you want to remove cells?: ([Y]/N)")
        if not any(do_remove_cells) or 'Y' in do_remove_cells:
            animal_p = remove_cells(animal_p, cells_to_keep)
            animal_r = remove_cells(animal_r, cells_to_keep)

            animal_p_still = remove_cells(animal_p_still, cells_to_keep)
            animal_r_still = remove_cells(animal_r_still, cells_to_keep)
        else:
            cells_to_keep = np.zeros((signal_p_og.shape[1],), dtype=bool)*True


    params["cells_to_keep"] = cells_to_keep
    with open(os.path.join(save_dir, mouse+"_params.pkl"), "wb") as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
    animal_dict = {
        fnamePre: animal_p,
        fnameRot: animal_r
    }
    with open(os.path.join(save_dir, mouse+"_df_dict.pkl"), "wb") as file:
        pickle.dump(animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    animal_still_dict = {
        fnamePre: animal_p_still,
        fnameRot: animal_r_still
    }
    with open(os.path.join(save_dir, mouse+"_df_still_dict.pkl"), "wb") as file:
        pickle.dump(animal_still_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE ROTATION                            |#
#|________________________________________________________________________|#

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx

def get_centroids(input_A, input_B, label_A, label_B, dir_A = None, dir_B = None, ndims = 2, nCentroids = 20):
    input_A = input_A[:,:ndims]
    input_B = input_B[:,:ndims]
    if label_A.ndim>1:
        label_A = label_A[:,0]
    if label_B.ndim>1:
        label_B = label_B[:,0]
    #compute label max and min to divide into centroids
    total_label = np.hstack((label_A, label_B))
    labelLimits = np.array([(np.percentile(total_label,5), np.percentile(total_label,95))]).T[:,0] 
    #find centroid size
    centSize = (labelLimits[1] - labelLimits[0]) / (nCentroids)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    centEdges = np.column_stack((np.linspace(labelLimits[0],labelLimits[0]+centSize*(nCentroids),nCentroids),
                                np.linspace(labelLimits[0],labelLimits[0]+centSize*(nCentroids),nCentroids)+centSize))

    if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
        centLabel_A = np.zeros((nCentroids,ndims))
        centLabel_B = np.zeros((nCentroids,ndims))
        
        ncentLabel_A = np.zeros((nCentroids,))
        ncentLabel_B = np.zeros((nCentroids,))
        for c in range(nCentroids):
            points_A = input_A[np.logical_and(label_A >= centEdges[c,0], label_A<centEdges[c,1]),:]
            centLabel_A[c,:] = np.median(points_A, axis=0)
            ncentLabel_A[c] = points_A.shape[0]
            
            points_B = input_B[np.logical_and(label_B >= centEdges[c,0], label_B<centEdges[c,1]),:]
            centLabel_B[c,:] = np.median(points_B, axis=0)
            ncentLabel_B[c] = points_B.shape[0]
    else:
        input_A_left = copy.deepcopy(input_A[dir_A[:,0]==1,:])
        label_A_left = copy.deepcopy(label_A[dir_A[:,0]==1])
        input_A_right = copy.deepcopy(input_A[dir_A[:,0]==2,:])
        label_A_right = copy.deepcopy(label_A[dir_A[:,0]==2])
        
        input_B_left = copy.deepcopy(input_B[dir_B[:,0]==1,:])
        label_B_left = copy.deepcopy(label_B[dir_B[:,0]==1])
        input_B_right = copy.deepcopy(input_B[dir_B[:,0]==2,:])
        label_B_right = copy.deepcopy(label_B[dir_B[:,0]==2])
        
        centLabel_A = np.zeros((2*nCentroids,ndims))
        centLabel_B = np.zeros((2*nCentroids,ndims))
        ncentLabel_A = np.zeros((2*nCentroids,))
        ncentLabel_B = np.zeros((2*nCentroids,))
        
        for c in range(nCentroids):
            points_A_left = input_A_left[np.logical_and(label_A_left >= centEdges[c,0], label_A_left<centEdges[c,1]),:]
            centLabel_A[2*c,:] = np.median(points_A_left, axis=0)
            ncentLabel_A[2*c] = points_A_left.shape[0]
            points_A_right = input_A_right[np.logical_and(label_A_right >= centEdges[c,0], label_A_right<centEdges[c,1]),:]
            centLabel_A[2*c+1,:] = np.median(points_A_right, axis=0)
            ncentLabel_A[2*c+1] = points_A_right.shape[0]

            points_B_left = input_B_left[np.logical_and(label_B_left >= centEdges[c,0], label_B_left<centEdges[c,1]),:]
            centLabel_B[2*c,:] = np.median(points_B_left, axis=0)
            ncentLabel_B[2*c] = points_B_left.shape[0]
            points_B_right = input_B_right[np.logical_and(label_B_right >= centEdges[c,0], label_B_right<centEdges[c,1]),:]
            centLabel_B[2*c+1,:] = np.median(points_B_right, axis=0)
            ncentLabel_B[2*c+1] = points_B_right.shape[0]

    del_cent_nan = np.all(np.isnan(centLabel_A), axis= 1)+ np.all(np.isnan(centLabel_B), axis= 1)
    del_cent_num = (ncentLabel_A<15) + (ncentLabel_B<15)
    del_cent = del_cent_nan + del_cent_num
    
    centLabel_A = np.delete(centLabel_A, del_cent, 0)
    centLabel_B = np.delete(centLabel_B, del_cent, 0)

    return centLabel_A, centLabel_B

def find_rotation(data_A, data_B, v):
    angles = np.linspace(-np.pi,np.pi,100)
    error = list()
    for angle in angles:
        #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        #https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
        a = np.cos(angle/2)
        b = np.sin(angle/2)*v[0,0]
        c = np.sin(angle/2)*v[1,0]
        d = np.sin(angle/2)*v[2,0]
        R = np.array([
                [a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d),a**2-b**2+c**2-d**2, 2*(c*d - a*b)],
                [2*(b*d - a*c), 2*(c*d + a*b), a**2-b**2-c**2+d**2]
            ])

        new_data =np.matmul(R, data_A.T).T
        error.append(np.sum(np.linalg.norm(new_data - data_B, axis=1)))

    return error


params = {
    'dim':3,
    'nNeigh': 120,
    'minDist': 0.1,
    'signalName': 'clean_traces',
}

miceList = ['CalbCh2_veh', 'CalbCh2_CNO', 'DD2_veh', 'DD2_CNO']
dataDir = '/home/julio/Documents/SP_project/LT_DREADD/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DREADD/rotation/'
rot_error_dict = dict()
for mouse in miceList:
    rot_error_dict[mouse] = dict()
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    saveDirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])


    signalPre = np.concatenate(animalPre[params['signalName']].values, axis = 0)
    posPre = np.concatenate(animalPre['pos'].values, axis = 0)
    dirMatPre = np.concatenate(animalPre['dir_mat'].values, axis=0)

    signalRot = np.concatenate(animalRot[params['signalName']].values, axis = 0)
    posRot = np.concatenate(animalRot['pos'].values, axis = 0)
    dirMatRot = np.concatenate(animalRot['dir_mat'].values, axis=0)


    indexPreRot = np.vstack((np.zeros((signalPre.shape[0],1)),np.zeros((signalRot.shape[0],1))+1))
    signalBoth = np.vstack((signalPre, signalRot))
    #umap
    print("\tFitting umap model...", sep= '', end = '')
    modelUmap = umap.UMAP(n_neighbors =100, n_components =params['dim'], min_dist=params['minDist'])
    modelUmap.fit(signalBoth)
    embBoth = modelUmap.transform(signalBoth)
    embPre = embBoth[indexPreRot[:,0]==0,:]
    embRot = embBoth[indexPreRot[:,0]==1,:]

    DPre = pairwise_distances(embPre)
    noiseIdxPre = filter_noisy_outliers(embPre,DPre)
    cembPre = embPre[~noiseIdxPre,:]
    cposPre = posPre[~noiseIdxPre,:]
    cdirMatPre = dirMatPre[~noiseIdxPre]

    DRot = pairwise_distances(embRot)
    noiseIdxRot = filter_noisy_outliers(embRot,DRot)
    cembRot = embRot[~noiseIdxRot,:]
    cposRot = posRot[~noiseIdxRot,:]
    cdirMatRot = dirMatRot[~noiseIdxRot]



    #compute centroids
    centPre, centRot = get_centroids(cembPre, cembRot, cposPre[:,0], cposRot[:,0], 
                                                    ndims = 3, nCentroids=30)   
    #find axis of rotatio                                                
    midPre = np.median(cembPre, axis=0).reshape(-1,1)
    midRot = np.median(cembRot, axis=0).reshape(-1,1)
    normVector =  midPre - midRot
    normVector = normVector/np.linalg.norm(normVector)
    k = np.dot(np.median(cembPre, axis=0), normVector)

    angles = np.linspace(-np.pi,np.pi,100)
    error = find_rotation(centPre-midPre.T, centRot-midRot.T, normVector)
    normError = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
    rotAngle = np.abs(angles[np.argmin(normError)])*180/np.pi
    plt.figure()
    plt.plot(angles, normError)
    print(f"\t{'umap'}: {rotAngle:2f} degrees")


    plt.figure()
    ax = plt.subplot(2,2,1, projection = '3d')
    ax.scatter(*cembPre[:,:3].T, color ='b', s=10)
    ax.scatter(*cembRot[:,:3].T, color = 'r', s=10)
    ax = plt.subplot(2,2,3, projection = '3d')
    ax.scatter(*cembPre[:,:3].T, c = cposPre[:,0], s=10, cmap = 'magma')
    ax.scatter(*cembRot[:,:3].T, c = cposRot[:,0], s=10, cmap = 'magma')
    ax = plt.subplot(1,2,2, projection = '3d')
    ax.scatter(*centPre[:,:3].T, color ='b', s=10)
    ax.scatter(*centRot[:,:3].T, color = 'r', s=10)

    rot_error_dict[mouse]['umap'] = {
        'embPre': embPre,
        'embRot': embRot,
        'posPre': posPre,
        'posRot': posRot,
        'dirMatPre': dirMatPre,
        'dirMatRot': dirMatRot,
        'noiseIdxPre': noiseIdxPre,
        'noiseIdxRot': noiseIdxRot,
        'centPre': centPre,
        'centRot': centRot,
        'midPre': midPre,
        'midRot': midRot,
        'normVector': normVector,
        'angles': angles,
        'error': error,
        'normError': normError,
        'rotAngle': rotAngle
    }

    with open(os.path.join(saveDir, f"rot_error_dict.pkl"), "wb") as file:
        pickle.dump(rot_error_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT EMBEDDING                             |#
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


dataDir = '/home/julio/Documents/SP_project/LT_DREADD/rotation/'
saveDir = '/home/julio/Documents/SP_project/LT_DREADD/rotation/emb_examples/'
rot_error_dict = load_pickle(dataDir, 'rot_error_dict.pkl')

view_init_dict = {
    'DD2_veh': [-150,-90],
    'DD2_CNO': [30, 30]
}
for mouse in miceList:
    noiseIdxPre = rot_error_dict[mouse]['umap']['noiseIdxPre']
    embPre = rot_error_dict[mouse]['umap']['embPre'][~noiseIdxPre,:]
    posPre = rot_error_dict[mouse]['umap']['posPre'][~noiseIdxPre,:]
    dirMatPre = rot_error_dict[mouse]['umap']['dirMatPre'][~noiseIdxPre]

    noiseIdxRot = rot_error_dict[mouse]['umap']['noiseIdxRot']
    embRot = rot_error_dict[mouse]['umap']['embRot'][~noiseIdxRot,:]
    posRot = rot_error_dict[mouse]['umap']['posRot'][~noiseIdxRot,:]
    dirMatRot = rot_error_dict[mouse]['umap']['dirMatRot'][~noiseIdxRot]


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

    if mouse in list(view_init_dict.keys()):
        view_init_values = view_init_dict[mouse]
    else:
        view_init_values = None
    fig = plt.figure(figsize=((15,5)))
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = 'b',s = 10)
    ax.scatter(*embRot[:,:3].T, c = 'r',s = 10)
    personalize_ax(ax,view_init_values)
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*embPre[:,:3].T, color = dirColorPre,s = 10)
    ax.scatter(*embRot[:,:3].T, color = dirColorRot,s = 10)
    personalize_ax(ax,view_init_values)
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], cmap='inferno',s = 10)
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], cmap='inferno',s = 10)
    personalize_ax(ax,view_init_values)
    fig.suptitle(mouse)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,mouse+'_manifold.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_manifold.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                              PLOT ROTATION                             |#
#|________________________________________________________________________|#

saveDir = '/home/julio/Documents/SP_project/LT_DREADD/rotation/'
miceList = ['CalbCh2_veh', 'CalbCh2_CNO', 'DD2_veh', 'DD2_CNO']

#PLOT BOXPLOTS
rotAngleList = list()
layerList = list()
mouseName = list()

rot_error_dict = load_pickle(saveDir, 'rot_error_dict.pkl')
for mouse in miceList:
    rotAngleList.append(rot_error_dict[mouse]['umap']['rotAngle'])
    if 'CNO' in mouse:
        layerList.append('CNO')
    elif 'veh' in mouse:
        layerList.append('veh')
    mouseName.append(mouse[:-4])

anglePD = pd.DataFrame(data={'angle': rotAngleList,
                            'layer': layerList,
                            'mouse': mouseName})

palette= ["#32e653", "#E632C5"]
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='layer', y='angle',data=anglePD, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='layer', y='angle', data=anglePD, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='layer', y= 'angle', data=anglePD, units = 'mouse', ax = ax, estimator = None, color = ".7", markers = True)
ax.set_ylabel('Angle Rotation')
ax.set_ylim([0,185])
ax.set_yticks([0,45,90,135,180])
plt.tight_layout()
plt.savefig(os.path.join(saveDir,f'vehCNO_rotation_boxplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(saveDir,f'vehCNO_rotation_boxplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)


#__________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE DISTANCE                            |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/LT_DREADD/rotation/'
save_dir = '/home/julio/Documents/SP_project/LT_DREADD/distance/'
miceList = ['CalbCh2_veh', 'CalbCh2_CNO', 'DD2_veh', 'DD2_CNO']

remap_dist_dict = dict()
rotation_dict = load_pickle(data_dir, 'rot_error_dict.pkl')
for mouse in miceList:
    print(f"Working on mouse {mouse}:")
    cent_pre = rotation_dict[mouse]['umap']['centPre']
    cent_rot = rotation_dict[mouse]['umap']['centRot']
    inter_dist = np.mean(pairwise_distances(cent_pre, cent_rot))
    intra_pre = np.percentile(pairwise_distances(cent_pre),95)
    intra_rot = np.percentile(pairwise_distances(cent_rot),95)
    remap_dist = inter_dist/np.max((intra_pre, intra_rot))
    print(f"Remmap Dist: {remap_dist:.4f}")

    remap_dist_dict[mouse] = {
        'cent_pre': cent_pre,
        'cent_rot': cent_rot,
        'inter_dist': inter_dist,
        'intra_pre': intra_pre,
        'intra_rot': intra_rot,
        'remap_dist': remap_dist,

    }
    with open(os.path.join(save_dir,'remap_distance_dict.pkl'), 'wb') as f:
        pickle.dump(remap_dist_dict, f)



#__________________________________________________________________________
#|                                                                        |#
#|                              PLOT DISTNACE                             |#
#|________________________________________________________________________|#

save_dir = '/home/julio/Documents/SP_project/LT_DREADD/distance/'

#PLOT BOXPLOTS
distance_list = list()
layer_list = list()
mouse_name_list = list()

remap_dist_dict = load_pickle(save_dir, 'remap_distance_dict.pkl')
for mouse in list(remap_dist_dict.keys()):
    distance_list.append(remap_dist_dict[mouse]['remap_dist'])
    if 'CNO' in mouse:
        layer_list.append('CNO')
    elif 'veh' in mouse:
        layer_list.append('veh')
    mouse_name_list.append(mouse[:-4])

pd_distance = pd.DataFrame(data={'angle': distance_list,
                            'layer': layer_list,
                            'mouse': mouse_name_list})

palette= ["#32e653", "#E632C5"]
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='layer', y='angle',data=pd_distance, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='layer', y='angle', data=pd_distance, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='layer', y= 'angle', data=pd_distance, units = 'mouse', ax = ax, estimator = None, color = ".7", markers = True)
ax.set_ylabel('Distance')
# ax.set_ylim([0,2])
# ax.set_yticks([0,45,90,135,180])
plt.tight_layout()
plt.savefig(os.path.join(save_dir,f'vehCNO_distance_barplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,f'vehCNO_distance_barplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)