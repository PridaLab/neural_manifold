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
from datetime import datetime
from neural_manifold import place_cells as pc
from scipy import stats
import skdim
import time
from scipy.stats import pearsonr

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

def compute_entanglement(points):
    distance_b = pairwise_distances(points)
    model_iso = Isomap(n_neighbors = 10, n_components = 1)
    emb = model_iso.fit_transform(points)
    distance_a = model_iso.dist_matrix_
    entanglement_boundary = np.max(distance_a[1:,0])/np.min(distance_b[1:,0])
    entanglement = np.max((distance_a[1:,0]/distance_b[1:,0]))
    return (entanglement-1)/entanglement_boundary

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
    del_cent_num = (ncentLabel_A<20) + (ncentLabel_B<20)
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


base_dir = '/home/julio/Documents/DeepSup_project/control_rot/'
mice_list = ['ChZ4', 'GC2']


def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))

#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#

def preprocess_traces(pd_struct_p, pd_struct_r, signal_field, output_signal_field, sigma = 5,sig_up = 4, sig_down = 12, peak_th=0.1):
    out_pd_p = copy.deepcopy(pd_struct_p)
    out_pd_r = copy.deepcopy(pd_struct_r)

    trial_id_mat_p = get_signal(out_pd_p, 'trial_id_mat').reshape(-1,)
    trial_id_mat_r = get_signal(out_pd_r, 'trial_id_mat').reshape(-1,)


    signal_p_og = get_signal(pd_struct_p,signal_field)
    lowpass_p = uniform_filter1d(signal_p_og, size = 4000, axis = 0)
    signal_p = gaussian_filter1d(signal_p_og, sigma = sigma, axis = 0)

    signal_r_og = get_signal(pd_struct_r,signal_field)
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


    out_pd_p[output_signal_field] = [bi_signal_p[trial_id_mat_p==out_pd_p["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_p.shape[0])]
    out_pd_r[output_signal_field] = [bi_signal_r[trial_id_mat_r==out_pd_r["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_r.shape[0])]
    return out_pd_p, out_pd_r


def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nnDist < np.percentile(nnDist, 20)
    return noiseIdx

vel_th = 6
sigma = 6
sig_up = 4
sig_down = 12
nn_val = 120
dim = 3

columns_to_drop = ['date','denoised_traces', '*spikes']
columns_to_rename = {'Fs':'sf','pos':'position', 'vel':'speed', 'index_mat': 'trial_idx_mat'}

for mouse in mice_list:
    print(f"Working on mouse: {mouse}")

    load_dir = os.path.join(base_dir,'data',mouse)
    save_dir = os.path.join(base_dir, 'processed_data',mouse)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    now = datetime.now()
    params = {
        "date": now.strftime("%d/%m/%Y %H:%M:%S"),
        "mouse": mouse,
        "load_dir": load_dir,
        "save_dir": save_dir,
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
    fname_pre = [fname for fname in fnames if 'ltpre' in fname][0]
    fname_post = [fname for fname in fnames if 'ltpost' in fname][0]
    animal_pre= copy.deepcopy(animal[fname_pre])
    animal_post= copy.deepcopy(animal[fname_post])

    #__________________________________________________________________________
    #|                                                                        |#
    #|               CHANGE COLUMN NAMES AND ADD NEW ONES                     |#
    #|________________________________________________________________________|#

    for column in columns_to_drop:
        if column in animal_pre.columns: animal_pre.drop(columns=[column], inplace=True)
        if column in animal_post.columns: animal_post.drop(columns=[column], inplace=True)

    for old, new in columns_to_rename.items():
        if old in animal_pre.columns: animal_pre.rename(columns={old:new}, inplace=True)
        if old in animal_post.columns: animal_post.rename(columns={old:new}, inplace=True)

    gu.add_trial_id_mat_field(animal_pre)
    gu.add_trial_id_mat_field(animal_post)

    gu.add_mov_direction_mat_field(animal_pre)
    gu.add_mov_direction_mat_field(animal_post)

    gu.add_trial_type_mat_field(animal_pre)
    gu.add_trial_type_mat_field(animal_post)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                          KEEP ONLY MOVING                              |#
    #|________________________________________________________________________|#

    # animal_pre = gu.select_trials(animal_pre,"dir == ['L','R','N']")
    # animal_post = gu.select_trials(animal_post,"dir == ['L','R','N']")
    if vel_th>0:
        animal_pre, animal_pre_still = gu.keep_only_moving(animal_pre, vel_th)
        animal_post, animal_post_still = gu.keep_only_moving(animal_post, vel_th)
    else:
        animal_pre_still = pd.DataFrame()
        animal_post_still = pd.DataFrame()

    animal_pre, _ = gu.keep_only_moving(animal_pre, vel_th)
    animal_post, _ = gu.keep_only_moving(animal_post, vel_th)


    #__________________________________________________________________________
    #|                                                                        |#
    #|                          PREPROCESS TRACES                             |#
    #|________________________________________________________________________|#

    for color in ['green', 'red']:
        animal_pre, animal_post = preprocess_traces(animal_pre, animal_post, 'raw_traces', 'clean_traces', sigma=sigma, sig_up = sig_up, sig_down = sig_down)
        animal_pre['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
        animal_post['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
        if vel_th>0:
            animal_pre_still, animal_post_still = preprocess_traces(animal_pre_still, animal_post_still, 'raw_traces', 'clean_traces', sigma=sigma, sig_up = sig_up, sig_down = sig_down)
            animal_pre_still['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
            animal_post_still['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                PLOT UMAP                               |#
    #|________________________________________________________________________|#


    signal_pre = get_signal(animal_pre, 'clean_traces')
    pos_pre = get_signal(animal_pre, 'position')
    dir_pre =get_signal(animal_pre, 'mov_direction')

    signal_post = get_signal(animal_post, 'clean_traces')
    pos_post = get_signal(animal_post, 'position')
    dir_post =get_signal(animal_post, 'mov_direction')

    #all data green
    index = np.vstack((np.zeros((signal_pre.shape[0],1)),np.ones((signal_post.shape[0],1))))
    concat_signal_green = np.vstack((signal_pre, signal_post))
    model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
    model.fit(concat_signal_green)
    emb_concat = model.transform(concat_signal_green)
    emb_pre = emb_concat[index[:,0]==0,:]
    emb_post = emb_concat[index[:,0]==1,:]

    #clean outliers
    D_pre = pairwise_distances(emb_pre)
    noise_idx_pre = filter_noisy_outliers(emb_pre,D=D_pre)
    csignal_pre = signal_pre[~noise_idx_pre,:]
    cemb_pre = emb_pre[~noise_idx_pre,:]
    cpos_pre = pos_pre[~noise_idx_pre,:]
    cdir_pre = dir_pre[~noise_idx_pre]

    D_post = pairwise_distances(emb_post)
    noise_idx_post = filter_noisy_outliers(emb_post,D=D_post)
    csignal_post = signal_post[~noise_idx_post,:]
    cemb_post = emb_post[~noise_idx_post,:]
    cpos_post = pos_post[~noise_idx_post,:]
    cdir_post = dir_post[~noise_idx_post]

    #PLOT
    def return_dir_color(dir_mat):
        dir_color = np.zeros((dir_mat.shape[0],3))
        for point in range(dir_mat.shape[0]):
            if dir_mat[point]==0:
                dir_color[point] = [14/255,14/255,143/255]
            elif dir_mat[point]==1:
                dir_color[point] = [12/255,136/255,249/255]
            else:
                dir_color[point] = [17/255,219/255,224/255]
        return dir_color
    dir_color_pre = return_dir_color(cdir_pre)
    dir_color_post = return_dir_color(cdir_post)

    plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*cemb_pre[:,:3].T, color ='b', s=10)
    ax.scatter(*cemb_post[:,:3].T, color = 'r', s=10)
    ax.set_title('Green')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*cemb_pre[:,:3].T, c = cpos_pre[:,0], s=10, cmap = 'magma')
    ax.scatter(*cemb_post[:,:3].T, c = cpos_post[:,0], s=10, cmap = 'magma')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*cemb_pre[:,:3].T, color=dir_color_pre, s=10)
    ax.scatter(*cemb_post[:,:3].T, color=dir_color_post, s=10)
    ax.set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb.png'), dpi = 400,bbox_inches="tight")


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                  SAVE                                  |#
    #|________________________________________________________________________|#
    with open(os.path.join(save_dir, mouse+"_params.pkl"), "wb") as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

    animal_dict = {
        fname_pre: animal_pre,
        fname_post: animal_post
    }
    with open(os.path.join(save_dir, mouse+"_df_dict.pkl"), "wb") as file:
        pickle.dump(animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    animal_still_dict = {
        fname_pre: animal_pre_still,
        fname_post: animal_post_still
    }
    with open(os.path.join(save_dir, mouse+"_df_still_dict.pkl"), "wb") as file:
        pickle.dump(animal_still_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, mouse+"_params.pkl"), "wb") as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

#__________________________________________________________________________
#|                                                                        |#
#|                              SAVE DIM RED                              |#
#|________________________________________________________________________|#

params = {
    'dim':3,
    'nNeigh': 120,
    'minDist': 0.1,
    'signalName': 'clean_traces',
}

for mouse in mice_list:
    dim_red_object = dict()

    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'

    file_path = os.path.join(base_dir, 'processed_data',mouse)
    save_dir_fig = os.path.join(file_path, 'figures')

    if not os.path.exists(save_dir_fig):
        os.mkdir(save_dir_fig)

    animal = load_pickle(file_path,file_name)
    fnames = list(animal.keys())
    fname_pre = [fname for fname in fnames if 'ltpre' in fname][0]
    fname_post = [fname for fname in fnames if 'ltpost' in fname][0]
    animal_pre= copy.deepcopy(animal[fname_pre])
    animal_post= copy.deepcopy(animal[fname_post])



    signal_pre = get_signal(animal_pre, params['signalName'])
    pos_pre = get_signal(animal_pre, 'position')
    dir_mat_pre = get_signal(animal_pre, 'mov_direction')
    index_mat_pre = get_signal(animal_pre, 'trial_id_mat')

    signal_post = get_signal(animal_post, params['signalName'])
    pos_post = get_signal(animal_post, 'position')
    dir_mat_post = get_signal(animal_post, 'mov_direction')
    index_mat_post = get_signal(animal_post, 'trial_id_mat')

    index_pre_post = np.vstack((np.zeros((signal_pre.shape[0],1)),np.zeros((signal_post.shape[0],1))+1))
    signal_concat = np.vstack((signal_pre, signal_post))

    #umap
    print("\tFitting umap model...", sep= '', end = '')
    model_umap = umap.UMAP(n_neighbors =params['nNeigh'], n_components =params['dim'], min_dist=params['minDist'])
    model_umap.fit(signal_concat)
    emb_concat = model_umap.transform(signal_concat)
    emb_pre = emb_concat[index_pre_post[:,0]==0,:]
    emb_post = emb_concat[index_pre_post[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*emb_post[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, c = pos_pre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*emb_post[:,:3].T, c = pos_post[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, c = dir_mat_pre, cmap = 'Accent',s = 30, vmin= -1, vmax = 7)
    ax.scatter(*emb_post[:,:3].T, c = dir_mat_post, cmap = 'Accent',s = 30, vmin= -1, vmax = 7)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_fig,mouse+'_saved_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(save_dir_fig,mouse+'_saved_umap.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animal_pre['umap'] = [emb_pre[index_mat_pre[:,0]==animal_pre["trial_id"][idx] ,:] 
                                   for idx in animal_pre.index]
    animal_post['umap'] = [emb_post[index_mat_post[:,0]==animal_post["trial_id"][idx] ,:] 
                                   for idx in animal_post.index]
    dim_red_object['umap'] = copy.deepcopy(model_umap)
    print("\b\b\b: Done")

    #isomap
    print("\tFitting isomap model...", sep= '', end = '')
    model_isomap = Isomap(n_neighbors =params['nNeigh'], n_components = signal_concat.shape[1])
    model_isomap.fit(signal_concat)
    emb_concat = model_isomap.transform(signal_concat)
    emb_pre = emb_concat[index_pre_post[:,0]==0,:]
    emb_post = emb_concat[index_pre_post[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*emb_post[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, c = pos_pre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*emb_post[:,:3].T, c = pos_post[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, c = dir_mat_pre, cmap = 'Accent',s = 30, vmin= -1, vmax = 7)
    ax.scatter(*emb_post[:,:3].T, c = dir_mat_post, cmap = 'Accent',s = 30, vmin= -1, vmax = 7)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_fig,mouse+'_saved_isomap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(save_dir_fig,mouse+'_saved_isomap.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animal_pre['isomap'] = [emb_pre[index_mat_pre[:,0]==animal_pre["trial_id"][idx] ,:] 
                                   for idx in animal_pre.index]
    animal_post['isomap'] = [emb_post[index_mat_post[:,0]==animal_post["trial_id"][idx] ,:] 
                                   for idx in animal_post.index]
    dim_red_object['isomap'] = copy.deepcopy(model_isomap)
    print("\b\b\b: Done")

    #pca
    print("\tFitting PCA model...", sep= '', end = '')
    model_pca = PCA(signal_concat.shape[1])
    model_pca.fit(signal_concat)
    emb_concat = model_pca.transform(signal_concat)
    emb_pre = emb_concat[index_pre_post[:,0]==0,:]
    emb_post = emb_concat[index_pre_post[:,0]==1,:]
    #%%
    fig = plt.figure()
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, color ='b', s= 30, cmap = 'magma')
    ax.scatter(*emb_post[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, c = pos_pre[:,0], s= 30, cmap = 'magma')
    ax.scatter(*emb_post[:,:3].T, c = pos_post[:,0], s= 30, cmap = 'magma')
    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, c = dir_mat_pre, cmap = 'Accent',s = 30, vmin= -1, vmax = 7)
    ax.scatter(*emb_post[:,:3].T, c = dir_mat_post, cmap = 'Accent',s = 30, vmin= -1, vmax = 7)
    plt.suptitle(f"{mouse}: {params['signalName']} - nn: {params['nNeigh']} - dim: {params['dim']}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_fig,mouse+'_saved_PCA.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(save_dir_fig,mouse+'_saved_PCA.svg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.close(fig)
    animal_pre['pca'] = [emb_pre[index_mat_pre[:,0]==animal_pre["trial_id"][idx] ,:] 
                                   for idx in animal_pre.index]
    animal_post['pca'] = [emb_post[index_mat_post[:,0]==animal_post["trial_id"][idx] ,:] 
                                   for idx in animal_post.index]
    dim_red_object['pca'] = copy.deepcopy(model_pca)
    print("\b\b\b: Done")

    new_animal_dict = {
        fname_pre: animal_pre,
        fname_post: animal_post
    }
    with open(os.path.join(file_path,file_name), "wb") as file:
        pickle.dump(new_animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(file_path, mouse+"_dim_red_object.pkl"), "wb") as file:
        pickle.dump(dim_red_object, file, protocol=pickle.HIGHEST_PROTOCOL)


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

def rotate_cloud_around_axis(point_cloud, angle, v):
    cloud_center = point_cloud.mean(axis=0)
    a = np.cos(angle/2)
    b = np.sin(angle/2)*v[0]
    c = np.sin(angle/2)*v[1]
    d = np.sin(angle/2)*v[2]
    R = np.array([
            [a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c)],
            [2*(b*c+a*d),a**2-b**2+c**2-d**2, 2*(c*d - a*b)],
            [2*(b*d - a*c), 2*(c*d + a*b), a**2-b**2-c**2+d**2]
        ])
    return  np.matmul(R, (point_cloud-cloud_center).T).T+cloud_center

def get_centroids(cloud_A, cloud_B, label_A, label_B, dir_A = None, dir_B = None, num_centroids = 20):
    dims = cloud_A.shape[1]
    if label_A.ndim>1:
        label_A = label_A[:,0]
    if label_B.ndim>1:
        label_B = label_B[:,0]
    #compute label max and min to divide into centroids
    total_label = np.hstack((label_A, label_B))
    label_lims = np.array([(np.percentile(total_label,5), np.percentile(total_label,95))]).T[:,0] 
    #find centroid size
    cent_size = (label_lims[1] - label_lims[0]) / (num_centroids)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    cent_edges = np.column_stack((np.linspace(label_lims[0],label_lims[0]+cent_size*(num_centroids),num_centroids),
                                np.linspace(label_lims[0],label_lims[0]+cent_size*(num_centroids),num_centroids)+cent_size))


    if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
        cent_A = np.zeros((num_centroids,dims))
        cent_B = np.zeros((num_centroids,dims))
        cent_label = np.mean(cent_edges,axis=1).reshape(-1,1)

        num_centroids_A = np.zeros((num_centroids,))
        num_centroids_B = np.zeros((num_centroids,))
        for c in range(num_centroids):
            points_A = cloud_A[np.logical_and(label_A >= cent_edges[c,0], label_A<cent_edges[c,1]),:]
            cent_A[c,:] = np.median(points_A, axis=0)
            num_centroids_A[c] = points_A.shape[0]
            
            points_B = cloud_B[np.logical_and(label_B >= cent_edges[c,0], label_B<cent_edges[c,1]),:]
            cent_B[c,:] = np.median(points_B, axis=0)
            num_centroids_B[c] = points_B.shape[0]
    else:
        cloud_A_left = copy.deepcopy(cloud_A[dir_A==-1,:])
        label_A_left = copy.deepcopy(label_A[dir_A==-1])
        cloud_A_right = copy.deepcopy(cloud_A[dir_A==1,:])
        label_A_right = copy.deepcopy(label_A[dir_A==1])
        
        cloud_B_left = copy.deepcopy(cloud_B[dir_B==-1,:])
        label_B_left = copy.deepcopy(label_B[dir_B==-1])
        cloud_B_right = copy.deepcopy(cloud_B[dir_B==1,:])
        label_B_right = copy.deepcopy(label_B[dir_B==1])
        
        cent_A = np.zeros((2*num_centroids,dims))
        cent_B = np.zeros((2*num_centroids,dims))
        num_centroids_A = np.zeros((2*num_centroids,))
        num_centroids_B = np.zeros((2*num_centroids,))
        
        cent_dir = np.zeros((2*num_centroids, ))
        cent_label = np.tile(np.mean(cent_edges,axis=1),(2,1)).T.reshape(-1,1)
        for c in range(num_centroids):
            points_A_left = cloud_A_left[np.logical_and(label_A_left >= cent_edges[c,0], label_A_left<cent_edges[c,1]),:]
            cent_A[2*c,:] = np.median(points_A_left, axis=0)
            num_centroids_A[2*c] = points_A_left.shape[0]
            points_A_right = cloud_A_right[np.logical_and(label_A_right >= cent_edges[c,0], label_A_right<cent_edges[c,1]),:]
            cent_A[2*c+1,:] = np.median(points_A_right, axis=0)
            num_centroids_A[2*c+1] = points_A_right.shape[0]

            points_B_left = cloud_B_left[np.logical_and(label_B_left >= cent_edges[c,0], label_B_left<cent_edges[c,1]),:]
            cent_B[2*c,:] = np.median(points_B_left, axis=0)
            num_centroids_B[2*c] = points_B_left.shape[0]
            points_B_right = cloud_B_right[np.logical_and(label_B_right >= cent_edges[c,0], label_B_right<cent_edges[c,1]),:]
            cent_B[2*c+1,:] = np.median(points_B_right, axis=0)
            num_centroids_B[2*c+1] = points_B_right.shape[0]

            cent_dir[2*c] = -1
            cent_dir[2*c+1] = 1

    del_cent_nan = np.all(np.isnan(cent_A), axis= 1)+ np.all(np.isnan(cent_B), axis= 1)
    del_cent_num = (num_centroids_A<15) + (num_centroids_B<15)
    del_cent = del_cent_nan + del_cent_num
    
    cent_A = np.delete(cent_A, del_cent, 0)
    cent_B = np.delete(cent_B, del_cent, 0)

    cent_label = np.delete(cent_label, del_cent, 0)

    if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
        return cent_A, cent_B, cent_label
    else:
        cent_dir = np.delete(cent_dir, del_cent, 0)
        return cent_A, cent_B, cent_label, cent_dir

def align_vectors(norm_vector_A, cloud_center_A, norm_vector_B, cloud_center_B):

    def find_rotation_align_vectors(a,b):
        v = np.cross(a,b)
        s = np.linalg.norm(v)
        c = np.dot(a,b)

        sscp = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]], [-v[1],v[0],0]])
        sscp2 = np.matmul(sscp,sscp)
        R = np.eye(3)+sscp+(sscp2*(1-c)/s**2)
        return R

    def check_norm_vector_direction(norm_vector, cloud_center, goal_point):
        og_dir = cloud_center+norm_vector
        op_dir = cloud_center+(-1*norm_vector)

        og_distance = np.linalg.norm(og_dir-goal_point)
        op_distance = np.linalg.norm(op_dir-goal_point)
        if og_distance<op_distance:
            return norm_vector
        else:
            return -norm_vector

    norm_vector_A = check_norm_vector_direction(norm_vector_A,cloud_center_A, cloud_center_B)
    norm_vector_B = check_norm_vector_direction(norm_vector_B,cloud_center_B, cloud_center_A)

    align_mat = find_rotation_align_vectors(norm_vector_A,-norm_vector_B)  #minus sign to avoid 180 flip
    align_angle = np.arccos(np.clip(np.dot(norm_vector_A, -norm_vector_B), -1.0, 1.0))*180/np.pi

    return align_angle, align_mat

def apply_rotation_to_cloud(point_cloud, rotation,center_of_rotation):
    return np.dot(point_cloud-center_of_rotation, rotation) + center_of_rotation

def parametrize_plane(point_cloud):
    '''
    point_cloud.shape = [p,d] (points x dimensions)
    based on Ger reply: https://stackoverflow.com/questions/35070178/
    fit-plane-to-a-set-of-points-in-3d-scipy-optimize-minimize-vs-scipy-linalg-lsts
    '''
    #get point cloud center
    cloud_center = point_cloud.mean(axis=0)
    # run SVD
    u, s, vh = np.linalg.svd(point_cloud - cloud_center)
    # unitary normal vector
    norm_vector = vh[2, :]
    return norm_vector, cloud_center

def project_onto_plane(point_cloud, norm_plane, point_plane):
    return point_cloud- np.multiply(np.tile(np.dot(norm_plane, (point_cloud-point_plane).T),(3,1)).T,norm_plane)

def find_rotation(data_A, data_B, v):
    center_A = data_A.mean(axis=0)
    angles = np.linspace(-np.pi,np.pi,200)
    error = list()
    for angle in angles:
        #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        #https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
        a = np.cos(angle/2)
        b = np.sin(angle/2)*v[0]
        c = np.sin(angle/2)*v[1]
        d = np.sin(angle/2)*v[2]
        R = np.array([
                [a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d),a**2-b**2+c**2-d**2, 2*(c*d - a*b)],
                [2*(b*d - a*c), 2*(c*d + a*b), a**2-b**2-c**2+d**2]
            ])

        new_data =np.matmul(R, (data_A-center_A).T).T + center_A
        error.append(np.sum(np.linalg.norm(new_data - data_B, axis=1)))

    return error

def plot_rotation(cloud_A, cloud_B, pos_A, pos_B, dir_A, dir_B, cent_A, cent_B, cent_pos, plane_cent_A, plane_cent_B, aligned_plane_cent_B, rotated_aligned_cent_rot, angles, error, rotation_angle):
    def process_axis(ax):
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_aspect('equal', adjustable='box')


    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(3,3,1, projection = '3d')
    ax.scatter(*cloud_A[:,:3].T, color ='b', s= 10)
    ax.scatter(*cloud_B[:,:3].T, color = 'r', s= 10)
    process_axis(ax)

    ax = plt.subplot(3,3,4, projection = '3d')
    ax.scatter(*cloud_A[:,:3].T, c = dir_A, s= 10, cmap = 'tab10')
    ax.scatter(*cloud_B[:,:3].T, c = dir_B, s= 10, cmap = 'tab10')
    process_axis(ax)

    ax = plt.subplot(3,3,7, projection = '3d')
    ax.scatter(*cloud_A[:,:3].T, c = pos_A[:,0], s= 10, cmap = 'viridis')
    ax.scatter(*cloud_B[:,:3].T, c = pos_B[:,0], s= 10, cmap = 'magma')
    process_axis(ax)

    ax = plt.subplot(3,3,2, projection = '3d')
    ax.scatter(*cent_A.T, color ='b', s= 30)
    ax.scatter(*plane_cent_A.T, color ='cyan', s= 30)

    ax.scatter(*cent_B.T, color = 'r', s= 30)
    ax.scatter(*plane_cent_B.T, color = 'orange', s= 30)
    ax.scatter(*aligned_plane_cent_B.T, color = 'khaki', s= 30)
    process_axis(ax)



    ax = plt.subplot(3,3,5, projection = '3d')
    ax.scatter(*plane_cent_A.T, c = cent_pos[:,0], s= 30, cmap = 'viridis')
    ax.scatter(*aligned_plane_cent_B[:,:3].T, c = cent_pos[:,0], s= 30, cmap = 'magma')
    for idx in range(cent_pos.shape[0]):
        ax.plot([plane_cent_A[idx,0], aligned_plane_cent_B[idx,0]], 
                [plane_cent_A[idx,1], aligned_plane_cent_B[idx,1]], 
                [plane_cent_A[idx,2], aligned_plane_cent_B[idx,2]], 
                color='gray', linewidth=0.5)
    process_axis(ax)


    ax = plt.subplot(3,3,8, projection = '3d')
    ax.scatter(*cent_A.T, c = cent_pos[:,0], s= 30, cmap = 'viridis')
    ax.scatter(*rotated_aligned_cent_rot.T, c = cent_pos[:,0], s= 30, cmap = 'magma')
    for idx in range(cent_pos.shape[0]):
        ax.plot([plane_cent_A[idx,0], rotated_aligned_cent_rot[idx,0]], 
                [plane_cent_A[idx,1], rotated_aligned_cent_rot[idx,1]], 
                [plane_cent_A[idx,2], rotated_aligned_cent_rot[idx,2]], 
                color='gray', linewidth=0.5)
    process_axis(ax)

    ax = plt.subplot(1,3,3)
    ax.plot(error, angles*180/np.pi)
    ax.plot([np.min(error),np.max(error)], [rotation_angle]*2, '--r')
    ax.set_yticks([-180, -90, 0 , 90, 180])
    ax.set_xlabel('Error')
    ax.set_ylabel('Angle')
    plt.tight_layout()

    return fig


save_dir = os.path.join(base_dir, 'rotation')
save_dir_fig = os.path.join(save_dir, 'figures')
for mouse in mice_list:
    print(f"Working on mouse {mouse}:")
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(base_dir, 'processed_data' , mouse)
    animal = load_pickle(file_path,file_name)
    fnames = list(animal.keys())
    fname_pre = [fname for fname in fnames if 'ltpre' in fname][0]
    fname_post = [fname for fname in fnames if 'ltpost' in fname][0]

    animal_pre= copy.deepcopy(animal[fname_pre])
    animal_post= copy.deepcopy(animal[fname_post])

    rotation_dict = dict()

    for emb_name in ['pca','isomap','umap']:

        emb_pre = get_signal(animal_pre, emb_name)[:,:3]
        pos_pre = get_signal(animal_pre, 'position')
        dir_pre =get_signal(animal_pre, 'mov_direction')

        emb_post = get_signal(animal_post, emb_name)[:,:3]
        pos_post = get_signal(animal_post, 'position')
        dir_post =get_signal(animal_post, 'mov_direction')



        D_pre = pairwise_distances(emb_pre)
        noise_pre = filter_noisy_outliers(emb_pre,D_pre)
        max_dist = np.nanmax(D_pre)
        emb_pre = emb_pre[~noise_pre,:]
        pos_pre = pos_pre[~noise_pre,:]
        dir_pre = dir_pre[~noise_pre]

        D_post = pairwise_distances(emb_post)
        noise_post = filter_noisy_outliers(emb_post,D_post)
        max_dist = np.nanmax(D_post)
        emb_post = emb_post[~noise_post,:]
        pos_post = pos_post[~noise_post,:]
        dir_post = dir_post[~noise_post]

        #compute centroids
        cent_pre, cent_post, cent_pos, cent_dir = get_centroids(emb_pre, emb_post, pos_pre[:,0], pos_post[:,0], 
                                                        dir_pre, dir_post, num_centroids=40) 

        #project into planes
        norm_vec_pre, cloud_center_pre = parametrize_plane(emb_pre)
        plane_emb_pre = project_onto_plane(emb_pre, norm_vec_pre, cloud_center_pre)

        norm_vec_post, cloud_center_post = parametrize_plane(emb_post)
        plane_emb_post = project_onto_plane(emb_post, norm_vec_post, cloud_center_post)

        plane_cent_pre, plane_cent_post, plane_cent_pos, plane_cent_dir = get_centroids(plane_emb_pre, plane_emb_post, 
                                                                                            pos_pre[:,0], pos_post[:,0], 
                                                                                            dir_pre, dir_post, num_centroids=40) 
        #align them
        align_angle, align_mat = align_vectors(norm_vec_pre, cloud_center_pre, norm_vec_post, cloud_center_post)

        aligned_emb_post =  apply_rotation_to_cloud(emb_post, align_mat, cloud_center_post)
        aligned_plane_emb_post =  apply_rotation_to_cloud(plane_emb_post, align_mat, cloud_center_post)

        aligned_cent_post =  apply_rotation_to_cloud(cent_post, align_mat, cloud_center_post)
        aligned_plane_cent_post =  apply_rotation_to_cloud(plane_cent_post, align_mat, cloud_center_post)

        #compute angle of rotation
        angles = np.linspace(-np.pi,np.pi,200)
        error = find_rotation(plane_cent_pre, plane_cent_post, -norm_vec_pre)
        norm_error = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
        signed_postation_angle = angles[np.argmin(norm_error)]*180/np.pi
        rotation_angle = np.abs(signed_postation_angle)
        print(f"\t{mouse} {emb_name}: {signed_postation_angle:2f} degrees")

        rotated_aligned_cent_post = rotate_cloud_around_axis(aligned_cent_post, (np.pi/180)*signed_postation_angle,norm_vec_pre)
        rotated_aligned_plane_cent_post = rotate_cloud_around_axis(aligned_plane_cent_post, (np.pi/180)*signed_postation_angle,norm_vec_pre)
        rotated_aligned_emb_post = rotate_cloud_around_axis(aligned_emb_post, (np.pi/180)*signed_postation_angle,norm_vec_pre)
        rotated_aligned_plane_emb_post = rotate_cloud_around_axis(aligned_plane_emb_post, (np.pi/180)*signed_postation_angle,norm_vec_pre)

        rotated_cent_post = rotate_cloud_around_axis(cent_post, (np.pi/180)*signed_postation_angle,norm_vec_pre)

        fig = plot_rotation(emb_pre, emb_post, pos_pre, pos_post, dir_pre, dir_post, 
                    cent_pre, cent_post, cent_pos, plane_cent_pre, plane_cent_post, 
                    aligned_plane_cent_post, rotated_aligned_plane_cent_post, angles, error, signed_postation_angle)
        plt.suptitle(f"{mouse} {emb_name}")
        plt.savefig(os.path.join(save_dir_fig,f'{mouse}_{emb_name}_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(save_dir_fig,f'{mouse}_{emb_name}_rotation_plot.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)

        rotation_dict[emb_name] = {
            #initial data
            'emb_pre': emb_pre,
            'pos_pre': pos_pre,
            'dir_pre': dir_pre,
            'noise_pre': noise_pre,

            'emb_post': emb_post,
            'pos_post': pos_post,
            'dir_post': dir_post,
            'noise_post': noise_post,
            #centroids
            'cent_pre': cent_pre,
            'cent_post': cent_post,
            'cent_pos': cent_pos,
            'cent_dir': cent_dir,

            #project into plane
            'norm_vec_pre': norm_vec_pre,
            'cloud_center_pre': cloud_center_pre,
            'plane_emb_pre': plane_emb_pre,

            'norm_vec_post': norm_vec_post,
            'cloud_center_post': cloud_center_post,
            'plane_emb_post': plane_emb_post,

            #plane centroids
            'plane_cent_pre': plane_cent_pre,
            'plane_cent_post': plane_cent_post,
            'plane_cent_pos': plane_cent_pos,
            'plane_cent_dir': plane_cent_dir,

            #align planes
            'align_angle': align_angle,
            'align_mat': align_mat,

            'aligned_emb_post': aligned_emb_post,
            'aligned_plane_emb_post': aligned_plane_emb_post,
            'aligned_cent_post': aligned_cent_post,
            'aligned_plane_cent_post': aligned_plane_cent_post,

            #compute angle of rotation
            'angles': angles,
            'error': error,
            'norm_error': norm_error,
            'signed_postation_angle': signed_postation_angle,
            'rotation_angle': rotation_angle,

            #rotate post session
            'rotated_cent_post': rotated_cent_post,
            'rotated_aligned_cent_post': rotated_aligned_cent_post,
            'rotated_aligned_plane_cent_post': rotated_aligned_plane_cent_post,
            'rotated_aligned_emb_post': rotated_aligned_emb_post,
            'rotated_aligned_plane_emb_post': rotated_aligned_plane_emb_post,
        }

        with open(os.path.join(save_dir, f"{mouse}_rotation_dict.pkl"), "wb") as file:
            pickle.dump(rotation_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

#__________________________________________________________________________
#|                                                                        |#
#|                     MEASURE REMMAPING DISTANCE                         |#
#|________________________________________________________________________|#

from sklearn.decomposition import PCA

def fit_ellipse(cloud_A, norm_vector):

    def find_rotation_align_vectors(a,b):
        v = np.cross(a,b)
        s = np.linalg.norm(v)
        c = np.dot(a,b)

        sscp = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]], [-v[1],v[0],0]])
        sscp2 = np.matmul(sscp,sscp)
        R = np.eye(3)+sscp+(sscp2*(1-c)/s**2)
        return R

    rot_2D = find_rotation_align_vectors([0,0,1], norm_vector)
    cloud_A_2D = (apply_rotation_to_cloud(cloud_A,rot_2D, cloud_A.mean(axis=0)) - cloud_A.mean(axis=0))[:,:2]
    X = cloud_A_2D[:,0:1]
    Y = cloud_A_2D[:,1:]

    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

    long_axis = ((x[0]+x[2])/2) + np.sqrt(((x[0]-x[2])/2)**2 + x[1]**2)
    short_axis = ((x[0]+x[2])/2) - np.sqrt(((x[0]-x[2])/2)**2 + x[1]**2)

    x_coord = np.linspace(2*np.min(cloud_A_2D[:,0]),2*np.max(cloud_A_2D[:,0]),1000)
    y_coord = np.linspace(2*np.min(cloud_A_2D[:,1]),2*np.max(cloud_A_2D[:,1]),1000)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord

    a = np.where(np.abs(Z_coord-1)<10e-4)
    fit_ellipse_points = np.zeros((a[0].shape[0],2))
    for point in range(a[0].shape[0]):
        x_coord = X_coord[a[0][point], a[1][point]]
        y_coord = Y_coord[a[0][point], a[1][point]]
        fit_ellipse_points[point,:] = [x_coord, y_coord]

    long_axis = np.max(pairwise_distances(fit_ellipse_points))/2
    short_axis = np.mean(pairwise_distances(fit_ellipse_points))/2


    R_ellipse = find_rotation_align_vectors(norm_vector, [0,0,1])

    fit_ellipse_points_3D = np.hstack((fit_ellipse_points, np.zeros((fit_ellipse_points.shape[0],1))))
    fit_ellipse_points_3D = apply_rotation_to_cloud(fit_ellipse_points_3D,R_ellipse, fit_ellipse_points_3D.mean(axis=0)) + cloud_A.mean(axis=0)


    return x, long_axis, short_axis, fit_ellipse_points,fit_ellipse_points_3D

def plot_distance(cent_A, cent_B, cent_pos, cent_dir, plane_cent_A, plane_cent_B, plane_cent_pos, plane_cent_dir, ellipse_A, ellipse_B):

    def process_axis(ax):
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_aspect('equal', adjustable='box')

    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(2,3,1, projection = '3d')
    ax.scatter(*cent_A[:,:3].T, color ='b', s= 20)
    ax.scatter(*cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*cent_B[:,:3].T, color = 'r', s= 20)
    ax.scatter(*cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([cent_A.mean(axis=0)[0], cent_B.mean(axis=0)[0]],
        [cent_A.mean(axis=0)[1], cent_B.mean(axis=0)[1]],
        [cent_A.mean(axis=0)[2], cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    process_axis(ax)

    ax = plt.subplot(2,3,2, projection = '3d')
    ax.scatter(*cent_A[:,:3].T, c = cent_dir, s= 20, cmap = 'tab10')
    ax.scatter(*cent_B[:,:3].T, c = cent_dir, s= 20, cmap = 'tab10')
    ax.scatter(*cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([cent_A.mean(axis=0)[0], cent_B.mean(axis=0)[0]],
        [cent_A.mean(axis=0)[1], cent_B.mean(axis=0)[1]],
        [cent_A.mean(axis=0)[2], cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    process_axis(ax)

    ax = plt.subplot(2,3,3, projection = '3d')
    ax.scatter(*cent_A[:,:3].T, c = cent_pos[:,0], s= 20, cmap = 'viridis')
    ax.scatter(*cent_B[:,:3].T, c = cent_pos[:,0], s= 20, cmap = 'magma')
    ax.scatter(*cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([cent_A.mean(axis=0)[0], cent_B.mean(axis=0)[0]],
        [cent_A.mean(axis=0)[1], cent_B.mean(axis=0)[1]],
        [cent_A.mean(axis=0)[2], cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    process_axis(ax)

    ax = plt.subplot(2,3,4, projection = '3d')
    ax.scatter(*plane_cent_A[:,:3].T, color ='b', s= 20)
    ax.scatter(*plane_cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*plane_cent_B[:,:3].T, color = 'r', s= 20)
    ax.scatter(*plane_cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([plane_cent_A.mean(axis=0)[0], plane_cent_B.mean(axis=0)[0]],
        [plane_cent_A.mean(axis=0)[1], plane_cent_B.mean(axis=0)[1]],
        [plane_cent_A.mean(axis=0)[2], plane_cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    ax.scatter(*ellipse_A[:,:3].T, color ='cyan', s= 5,alpha=0.3)
    ax.scatter(*ellipse_B[:,:3].T, color ='orange', s= 5,alpha=0.3)
    process_axis(ax)

    ax = plt.subplot(2,3,5, projection = '3d')
    ax.scatter(*plane_cent_A[:,:3].T, c = plane_cent_dir, s= 20, cmap = 'tab10')
    ax.scatter(*plane_cent_B[:,:3].T, c = plane_cent_dir, s= 20, cmap = 'tab10')
    ax.scatter(*plane_cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*plane_cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([plane_cent_A.mean(axis=0)[0], plane_cent_B.mean(axis=0)[0]],
        [plane_cent_A.mean(axis=0)[1], plane_cent_B.mean(axis=0)[1]],
        [plane_cent_A.mean(axis=0)[2], plane_cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    ax.scatter(*ellipse_A[:,:3].T, color ='cyan', s= 5,alpha=0.3)
    ax.scatter(*ellipse_B[:,:3].T, color ='orange', s= 5,alpha=0.3)
    process_axis(ax)

    ax = plt.subplot(2,3,6, projection = '3d')
    ax.scatter(*plane_cent_A[:,:3].T, c = plane_cent_pos[:,0], s= 20, cmap = 'viridis')
    ax.scatter(*plane_cent_B[:,:3].T, c = plane_cent_pos[:,0], s= 20, cmap = 'magma')
    ax.scatter(*plane_cent_A.mean(axis=0).T, color ='b', s= 40)
    ax.scatter(*plane_cent_B.mean(axis=0).T, color ='r', s= 40)
    ax.plot([plane_cent_A.mean(axis=0)[0], plane_cent_B.mean(axis=0)[0]],
        [plane_cent_A.mean(axis=0)[1], plane_cent_B.mean(axis=0)[1]],
        [plane_cent_A.mean(axis=0)[2], plane_cent_B.mean(axis=0)[2]],
        color='k', linewidth=2)
    ax.scatter(*ellipse_A[:,:3].T, color ='cyan', s= 5,alpha=0.3)
    ax.scatter(*ellipse_B[:,:3].T, color ='orange', s= 5,alpha=0.3)
    process_axis(ax)
    plt.tight_layout()
    return fig





data_dir = os.path.join(base_dir, 'rotation')
save_dir = os.path.join(base_dir, 'distance')
save_dir_fig = os.path.join(save_dir, 'figures')


for mouse in mice_list:

    print(f"Working on mouse {mouse}:")
    rotation_dict = load_pickle(data_dir, mouse+'_rotation_dict.pkl')
    distance_dict = dict()


    for emb_name in ['umap', 'isomap', 'pca']:

        cent_pre = rotation_dict[emb_name]['cent_pre']
        cent_post = rotation_dict[emb_name]['cent_post']
        cent_pos = rotation_dict[emb_name]['cent_pos']
        cent_dir = rotation_dict[emb_name]['cent_dir']

        inter_dist = np.linalg.norm(cent_pre.mean(axis=0)-cent_post.mean(axis=0))
        intra_dist_pre = np.percentile(pairwise_distances(cent_pre),95)/2
        intra_dist_post = np.percentile(pairwise_distances(cent_post),95)/2
        remap_dist = inter_dist/np.mean((intra_dist_pre, intra_dist_post))

        plane_cent_pre = rotation_dict[emb_name]['plane_cent_pre']
        plane_cent_post = rotation_dict[emb_name]['plane_cent_post']
        norm_vector_pre = rotation_dict[emb_name]['norm_vec_pre']
        plane_cent_pos = rotation_dict[emb_name]['plane_cent_pos']
        plane_cent_dir = rotation_dict[emb_name]['plane_cent_dir']
        norm_vector_post = rotation_dict[emb_name]['norm_vec_post']


        plane_inter_dist = np.linalg.norm(plane_cent_pre.mean(axis=0)-plane_cent_post.mean(axis=0))
        ellipse_pre_params, ellipse_pre_long_axis, ellipse_pre_short_axis, ellipse_pre_fit, ellipse_pre_fit_3D = fit_ellipse(plane_cent_pre, norm_vector_pre)
        ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(ellipse_pre_long_axis+ellipse_pre_short_axis)**2)

        ellipse_post_params, ellipse_post_long_axis, ellipse_post_short_axis, ellipse_post_fit, ellipse_post_fit_3D = fit_ellipse(plane_cent_post, norm_vector_post)
        ellipse_post_perimeter = 2*np.pi*np.sqrt(0.5*(ellipse_post_long_axis+ellipse_post_short_axis)**2)

        plane_remap_dist = plane_inter_dist/np.mean((ellipse_pre_perimeter, ellipse_post_perimeter))

        print(f"\t{mouse} {emb_name}: {remap_dist:.2f} remap dist | {plane_remap_dist:.2f} remap dist plane")

        fig = plot_distance(cent_pre,cent_post,cent_pos,cent_dir,
                plane_cent_pre,plane_cent_post, plane_cent_pos, plane_cent_dir,
                ellipse_pre_fit_3D, ellipse_post_fit_3D)
        plt.suptitle(f"{mouse} {emb_name}")
        plt.savefig(os.path.join(save_dir_fig,f'{mouse}_{emb_name}_distance_plot.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(save_dir_fig,f'{mouse}_{emb_name}_distance_plot.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)

        distance_dict[emb_name] = {

            #cent
            'cent_pre': cent_pre,
            'cent_post': cent_post,
            'cent_pos': cent_pos,
            'noise_pre': cent_dir,
            #distance og
            'inter_dist': inter_dist,
            'intra_dist_pre': intra_dist_pre,
            'intra_dist_post': intra_dist_post,
            'remap_dist': remap_dist,

            #plane
            'plane_cent_pre': cent_pre,
            'norm_vector_pre': norm_vector_pre,
            'plane_cent_post': plane_cent_post,
            'norm_vector_post': norm_vector_post,
            'plane_cent_pos': plane_cent_pos,
            'plane_cent_dir': plane_cent_dir,

            #ellipse
            'ellipse_pre_params': ellipse_pre_params,
            'ellipse_pre_long_axis': ellipse_pre_long_axis,
            'ellipse_pre_short_axis': ellipse_pre_short_axis,
            'ellipse_pre_fit': ellipse_pre_fit,
            'ellipse_pre_fit_3D': ellipse_pre_fit_3D,

            'ellipse_post_params': ellipse_post_params,
            'ellipse_post_long_axis': ellipse_post_long_axis,
            'ellipse_post_short_axis': ellipse_post_short_axis,
            'ellipse_post_fit': ellipse_post_fit,
            'ellipse_post_fit_3D': ellipse_post_fit_3D,

            #distance ellipse
            'plane_inter_dist': plane_inter_dist,
            'ellipse_pre_perimeter': ellipse_pre_perimeter,
            'ellipse_post_perimeter': ellipse_post_perimeter,
            'plane_remap_dist': plane_remap_dist,
        }

    with open(os.path.join(save_dir,f'{mouse}_distance_dict.pkl'), 'wb') as f:
        pickle.dump(distance_dict, f)


#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT ROTATION                               |#
#|________________________________________________________________________|#

data_dir_control = os.path.join(base_dir, 'rotation')
data_dir_rotation = '/home/julio/Documents/DeepSup_project/DeepSup/rotation/'

#PLOT LINES
for embName in ['pca', 'isomap', 'umap']:
    rot_error_control = np.zeros((200, len(mice_list)))
    rot_error_rotation = np.zeros((200, len(mice_list)))

    for idx, mouse in enumerate(mice_list):
        rotation_dict_control = load_pickle(data_dir_control, f'{mouse}_rotation_dict.pkl')
        rotation_dict_rotation = load_pickle(data_dir_rotation, f'{mouse}_rotation_dict.pkl')
        rot_error_control[:,idx] = rotation_dict_control[embName]['norm_error']
        rot_error_rotation[:,idx] = rotation_dict_rotation[embName]['norm_error']

    angle_degrees = rotation_dict_control[embName]['angles']

    plt.figure()
    ax = plt.subplot(111)
    m = np.mean(rot_error_control,axis=1)
    sd = np.std(rot_error_control,axis=1)
    ax.plot(angle_degrees, m, color = '#32E653',label = 'Control')
    ax.fill_between(angle_degrees, m-sd, m+sd, color = '#32E653', alpha = 0.3)
    m = np.mean(rot_error_rotation,axis=1)
    sd = np.std(rot_error_rotation,axis=1)
    ax.plot(angle_degrees, m, color = '#E632C5', label = 'Rotation')
    ax.fill_between(angle_degrees, m-sd, m+sd, color = '#E632C5', alpha = 0.3)
    ax.set_xlabel('Angle of rotation (º)')
    ax.set_ylabel('Aligment Error')
    ax.set_title(embName)
    ax.legend()
    plt.savefig(os.path.join(data_dir_control,f'control_{embName}_rotation_error.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(data_dir_control,f'control_{embName}_rotation_error.svg'), dpi = 400,bbox_inches="tight",transparent=True)

#PLOT BOXPLOTS
rot_angle_list = list()
condition_list = list()
emb_list = list()
for mouse in mice_list:
    rotation_dict_control = load_pickle(data_dir_control, f'{mouse}_rotation_dict.pkl')
    rotation_dict_rotation = load_pickle(data_dir_rotation, f'{mouse}_rotation_dict.pkl')

    for embName in['pca', 'isomap', 'umap']:
        rot_angle_list.append(rotation_dict_control[embName]['rotation_angle'])
        emb_list.append(embName)
        condition_list.append('control')

        rot_angle_list.append(rotation_dict_rotation[embName]['rotation_angle'])
        emb_list.append(embName)
        condition_list.append('rotation')

angle_pd = pd.DataFrame(data={'angle': rot_angle_list,
                            'emb': emb_list,
                            'condition': condition_list})

palette= ["#32e653", "#E632C5"]
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='emb', y='angle', hue='condition', data=angle_pd, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='emb', y='angle', hue = 'condition', data=angle_pd, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylabel('Angle Rotation')
plt.tight_layout()

plt.savefig(os.path.join(data_dir_control,f'control_rotation_boxplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(data_dir_control,f'control_rotation_boxplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)


#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT DISTANCE                               |#
#|________________________________________________________________________|#


data_dir_control = os.path.join(base_dir, 'distance')
data_dir_rotation = '/home/julio/Documents/DeepSup_project/DeepSup/distance/'


#PLOT BOXPLOTS
distance_list = list()
condition_list = list()
emb_list = list()
mouse_list = list()
for mouse in mice_list:
    distance_dict_control = load_pickle(data_dir_control, f'{mouse}_distance_dict.pkl')
    distance_dict_rotation = load_pickle(data_dir_rotation, f'{mouse}_distance_dict.pkl')

    for embName in['pca', 'isomap', 'umap']:
        distance_list.append(distance_dict_control[embName]['plane_remap_dist'])
        emb_list.append(embName)
        condition_list.append('control')

        distance_list.append(distance_dict_rotation[embName]['plane_remap_dist'])
        emb_list.append(embName)
        condition_list.append('rotation')

        mouse_list += [mouse]*2

distance_pd = pd.DataFrame(data={'plane_distance': distance_list,
                            'emb': emb_list,
                            'condition': condition_list,
                            'mouse': mouse_list})

palette= ["#1be4aaff", "#e41b55ff"]
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='emb', y='plane_distance', hue='condition', data=distance_pd, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='emb', y='plane_distance', hue = 'condition', data=distance_pd, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
plt.tight_layout()

plt.savefig(os.path.join(data_dir_control,f'control_plane_distance_barplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(data_dir_control,f'control_plane_distance_barplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)




#PLOT BOXPLOTS
distance_list = list()
condition_list = list()
emb_list = list()
mouse_list = list()
for mouse in mice_list:
    distance_dict_control = load_pickle(data_dir_control, f'{mouse}_distance_dict.pkl')
    distance_dict_rotation = load_pickle(data_dir_rotation, f'{mouse}_distance_dict.pkl')

    for embName in['pca', 'isomap', 'umap']:
        distance_list.append(distance_dict_control[embName]['remap_dist'])
        emb_list.append(embName)
        condition_list.append('control')

        distance_list.append(distance_dict_rotation[embName]['remap_dist'])
        emb_list.append(embName)
        condition_list.append('rotation')

        mouse_list += [mouse]*2

angle_pd = pd.DataFrame(data={'remap distance': distance_list,
                            'emb': emb_list,
                            'condition': condition_list,
                            'mouse': mouse_list})

palette= ["#1be4aaff", "#e41b55ff"]
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='emb', y='remap distance', hue='condition', data=angle_pd, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='emb', y='remap distance', hue = 'condition', data=angle_pd, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
plt.tight_layout()

plt.savefig(os.path.join(data_dir_control,f'control_distance_barplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(data_dir_control,f'control_distance_barplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)

#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT EMBEDDING                              |#
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

dataDir = '/home/julio/Documents/SP_project/Fig3_rotControl/processed_data/'
saveDir = '/home/julio/Documents/SP_project/Fig3_rotControl/rotation/emb_example/'

view_init_dict = {
    'GC2': [25,-175],
    'ChZ4': [20, 135]
}

for mouse in mice_list:
    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    animal = load_pickle(filePath,file_name)
    fnames = list(animal.keys())

    fname_pre = [fname for fname in fnames if 'pre' in fname][0]
    fnameRot = [fname for fname in fnames if 'post' in fname][0]

    animal_pre= copy.deepcopy(animal[fname_pre])
    animalRot= copy.deepcopy(animal[fnameRot])

    pos_pre = copy.deepcopy(np.concatenate(animal_pre['pos'].values, axis=0))
    dir_mat_pre = copy.deepcopy(np.concatenate(animal_pre['dir_mat'].values, axis=0))
    emb_pre = copy.deepcopy(np.concatenate(animal_pre['umap'].values, axis=0))

    pos_post = copy.deepcopy(np.concatenate(animalRot['pos'].values, axis=0))
    dirMatRot = copy.deepcopy(np.concatenate(animalRot['dir_mat'].values, axis=0))
    embRot = copy.deepcopy(np.concatenate(animalRot['umap'].values, axis=0))

    D_pre= pairwise_distances(emb_pre)
    noiseIdx = filter_noisy_outliers(emb_pre,D_pre)
    emb_pre = emb_pre[~noiseIdx,:]
    pos_pre = pos_pre[~noiseIdx,:]
    dir_mat_pre = dir_mat_pre[~noiseIdx]

    D_rot= pairwise_distances(embRot)
    noiseIdx = filter_noisy_outliers(embRot,D_rot)
    embRot = embRot[~noiseIdx,:]
    pos_post = pos_post[~noiseIdx,:]
    dirMatRot = dirMatRot[~noiseIdx]

    dirColorPre = np.zeros((dir_mat_pre.shape[0],3))
    for point in range(dir_mat_pre.shape[0]):
        if dir_mat_pre[point]==0:
            dirColorPre[point] = [14/255,14/255,143/255]
        elif dir_mat_pre[point]==1:
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
    ax.scatter(*emb_pre[:,:3].T, color = '#00fe42ff',s = 20)
    ax.scatter(*embRot[:,:3].T, color = 'b',s = 20)
    personalize_ax(ax,view_init_values)

    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, color = dirColorPre, s = 20)
    ax.scatter(*embRot[:,:3].T, color = dirColorRot, s = 20)
    personalize_ax(ax,view_init_values)

    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*emb_pre[:,:3].T, c = pos_pre[:,0], cmap = 'inferno',s = 20)
    ax.scatter(*embRot[:,:3].T, c = pos_post[:,0], cmap = 'inferno',s = 20)
    personalize_ax(ax,view_init_values)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,f'{mouse}_control_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,f'{mouse}_control_emb.png'), dpi = 400,bbox_inches="tight")