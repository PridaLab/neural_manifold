import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import os, copy, pickle
import neural_manifold.general_utils as gu
from datetime import datetime

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nnDist < np.percentile(nnDist, 20)
    return noiseIdx

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))

mice_list = ['ThyCalbRCaMP2','ThyCalbRCaMP8','ThyCalbRCaMP9']
base_dir = '/home/julio/Documents/DeepSup_project/DualColor/ThyCalbRCaMP/'

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

vel_th = 4
sigma = 3
sig_up = 2
sig_down = 6
nn_val = 60
dim = 3


columns_to_drop = ['date','denoised_traces', '*spikes']
columns_to_rename = {'Fs':'sf','pos':'position', 'vel':'speed', 'index_mat': 'trial_idx_mat'}

for mouse in mice_list:
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
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]
    animal_pre= copy.deepcopy(animal[fname_pre])
    animal_rot= copy.deepcopy(animal[fname_rot])

    #__________________________________________________________________________
    #|                                                                        |#
    #|               CHANGE COLUMN NAMES AND ADD NEW ONES                     |#
    #|________________________________________________________________________|#

    for column in columns_to_drop:
        if column in animal_pre.columns: animal_pre.drop(columns=[column], inplace=True)
        if column in animal_rot.columns: animal_rot.drop(columns=[column], inplace=True)

    for old, new in columns_to_rename.items():
        if old in animal_pre.columns: animal_pre.rename(columns={old:new}, inplace=True)
        if old in animal_rot.columns: animal_rot.rename(columns={old:new}, inplace=True)

    gu.add_trial_id_mat_field(animal_pre)
    gu.add_trial_id_mat_field(animal_rot)

    gu.add_mov_direction_mat_field(animal_pre)
    gu.add_mov_direction_mat_field(animal_rot)

    gu.add_trial_type_mat_field(animal_pre)
    gu.add_trial_type_mat_field(animal_rot)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                          KEEP ONLY MOVING                              |#
    #|________________________________________________________________________|#

    # animal_pre = gu.select_trials(animal_pre,"dir == ['L','R','N']")
    # animal_rot = gu.select_trials(animal_rot,"dir == ['L','R','N']")
    if vel_th>0:
        animal_pre, animal_pre_still = gu.keep_only_moving(animal_pre, vel_th)
        animal_rot, animal_rot_still = gu.keep_only_moving(animal_rot, vel_th)
    else:
        animal_pre_still = pd.DataFrame()
        animal_rot_still = pd.DataFrame()

    animal_pre, _ = gu.keep_only_moving(animal_pre, vel_th)
    animal_rot, _ = gu.keep_only_moving(animal_rot, vel_th)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                          PREPROCESS TRACES                             |#
    #|________________________________________________________________________|#

    for color in ['green', 'red']:
        animal_pre, animal_rot = preprocess_traces(animal_pre, animal_rot, color+'_raw_traces', color+'_clean_traces', sigma=sigma, sig_up = sig_up, sig_down = sig_down)
        animal_pre['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
        animal_rot['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
        if vel_th>0:
            animal_pre_still, animal_rot_still = preprocess_traces(animal_pre_still, animal_rot_still, color+'_raw_traces', color+'_clean_traces', sigma=sigma, sig_up = sig_up, sig_down = sig_down)
            animal_pre_still['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
            animal_rot_still['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}

    #__________________________________________________________________________
    #|                                                                        |#
    #|                           REMOVE DUPLICATED CELLS                      |#
    #|________________________________________________________________________|#

    signal_green_pre = get_signal(animal_pre, 'green_clean_traces')
    signal_red_pre = get_signal(animal_pre, 'red_clean_traces')

    signal_green_rot = get_signal(animal_rot, 'green_clean_traces')
    signal_red_rot = get_signal(animal_rot, 'red_clean_traces')

    corr_channels = np.zeros((signal_green_pre.shape[1], signal_red_pre.shape[1]))
    for neu_g in range(signal_green_pre.shape[1]):
        for neu_r in range(signal_red_pre.shape[1]):
            corr_channels[neu_g, neu_r] = np.corrcoef(signal_green_pre[:, neu_g], signal_red_pre[:, neu_r])[0,1]

    compromised_cells = np.where(corr_channels>0.8)
    print(f"\tChecking duplicated cells: {len(compromised_cells)} cells duplicated out of",
        f"{signal_green_pre.shape[1]} green and {signal_red_pre.shape[1]} red cells")

    animal_pre['green_clean_traces'] = [np.delete(animal_pre['green_clean_traces'][idx],compromised_cells[0],1)
                                                            for idx in range(animal_pre.shape[0])]
    animal_pre['green_cell_idx'] = [np.delete(animal_pre['green_cell_idx'][idx],compromised_cells[0])
                                                            for idx in range(animal_pre.shape[0])]

    animal_pre['red_clean_traces'] = [np.delete(animal_pre['red_clean_traces'][idx],compromised_cells[1],1)
                                                            for idx in range(animal_pre.shape[0])]
    animal_pre['red_cell_idx'] = [np.delete(animal_pre['red_cell_idx'][idx],compromised_cells[1])
                                                            for idx in range(animal_pre.shape[0])]

    animal_rot['green_clean_traces'] = [np.delete(animal_rot['green_clean_traces'][idx],compromised_cells[0],1)
                                                            for idx in range(animal_rot.shape[0])]
    animal_rot['green_cell_idx'] = [np.delete(animal_rot['green_cell_idx'][idx],compromised_cells[0])
                                                            for idx in range(animal_rot.shape[0])]

    animal_rot['red_clean_traces'] = [np.delete(animal_rot['red_clean_traces'][idx],compromised_cells[1],1)
                                                            for idx in range(animal_rot.shape[0])]
    animal_rot['red_cell_idx'] = [np.delete(animal_rot['red_cell_idx'][idx],compromised_cells[1])
                                                            for idx in range(animal_rot.shape[0])]

    #__________________________________________________________________________
    #|                                                                        |#
    #|                              COMPUTE UMAP                              |#
    #|________________________________________________________________________|#

    signal_green_pre = get_signal(animal_pre, 'green_clean_traces')
    signal_red_pre = get_signal(animal_pre, 'red_clean_traces')
    pos_pre = get_signal(animal_pre, 'position')
    dir_pre =get_signal(animal_pre, 'mov_direction')

    signal_green_rot = get_signal(animal_rot, 'green_clean_traces')
    signal_red_rot = get_signal(animal_rot, 'red_clean_traces')
    pos_rot = get_signal(animal_rot, 'position')
    dir_rot =get_signal(animal_rot, 'mov_direction')

    #all data green
    index = np.vstack((np.zeros((signal_green_pre.shape[0],1)),np.ones((signal_green_rot.shape[0],1))))
    concat_signal_green = np.vstack((signal_green_pre, signal_green_rot))
    model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
    model.fit(concat_signal_green)
    emb_green_concat = model.transform(concat_signal_green)
    emb_green_pre = emb_green_concat[index[:,0]==0,:]
    emb_green_rot = emb_green_concat[index[:,0]==1,:]

    #all data red
    index = np.vstack((np.zeros((signal_red_pre.shape[0],1)),np.ones((signal_red_rot.shape[0],1))))
    concat_signal_red = np.vstack((signal_red_pre, signal_red_rot))
    model = umap.UMAP(n_neighbors=nn_val, n_components =dim, min_dist=0.1)
    model.fit(concat_signal_red)
    emb_red_concat = model.transform(concat_signal_red)
    emb_red_pre = emb_red_concat[index[:,0]==0,:]
    emb_red_rot = emb_red_concat[index[:,0]==1,:]

    #all data both colors
    signal_both_pre = np.hstack((signal_green_pre, signal_red_pre))
    signal_both_rot = np.hstack((signal_green_rot, signal_red_rot))
    index = np.vstack((np.zeros((signal_both_pre.shape[0],1)),np.ones((signal_both_rot.shape[0],1))))
    concat_signal_both = np.vstack((signal_both_pre, signal_both_rot))
    model = umap.UMAP(n_neighbors=nn_val, n_components =dim, min_dist=0.1)
    model.fit(concat_signal_both)
    emb_both_concat = model.transform(concat_signal_both)
    emb_both_pre = emb_both_concat[index[:,0]==0,:]
    emb_both_rot = emb_both_concat[index[:,0]==1,:]

    #clean outliers
    D = pairwise_distances(emb_green_pre)
    noiseIdxGPre = filter_noisy_outliers(emb_green_pre,D=D)
    csignal_green_pre = signal_green_pre[~noiseIdxGPre,:]
    cemb_green_pre = emb_green_pre[~noiseIdxGPre,:]
    cpos_green_pre = pos_pre[~noiseIdxGPre,:]
    cdir_green_pre = dir_pre[~noiseIdxGPre]

    D = pairwise_distances(emb_red_pre)
    noiseIdxRPre = filter_noisy_outliers(emb_red_pre,D=D)
    csignal_red_pre = signal_red_pre[~noiseIdxRPre,:]
    cemb_red_pre = emb_red_pre[~noiseIdxRPre,:]
    cpos_red_pre = pos_pre[~noiseIdxRPre,:]
    cdir_red_pre = dir_pre[~noiseIdxRPre]

    D = pairwise_distances(emb_both_pre)
    noiseIdxBPre = filter_noisy_outliers(emb_both_pre,D=D)
    csignal_both_pre = signal_red_pre[~noiseIdxRPre,:]
    cemb_both_pre = emb_both_pre[~noiseIdxBPre,:]
    cpos_both_pre = pos_pre[~noiseIdxBPre,:]
    cdir_both_pre = dir_pre[~noiseIdxBPre]

    D = pairwise_distances(emb_green_rot)
    noiseIdxGRot = filter_noisy_outliers(emb_green_rot,D=D)
    csignal_green_rot = signal_green_rot[~noiseIdxGRot,:]
    cemb_green_rot = emb_green_rot[~noiseIdxGRot,:]
    cpos_green_rot = pos_rot[~noiseIdxGRot,:]
    cdir_green_rot = dir_rot[~noiseIdxGRot]

    D = pairwise_distances(emb_red_rot)
    noiseIdxRRot = filter_noisy_outliers(emb_red_rot,D=D)
    csignal_red_rot = signal_red_rot[~noiseIdxRRot,:]
    cemb_red_rot = emb_red_rot[~noiseIdxRRot,:]
    cpos_red_rot = pos_rot[~noiseIdxRRot,:]
    cdir_red_rot = dir_rot[~noiseIdxRRot]    

    D = pairwise_distances(emb_both_rot)
    noiseIdxBRot = filter_noisy_outliers(emb_both_rot,D=D)
    csignal_both_rot = signal_red_rot[~noiseIdxRRot,:]
    cemb_both_rot = emb_both_rot[~noiseIdxBRot,:]
    cpos_both_rot = pos_rot[~noiseIdxBRot,:]
    cdir_both_rot = dir_rot[~noiseIdxBRot]

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
    dir_color_green_pre = return_dir_color(cdir_green_pre)
    dir_color_red_pre = return_dir_color(cdir_red_pre)
    dir_color_green_rot = return_dir_color(cdir_green_rot)
    dir_color_red_rot = return_dir_color(cdir_red_rot)
    dir_color_both_pre = return_dir_color(cdir_both_pre)
    dir_color_both_rot = return_dir_color(cdir_both_rot)

    plt.figure()
    ax = plt.subplot(3,3,1, projection = '3d')
    ax.scatter(*cemb_green_pre[:,:3].T, color ='b', s=10)
    ax.scatter(*cemb_green_rot[:,:3].T, color = 'r', s=10)
    ax.set_title('Green')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(3,3,2, projection = '3d')
    ax.scatter(*cemb_green_pre[:,:3].T, c = cpos_green_pre[:,0], s=10, cmap = 'magma')
    ax.scatter(*cemb_green_rot[:,:3].T, c = cpos_green_rot[:,0], s=10, cmap = 'magma')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(3,3,3, projection = '3d')
    ax.scatter(*cemb_green_pre[:,:3].T, color=dir_color_green_pre, s=10)
    ax.scatter(*cemb_green_rot[:,:3].T, color=dir_color_green_rot, s=10)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(3,3,4, projection = '3d')
    ax.scatter(*cemb_red_pre[:,:3].T, color ='b', s=10)
    ax.scatter(*cemb_red_rot[:,:3].T, color = 'r', s=10)
    ax.set_title('Red')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(3,3,5, projection = '3d')
    ax.scatter(*cemb_red_pre[:,:3].T, c = cpos_red_pre[:,0], s=10, cmap = 'magma')
    ax.scatter(*cemb_red_rot[:,:3].T, c = cpos_red_rot[:,0], s=10, cmap = 'magma')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(3,3,6, projection = '3d')
    ax.scatter(*cemb_red_pre[:,:3].T, color=dir_color_red_pre, s=10)
    ax.scatter(*cemb_red_rot[:,:3].T, color=dir_color_red_rot, s=10)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(3,3,7, projection = '3d')
    ax.scatter(*cemb_both_pre[:,:3].T, color ='b', s=10)
    ax.scatter(*cemb_both_rot[:,:3].T, color = 'r', s=10)
    ax.set_title('Both')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(3,3,8, projection = '3d')
    ax.scatter(*cemb_both_pre[:,:3].T, c = cpos_both_pre[:,0], s=10, cmap = 'magma')
    ax.scatter(*cemb_both_rot[:,:3].T, c = cpos_both_rot[:,0], s=10, cmap = 'magma')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(3,3,9, projection = '3d')
    ax.scatter(*cemb_both_pre[:,:3].T, color=dir_color_both_pre, s=10)
    ax.scatter(*cemb_both_rot[:,:3].T, color=dir_color_both_rot, s=10)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.suptitle(f"{mouse} - {vel_th} | {nn_val}")

    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb.png'), dpi = 400,bbox_inches="tight")

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                SAVE DATA                               |#
    #|________________________________________________________________________|#

    trial_id_mat_p = np.concatenate(animal_pre['trial_id_mat'].values, axis=0)
    animal_pre['green_umap'] = [emb_green_pre[trial_id_mat_p[:,0]==animal_pre["trial_id"][idx] ,:] 
                                                    for idx in animal_pre.index]
    animal_pre['red_umap'] = [emb_red_pre[trial_id_mat_p[:,0]==animal_pre["trial_id"][idx] ,:] 
                                                    for idx in animal_pre.index]
    animal_pre['both_umap'] = [emb_both_pre[trial_id_mat_p[:,0]==animal_pre["trial_id"][idx] ,:] 
                                                    for idx in animal_pre.index]
    animal_pre['both_clean_traces'] = [signal_both_pre[trial_id_mat_p[:,0]==animal_pre["trial_id"][idx] ,:] 
                                                    for idx in animal_pre.index]


    trial_id_mat_r = np.concatenate(animal_rot['trial_id_mat'].values, axis=0)
    animal_rot['green_umap'] = [emb_green_rot[trial_id_mat_r[:,0]==animal_rot["trial_id"][idx] ,:] 
                                                    for idx in animal_rot.index]
    animal_rot['red_umap'] = [emb_red_rot[trial_id_mat_r[:,0]==animal_rot["trial_id"][idx] ,:] 
                                                    for idx in animal_rot.index]
    animal_rot['both_umap'] = [emb_both_rot[trial_id_mat_r[:,0]==animal_rot["trial_id"][idx] ,:] 
                                                    for idx in animal_rot.index]
    animal_rot['both_clean_traces'] = [signal_both_rot[trial_id_mat_r[:,0]==animal_rot["trial_id"][idx] ,:] 
                                                    for idx in animal_rot.index]

    #__________________________________________________________________________
    #|                                                                        |#
    #|                           FIX COLUMN NAMES                             |#
    #|________________________________________________________________________|#

    for column in animal_pre.columns:
        if 'green' in column: animal_pre.rename(columns={column:column.replace('green', 'deep')}, inplace=True)
        if 'red' in column: animal_pre.rename(columns={column:column.replace('red', 'sup')}, inplace=True)
        if 'both' in column: animal_pre.rename(columns={column:column.replace('both', 'all')}, inplace=True)

    for column in animal_rot.columns:
        if 'green' in column: animal_rot.rename(columns={column:column.replace('green', 'deep')}, inplace=True)
        if 'red' in column: animal_rot.rename(columns={column:column.replace('red', 'sup')}, inplace=True)
        if 'both' in column: animal_rot.rename(columns={column:column.replace('both', 'all')}, inplace=True)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                SAVE DATA                               |#
    #|________________________________________________________________________|#

    animal_dict = {
        fname_pre: animal_pre,
        fname_rot: animal_rot
    }

    with open(os.path.join(save_dir, mouse+"_df_dict.pkl"), "wb") as file:
        pickle.dump(animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    animal_still_dict = {
        fname_pre: animal_pre_still,
        fname_rot: animal_rot_still
    }

    with open(os.path.join(save_dir, mouse+"_df_still_dict.pkl"), "wb") as file:
        pickle.dump(animal_still_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    params["compromised_cells"] = compromised_cells
    with open(os.path.join(save_dir, mouse+"_params.pkl"), "wb") as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
#_________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE ROTATION                            |#
#|________________________________________________________________________|#

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


    fig = plt.figure()
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


for mouse in mice_list:

    load_dir = os.path.join(base_dir,'processed_data',mouse)
    save_dir = os.path.join(base_dir, 'rotation', mouse)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    animal = load_pickle(load_dir, mouse+'_df_dict.pkl')
    fnames = list(animal.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]
    animal_pre= copy.deepcopy(animal[fname_pre])
    animal_rot= copy.deepcopy(animal[fname_rot])

    rotation_dict = {}
    #__________________________________________________________________________
    #|                                                                        |#
    #|                                  DEEP                                  |#
    #|________________________________________________________________________|#

    deep_emb_pre = get_signal(animal_pre, 'deep_umap')
    deep_pos_pre = get_signal(animal_pre, 'position')
    deep_dir_pre =get_signal(animal_pre, 'mov_direction')

    deep_emb_rot = get_signal(animal_rot, 'deep_umap')
    deep_pos_rot = get_signal(animal_rot, 'position')
    deep_dir_rot =get_signal(animal_rot, 'mov_direction')

    #clean outliers
    D = pairwise_distances(deep_emb_pre)
    noise_deep_pre = filter_noisy_outliers(deep_emb_pre,D=D)
    deep_emb_pre = deep_emb_pre[~noise_deep_pre,:]
    deep_pos_pre = deep_pos_pre[~noise_deep_pre,:]
    deep_dir_pre = deep_dir_pre[~noise_deep_pre]

    D = pairwise_distances(deep_emb_rot)
    noise_deep_rot = filter_noisy_outliers(deep_emb_rot,D=D)
    deep_emb_rot = deep_emb_rot[~noise_deep_rot,:]
    deep_pos_rot = deep_pos_rot[~noise_deep_rot,:]
    deep_dir_rot = deep_dir_rot[~noise_deep_rot]


    #compute centroids
    deep_cent_pre, deep_cent_rot, deep_cent_pos, deep_cent_dir = get_centroids(deep_emb_pre, deep_emb_rot, deep_pos_pre[:,0], deep_pos_rot[:,0], 
                                                    deep_dir_pre, deep_dir_rot, num_centroids=40) 

    #project into planes
    deep_norm_vec_pre, deep_cloud_center_pre = parametrize_plane(deep_emb_pre)
    plane_deep_emb_pre = project_onto_plane(deep_emb_pre, deep_norm_vec_pre, deep_cloud_center_pre)

    deep_norm_vec_rot, deep_cloud_center_rot = parametrize_plane(deep_emb_rot)
    plane_deep_emb_rot = project_onto_plane(deep_emb_rot, deep_norm_vec_rot, deep_cloud_center_rot)

    plane_deep_cent_pre, plane_deep_cent_rot, plane_deep_cent_pos, plane_deep_cent_dir = get_centroids(plane_deep_emb_pre, plane_deep_emb_rot, 
                                                                                        deep_pos_pre[:,0], deep_pos_rot[:,0], 
                                                                                        deep_dir_pre, deep_dir_rot, num_centroids=40) 
    #align them
    deep_align_angle, deep_align_mat = align_vectors(deep_norm_vec_pre, deep_cloud_center_pre, deep_norm_vec_rot, deep_cloud_center_rot)

    aligned_deep_emb_rot =  apply_rotation_to_cloud(deep_emb_rot, deep_align_mat, deep_cloud_center_rot)
    aligned_plane_deep_emb_rot =  apply_rotation_to_cloud(plane_deep_emb_rot, deep_align_mat, deep_cloud_center_rot)

    aligned_deep_cent_rot =  apply_rotation_to_cloud(deep_cent_rot, deep_align_mat, deep_cloud_center_rot)
    aligned_plane_deep_cent_rot =  apply_rotation_to_cloud(plane_deep_cent_rot, deep_align_mat, deep_cloud_center_rot)

    #compute angle of rotation
    deep_angles = np.linspace(-np.pi,np.pi,200)
    deep_error = find_rotation(plane_deep_cent_pre, plane_deep_cent_rot, -deep_norm_vec_pre)
    norm_deep_error = (np.array(deep_error)-np.min(deep_error))/(np.max(deep_error)-np.min(deep_error))
    signed_deep_rotation_angle = deep_angles[np.argmin(norm_deep_error)]*180/np.pi
    deep_rotation_angle = np.abs(signed_deep_rotation_angle)
    print(f"\tDeep: {signed_deep_rotation_angle:2f} degrees")

    rotated_aligned_deep_cent_rot = rotate_cloud_around_axis(aligned_deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_plane_deep_cent_rot = rotate_cloud_around_axis(aligned_plane_deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_deep_emb_rot = rotate_cloud_around_axis(aligned_deep_emb_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_plane_deep_emb_rot = rotate_cloud_around_axis(aligned_plane_deep_emb_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)

    rotated_deep_cent_rot = rotate_cloud_around_axis(deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)

    fig = plot_rotation(deep_emb_pre, deep_emb_rot, deep_pos_pre, deep_pos_rot, deep_dir_pre, deep_dir_rot, 
                deep_cent_pre, deep_cent_rot, deep_cent_pos, plane_deep_cent_pre, plane_deep_cent_rot, 
                aligned_plane_deep_cent_rot, rotated_aligned_plane_deep_cent_rot, deep_angles, deep_error, signed_deep_rotation_angle)
    plt.suptitle(f"{mouse} deep")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_rotation_plot.png'), dpi = 400,bbox_inches="tight")


    rotation_dict['deep'] = {
        #initial data
        'deep_emb_pre': deep_emb_pre,
        'deep_pos_pre': deep_pos_pre,
        'deep_dir_pre': deep_dir_pre,
        'noise_deep_pre': noise_deep_pre,

        'deep_emb_rot': deep_emb_rot,
        'deep_pos_rot': deep_pos_rot,
        'deep_dir_rot': deep_dir_rot,
        'noise_deep_rot': noise_deep_rot,
        #centroids
        'deep_cent_pre': deep_cent_pre,
        'deep_cent_rot': deep_cent_rot,
        'deep_cent_pos': deep_cent_pos,
        'deep_cent_dir': deep_cent_dir,

        #project into plane
        'deep_norm_vec_pre': deep_norm_vec_pre,
        'deep_cloud_center_pre': deep_cloud_center_pre,
        'plane_deep_emb_pre': plane_deep_emb_pre,

        'deep_norm_vec_rot': deep_norm_vec_rot,
        'deep_cloud_center_rot': deep_cloud_center_rot,
        'plane_deep_emb_rot': plane_deep_emb_rot,

        #plane centroids
        'plane_deep_cent_pre': plane_deep_cent_pre,
        'plane_deep_cent_rot': plane_deep_cent_rot,
        'plane_deep_cent_pos': plane_deep_cent_pos,
        'plane_deep_cent_dir': plane_deep_cent_dir,

        #align planes
        'deep_align_angle': deep_align_angle,
        'deep_align_mat': deep_align_mat,

        'aligned_deep_emb_rot': aligned_deep_emb_rot,
        'aligned_plane_deep_emb_rot': aligned_plane_deep_emb_rot,
        'aligned_deep_cent_rot': aligned_deep_cent_rot,
        'aligned_plane_deep_cent_rot': aligned_plane_deep_cent_rot,

        #compute angle of rotation
        'deep_angles': deep_angles,
        'deep_error': deep_error,
        'norm_deep_error': norm_deep_error,
        'signed_deep_rotation_angle': signed_deep_rotation_angle,
        'deep_rotation_angle': deep_rotation_angle,

        #rotate post session
        'rotated_deep_cent_rot': rotated_deep_cent_rot,
        'rotated_aligned_deep_cent_rot': rotated_aligned_deep_cent_rot,
        'rotated_aligned_plane_deep_cent_rot': rotated_aligned_plane_deep_cent_rot,
        'rotated_aligned_deep_emb_rot': rotated_aligned_deep_emb_rot,
        'rotated_aligned_plane_deep_emb_rot': rotated_aligned_plane_deep_emb_rot,
    }

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                  SUP                                  |#
    #|________________________________________________________________________|#

    sup_emb_pre = get_signal(animal_pre, 'sup_umap')
    sup_pos_pre = get_signal(animal_pre, 'position')
    sup_dir_pre =get_signal(animal_pre, 'mov_direction')

    sup_emb_rot = get_signal(animal_rot, 'sup_umap')
    sup_pos_rot = get_signal(animal_rot, 'position')
    sup_dir_rot =get_signal(animal_rot, 'mov_direction')

    #clean outliers
    D = pairwise_distances(sup_emb_pre)
    noise_sup_pre = filter_noisy_outliers(sup_emb_pre,D=D)
    sup_emb_pre = sup_emb_pre[~noise_sup_pre,:]
    sup_pos_pre = sup_pos_pre[~noise_sup_pre,:]
    sup_dir_pre = sup_dir_pre[~noise_sup_pre]

    D = pairwise_distances(sup_emb_rot)
    noise_sup_rot = filter_noisy_outliers(sup_emb_rot,D=D)
    sup_emb_rot = sup_emb_rot[~noise_sup_rot,:]
    sup_pos_rot = sup_pos_rot[~noise_sup_rot,:]
    sup_dir_rot = sup_dir_rot[~noise_sup_rot]


    #compute centroids
    sup_cent_pre, sup_cent_rot, sup_cent_pos, sup_cent_dir = get_centroids(sup_emb_pre, sup_emb_rot, sup_pos_pre[:,0], sup_pos_rot[:,0], 
                                                    sup_dir_pre, sup_dir_rot, num_centroids=40) 

    #project into planes
    sup_norm_vec_pre, sup_cloud_center_pre = parametrize_plane(sup_emb_pre)
    plane_sup_emb_pre = project_onto_plane(sup_emb_pre, sup_norm_vec_pre, sup_cloud_center_pre)

    sup_norm_vec_rot, sup_cloud_center_rot = parametrize_plane(sup_emb_rot)
    plane_sup_emb_rot = project_onto_plane(sup_emb_rot, sup_norm_vec_rot, sup_cloud_center_rot)

    plane_sup_cent_pre, plane_sup_cent_rot, plane_sup_cent_pos, plane_sup_cent_dir = get_centroids(plane_sup_emb_pre, plane_sup_emb_rot, 
                                                                                        sup_pos_pre[:,0], sup_pos_rot[:,0], 
                                                                                        sup_dir_pre, sup_dir_rot, num_centroids=40) 
    #align them
    sup_align_angle, sup_align_mat = align_vectors(sup_norm_vec_pre, sup_cloud_center_pre, sup_norm_vec_rot, sup_cloud_center_rot)

    aligned_sup_emb_rot =  apply_rotation_to_cloud(sup_emb_rot, sup_align_mat, sup_cloud_center_rot)
    aligned_plane_sup_emb_rot =  apply_rotation_to_cloud(plane_sup_emb_rot, sup_align_mat, sup_cloud_center_rot)

    aligned_sup_cent_rot =  apply_rotation_to_cloud(sup_cent_rot, sup_align_mat, sup_cloud_center_rot)
    aligned_plane_sup_cent_rot =  apply_rotation_to_cloud(plane_sup_cent_rot, sup_align_mat, sup_cloud_center_rot)

    #compute angle of rotation
    sup_angles = np.linspace(-np.pi,np.pi,200)
    sup_error = find_rotation(plane_sup_cent_pre, aligned_plane_sup_cent_rot, -sup_norm_vec_pre)
    norm_sup_error = (np.array(sup_error)-np.min(sup_error))/(np.max(sup_error)-np.min(sup_error))
    signed_sup_rotation_angle = sup_angles[np.argmin(norm_sup_error)]*180/np.pi
    sup_rotation_angle = np.abs(signed_sup_rotation_angle)
    print(f"\tsup: {signed_sup_rotation_angle:2f} degrees")

    rotated_aligned_sup_cent_rot = rotate_cloud_around_axis(aligned_sup_cent_rot, (np.pi/180)*signed_sup_rotation_angle,sup_norm_vec_pre)
    rotated_aligned_plane_sup_cent_rot = rotate_cloud_around_axis(aligned_plane_sup_cent_rot, (np.pi/180)*signed_sup_rotation_angle,sup_norm_vec_pre)
    rotated_aligned_sup_emb_rot = rotate_cloud_around_axis(aligned_sup_emb_rot, (np.pi/180)*signed_sup_rotation_angle,sup_norm_vec_pre)
    rotated_aligned_plane_sup_emb_rot = rotate_cloud_around_axis(aligned_plane_sup_emb_rot, (np.pi/180)*signed_sup_rotation_angle,sup_norm_vec_pre)


    fig = plot_rotation(sup_emb_pre, sup_emb_rot, sup_pos_pre, sup_pos_rot, sup_dir_pre, sup_dir_rot, 
                sup_cent_pre, sup_cent_rot, sup_cent_pos, plane_sup_cent_pre, plane_sup_cent_rot, 
                aligned_plane_sup_cent_rot, rotated_aligned_sup_cent_rot, sup_angles, sup_error, signed_sup_rotation_angle)

    plt.suptitle(f"{mouse} sup")

    plt.savefig(os.path.join(save_dir,f'{mouse}_sup_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_sup_rotation_plot.png'), dpi = 400,bbox_inches="tight")


    rotation_dict['sup'] = {
        #initial data
        'sup_emb_pre': sup_emb_pre,
        'sup_pos_pre': sup_pos_pre,
        'sup_dir_pre': sup_dir_pre,
        'noise_sup_pre': noise_sup_pre,

        'sup_emb_rot': sup_emb_rot,
        'sup_pos_rot': sup_pos_rot,
        'sup_dir_rot': sup_dir_rot,
        'noise_sup_rot': noise_sup_rot,
        #centroids
        'sup_cent_pre': sup_cent_pre,
        'sup_cent_rot': sup_cent_rot,
        'sup_cent_pos': sup_cent_pos,
        'sup_cent_dir': sup_cent_dir,

        #project into plane
        'sup_norm_vec_pre': sup_norm_vec_pre,
        'sup_cloud_center_pre': sup_cloud_center_pre,
        'plane_sup_emb_pre': plane_sup_emb_pre,

        'sup_norm_vec_rot': sup_norm_vec_rot,
        'sup_cloud_center_rot': sup_cloud_center_rot,
        'plane_sup_emb_rot': plane_sup_emb_rot,

        #plane centroids
        'plane_sup_cent_pre': plane_sup_cent_pre,
        'plane_sup_cent_rot': plane_sup_cent_rot,
        'plane_sup_cent_pos': plane_sup_cent_pos,
        'plane_sup_cent_dir': plane_sup_cent_dir,

        #align planes
        'sup_align_angle': sup_align_angle,
        'sup_align_mat': sup_align_mat,

        'aligned_sup_emb_rot': aligned_sup_emb_rot,
        'aligned_plane_sup_emb_rot': aligned_plane_sup_emb_rot,
        'aligned_sup_cent_rot': aligned_sup_cent_rot,
        'aligned_plane_sup_cent_rot': aligned_plane_sup_cent_rot,

        #compute angle of rotation
        'sup_angles': sup_angles,
        'sup_error': sup_error,
        'norm_sup_error': norm_sup_error,
        'signed_sup_rotation_angle': signed_sup_rotation_angle,
        'sup_rotation_angle': sup_rotation_angle,


        #rotate post session
        'rotated_aligned_sup_cent_rot': rotated_aligned_sup_cent_rot,
        'rotated_aligned_plane_sup_cent_rot': rotated_aligned_plane_sup_cent_rot,
        'rotated_aligned_sup_emb_rot': rotated_aligned_sup_emb_rot,
        'rotated_aligned_plane_sup_emb_rot': rotated_aligned_plane_sup_emb_rot,

    }


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                  all                                  |#
    #|________________________________________________________________________|#

    all_emb_pre = get_signal(animal_pre, 'all_umap')
    all_pos_pre = get_signal(animal_pre, 'position')
    all_dir_pre =get_signal(animal_pre, 'mov_direction')

    all_emb_rot = get_signal(animal_rot, 'all_umap')
    all_pos_rot = get_signal(animal_rot, 'position')
    all_dir_rot =get_signal(animal_rot, 'mov_direction')

    #clean outliers
    D = pairwise_distances(all_emb_pre)
    noise_all_pre = filter_noisy_outliers(all_emb_pre,D=D)
    all_emb_pre = all_emb_pre[~noise_all_pre,:]
    all_pos_pre = all_pos_pre[~noise_all_pre,:]
    all_dir_pre = all_dir_pre[~noise_all_pre]

    D = pairwise_distances(all_emb_rot)
    noise_all_rot = filter_noisy_outliers(all_emb_rot,D=D)
    all_emb_rot = all_emb_rot[~noise_all_rot,:]
    all_pos_rot = all_pos_rot[~noise_all_rot,:]
    all_dir_rot = all_dir_rot[~noise_all_rot]


    #compute centroids
    all_cent_pre, all_cent_rot, all_cent_pos, all_cent_dir = get_centroids(all_emb_pre, all_emb_rot, all_pos_pre[:,0], all_pos_rot[:,0], 
                                                    all_dir_pre, all_dir_rot, num_centroids=40) 

    #project into planes
    all_norm_vec_pre, all_cloud_center_pre = parametrize_plane(all_emb_pre)
    plane_all_emb_pre = project_onto_plane(all_emb_pre, all_norm_vec_pre, all_cloud_center_pre)

    all_norm_vec_rot, all_cloud_center_rot = parametrize_plane(all_emb_rot)
    plane_all_emb_rot = project_onto_plane(all_emb_rot, all_norm_vec_rot, all_cloud_center_rot)

    plane_all_cent_pre, plane_all_cent_rot, plane_all_cent_pos, plane_all_cent_dir = get_centroids(plane_all_emb_pre, plane_all_emb_rot, 
                                                                                        all_pos_pre[:,0], all_pos_rot[:,0], 
                                                                                        all_dir_pre, all_dir_rot, num_centroids=40) 
    #align them
    all_align_angle, all_align_mat = align_vectors(all_norm_vec_pre, all_cloud_center_pre, all_norm_vec_rot, all_cloud_center_rot)

    aligned_all_emb_rot =  apply_rotation_to_cloud(all_emb_rot, all_align_mat, all_cloud_center_rot)
    aligned_plane_all_emb_rot =  apply_rotation_to_cloud(plane_all_emb_rot, all_align_mat, all_cloud_center_rot)

    aligned_all_cent_rot =  apply_rotation_to_cloud(all_cent_rot, all_align_mat, all_cloud_center_rot)
    aligned_plane_all_cent_rot =  apply_rotation_to_cloud(plane_all_cent_rot, all_align_mat, all_cloud_center_rot)

    #compute angle of rotation
    all_angles = np.linspace(-np.pi,np.pi,200)
    all_error = find_rotation(plane_all_cent_pre, aligned_plane_all_cent_rot, -all_norm_vec_pre)
    norm_all_error = (np.array(all_error)-np.min(all_error))/(np.max(all_error)-np.min(all_error))
    signed_all_rotation_angle = all_angles[np.argmin(norm_all_error)]*180/np.pi
    all_rotation_angle = np.abs(signed_all_rotation_angle)
    print(f"\tall: {signed_all_rotation_angle:2f} degrees")

    rotated_aligned_all_cent_rot = rotate_cloud_around_axis(aligned_all_cent_rot, (np.pi/180)*signed_all_rotation_angle,all_norm_vec_pre)
    rotated_aligned_plane_all_cent_rot = rotate_cloud_around_axis(aligned_plane_all_cent_rot, (np.pi/180)*signed_all_rotation_angle,all_norm_vec_pre)
    rotated_aligned_all_emb_rot = rotate_cloud_around_axis(aligned_all_emb_rot, (np.pi/180)*signed_all_rotation_angle,all_norm_vec_pre)
    rotated_aligned_plane_all_emb_rot = rotate_cloud_around_axis(aligned_plane_all_emb_rot, (np.pi/180)*signed_all_rotation_angle,all_norm_vec_pre)


    fig = plot_rotation(all_emb_pre, all_emb_rot, all_pos_pre, all_pos_rot, all_dir_pre, all_dir_rot, 
                all_cent_pre, all_cent_rot, all_cent_pos, plane_all_cent_pre, plane_all_cent_rot, 
                aligned_plane_all_cent_rot, rotated_aligned_all_cent_rot, all_angles, all_error, signed_all_rotation_angle)

    plt.suptitle(f"{mouse} all")

    plt.savefig(os.path.join(save_dir,f'{mouse}_all_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_all_rotation_plot.png'), dpi = 400,bbox_inches="tight")

    rotation_dict['all'] = {
        #initial data
        'all_emb_pre': all_emb_pre,
        'all_pos_pre': all_pos_pre,
        'all_dir_pre': all_dir_pre,
        'noise_all_pre': noise_all_pre,

        'all_emb_rot': all_emb_rot,
        'all_pos_rot': all_pos_rot,
        'all_dir_rot': all_dir_rot,
        'noise_all_rot': noise_all_rot,
        #centroids
        'all_cent_pre': all_cent_pre,
        'all_cent_rot': all_cent_rot,
        'all_cent_pos': all_cent_pos,
        'all_cent_dir': all_cent_dir,

        #project into plane
        'all_norm_vec_pre': all_norm_vec_pre,
        'all_cloud_center_pre': all_cloud_center_pre,
        'plane_all_emb_pre': plane_all_emb_pre,

        'all_norm_vec_rot': all_norm_vec_rot,
        'all_cloud_center_rot': all_cloud_center_rot,
        'plane_all_emb_rot': plane_all_emb_rot,

        #plane centroids
        'plane_all_cent_pre': plane_all_cent_pre,
        'plane_all_cent_rot': plane_all_cent_rot,
        'plane_all_cent_pos': plane_all_cent_pos,
        'plane_all_cent_dir': plane_all_cent_dir,

        #align planes
        'all_align_angle': all_align_angle,
        'all_align_mat': all_align_mat,

        'aligned_all_emb_rot': aligned_all_emb_rot,
        'aligned_plane_all_emb_rot': aligned_plane_all_emb_rot,
        'aligned_all_cent_rot': aligned_all_cent_rot,
        'aligned_plane_all_cent_rot': aligned_plane_all_cent_rot,

        #compute angle of rotation
        'all_angles': all_angles,
        'all_error': all_error,
        'norm_all_error': norm_all_error,
        'signed_all_rotation_angle': signed_all_rotation_angle,
        'all_rotation_angle': all_rotation_angle,


        #rotate post session
        'rotated_aligned_all_cent_rot': rotated_aligned_all_cent_rot,
        'rotated_aligned_plane_all_cent_rot': rotated_aligned_plane_all_cent_rot,
        'rotated_aligned_all_emb_rot': rotated_aligned_all_emb_rot,
        'rotated_aligned_plane_all_emb_rot': rotated_aligned_plane_all_emb_rot,

    }

    with open(os.path.join(save_dir, mouse+"_rotation_dict.pkl"), "wb") as file:
        pickle.dump(rotation_dict, file, protocol=pickle.HIGHEST_PROTOCOL)



#_________________________________________________________________________
#|                                                                        |#
#|                           COMPUTE DISTANCE                             |#
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


for mouse in mice_list:
    print(f"Working on {mouse}:")
    load_dir = os.path.join(base_dir,'rotation',mouse)
    save_dir = os.path.join(base_dir, 'distance', mouse)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    rotation_dict = load_pickle(load_dir, mouse+'_rotation_dict.pkl')
    distance_dict = dict()

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                   DEEP                                  |#
    #|________________________________________________________________________|#

    deep_cent_pre = rotation_dict['deep']['deep_cent_pre']
    deep_cent_rot = rotation_dict['deep']['deep_cent_rot']
    deep_cent_pos = rotation_dict['deep']['deep_cent_pos']
    deep_cent_dir = rotation_dict['deep']['deep_cent_dir']

    deep_inter_dist = np.linalg.norm(deep_cent_pre.mean(axis=0)-deep_cent_rot.mean(axis=0))
    deep_intra_dist_pre = np.percentile(pairwise_distances(deep_cent_pre),95)/2
    deep_intra_dist_rot = np.percentile(pairwise_distances(deep_cent_rot),95)/2
    deep_remap_dist = deep_inter_dist/np.mean((deep_intra_dist_pre, deep_intra_dist_rot))

    plane_deep_cent_pre = rotation_dict['deep']['plane_deep_cent_pre']
    plane_deep_cent_rot = rotation_dict['deep']['plane_deep_cent_rot']
    deep_norm_vector_pre = rotation_dict['deep']['deep_norm_vec_pre']
    plane_deep_cent_pos = rotation_dict['deep']['plane_deep_cent_pos']
    plane_deep_cent_dir = rotation_dict['deep']['plane_deep_cent_dir']
    deep_norm_vector_rot = rotation_dict['deep']['deep_norm_vec_rot']


    plane_deep_inter_dist = np.linalg.norm(plane_deep_cent_pre.mean(axis=0)-plane_deep_cent_rot.mean(axis=0))
    deep_ellipse_pre_params, deep_ellipse_pre_long_axis, deep_ellipse_pre_short_axis, deep_ellipse_pre_fit, deep_ellipse_pre_fit_3D = fit_ellipse(plane_deep_cent_pre, deep_norm_vector_pre)
    deep_ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(deep_ellipse_pre_long_axis+deep_ellipse_pre_short_axis)**2)

    deep_ellipse_rot_params, deep_ellipse_rot_long_axis, deep_ellipse_rot_short_axis, deep_ellipse_rot_fit, deep_ellipse_rot_fit_3D = fit_ellipse(plane_deep_cent_rot, deep_norm_vector_rot)
    deep_ellipse_rot_perimeter = 2*np.pi*np.sqrt(0.5*(deep_ellipse_rot_long_axis+deep_ellipse_rot_short_axis)**2)

    plane_deep_remap_dist = plane_deep_inter_dist/np.mean((deep_ellipse_pre_perimeter, deep_ellipse_rot_perimeter))

    print(f"\tdeep: {deep_remap_dist:.2f} remap dist | {plane_deep_remap_dist:.2f} remap dist plane")

    fig = plot_distance(deep_cent_pre,deep_cent_rot,deep_cent_pos,deep_cent_dir,
            plane_deep_cent_pre,plane_deep_cent_rot, plane_deep_cent_pos, plane_deep_cent_dir,
            deep_ellipse_pre_fit_3D, deep_ellipse_rot_fit_3D)
    plt.suptitle(f"{mouse} deep")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_distance_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_distance_plot.png'), dpi = 400,bbox_inches="tight")

    distance_dict['deep'] = {

        #cent
        'deep_cent_pre': deep_cent_pre,
        'deep_cent_rot': deep_cent_rot,
        'deep_cent_pos': deep_cent_pos,
        'noise_deep_pre': deep_cent_dir,
        #distance og
        'deep_inter_dist': deep_inter_dist,
        'deep_intra_dist_pre': deep_intra_dist_pre,
        'deep_intra_dist_rot': deep_intra_dist_rot,
        'deep_remap_dist': deep_remap_dist,

        #plane
        'plane_deep_cent_pre': deep_cent_pre,
        'deep_norm_vector_pre': deep_norm_vector_pre,
        'plane_deep_cent_rot': plane_deep_cent_rot,
        'deep_norm_vector_rot': deep_norm_vector_rot,
        'plane_deep_cent_pos': plane_deep_cent_pos,
        'plane_deep_cent_dir': plane_deep_cent_dir,

        #ellipse
        'deep_ellipse_pre_params': deep_ellipse_pre_params,
        'deep_ellipse_pre_long_axis': deep_ellipse_pre_long_axis,
        'deep_ellipse_pre_short_axis': deep_ellipse_pre_short_axis,
        'deep_ellipse_pre_fit': deep_ellipse_pre_fit,
        'deep_ellipse_pre_fit_3D': deep_ellipse_pre_fit_3D,

        'deep_ellipse_rot_params': deep_ellipse_rot_params,
        'deep_ellipse_rot_long_axis': deep_ellipse_rot_long_axis,
        'deep_ellipse_rot_short_axis': deep_ellipse_rot_short_axis,
        'deep_ellipse_rot_fit': deep_ellipse_rot_fit,
        'deep_ellipse_rot_fit_3D': deep_ellipse_rot_fit_3D,

        #distance ellipse
        'plane_deep_inter_dist': plane_deep_inter_dist,
        'deep_ellipse_pre_perimeter': deep_ellipse_pre_perimeter,
        'deep_ellipse_rot_perimeter': deep_ellipse_rot_perimeter,
        'plane_deep_remap_dist': plane_deep_remap_dist,
    }


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                   SUP                                  |#
    #|________________________________________________________________________|#

    sup_cent_pre = rotation_dict['sup']['sup_cent_pre']
    sup_cent_rot = rotation_dict['sup']['sup_cent_rot']
    sup_cent_pos = rotation_dict['sup']['sup_cent_pos']
    sup_cent_dir = rotation_dict['sup']['sup_cent_dir']

    sup_inter_dist = np.linalg.norm(sup_cent_pre.mean(axis=0)-sup_cent_rot.mean(axis=0))
    sup_intra_dist_pre = np.percentile(pairwise_distances(sup_cent_pre),95)/2
    sup_intra_dist_rot = np.percentile(pairwise_distances(sup_cent_rot),95)/2
    sup_remap_dist = sup_inter_dist/np.mean((sup_intra_dist_pre, sup_intra_dist_rot))

    plane_sup_cent_pre = rotation_dict['sup']['plane_sup_cent_pre']
    plane_sup_cent_rot = rotation_dict['sup']['plane_sup_cent_rot']
    sup_norm_vector_pre = rotation_dict['sup']['sup_norm_vec_pre']
    plane_sup_cent_pos = rotation_dict['sup']['plane_sup_cent_pos']
    plane_sup_cent_dir = rotation_dict['sup']['plane_sup_cent_dir']
    sup_norm_vector_rot = rotation_dict['sup']['sup_norm_vec_rot']


    plane_sup_inter_dist = np.linalg.norm(plane_sup_cent_pre.mean(axis=0)-plane_sup_cent_rot.mean(axis=0))
    sup_ellipse_pre_params, sup_ellipse_pre_long_axis, sup_ellipse_pre_short_axis, sup_ellipse_pre_fit, sup_ellipse_pre_fit_3D = fit_ellipse(plane_sup_cent_pre, sup_norm_vector_pre)
    sup_ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(sup_ellipse_pre_long_axis+sup_ellipse_pre_short_axis)**2)

    sup_ellipse_rot_params, sup_ellipse_rot_long_axis, sup_ellipse_rot_short_axis, sup_ellipse_rot_fit, sup_ellipse_rot_fit_3D = fit_ellipse(plane_sup_cent_rot, sup_norm_vector_rot)
    sup_ellipse_rot_perimeter = 2*np.pi*np.sqrt(0.5*(sup_ellipse_rot_long_axis+sup_ellipse_rot_short_axis)**2)

    plane_sup_remap_dist = plane_sup_inter_dist/np.mean((sup_ellipse_pre_perimeter, sup_ellipse_rot_perimeter))

    print(f"\tsup: {sup_remap_dist:.2f} remap dist | {plane_sup_remap_dist:.2f} remap dist plane")

    fig = plot_distance(sup_cent_pre,sup_cent_rot,sup_cent_pos,sup_cent_dir,
            plane_sup_cent_pre,plane_sup_cent_rot, plane_sup_cent_pos, plane_sup_cent_dir,
            sup_ellipse_pre_fit_3D, sup_ellipse_rot_fit_3D)
    plt.suptitle(f"{mouse} sup")
    plt.savefig(os.path.join(save_dir,f'{mouse}_sup_distance_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_sup_distance_plot.png'), dpi = 400,bbox_inches="tight")

    distance_dict['sup'] = {

        #cent
        'sup_cent_pre': sup_cent_pre,
        'sup_cent_rot': sup_cent_rot,
        'sup_cent_pos': sup_cent_pos,
        'noise_sup_pre': sup_cent_dir,
        #distance og
        'sup_inter_dist': sup_inter_dist,
        'sup_intra_dist_pre': sup_intra_dist_pre,
        'sup_intra_dist_rot': sup_intra_dist_rot,
        'sup_remap_dist': sup_remap_dist,

        #plane
        'plane_sup_cent_pre': sup_cent_pre,
        'sup_norm_vector_pre': sup_norm_vector_pre,
        'plane_sup_cent_rot': plane_sup_cent_rot,
        'sup_norm_vector_rot': sup_norm_vector_rot,
        'plane_sup_cent_pos': plane_sup_cent_pos,
        'plane_sup_cent_dir': plane_sup_cent_dir,

        #ellipse
        'sup_ellipse_pre_params': sup_ellipse_pre_params,
        'sup_ellipse_pre_long_axis': sup_ellipse_pre_long_axis,
        'sup_ellipse_pre_short_axis': sup_ellipse_pre_short_axis,
        'sup_ellipse_pre_fit': sup_ellipse_pre_fit,
        'sup_ellipse_pre_fit_3D': sup_ellipse_pre_fit_3D,

        'sup_ellipse_rot_params': sup_ellipse_rot_params,
        'sup_ellipse_rot_long_axis': sup_ellipse_rot_long_axis,
        'sup_ellipse_rot_short_axis': sup_ellipse_rot_short_axis,
        'sup_ellipse_rot_fit': sup_ellipse_rot_fit,
        'sup_ellipse_rot_fit_3D': sup_ellipse_rot_fit_3D,

        #distance ellipse
        'plane_sup_inter_dist': plane_sup_inter_dist,
        'sup_ellipse_pre_perimeter': sup_ellipse_pre_perimeter,
        'sup_ellipse_rot_perimeter': sup_ellipse_rot_perimeter,
        'plane_sup_remap_dist': plane_sup_remap_dist,
    }

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                   ALL                                  |#
    #|________________________________________________________________________|#

    all_cent_pre = rotation_dict['all']['all_cent_pre']
    all_cent_rot = rotation_dict['all']['all_cent_rot']
    all_cent_pos = rotation_dict['all']['all_cent_pos']
    all_cent_dir = rotation_dict['all']['all_cent_dir']

    all_inter_dist = np.linalg.norm(all_cent_pre.mean(axis=0)-all_cent_rot.mean(axis=0))
    all_intra_dist_pre = np.percentile(pairwise_distances(all_cent_pre),95)/2
    all_intra_dist_rot = np.percentile(pairwise_distances(all_cent_rot),95)/2
    all_remap_dist = all_inter_dist/np.mean((all_intra_dist_pre, all_intra_dist_rot))

    plane_all_cent_pre = rotation_dict['all']['plane_all_cent_pre']
    plane_all_cent_rot = rotation_dict['all']['plane_all_cent_rot']
    all_norm_vector_pre = rotation_dict['all']['all_norm_vec_pre']
    plane_all_cent_pos = rotation_dict['all']['plane_all_cent_pos']
    plane_all_cent_dir = rotation_dict['all']['plane_all_cent_dir']
    all_norm_vector_rot = rotation_dict['all']['all_norm_vec_rot']

    plane_all_inter_dist = np.linalg.norm(plane_all_cent_pre.mean(axis=0)-plane_all_cent_rot.mean(axis=0))
    all_ellipse_pre_params, all_ellipse_pre_long_axis, all_ellipse_pre_short_axis, all_ellipse_pre_fit, all_ellipse_pre_fit_3D = fit_ellipse(plane_all_cent_pre, all_norm_vector_pre)
    all_ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(all_ellipse_pre_long_axis+all_ellipse_pre_short_axis)**2)

    all_ellipse_rot_params, all_ellipse_rot_long_axis, all_ellipse_rot_short_axis, all_ellipse_rot_fit, all_ellipse_rot_fit_3D = fit_ellipse(plane_all_cent_rot, all_norm_vector_rot)
    all_ellipse_rot_perimeter = 2*np.pi*np.sqrt(0.5*(all_ellipse_rot_long_axis+all_ellipse_rot_short_axis)**2)

    plane_all_remap_dist = plane_all_inter_dist/np.mean((all_ellipse_pre_perimeter, all_ellipse_rot_perimeter))

    print(f"\tall: {all_remap_dist:.2f} remap dist | {plane_all_remap_dist:.2f} remap dist plane")

    fig = plot_distance(all_cent_pre,all_cent_rot,all_cent_pos,all_cent_dir,
            plane_all_cent_pre,plane_all_cent_rot, plane_all_cent_pos, plane_all_cent_dir,
            all_ellipse_pre_fit_3D, all_ellipse_rot_fit_3D)
    plt.suptitle(f"{mouse} all")
    plt.savefig(os.path.join(save_dir,f'{mouse}_all_distance_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_all_distance_plot.png'), dpi = 400,bbox_inches="tight")

    distance_dict['all'] = {

        #cent
        'all_cent_pre': all_cent_pre,
        'all_cent_rot': all_cent_rot,
        'all_cent_pos': all_cent_pos,
        'noise_all_pre': all_cent_dir,
        #distance og
        'all_inter_dist': all_inter_dist,
        'all_intra_dist_pre': all_intra_dist_pre,
        'all_intra_dist_rot': all_intra_dist_rot,
        'all_remap_dist': all_remap_dist,

        #plane
        'plane_all_cent_pre': all_cent_pre,
        'all_norm_vector_pre': all_norm_vector_pre,
        'plane_all_cent_rot': plane_all_cent_rot,
        'all_norm_vector_rot': all_norm_vector_rot,
        'plane_all_cent_pos': plane_all_cent_pos,
        'plane_all_cent_dir': plane_all_cent_dir,

        #ellipse
        'all_ellipse_pre_params': all_ellipse_pre_params,
        'all_ellipse_pre_long_axis': all_ellipse_pre_long_axis,
        'all_ellipse_pre_short_axis': all_ellipse_pre_short_axis,
        'all_ellipse_pre_fit': all_ellipse_pre_fit,
        'all_ellipse_pre_fit_3D': all_ellipse_pre_fit_3D,

        'all_ellipse_rot_params': all_ellipse_rot_params,
        'all_ellipse_rot_long_axis': all_ellipse_rot_long_axis,
        'all_ellipse_rot_short_axis': all_ellipse_rot_short_axis,
        'all_ellipse_rot_fit': all_ellipse_rot_fit,
        'all_ellipse_rot_fit_3D': all_ellipse_rot_fit_3D,

        #distance ellipse
        'plane_all_inter_dist': plane_all_inter_dist,
        'all_ellipse_pre_perimeter': all_ellipse_pre_perimeter,
        'all_ellipse_rot_perimeter': all_ellipse_rot_perimeter,
        'plane_all_remap_dist': plane_all_remap_dist,
    }


    with open(os.path.join(save_dir, mouse+"_distance_dict.pkl"), "wb") as file:
        pickle.dump(distance_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
