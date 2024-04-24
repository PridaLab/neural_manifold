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
from sklearn.decomposition import PCA


from sklearn.metrics import pairwise_distances
from datetime import datetime

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data



mice_list = ['CalbCharly2', 'CalbCharly11_concat', 'CalbV23', 'DD2']
base_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'
palette= ['#666666ff', '#aa0007ff']
#__________________________________________________________________________
#|                                                                        |#
#|                             PREPROCESS DATA                            |#
#|________________________________________________________________________|#

def preprocess_traces(pd_struct_p, pd_struct_r, signal_field, sigma = 5,sig_up = 4, sig_down = 12, peak_th=0.1):
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


    out_pd_p['clean_traces'] = [bi_signal_p[trial_id_mat_p==out_pd_p["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_p.shape[0])]
    out_pd_r['clean_traces'] = [bi_signal_r[trial_id_mat_r==out_pd_r["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_r.shape[0])]
    return out_pd_p, out_pd_r

def remove_cells(pd_struct, cells_to_keep):
    out_pd = copy.deepcopy(pd_struct)

    out_pd["index_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            out_pd["trial_id"][idx]
                            for idx in out_pd.index]

    index_mat = np.concatenate(out_pd["index_mat"].values, axis=0)
    neuro_fields = gu.get_neuronal_fields(out_pd, 'raw_traces')
    for field in neuro_fields:
        signal = np.concatenate(out_pd[field].values, axis=0)
        out_pd[field] = [signal[index_mat[:,0]==out_pd["trial_id"][idx]][:, cells_to_keep] 
                                                for idx in range(out_pd.shape[0])]
    return out_pd

def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))

signal_field = 'raw_traces'
vel_th = 4
sigma = 6
sig_up = 4
sig_down =12
nn_val = 120
dim = 3

columns_to_drop = ['date','denoised_traces', '*spikes']
columns_to_rename = {'Fs':'sf','pos':'position', 'vel':'speed', 'index_mat': 'trial_idx_mat'}

for mouse in mice_list:
    print(f"Working on mouse: {mouse}")
    mouse_dir = os.path.join(base_dir,'data',mouse)

    for case in ['veh', 'CNO']:
        print(f"\tcondition: {case}")
        case_dir = os.path.join(mouse_dir, mouse+'_'+case)
        save_dir = os.path.join(base_dir, 'processed_data',mouse+'_'+case)

        if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        now = datetime.now()
        params = {
            "date": now.strftime("%d/%m/%Y %H:%M:%S"),
            "mouse": mouse,
            "load_dir": case_dir,
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
        animal = gu.load_files(case_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")
        fnames = list(animal.keys())

        fname_pre = [fname for fname in fnames if 'lt' in fname][0]
        fname_rot = [fname for fname in fnames if 'rot' in fname][0]

        animal_p = copy.deepcopy(animal[fname_pre])
        animal_r = copy.deepcopy(animal[fname_rot])

        #__________________________________________________________________________
        #|                                                                        |#
        #|               CHANGE COLUMN NAMES AND ADD NEW ONES                     |#
        #|________________________________________________________________________|#

        for column in columns_to_drop:
            if column in animal_p.columns: animal_p.drop(columns=[column], inplace=True)
            if column in animal_r.columns: animal_r.drop(columns=[column], inplace=True)

        for old, new in columns_to_rename.items():
            if old in animal_p.columns: animal_p.rename(columns={old:new}, inplace=True)
            if old in animal_r.columns: animal_r.rename(columns={old:new}, inplace=True)

        gu.add_trial_id_mat_field(animal_p)
        gu.add_trial_id_mat_field(animal_r)

        gu.add_mov_direction_mat_field(animal_p)
        gu.add_mov_direction_mat_field(animal_r)

        gu.add_trial_type_mat_field(animal_p)
        gu.add_trial_type_mat_field(animal_r)

        #__________________________________________________________________________
        #|                                                                        |#
        #|                          KEEP ONLY MOVING                              |#
        #|________________________________________________________________________|#

        if vel_th>0:
            animal_p, animal_p_still = gu.keep_only_moving(animal_p, vel_th)
            animal_r, animal_r_still = gu.keep_only_moving(animal_r, vel_th)
        else:
            animal_p_still = pd.DataFrame()
            animal_r_still = pd.DataFrame()

        #__________________________________________________________________________
        #|                                                                        |#
        #|                          PREPROCESS TRACES                             |#
        #|________________________________________________________________________|#

        animal_p, animal_r = preprocess_traces(animal_p, animal_r, signal_field, sigma=sigma, sig_up = sig_up, sig_down = sig_down)
        animal_p['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
        animal_r['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}

        if vel_th>0:
            animal_p_still, animal_r_still = preprocess_traces(animal_p_still, animal_r_still, signal_field, sigma=sigma, sig_up = sig_up, sig_down = sig_down)
            animal_p_still['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
            animal_r_still['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
        #__________________________________________________________________________
        #|                                                                        |#
        #|                               REMOVE CELLS                             |#
        #|________________________________________________________________________|#

        signal_p = get_signal(animal_p,'clean_traces')
        pos_p = get_signal(animal_p, 'position')

        signal_r = get_signal(animal_r,'clean_traces')
        pos_r = get_signal(animal_r, 'position')

        #%%all data
        index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
        concat_signal = np.vstack((signal_p, signal_r))
        model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
        model.fit(concat_signal)
        concat_emb = model.transform(concat_signal)
        emb_p = concat_emb[index[:,0]==0,:]
        emb_r = concat_emb[index[:,0]==1,:]

        #%%
        plt.figure()
        ax = plt.subplot(1,2,1, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, color ='b', s= 20)
        ax.scatter(*emb_r[:,:3].T, color = 'r', s= 20)
        ax.set_aspect('equal', adjustable='box')

        ax.set_title('All')
        ax = plt.subplot(1,2,2, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 20, cmap = 'magma')
        ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 20, cmap = 'magma')
        ax.set_aspect('equal', adjustable='box')

        plt.suptitle(f"{mouse}: clean_traces - vel: {vel_th} - nn: {nn_val} - dim: {dim}")
        plt.savefig(os.path.join(save_dir,mouse+'_'+case+'_original_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)


        #check std as a proxy for noise
        cells_std = np.vstack((np.std(signal_p, axis=0),np.std(signal_r, axis=0)))
        cells_std_diff = np.diff(cells_std,axis=0);
        max_diff = np.max([0.1, np.percentile(cells_std_diff,80)])
        cells_with_std_diff_flag = np.where(cells_std_diff>max_diff)[0];
        max_std = np.percentile(cells_std,80)
        cells_with_std_flag  = np.where(np.any(cells_std>max_std,axis=0))[0];

        #check activity rate
        cells_er = np.zeros((signal_p.shape[1],))
        for neu in range(signal_p.shape[1]):
            peaks_p,_ =find_peaks(signal_p[:,neu],height=0.1)
            cells_er[neu] = len(peaks_p)
            peaks_r,_ =find_peaks(signal_r[:,neu],height=0.1)
            cells_er[neu] += len(peaks_r)
        cells_with_er_flag = np.where(cells_er>0.5*signal_p.shape[0]/animal_p['sf'][0])[0]

        #cells to check
        cells_to_check = np.unique(np.concatenate((cells_with_std_flag, 
                                                    cells_with_std_flag,
                                                    cells_with_er_flag)))

        cells_to_keep = np.zeros((signal_p.shape[1],), dtype=bool)+True
        do_cells = input(f"{len(cells_to_check)} cells to check out of {signal_p.shape[1]} cells. Do you want to review them?: ([Y]/N)")
        if not any(do_cells) or 'Y' in do_cells:
            signal_p_og = get_signal(animal_p,signal_field)
            signal_r_og = get_signal(animal_r,signal_field)
            lowpass_r = uniform_filter1d(signal_r_og, size = 4000, axis = 0)
            lowpass_p = uniform_filter1d(signal_p_og, size = 4000, axis = 0)

            for idx, neu in enumerate(cells_to_check):
                ylims = [np.min([np.min(signal_p_og[:,neu]), np.min(signal_r_og[:,neu])]),
                        1.1*np.max([np.max(signal_p_og[:,neu]), np.max(signal_r_og[:,neu])]) ]

                f = plt.figure(figsize=(18,6))
                ax = plt.subplot(2,2,1)
                ax.plot(signal_p_og[:,neu])
                base = np.histogram(signal_p_og[:,neu], 100)
                base_p = base[1][np.argmax(base[0])]
                base_p = base_p + lowpass_p[:,neu] - np.min(lowpass_p[:,neu])   
                ax.plot(base_p, color = 'r')
                ax.set_ylim(ylims)
                ax = plt.subplot(2,2,2)
                ax.plot(signal_r_og[:,neu])
                base = np.histogram(signal_r_og[:,neu], 100)
                base_r = base[1][np.argmax(base[0])]
                base_r = base_r + lowpass_r[:,neu] - np.min(lowpass_r[:,neu])   
                ax.plot(base_r, color = 'r')
                ax.set_ylim(ylims)
                ax = plt.subplot(2,2,3)
                ax.plot(signal_p[:,neu])
                ax.set_ylim([-0.05, 1.5])
                ax = plt.subplot(2,2,4)
                ax.plot(signal_r[:,neu])
                ax.set_ylim([-0.05, 1.5])
                plt.tight_layout()
                plt.suptitle(f"{neu}/{signal_p.shape[1]} ({idx+1}/{len(cells_to_check)})")
                a = input()
                cells_to_keep[neu] = not any(a)
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
            ax.scatter(*emb_p[:,:3].T, color ='b', s= 20)
            ax.scatter(*emb_r[:,:3].T, color = 'r', s= 20)
            ax.set_title('All cells')
            ax.set_aspect('equal', adjustable='box')
            ax = plt.subplot(2,2,2, projection = '3d')
            ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 20, cmap = 'magma')
            ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 20, cmap = 'magma')
            ax.set_aspect('equal', adjustable='box')
            ax = plt.subplot(2,2,3, projection = '3d')
            ax.scatter(*emb_p_clean[:,:3].T, color ='b', s= 20)
            ax.scatter(*emb_r_clean[:,:3].T, color = 'r', s= 20)
            ax.set_title(f'Clean cells: {np.sum(~cells_to_keep)} cells removed ({100*np.sum(~cells_to_keep)/cells_to_keep.shape[0]:.2f} %)')
            ax.set_aspect('equal', adjustable='box')
            ax = plt.subplot(2,2,4, projection = '3d')
            ax.scatter(*emb_p_clean[:,:3].T, c = pos_p[:,0], s= 20, cmap = 'magma')
            ax.scatter(*emb_r_clean[:,:3].T, c = pos_r[:,0], s= 20, cmap = 'magma')
            plt.tight_layout()
            plt.suptitle(f"{mouse}: clean_traces vel: {vel_th}")
            do_remove_cells = input(f"Do you want to remove cells?: ([Y]/N)")

            if not any(do_remove_cells) or 'Y' in do_remove_cells:
                animal_p = remove_cells(animal_p, cells_to_keep)
                animal_r = remove_cells(animal_r, cells_to_keep)

                animal_p_still = remove_cells(animal_p_still, cells_to_keep)
                animal_r_still = remove_cells(animal_r_still, cells_to_keep)
            else:
                cells_to_keep = np.zeros((signal_p_og.shape[1],), dtype=bool)*True

        params["cells_to_keep"] = cells_to_keep
        with open(os.path.join(save_dir, mouse+"_"+case+"_params.pkl"), "wb") as file:
            pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
        animal_dict = {
            fname_pre: animal_p,
            fname_rot: animal_r
        }
        with open(os.path.join(save_dir, mouse+"_"+case+"_df_dict.pkl"), "wb") as file:
            pickle.dump(animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        animal_still_dict = {
            fname_pre: animal_p_still,
            fname_rot: animal_r_still
        }

        with open(os.path.join(save_dir, mouse+"_"+case+"_df_still_dict.pkl"), "wb") as file:
            pickle.dump(animal_still_dict, file, protocol=pickle.HIGHEST_PROTOCOL)



#__________________________________________________________________________
#|                                                                        |#
#|                              SAVE DIM RED                              |#
#|________________________________________________________________________|#

params = {
    'dim':3,
    'num_neigh': 120,
    'min_dist': 0.1,
    'signal_name': 'clean_traces',
}

for mouse in mice_list:
    print(f"Working on mouse: {mouse}")
    for case in ['veh', 'CNO']:
        print(f"\tCondition: {case}")
        case_dir = os.path.join(base_dir, 'processed_data', mouse+'_'+case)

        dim_red_object = dict()

        file_name =  f"{mouse}_{case}_df_dict.pkl"
        save_fig_dir = os.path.join(case_dir, 'figures')
        if not os.path.exists(save_fig_dir):
            os.mkdir(save_fig_dir)


        animal = load_pickle(case_dir,file_name)
        fnames = list(animal.keys())
        fname_pre = [fname for fname in fnames if 'lt' in fname][0]
        fname_rot = [fname for fname in fnames if 'rot' in fname][0]
        animal_p= copy.deepcopy(animal[fname_pre])
        animal_r= copy.deepcopy(animal[fname_rot])


        signal_p = get_signal(animal_p,'clean_traces')
        pos_p = get_signal(animal_p, 'position')
        mov_dir_p = get_signal(animal_p, 'mov_direction')
        trial_id_mat_p = get_signal(animal_p, 'trial_id_mat')

        signal_r = get_signal(animal_r,'clean_traces')
        pos_r = get_signal(animal_r, 'position')
        mov_dir_r = get_signal(animal_r, 'mov_direction')
        trial_id_mat_r = get_signal(animal_r, 'trial_id_mat')

        pre_rot_index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
        signal_both = np.vstack((signal_p, signal_r))


        #umap
        print("\t\tFitting umap model...", sep= '', end = '')
        model_umap = umap.UMAP(n_neighbors=params['num_neigh'], n_components =params['dim'], min_dist=params['min_dist'])
        model_umap.fit(signal_both)
        emb_both = model_umap.transform(signal_both)
        emb_p = emb_both[pre_rot_index[:,0]==0,:]
        emb_r = emb_both[pre_rot_index[:,0]==1,:]

        #%%
        fig = plt.figure()
        ax = plt.subplot(1,3,1, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, color ='b', s = 20)
        ax.scatter(*emb_r[:,:3].T, color = 'r', s = 20)
        ax = plt.subplot(1,3,2, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s = 20, cmap = 'magma')
        ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s = 20, cmap = 'magma')
        ax = plt.subplot(1,3,3, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = mov_dir_p, cmap = 'Accent', s = 20, vmin= 0, vmax = 8)
        ax.scatter(*emb_r[:,:3].T, c = mov_dir_r, cmap = 'Accent', s = 20, vmin= 0, vmax = 8)
        plt.suptitle(f"{mouse}: {params['signal_name']} - nn: {params['num_neigh']} - dim: {params['dim']}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_fig_dir,mouse+'_saved_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
        plt.savefig(os.path.join(save_fig_dir,mouse+'_saved_umap.svg'), dpi = 400,bbox_inches="tight",transparent=True)
        plt.close(fig)
        animal_p['umap'] = [emb_p[trial_id_mat_p[:,0]==animal_p["trial_id"][idx] ,:] 
                                       for idx in animal_p.index]
        animal_r['umap'] = [emb_r[trial_id_mat_r[:,0]==animal_r["trial_id"][idx] ,:] 
                                       for idx in animal_r.index]

        dim_red_object['umap'] = copy.deepcopy(model_umap)
        print("\b\b\b: Done")

        #isomap
        print("\t\tFitting isomap model...", sep= '', end = '')
        model_isomap = Isomap(n_neighbors =params['num_neigh'], n_components = signal_both.shape[1])
        model_isomap.fit(signal_both)
        emb_both = model_isomap.transform(signal_both)
        emb_p = emb_both[pre_rot_index[:,0]==0,:]
        emb_r = emb_both[pre_rot_index[:,0]==1,:]
        #%%
        fig = plt.figure()
        ax = plt.subplot(1,3,1, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, color ='b', s = 20)
        ax.scatter(*emb_r[:,:3].T, color = 'r', s = 20)
        ax = plt.subplot(1,3,2, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s = 20, cmap = 'magma')
        ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s = 20, cmap = 'magma')
        ax = plt.subplot(1,3,3, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = mov_dir_p, cmap = 'Accent', s = 20, vmin= 0, vmax = 8)
        ax.scatter(*emb_r[:,:3].T, c = mov_dir_r, cmap = 'Accent', s = 20, vmin= 0, vmax = 8)
        plt.suptitle(f"{mouse}: {params['signal_name']} - nn: {params['num_neigh']} - dim: {params['dim']}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_fig_dir,mouse+'_saved_isomap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
        plt.savefig(os.path.join(save_fig_dir,mouse+'_saved_isomap.svg'), dpi = 400,bbox_inches="tight",transparent=True)
        plt.close(fig)
        animal_p['isomap'] = [emb_p[trial_id_mat_p[:,0]==animal_p["trial_id"][idx] ,:] 
                                       for idx in animal_p.index]
        animal_r['isomap'] = [emb_r[trial_id_mat_r[:,0]==animal_r["trial_id"][idx] ,:] 
                                       for idx in animal_r.index]
        dim_red_object['isomap'] = copy.deepcopy(model_isomap)
        print("\b\b\b: Done")

        #pca
        print("\t\tFitting PCA model...", sep= '', end = '')
        model_pca = PCA(signal_both.shape[1])
        model_pca.fit(signal_both)
        emb_both = model_pca.transform(signal_both)
        emb_p = emb_both[pre_rot_index[:,0]==0,:]
        emb_r = emb_both[pre_rot_index[:,0]==1,:]
        #%%
        fig = plt.figure()
        ax = plt.subplot(1,3,1, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, color ='b', s = 20)
        ax.scatter(*emb_r[:,:3].T, color = 'r', s = 20)
        ax = plt.subplot(1,3,2, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s = 20, cmap = 'magma')
        ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s = 20, cmap = 'magma')
        ax = plt.subplot(1,3,3, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = mov_dir_p, cmap = 'Accent', s = 20, vmin= 0, vmax = 8)
        ax.scatter(*emb_r[:,:3].T, c = mov_dir_r, cmap = 'Accent', s = 20, vmin= 0, vmax = 8)
        plt.suptitle(f"{mouse}: {params['signal_name']} - nn: {params['num_neigh']} - dim: {params['dim']}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_fig_dir,mouse+'_saved_PCA.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
        plt.savefig(os.path.join(save_fig_dir,mouse+'_saved_PCA.svg'), dpi = 400,bbox_inches="tight",transparent=True)
        plt.close(fig)
        animal_p['pca'] = [emb_p[trial_id_mat_p[:,0]==animal_p["trial_id"][idx] ,:] 
                                       for idx in animal_p.index]
        animal_r['pca'] = [emb_r[trial_id_mat_r[:,0]==animal_r["trial_id"][idx] ,:] 
                                       for idx in animal_r.index]
        dim_red_object['pca'] = copy.deepcopy(model_pca)
        print("\b\b\b: Done")

        animal_updated = {
            fname_pre: copy.deepcopy(animal_p),
            fname_rot: copy.deepcopy(animal_r)
        }
        with open(os.path.join(case_dir,file_name), "wb") as file:
            pickle.dump(animal_updated, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(case_dir, f"{mouse}_{case}_dim_red_object.pkl"), "wb") as file:
            pickle.dump(dim_red_object, file, protocol=pickle.HIGHEST_PROTOCOL)

#_________________________________________________________________________
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


def get_centroids(cloud_A, cloud_B, label_A, label_B, dir_A = None, dir_B = None, num_cent = 20):

    dims = cloud_A.shape[1]
    if label_A.ndim>1:
        label_A = label_A[:,0]
    if label_B.ndim>1:
        label_B = label_B[:,0]
    #compute label max and min to divide into centroids
    total_label = np.hstack((label_A, label_B))
    label_lims = np.array([(np.percentile(total_label,5), np.percentile(total_label,95))]).T[:,0] 
    #find centroid size
    cent_size = (label_lims[1] - label_lims[0]) / (num_cent)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    cent_edges = np.column_stack((np.linspace(label_lims[0],label_lims[0]+cent_size*(num_cent),num_cent),
                                np.linspace(label_lims[0],label_lims[0]+cent_size*(num_cent),num_cent)+cent_size))


    if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
        cent_A = np.zeros((num_cent,dims))
        cent_B = np.zeros((num_cent,dims))
        cent_label = np.mean(cent_edges,axis=1)

        num_cent_A = np.zeros((num_cent,))
        num_cent_B = np.zeros((num_cent,))
        for c in range(num_cent):
            points_A = cloud_A[np.logical_and(label_A >= cent_edges[c,0], label_A<cent_edges[c,1]),:]
            cent_A[c,:] = np.median(points_A, axis=0)
            num_cent_A[c] = points_A.shape[0]
            
            points_B = cloud_B[np.logical_and(label_B >= cent_edges[c,0], label_B<cent_edges[c,1]),:]
            cent_B[c,:] = np.median(points_B, axis=0)
            num_cent_B[c] = points_B.shape[0]
    else:
        cloud_A_left = copy.deepcopy(cloud_A[dir_A==-1,:])
        label_A_left = copy.deepcopy(label_A[dir_A==-1])
        cloud_A_right = copy.deepcopy(cloud_A[dir_A==1,:])
        label_A_right = copy.deepcopy(label_A[dir_A==1])
        
        cloud_B_left = copy.deepcopy(cloud_B[dir_B==-1,:])
        label_B_left = copy.deepcopy(label_B[dir_B==-1])
        cloud_B_right = copy.deepcopy(cloud_B[dir_B==1,:])
        label_B_right = copy.deepcopy(label_B[dir_B==1])
        
        cent_A = np.zeros((2*num_cent,dims))
        cent_B = np.zeros((2*num_cent,dims))
        num_cent_A = np.zeros((2*num_cent,))
        num_cent_B = np.zeros((2*num_cent,))
        
        cent_dir = np.zeros((2*num_cent, ))
        cent_label = np.tile(np.mean(cent_edges,axis=1),(2,1)).T.reshape(-1,1)
        for c in range(num_cent):
            points_A_left = cloud_A_left[np.logical_and(label_A_left >= cent_edges[c,0], label_A_left<cent_edges[c,1]),:]
            cent_A[2*c,:] = np.median(points_A_left, axis=0)
            num_cent_A[2*c] = points_A_left.shape[0]
            points_A_right = cloud_A_right[np.logical_and(label_A_right >= cent_edges[c,0], label_A_right<cent_edges[c,1]),:]
            cent_A[2*c+1,:] = np.median(points_A_right, axis=0)
            num_cent_A[2*c+1] = points_A_right.shape[0]

            points_B_left = cloud_B_left[np.logical_and(label_B_left >= cent_edges[c,0], label_B_left<cent_edges[c,1]),:]
            cent_B[2*c,:] = np.median(points_B_left, axis=0)
            num_cent_B[2*c] = points_B_left.shape[0]
            points_B_right = cloud_B_right[np.logical_and(label_B_right >= cent_edges[c,0], label_B_right<cent_edges[c,1]),:]
            cent_B[2*c+1,:] = np.median(points_B_right, axis=0)
            num_cent_B[2*c+1] = points_B_right.shape[0]

            cent_dir[2*c] = -1
            cent_dir[2*c+1] = 1

    del_cent_nan = np.all(np.isnan(cent_A), axis= 1)+ np.all(np.isnan(cent_B), axis= 1)
    del_cent_num = (num_cent_A<15) + (num_cent_B<15)
    del_cent = del_cent_nan + del_cent_num
    
    cent_A = np.delete(cent_A, del_cent, 0)
    cent_B = np.delete(cent_B, del_cent, 0)

    cent_label = np.delete(cent_label, del_cent, 0)

    if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
        return cent_A, cent_B, cent_label
    else:
        cent_dir = np.delete(cent_dir, del_cent, 0)
        return cent_A, cent_B, cent_label, cent_dir


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


def create_plane_from_equation(norm_vector, cloud_center, xlims, ylims):
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -cloud_center.dot(norm_vector)
    # create x,y
    xx, yy = np.meshgrid(np.linspace(xlims[0], xlims[1], 10), 
                        np.linspace(ylims[0], ylims[1], 10))
    # calculate corresponding z
    zz = (-norm_vector[0] * xx - norm_vector[1] * yy - d) * 1. /norm_vector[2]

    return xx, yy, zz


def find_rotation_align_vectors(a,b):
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)

    sscp = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]], [-v[1],v[0],0]])
    sscp2 = np.matmul(sscp,sscp)
    R = np.eye(3)+sscp+(sscp2*(1-c)/s**2)
    return R


def apply_rotation_to_cloud(point_cloud, rotation,center_of_rotation):
    return np.dot(point_cloud-center_of_rotation, rotation) + center_of_rotation
    

def check_norm_vector_direction(norm_vector, cloud_center, goal_point):
    og_dir = cloud_center+norm_vector
    op_dir = cloud_center+(-1*norm_vector)

    og_distance = np.linalg.norm(og_dir-goal_point)
    op_distance = np.linalg.norm(op_dir-goal_point)
    if og_distance<op_distance:
        return norm_vector
    else:
        return -norm_vector


def find_rotation(data_A, data_B, v, angles):
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


def create_color_mov_dir(mov_dir):
    mov_dir_color = np.zeros((mov_dir.shape[0],3))
    for point in range(mov_dir.shape[0]):
        if mov_dir[point]==0:
            mov_dir_color[point] = [14/255,14/255,143/255]
        elif mov_dir[point]==-1:
            mov_dir_color[point] = [12/255,136/255,249/255]
        elif mov_dir[point]==1:
            mov_dir_color[point] = [17/255,219/255,224/255]
    return mov_dir_color


angles = np.linspace(-np.pi,np.pi,200)


mov_dir_p = get_signal(animal_p, 'mov_direction')
trial_id_mat_p = get_signal(animal_p, 'trial_id_mat')
mov_dir_p[mov_dir_p==-1] = -2
mov_dir_p[mov_dir_p==1] = -1
mov_dir_p[mov_dir_p==-2] = 1

animal_p['mov_direction'] = [mov_dir_p[trial_id_mat_p[:,0]==animal_p["trial_id"][idx]]
                               for idx in animal_p.index]


animal_updated = {
    fname_pre: copy.deepcopy(animal_p),
    fname_rot: copy.deepcopy(animal_r)
}
with open(os.path.join(case_dir,file_name), "wb") as file:
    pickle.dump(animal_updated, file, protocol=pickle.HIGHEST_PROTOCOL)


for mouse in mice_list:
    print(f"Working on mouse: {mouse}")
    for case in ['veh', 'CNO']:
        print(f"\tCondition: {case}")
        file_name =  f"{mouse}_{case}_df_dict.pkl"
        case_dir = os.path.join(base_dir, 'processed_data', mouse+'_'+case)
        save_dir = os.path.join(base_dir, 'rotation', mouse+'_'+case)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        rotation_dict = {}
        animal = load_pickle(case_dir,file_name)
        fnames = list(animal.keys())
        fname_pre = [fname for fname in fnames if 'lt' in fname][0]
        fname_rot = [fname for fname in fnames if 'rot' in fname][0]
        animal_p= copy.deepcopy(animal[fname_pre])
        animal_r= copy.deepcopy(animal[fname_rot])

        pos_p = get_signal(animal_p, 'position')
        mov_dir_p = get_signal(animal_p, 'mov_direction')
        pos_r = get_signal(animal_r, 'position')
        mov_dir_r = get_signal(animal_r, 'mov_direction')

        for emb in ['umap', 'isomap','pca']:

            emb_p = get_signal(animal_p,emb)[:,:3]
            emb_r = get_signal(animal_r,emb)[:,:3]

            D_pre = pairwise_distances(emb_p)
            noise_idx_pre = filter_noisy_outliers(emb_p,D_pre)
            cemb_p = emb_p[~noise_idx_pre,:]
            cpos_p = pos_p[~noise_idx_pre,:]
            cmov_dir_p = mov_dir_p[~noise_idx_pre]

            D_rot = pairwise_distances(emb_r)
            noise_idx_rot = filter_noisy_outliers(emb_r,D_rot)
            cemb_r = emb_r[~noise_idx_rot,:]
            cpos_r = pos_r[~noise_idx_rot,:]
            cmov_dir_r = mov_dir_r[~noise_idx_rot]

            #compute centroids
            cent_p, cent_r, cent_pos, cent_dir = get_centroids(cemb_p, cemb_r, cpos_p[:,0], cpos_r[:,0], 
                                                    cmov_dir_p, cmov_dir_r, num_centroids=40)   
            #parametrize planes
            norm_vector_p, cloud_center_p = parametrize_plane(cemb_p)
            norm_vector_r, cloud_center_r = parametrize_plane(cemb_r)

            #project into planes
            plane_emb_p = project_onto_plane(cemb_p, norm_vector_p, cloud_center_p)
            plane_emb_r = project_onto_plane(cemb_r, norm_vector_r, cloud_center_r)


            plane_cent_p, plane_cent_r, plane_cent_pos, plane_cent_dir = get_centroids(plane_emb_p, plane_emb_r, 
                                                                                            cpos_p[:,0], cpos_r[:,0], 
                                                                                            cmov_dir_p, cmov_dir_r, num_centroids=40) 
            
            #align them
            align_angle, align_mat = align_vectors(norm_vector_p, cloud_center_p, norm_vector_r, cloud_center_r)

            aligned_emb_r =  apply_rotation_to_cloud(cemb_r, align_mat, cloud_center_r)
            aligned_plane_emb_r =  apply_rotation_to_cloud(plane_emb_r, align_mat, cloud_center_r)

            aligned_cent_r =  apply_rotation_to_cloud(cent_r, align_mat, cloud_center_r)
            aligned_plane_cent_r =  apply_rotation_to_cloud(plane_cent_r, align_mat, cloud_center_r)

            #compute angle of rotation
            angles = np.linspace(-np.pi,np.pi,200)
            error = find_rotation(plane_cent_p, plane_cent_r, -norm_vector_p)
            norm_error = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
            signed_rotation_angle = angles[np.argmin(norm_error)]*180/np.pi
            rotation_angle = np.abs(signed_rotation_angle)
            print(f"\t{mouse} {case}: {signed_rotation_angle:2f} degrees")

            rotated_aligned_cent_r = rotate_cloud_around_axis(aligned_cent_r, (np.pi/180)*signed_rotation_angle,norm_vector_p)
            rotated_aligned_plane_cent_r = rotate_cloud_around_axis(aligned_plane_cent_r, (np.pi/180)*signed_rotation_angle,norm_vector_p)
            rotated_aligned_emb_r = rotate_cloud_around_axis(aligned_emb_r, (np.pi/180)*signed_rotation_angle,norm_vector_p)
            rotated_aligned_plane_emb_r = rotate_cloud_around_axis(aligned_plane_emb_r, (np.pi/180)*signed_rotation_angle,norm_vector_p)

            rotated_cent_r = rotate_cloud_around_axis(cent_r, (np.pi/180)*signed_rotation_angle,norm_vector_p)

            fig = plot_rotation(cemb_p, cemb_r, cpos_p, cpos_r, cmov_dir_p, cmov_dir_r, 
                        cent_p, cent_r, cent_pos, plane_cent_p, plane_cent_r, 
                        aligned_plane_cent_r, rotated_aligned_plane_cent_r, angles, error, signed_rotation_angle)
            plt.suptitle(f"{mouse} {case}")
            plt.savefig(os.path.join(save_dir,f'{mouse}_{case}_{emb}_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_{case}_{emb}_rotation_plot.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            rotation_dict[emb] = {
                #initial data
                'emb_pre': cemb_p,
                'pos_pre': cpos_p,
                'dir_pre': cmov_dir_p,

                'emb_rot': cemb_r,
                'pos_rot': cpos_r,
                'dir_rot': cmov_dir_r,
                #centroids
                'cent_pre': cent_p,
                'cent_rot': cent_r,
                'cent_pos': cent_pos,
                'cent_dir': cent_dir,

                #project into plane
                'norm_vec_pre': norm_vector_p,
                'cloud_center_pre': cloud_center_p,
                'plane_emb_p': plane_emb_p,

                'norm_vec_rot': norm_vector_r,
                'cloud_center_rot': cloud_center_r,
                'plane_emb_rot': plane_emb_r,

                #plane centroids
                'plane_cent_pre': plane_cent_p,
                'plane_cent_rot': plane_cent_r,
                'plane_cent_pos': plane_cent_pos,
                'plane_cent_dir': plane_cent_dir,

                #align planes
                'align_angle': align_angle,
                'align_mat': align_mat,

                'aligned_emb_rot': aligned_emb_r,
                'aligned_plane_emb_rot': aligned_plane_emb_r,
                'aligned_cent_rot': aligned_cent_r,
                'aligned_plane_cent_rot': aligned_plane_cent_r,

                #compute angle of rotation
                'angles': angles,
                'error': error,
                'norm_error': norm_error,
                'signed_rotation_angle': signed_rotation_angle,
                'rotation_angle': rotation_angle,

                #rotate post session
                'rotated_cent_rot': rotated_cent_r,
                'rotated_aligned_cent_rot': rotated_aligned_cent_r,
                'rotated_aligned_plane_cent_rot': rotated_aligned_plane_cent_r,
                'rotated_aligned_emb_rot': rotated_aligned_emb_r,
                'rotated_aligned_plane_emb_rot': rotated_aligned_plane_emb_r
            }


            with open(os.path.join(save_dir, f"{mouse}_{case}_rotation_dict.pkl"), "wb") as file:
                pickle.dump(rotation_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE DISTANCES                            |#
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
    print(f"Working on mouse: {mouse}")

    for case in ['veh', 'CNO']:

        print(f"\tCondition: {case}")
        data_dir = os.path.join(base_dir, 'rotation', mouse+'_'+case)
        rotation_dict = load_pickle(data_dir, f"{mouse}_{case}_rotation_dict.pkl")

        save_dir = os.path.join(base_dir, 'distance', mouse+'_'+case)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        distance_dict = dict()

        for emb_name in ['umap','isomap','pca']:

            cent_pre = rotation_dict[emb_name]['cent_pre']
            cent_rot = rotation_dict[emb_name]['cent_rot']
            cent_pos = rotation_dict[emb_name]['cent_pos']
            cent_dir = rotation_dict[emb_name]['cent_dir']

            inter_dist = np.linalg.norm(cent_pre.mean(axis=0)-cent_rot.mean(axis=0))
            intra_dist_pre = np.percentile(pairwise_distances(cent_pre),95)/2
            intra_dist_rot = np.percentile(pairwise_distances(cent_rot),95)/2
            remap_dist = inter_dist/np.mean((intra_dist_pre, intra_dist_rot))

            plane_cent_pre = rotation_dict[emb_name]['plane_cent_pre']
            plane_cent_rot = rotation_dict[emb_name]['plane_cent_rot']
            norm_vector_pre = rotation_dict[emb_name]['norm_vec_pre']
            plane_cent_pos = rotation_dict[emb_name]['plane_cent_pos']
            plane_cent_dir = rotation_dict[emb_name]['plane_cent_dir']
            norm_vector_rot = rotation_dict[emb_name]['norm_vec_rot']


            plane_inter_dist = np.linalg.norm(plane_cent_pre.mean(axis=0)-plane_cent_rot.mean(axis=0))
            ellipse_pre_params, ellipse_pre_long_axis, ellipse_pre_short_axis, ellipse_pre_fit, ellipse_pre_fit_3D = fit_ellipse(plane_cent_pre, norm_vector_pre)
            ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(ellipse_pre_long_axis+ellipse_pre_short_axis)**2)

            ellipse_rot_params, ellipse_rot_long_axis, ellipse_rot_short_axis, ellipse_rot_fit, ellipse_rot_fit_3D = fit_ellipse(plane_cent_rot, norm_vector_rot)
            ellipse_rot_perimeter = 2*np.pi*np.sqrt(0.5*(ellipse_rot_long_axis+ellipse_rot_short_axis)**2)

            plane_remap_dist = plane_inter_dist/np.mean((ellipse_pre_perimeter, ellipse_rot_perimeter))

            print(f"\t{mouse} {case} {emb_name}: {remap_dist:.2f} remap dist | {plane_remap_dist:.2f} remap dist plane")



            fig = plot_distance(cent_pre,cent_rot,cent_pos,cent_dir,
                    plane_cent_pre,plane_cent_rot, plane_cent_pos, plane_cent_dir,
                    ellipse_pre_fit_3D, ellipse_rot_fit_3D)
            plt.suptitle(f"{mouse} {emb_name}")
            plt.savefig(os.path.join(save_dir,f'{mouse}_{case}_{emb_name}_distance_plot.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_{case}_{emb_name}_distance_plot.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            distance_dict[emb_name] = {

                #cent
                'cent_pre': cent_pre,
                'cent_rot': cent_rot,
                'cent_pos': cent_pos,
                'noise_pre': cent_dir,
                #distance og
                'inter_dist': inter_dist,
                'intra_dist_pre': intra_dist_pre,
                'intra_dist_rot': intra_dist_rot,
                'remap_dist': remap_dist,

                #plane
                'plane_cent_pre': cent_pre,
                'norm_vector_pre': norm_vector_pre,
                'plane_cent_rot': plane_cent_rot,
                'norm_vector_rot': norm_vector_rot,
                'plane_cent_pos': plane_cent_pos,
                'plane_cent_dir': plane_cent_dir,

                #ellipse
                'ellipse_pre_params': ellipse_pre_params,
                'ellipse_pre_long_axis': ellipse_pre_long_axis,
                'ellipse_pre_short_axis': ellipse_pre_short_axis,
                'ellipse_pre_fit': ellipse_pre_fit,
                'ellipse_pre_fit_3D': ellipse_pre_fit_3D,

                'ellipse_rot_params': ellipse_rot_params,
                'ellipse_rot_long_axis': ellipse_rot_long_axis,
                'ellipse_rot_short_axis': ellipse_rot_short_axis,
                'ellipse_rot_fit': ellipse_rot_fit,
                'ellipse_rot_fit_3D': ellipse_rot_fit_3D,

                #distance ellipse
                'plane_inter_dist': plane_inter_dist,
                'ellipse_pre_perimeter': ellipse_pre_perimeter,
                'ellipse_rot_perimeter': ellipse_rot_perimeter,
                'plane_remap_dist': plane_remap_dist,
            }

            with open(os.path.join(save_dir, f"{mouse}_{case}_distance_dict.pkl"), "wb") as file:
                pickle.dump(distance_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                               PLOT EMB                                 |#
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

view_init_dict = {
    'CalbCharly1_veh': [-60,-130],
    'CalbCharly1_CNO': [-160, -130],

    'CalbCharly2_veh': [-109,-170],
    'CalbCharly2_CNO': [-48, 6]
}
for mouse in mice_list:

    for case in ['veh','CNO']:

        data_dir = os.path.join(base_dir, 'DREADDs', 'rotation', 'ChRNA7', mouse+'_'+case)

        rotation_dict = load_pickle(data_dir, f"{mouse}_{case}_rotation_dict.pkl")

        noise_idx_pre = rotation_dict['umap']['noise_idx_pre']
        emb_p = rotation_dict['umap']['emb_p'][~noise_idx_pre,:]
        pos_p = rotation_dict['umap']['pos_p'][~noise_idx_pre,:]
        mov_dir_p = rotation_dict['umap']['mov_dir_p'][~noise_idx_pre]

        noise_idx_rot = rotation_dict['umap']['noise_idx_rot']
        emb_r = rotation_dict['umap']['emb_r'][~noise_idx_rot,:]
        pos_r = rotation_dict['umap']['pos_r'][~noise_idx_rot,:]
        mov_dir_r = rotation_dict['umap']['mov_dir_r'][~noise_idx_rot]

        mov_dir_color_p = np.zeros((mov_dir_p.shape[0],3))
        for point in range(mov_dir_p.shape[0]):
            if mov_dir_p[point]== 0:
                mov_dir_color_p[point] = [14/255,14/255,143/255]
            elif mov_dir_p[point]== -1:
                mov_dir_color_p[point] = [12/255,136/255,249/255]
            elif mov_dir_p[point]== 1:
                mov_dir_color_p[point] = [17/255,219/255,224/255]


        mov_dir_color_r = np.zeros((mov_dir_r.shape[0],3))
        for point in range(mov_dir_r.shape[0]):
            if mov_dir_r[point]==0:
                mov_dir_color_r[point] = [14/255,14/255,143/255]
            elif mov_dir_r[point]== -1:
                mov_dir_color_r[point] = [12/255,136/255,249/255]
            elif mov_dir_r[point]== 1:
                mov_dir_color_r[point] = [17/255,219/255,224/255]

        if mouse+'_'+case in list(view_init_dict.keys()):
            view_init_values = view_init_dict[mouse+'_'+case]
        else:
            view_init_values = None
        fig = plt.figure(figsize=((15,5)))
        ax = plt.subplot(1,3,1, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = 'b',s = 10)
        ax.scatter(*emb_r[:,:3].T, c = 'r',s = 10)
        personalize_ax(ax,view_init_values)
        ax = plt.subplot(1,3,2, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, color = mov_dir_color_p,s = 10)
        ax.scatter(*emb_r[:,:3].T, color = mov_dir_color_r,s = 10)
        personalize_ax(ax,view_init_values)
        ax = plt.subplot(1,3,3, projection = '3d')
        ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], cmap='inferno',s = 10)
        ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], cmap='inferno',s = 10)
        personalize_ax(ax,view_init_values)
        fig.suptitle(mouse)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir,f"{mouse}_{case}_umap_emb.svg"), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(data_dir,f"{mouse}_{case}_umap_emb.png"), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT ROTATION                               |#
#|________________________________________________________________________|#

case_list = list()
mouse_list = list()
rotation_list = list()

for mouse in mice_list:
    for case in ['veh','CNO']:
        data_dir = os.path.join(base_dir, 'rotation', mouse+'_'+case)
        rot_error_dict = load_pickle(data_dir, f"{mouse}_{case}_rotation_dict.pkl")
        case_list.append(case)
        mouse_list.append(mouse)
        rotation_list.append(rot_error_dict['umap']['rotation_angle'])

rotation_pd = pd.DataFrame(data={'rotation_angle': rotation_list,
                            'case': case_list,
                            'mouse': mouse_list})


save_dir = os.path.join(base_dir, 'rotation')

fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='case', y='rotation_angle',data=rotation_pd, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='case', y='rotation_angle', data=rotation_pd, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='case', y= 'rotation_angle', data=rotation_pd, units = 'mouse', ax = ax, estimator = None, color = ".7", markers = True)
ax.set_ylabel('Angle Rotation')
ax.set_ylim([0,190])
plt.tight_layout()
plt.savefig(os.path.join(save_dir,f'Calb_rotation_boxplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,f'Calb_rotation_boxplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)


from scipy import stats
vehAngle = rotation_pd.loc[rotation_pd['case']=='veh']['rotation_angle']
CNOAngle = rotation_pd.loc[rotation_pd['case']=='CNO']['rotation_angle']

vehAngle_norm = stats.shapiro(vehAngle)
CNOAngle_norm = stats.shapiro(CNOAngle)


if vehAngle_norm.pvalue<=0.05 or CNOAngle_norm.pvalue<=0.05:
    print('vehAngle vs CNOAngle:',stats.ks_2samp(vehAngle, CNOAngle))
else:
    print('vehAngle vs CNOAngle:', stats.ttest_rel(vehAngle, CNOAngle))

from statsmodels.formula.api import ols
import statsmodels.api as sm
#perform two-way ANOVA
model = ols('rotation_angle ~ C(case) + C(mouse) + C(case):C(mouse)', data=rotation_pd).fit()
sm.stats.anova_lm(model, typ=2)

#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT DISTANCE                              |#
#|________________________________________________________________________|#

case_list = list()
mouse_list = list()
remap_dist_list = list()

for mouse in mice_list:
    for case in ['veh','CNO']:
        data_dir = os.path.join(base_dir,'distance', mouse+'_'+case)
        rot_error_dict = load_pickle(data_dir, f"{mouse}_{case}_distance_dict.pkl")
        case_list.append(case)
        mouse_list.append(mouse)
        remap_dist_list.append(rot_error_dict['umap']['remap_dist'])

rotation_pd = pd.DataFrame(data={'remap_dist': remap_dist_list,
                            'case': case_list,
                            'mouse': mouse_list})
save_dir = os.path.join(base_dir,'distance')


fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='case', y='remap_dist',data=rotation_pd, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='case', y='remap_dist', data=rotation_pd, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='case', y= 'remap_dist', data=rotation_pd, units = 'mouse', ax = ax, estimator = None, color = ".7", markers = True)
ax.set_ylabel('Remap distance')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,f'Calb_distance_barplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,f'Calb_distance_barplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)

fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='case', y='remap_dist',data=rotation_pd, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='case', y='remap_dist', data=rotation_pd, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='case', y= 'remap_dist', data=rotation_pd, units = 'mouse', ax = ax, estimator = None, color = ".7", markers = True)
ax.set_ylabel('Remap distance')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,f'Calb_distance_boxplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,f'Calb_distance_boxplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)

#__________________________________________________________________________
#|                                                                        |#
#|                           ALIGNMENT ANGLE                              |#
#|________________________________________________________________________|#

case_list = list()
mouse_list = list()
rotation_list = list()

for mouse in mice_list:
    for case in ['veh','CNO']:
        data_dir = os.path.join(base_dir, 'rotation', mouse+'_'+case)
        rot_error_dict = load_pickle(data_dir, f"{mouse}_{case}_rotation_dict.pkl")
        case_list.append(case)
        mouse_list.append(mouse)
        rotation_list.append(rot_error_dict['umap']['align_angle'])

rotation_pd = pd.DataFrame(data={'angle_of_alignment': rotation_list,
                            'case': case_list,
                            'mouse': mouse_list})


save_dir = os.path.join(base_dir, 'rotation')
fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.boxplot(x='case', y='angle_of_alignment',data=rotation_pd, palette = palette, linewidth = 1, width= .5, ax = ax)
sns.stripplot(x='case', y='angle_of_alignment', data=rotation_pd, dodge=True, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='case', y= 'angle_of_alignment', data=rotation_pd, units = 'mouse', ax = ax, estimator = None, color = ".7", markers = True)
ax.set_ylabel('Angle Alignment')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,f'Calb_alignment_boxplot.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,f'Calb_alignment_boxplot.svg'), dpi = 400,bbox_inches="tight",transparent=True)









#######################################################################################################################################################

#__________________________________________________________________________
#|                                                                        |#
#|                        COMPUTE ROTATION OLD                            |#
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
        input_A_left = copy.deepcopy(input_A[dir_A==-1,:])
        label_A_left = copy.deepcopy(label_A[dir_A==-1])
        input_A_right = copy.deepcopy(input_A[dir_A==1,:])
        label_A_right = copy.deepcopy(label_A[dir_A==1])
        
        input_B_left = copy.deepcopy(input_B[dir_B==-1,:])
        label_B_left = copy.deepcopy(label_B[dir_B==-1])
        input_B_right = copy.deepcopy(input_B[dir_B==1,:])
        label_B_right = copy.deepcopy(label_B[dir_B==1])
        
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


for mouse in mice_list:
    print(f"Working on mouse: {mouse}")
    for case in ['veh', 'CNO']:
        print(f"\tCondition: {case}")

        file_name =  f"{mouse}_{case}_df_dict.pkl"
        case_dir = os.path.join(base_dir, 'processed_data', mouse+'_'+case)
        save_dir = os.path.join(base_dir, 'rotation', mouse+'_'+case)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        rot_error_dict = dict()


        animal = load_pickle(case_dir,file_name)
        fnames = list(animal.keys())
        fname_pre = [fname for fname in fnames if 'lt' in fname][0]
        fname_rot = [fname for fname in fnames if 'rot' in fname][0]
        animal_p= copy.deepcopy(animal[fname_pre])
        animal_r= copy.deepcopy(animal[fname_rot])


        emb_p = get_signal(animal_p,'umap')
        pos_p = get_signal(animal_p, 'position')
        mov_dir_p = get_signal(animal_p, 'mov_direction')

        emb_r = get_signal(animal_r,'umap')
        pos_r = get_signal(animal_r, 'position')
        mov_dir_r = get_signal(animal_r, 'mov_direction')

        D_pre = pairwise_distances(emb_p)
        noise_idx_pre = filter_noisy_outliers(emb_p,D_pre)
        cemb_p = emb_p[~noise_idx_pre,:]
        cpos_p = pos_p[~noise_idx_pre,:]
        cmov_dir_p = mov_dir_p[~noise_idx_pre]

        D_rot = pairwise_distances(emb_r)
        noise_idx_rot = filter_noisy_outliers(emb_r,D_rot)
        cemb_r = emb_r[~noise_idx_rot,:]
        cpos_r = pos_r[~noise_idx_rot,:]
        cmov_dir_r = mov_dir_r[~noise_idx_rot]


        #compute centroids
        cent_pre, cent_rot = get_centroids(cemb_p, cemb_r, cpos_p[:,0], cpos_r[:,0], 
                                                cmov_dir_p, cmov_dir_r, ndims = 3, nCentroids=30)   
        #find axis of rotatio                                                
        mid_pre = np.median(cemb_p, axis=0).reshape(-1,1)
        mid_rot = np.median(cemb_r, axis=0).reshape(-1,1)
        norm_vector =  mid_pre - mid_rot
        norm_vector = norm_vector/np.linalg.norm(norm_vector)
        k = np.dot(np.median(cemb_p, axis=0), norm_vector)

        angles = np.linspace(-np.pi,np.pi,200)
        error = find_rotation(cent_pre-mid_pre.T, cent_rot-mid_rot.T, norm_vector)
        norm_error = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
        rot_angle = np.abs(angles[np.argmin(norm_error)])*180/np.pi
        print(f"\t{'umap'}: {rot_angle:2f} degrees")

        plt.figure()
        ax = plt.subplot(2,2,1, projection = '3d')
        ax.scatter(*cemb_p[:,:3].T, color ='b', s=10)
        ax.scatter(*cemb_r[:,:3].T, color = 'r', s=10)
        ax = plt.subplot(2,2,3, projection = '3d')
        ax.scatter(*cemb_p[:,:3].T, c = cpos_p[:,0], s=10, cmap = 'magma')
        ax.scatter(*cemb_r[:,:3].T, c = cpos_r[:,0], s=10, cmap = 'magma')
        ax = plt.subplot(1,2,2, projection = '3d')
        ax.scatter(*cent_pre[:,:3].T, color ='b', s=10)
        ax.scatter(*cent_rot[:,:3].T, color = 'r', s=10)
        plt.savefig(os.path.join(save_dir,mouse+'_rotation_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
        plt.savefig(os.path.join(save_dir,mouse+'_rotation_umap.svg'), dpi = 400,bbox_inches="tight",transparent=True)

        rot_error_dict['umap'] = {
            'emb_p': emb_p,
            'emb_r': emb_r,
            'pos_p': pos_p,
            'pos_r': pos_r,
            'mov_dir_p': mov_dir_p,
            'mov_dir_r': mov_dir_r,
            'noise_idx_pre': noise_idx_pre,
            'noise_idx_rot': noise_idx_rot,
            'cent_pre': cent_pre,
            'cent_rot': cent_rot,
            'mid_pre': mid_pre,
            'mid_rot': mid_rot,
            'norm_vector': norm_vector,
            'angles': angles,
            'error': error,
            'norm_error': norm_error,
            'rot_angle': rot_angle
        }

        with open(os.path.join(save_dir, f"{mouse}_{case}_rotation_dict.pkl"), "wb") as file:
            pickle.dump(rot_error_dict, file, protocol=pickle.HIGHEST_PROTOCOL)






#__________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE DISTANCES                            |#
#|________________________________________________________________________|#

for mouse in mice_list:
    print(f"Working on mouse: {mouse}")
    for case in ['veh', 'CNO']:
        print(f"\tCondition: {case}")

        data_dir = os.path.join(base_dir, 'rotation', mouse+'_'+case)
        rot_error_dict = load_pickle(data_dir, f"{mouse}_{case}_rotation_dict.pkl")
        save_dir = os.path.join(base_dir, 'distance', mouse+'_'+case)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        remap_dist_dict = dict()
        
        cent_pre = rot_error_dict['umap']['cent_pre']
        cent_rot = rot_error_dict['umap']['cent_rot']


        inter_dist = np.mean(pairwise_distances(cent_pre, cent_rot))
        intra_pre = np.percentile(pairwise_distances(cent_pre),95)
        intra_rot = np.percentile(pairwise_distances(cent_rot),95)
        remap_dist = inter_dist/np.max((intra_pre, intra_rot))
        print(f"\t{'umap'}: {remap_dist:2f}")


        remap_dist_dict['umap'] = {
            'cent_pre': cent_pre,
            'cent_rot': cent_rot,
            'inter_dist': inter_dist,
            'intra_pre': intra_pre,
            'intra_rot': intra_rot,
            'remap_dist': remap_dist,

        }

        with open(os.path.join(save_dir, f"{mouse}_{case}_distance_dict.pkl"), "wb") as file:
            pickle.dump(remap_dist_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
