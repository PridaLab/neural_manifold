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




signal_name = 'clean_traces'
n_neigh = 120
dim = 3
min_dist = 0.1
iters = 100

data_dir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/paper/Fig3/data/'
celltype_dir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/paper/Fig3/place_cells/'
save_dir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/paper/Fig3/delete_cells/'

######################################################################################################################################
mouse = 'GC2'
miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    sub_save_dir = os.path.join(save_dir, mouse)
    if not os.path.exists(sub_save_dir):
        os.mkdir(sub_save_dir)

    animal_dict = load_pickle(file_path,file_name)
    fnames = list(animal_dict.keys())
    animal_p= copy.deepcopy(animal_dict[fnames[0]])
    animal_r= copy.deepcopy(animal_dict[fnames[1]])

    signal_p = copy.deepcopy(np.concatenate(animal_p[signal_name].values, axis=0))
    pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
    index_mat_p = copy.deepcopy(np.concatenate(animal_p['index_mat'].values, axis=0))
    dir_mat_p = copy.deepcopy(np.concatenate(animal_p['dir_mat'].values, axis=0))
    vfeat_p = np.concatenate((pos_p[:,0].reshape(-1,1),dir_mat_p),axis=1)

    signal_r = copy.deepcopy(np.concatenate(animal_r[signal_name].values, axis=0))
    pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
    index_mat_r = copy.deepcopy(np.concatenate(animal_p['index_mat'].values, axis=0))
    dir_mat_r = copy.deepcopy(np.concatenate(animal_r['dir_mat'].values, axis=0))
    vfeat_r = np.concatenate((pos_r[:,0].reshape(-1,1),dir_mat_r),axis=1)

    #%%all data
    index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
    concat_signal = np.vstack((signal_p, signal_r))

    cellType_path = os.path.join(celltype_dir, mouse)
    cellType_name = mouse+'_cellType.npy'
    cellType = np.load(os.path.join(cellType_path,cellType_name))

    deep_cells = np.where(np.logical_and(cellType<4,cellType>0))[0]

    rot_error_GC2 = np.zeros((100, iters, deep_cells.shape[0]))
    rot_angle_GC2 = np.zeros((iters, deep_cells.shape[0]))
    SI_val = np.zeros((2,iters,deep_cells.shape[0]))

    for it in range(iters):
        new_order = np.random.permutation(concat_signal.shape[1])
        sconcat_signal = copy.deepcopy(concat_signal[:, new_order])
        scellType = cellType[new_order]
        sdeep_cells = np.where(np.logical_and(scellType<4,scellType>0))[0]
        for idx in range(sdeep_cells.shape[0]):
            temp_concat_signal = copy.deepcopy(sconcat_signal)
            temp_concat_signal = np.delete(temp_concat_signal, sdeep_cells[:idx+1], axis = 1)

            model = umap.UMAP(n_neighbors =n_neigh, n_components =dim, min_dist=min_dist)
            model.fit(temp_concat_signal)
            concat_emb = model.transform(temp_concat_signal)
            emb_p = concat_emb[index[:,0]==0,:]
            emb_r = concat_emb[index[:,0]==1,:]

            D_p = pairwise_distances(emb_p)
            noiseIdx_p = filter_noisy_outliers(emb_p,D_p)
            max_dist = np.nanmax(D_p)
            cemb_p = emb_p[~noiseIdx_p,:]
            cpos_p = pos_p[~noiseIdx_p,:]
            cdir_mat_p = dir_mat_p[~noiseIdx_p]

            D_r = pairwise_distances(emb_r)
            noiseIdx_r = filter_noisy_outliers(emb_r,D_r)
            max_dist = np.nanmax(D_r)
            cemb_r = emb_r[~noiseIdx_r,:]
            cpos_r = pos_r[~noiseIdx_r,:]
            cdir_mat_r = dir_mat_r[~noiseIdx_r]

            #compute centroids
            cent_p, cent_r = get_centroids(cemb_p, cemb_r, cpos_p[:,0], cpos_r[:,0], 
                                                            cdir_mat_p, cdir_mat_r, ndims = 3, nCentroids=40)   
            #find axis of rotatio                                                
            mid_p = np.median(cemb_p, axis=0).reshape(-1,1)
            mid_r = np.median(cemb_r, axis=0).reshape(-1,1)
            norm_vector =  mid_p - mid_r
            norm_vector = norm_vector/np.linalg.norm(norm_vector)
            k = np.dot(np.median(cemb_p, axis=0), norm_vector)

            angles = np.linspace(-np.pi,np.pi,100)
            error = find_rotation(cent_p-mid_p.T, cent_r-mid_r.T, norm_vector)
            rot_error_GC2[:,it, idx] = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
            rot_angle_GC2[it, idx] = angles[np.argmin(error)]
            #compute SI of emb
            SI_val[0,it, idx],_,_,_ = compute_structure_index(emb_p, vfeat_p, 
                                                        n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)

            SI_val[1,it, idx],_,_,_ = compute_structure_index(emb_r, vfeat_r, 
                                                        n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)

            print(f"Iter: {it+1}/{iters} | Idx: {idx+1}/{sdeep_cells.shape[0]} | Rot: {(rot_angle_GC2[it,idx]*180/np.pi):.2f} | SI_p: {SI_val[0,it,idx]:2f} | SI_r: {SI_val[1,it,idx]:2f}")

    rot_GC2 = {
        'rotAngle': rot_angle_GC2,
        'rotError': rot_error_GC2,
        'SI_val': SI_val
    }
    with open(os.path.join(sub_save_dir,mouse+'rot_it_dict.pkl'), 'wb') as f:
        pickle.dump(rot_GC2, f)


miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
rotAngle = np.zeros((100, 200, len(miceList)))*np.nan
for idx, mouse in enumerate(miceList):
    sub_save_dir = os.path.join(save_dir, mouse)
    rot_dict = load_pickle(sub_save_dir,mouse+'rot_it_dict.pkl')
    rotAngleTemp = rot_dict['rotAngle']
    rotAngle[:,:rotAngleTemp.shape[1],idx] = rotAngleTemp


m = np.mean(np.abs(rot_angle_GC2)*180/np.pi,axis=0)
sd = np.std(np.abs(rot_angle_GC2)*180/np.pi,axis=0)

plt.figure()
for idx, mouse in enumerate(miceList):
    m = np.mean(np.abs(rotAngle[:,:,idx])*180/np.pi,axis=0)
    plt.plot(m, label = mouse)
plt.legend()
plt.xlabel('# rot cells removed')
plt.ylabel('Rotation angle')
plt.fill_between(np.arange(rot_angle_GC2.shape[1]), m-sd, m+sd, alpha = 0.3)

######################################################################################################################################

mouse = 'CZ6'
miceList = ['CZ3', 'CZ8', 'CZ9', 'CGrin1']
for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    sub_save_dir = os.path.join(save_dir, mouse)
    if not os.path.exists(sub_save_dir):
        os.mkdir(sub_save_dir)

    animal_dict = load_pickle(file_path,file_name)
    fnames = list(animal_dict.keys())
    animal_p= copy.deepcopy(animal_dict[fnames[0]])
    animal_r= copy.deepcopy(animal_dict[fnames[1]])

    signal_p = copy.deepcopy(np.concatenate(animal_p[signal_name].values, axis=0))
    pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
    index_mat_p = copy.deepcopy(np.concatenate(animal_p['index_mat'].values, axis=0))
    dir_mat_p = copy.deepcopy(np.concatenate(animal_p['dir_mat'].values, axis=0))
    vfeat_p = np.concatenate((pos_p[:,0].reshape(-1,1),dir_mat_p),axis=1)

    signal_r = copy.deepcopy(np.concatenate(animal_r[signal_name].values, axis=0))
    pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
    index_mat_r = copy.deepcopy(np.concatenate(animal_p['index_mat'].values, axis=0))
    dir_mat_r = copy.deepcopy(np.concatenate(animal_r['dir_mat'].values, axis=0))
    vfeat_r = np.concatenate((pos_r[:,0].reshape(-1,1),dir_mat_r),axis=1)

    #%%all data
    index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
    concat_signal = np.vstack((signal_p, signal_r))

    cellType_path = os.path.join(celltype_dir, mouse)
    cellType_name = mouse+'_cellType.npy'
    cellType = np.load(os.path.join(cellType_path,cellType_name))
    deep_cells = np.where(cellType==0)[0]

    rot_error_CZ6 = np.zeros((100, iters, deep_cells.shape[0]))
    rot_angle_CZ6 = np.zeros((iters, deep_cells.shape[0]))
    SI_val = np.zeros((2,iters,deep_cells.shape[0]))

    for it in range(iters):
        new_order = np.random.permutation(concat_signal.shape[1])
        sconcat_signal = copy.deepcopy(concat_signal[:, new_order])
        scellType = cellType[new_order]
        sdeep_cells = np.where(scellType==0)[0]
        for idx in range(sdeep_cells.shape[0]):
            temp_concat_signal = copy.deepcopy(sconcat_signal)
            temp_concat_signal = np.delete(temp_concat_signal, sdeep_cells[:idx+1], axis = 1)

            model = umap.UMAP(n_neighbors =n_neigh, n_components =dim, min_dist=min_dist)
            model.fit(temp_concat_signal)
            concat_emb = model.transform(temp_concat_signal)
            emb_p = concat_emb[index[:,0]==0,:]
            emb_r = concat_emb[index[:,0]==1,:]

            D_p = pairwise_distances(emb_p)
            noiseIdx_p = filter_noisy_outliers(emb_p,D_p)
            max_dist = np.nanmax(D_p)
            cemb_p = emb_p[~noiseIdx_p,:]
            cpos_p = pos_p[~noiseIdx_p,:]
            cdir_mat_p = dir_mat_p[~noiseIdx_p]

            D_r = pairwise_distances(emb_r)
            noiseIdx_r = filter_noisy_outliers(emb_r,D_r)
            max_dist = np.nanmax(D_r)
            cemb_r = emb_r[~noiseIdx_r,:]
            cpos_r = pos_r[~noiseIdx_r,:]
            cdir_mat_r = dir_mat_r[~noiseIdx_r]

            #compute centroids
            cent_p, cent_r = get_centroids(cemb_p, cemb_r, cpos_p[:,0], cpos_r[:,0], 
                                                            cdir_mat_p, cdir_mat_r, ndims = 3, nCentroids=40)   
            #find axis of rotatio                                                
            mid_p = np.median(cemb_p, axis=0).reshape(-1,1)
            mid_r = np.median(cemb_r, axis=0).reshape(-1,1)
            norm_vector =  mid_p - mid_r
            norm_vector = norm_vector/np.linalg.norm(norm_vector)
            k = np.dot(np.median(cemb_p, axis=0), norm_vector)

            angles = np.linspace(-np.pi,np.pi,100)
            error = find_rotation(cent_p-mid_p.T, cent_r-mid_r.T, norm_vector)
            rot_error_CZ6[:,it, idx] = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
            rot_angle_CZ6[it, idx] = angles[np.argmin(error)]
            #compute SI of emb
            SI_val[0,it, idx],_,_,_ = compute_structure_index(emb_p, vfeat_p, 
                                                        n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)

            SI_val[1,it, idx],_,_,_ = compute_structure_index(emb_r, vfeat_r, 
                                                        n_neighbors=20, discrete_label=[False, True], num_shuffles=0, verbose=False)

            print(f"Iter: {it+1}/{iters} | Idx: {idx+1}/{sdeep_cells.shape[0]} | Rot: {(rot_angle_CZ6[it,idx]*180/np.pi):.2f} | SI_p: {SI_val[0,it,idx]:2f} | SI_r: {SI_val[1,it,idx]:2f}")

        rot_CZ6 = {
            'rotAngle': rot_angle_CZ6,
            'rotError': rot_error_CZ6,
            'SI_val': SI_val
        }
        with open(os.path.join(sub_save_dir,mouse+'rot_it_dict.pkl'), 'wb') as f:
            pickle.dump(rot_CZ6, f)


m = np.mean(np.abs(rot_angle_CZ6)*180/np.pi,axis=0)
sd = np.std(np.abs(rot_angle_CZ6)*180/np.pi,axis=0)

plt.figure()
plt.plot(m)
plt.fill_between(np.arange(rot_angle_CZ6.shape[1]), m-sd, m+sd, alpha = 0.3)



######################################################################################################################################
import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neural_manifold.general_utils as gu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from neural_manifold import dimensionality_reduction as dim_red
import seaborn as sns
# from neural_manifold.dimensionality_reduction import validation as dim_validation 
import umap
# import math
# from kneed import KneeLocator
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from structure_index import compute_structure_index, draw_graph
from neural_manifold import decoders as dec
from datetime import datetime
from neural_manifold import place_cells as pc

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data



deep_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
sup_list = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']


miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']


data_dir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/paper/Fig3/data/'
saveDir = '/home/julio/Pictures/'

save_dir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/paper/Fig3/manifold_cells/'

params = {
    'sF': 20,
    'bin_num': [10,4],
    'std_pos': 0,
    'std_pdf': 0.5,
    'method': 'spatial_info',
    'ignore_edges': 0,
    'num_shuffles': 10,
    'min_shift': 10,
    'th_metric': 95,
    'vel_th': 0
    }

for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    save_path = os.path.join(save_dir, mouse)

    animal_dict = load_pickle(file_path,file_name)
    fnames = list(animal_dict.keys())
    mouse_pc = dict()
    fname = fnames[0]
    print(f'\t{fname}')
    pd_struct = copy.deepcopy(animal_dict[fname])

    neu_signal = np.concatenate(pd_struct['clean_traces'].values, axis=0)
    pos_signal = np.concatenate(pd_struct['umap'].values, axis=0)
    vel_signal = np.concatenate(pd_struct['vel'].values, axis=0)
    ppos_signal = np.concatenate(pd_struct['pos'].values, axis=0)


    mouse_pc[fname] = pc.get_place_cells(pos_signal, neu_signal, vel_signal = vel_signal, dim = 2,
                           mouse = mouse, save_dir = save_path, **params)

    neu_pdf = np.load('/home/julio/Pictures/'+mouse+'_neu_pdf.npy') 
    mapAxis = mouse_pc[fname]['mapAxis']
    neu_pdf = mouse_pc[fname]['neu_pdf']

    neu_pdf_norm = np.zeros((neu_pdf.shape[0],neu_pdf.shape[1],neu_pdf.shape[2]))
    for c in range(neu_pdf.shape[2]):
        neu_pdf_norm[:,:,c] = neu_pdf[:,:,c,0]/np.max(neu_pdf[:,:,c,0])
    mneu_pdf = np.nanmean(neu_pdf_norm, axis=2)

    msignal_r = np.zeros((pos_signal.shape[0]))
    for p in range(pos_signal.shape[0]):
        try:

            x = np.where(mapAxis[0]<=pos_signal[p,0])[0][-1]
        except: 
            x = 0
        try:
            y = np.where(mapAxis[1]<=pos_signal[p,1])[0][-1]
        except:
            y = 0
        msignal_r[p] = mneu_pdf[x,y]
    msignal_r = gaussian_filter1d(msignal_r, sigma = 3, axis = 0)


    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(1,2,1, projection = '3d')
    b = ax.scatter(*pos_signal[:,:3].T, c = ppos_signal[:,0],s = 20, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(130,110,90)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax = plt.subplot(1,2,2, projection = '3d')
    b = ax.scatter(*pos_signal[:,:3].T, c = msignal_r,s = 20,vmin=0.25, vmax=0.35)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.view_init(130,110,90)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,mouse+'_umap_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_umap_emb.png'), dpi = 400,bbox_inches="tight")






plt.figure(); plt.matshow(mneu_pdf)

fig, ax = plt.subplots(1,2,figsize=(15,5))
b = ax[0].matshow(np.nanmean(GC2_pdf,axis=2)/np.max(np.nanmean(GC2_pdf,axis=2)), vmin= 0.4, vmax = 1)
cbar = fig.colorbar(b, ax=ax[0], location='right',anchor=(0, 0.3), shrink=0.8)
b = ax[1].matshow(np.nanmean(CZ6_pdf,axis=2)/np.max(np.nanmean(CZ6_pdf,axis=2)), vmin= 0.4, vmax = 1)
cbar = fig.colorbar(b, ax=ax[1], location='right',anchor=(0, 0.3), shrink=0.8)




plt.savefig('CZ6_manifold_place.svg', dpi = 400,bbox_inches="tight")
plt.savefig('CZ6_manifold_place.png', dpi = 400,bbox_inches="tight")

plt.savefig(os.path.join(data_dir,f'noise_dec_{label_name}_test_line.png'), dpi = 400,bbox_inches="tight")


######################################################################################################################################
place_cell_dir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/paper/Fig3/place_cells/'

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,10), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx


saveDir = '/home/julio/Pictures/'


signal_name = 'clean_traces'
n_neigh = 120
dim = 3
min_dist = 0.1

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

for mouse in miceList:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    animal_dict = load_pickle(file_path,file_name)
    fnames = list(animal_dict.keys())
    animal_r= copy.deepcopy(animal_dict[fnames[1]])
    signal_r = copy.deepcopy(np.concatenate(animal_r[signal_name].values, axis=0))
    pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))


    place_cell = load_pickle(os.path.join(place_cell_dir, mouse),mouse+'_pc_dict.pkl')[fnames[1]]
    pc_idx = place_cell['place_cells_idx']

    # #%%all data
    # index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
    # concat_signal = np.vstack((signal_p, signal_r))
    # model = umap.UMAP(n_neighbors =n_neigh, n_components =dim, min_dist=min_dist)
    # # model = umap.UMAP(n_neighbors = 600, n_components =4, min_dist=0.5)
    # model.fit(concat_signal)
    # concat_emb = model.transform(concat_signal)
    # emb_p = concat_emb[index[:,0]==0,:]
    # emb_r = concat_emb[index[:,0]==1,:]


    model = umap.UMAP(n_neighbors =n_neigh, n_components =dim, min_dist=min_dist)
    model.fit(signal_r)
    emb_r = model.transform(signal_r)

    D_r = pairwise_distances(emb_r)
    noiseIdx_r = filter_noisy_outliers(emb_r,D_r)
    max_dist = np.nanmax(D_r)
    cemb_r = emb_r[~noiseIdx_r,:]
    csignal_r = signal_r[~noiseIdx_r,:]
    cpos_r = pos_r[~noiseIdx_r,:]
    msignal_r = np.sum(csignal_r[:,pc_idx], axis=1)


    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,2,1, projection = '3d')
    b = ax.scatter(*cemb_r[:,:3].T, c = cpos_r[:,0],s = 20, cmap = 'inferno')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    # ax.view_init(140, -115)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax = plt.subplot(1,2,2, projection = '3d')
    b = ax.scatter(*cemb_r[:,:3].T, c = msignal_r,s = 20)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    # ax.view_init(140, -115)
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,mouse+'_umap_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_umap_emb.svg'), dpi = 400,bbox_inches="tight")


vel_th = 0

animal_p= copy.deepcopy(animal[fnames[0]])
animal_r= copy.deepcopy(animal[fnames[1]])
animal_p = add_dir_mat_field(animal_p)
animal_r = add_dir_mat_field(animal_r)

animal_p = gu.select_trials(animal_p,"dir == ['L','R']")
animal_r = gu.select_trials(animal_r,"dir == ['L','R']")
animal_p, animal_p_still = gu.keep_only_moving(animal_p, vel_th)
animal_r, animal_r_still = gu.keep_only_moving(animal_r, vel_th)
animal_p, animal_r = preprocess_traces(animal_p, animal_r, signal_field, sigma=sigma, sig_up = sig_up, sig_down = sig_down)

######################################################################################################################################

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


miceList = ['GC2','CZ3']
dataDir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/JP_data/Castle_inscopix/data/'
saveDir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/JP_data/Castle_inscopix/processed_data/'

#%% PARAMS
sigma = 6
upSig = 4
downSig = 12
signalField = 'raw_traces'
peakTh = 0.1
velTh = 0
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
    pdMouse = gu.load_files(mouseDataDir, '*_PyalData_struct*.mat', verbose=verbose)
    if verbose:
        gu.print_time_verbose(localTime, globalTime)
    #%% KEEP ONLY LAST SESSION
    print('\n### 2. PROCESS DATA ###')
    #%% 2. PROCESS DATA
    # pdMouse = add_dir_mat_field(pdMouse)
    #2.1 keep only moving epochs
    print(f'2.1 Dividing into moving/still data w/ velTh= {velTh:.2f}.')
    if velTh>0:
        og_dur = np.concatenate(pdMouse["pos"].values, axis=0).shape[0]
        pdMouse, still_pdMouse = gu.keep_only_moving(pdMouse, velTh)
        move_dur = np.concatenate(pdMouse["pos"].values, axis=0).shape[0]
        print(f"\tOg={og_dur} ({og_dur/20}s) Move= {move_dur} ({move_dur/20}s)")
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



miceList = ['GC2','CZ3']
params = {
    'dim':3,
    'nNeigh': 120,
    'minDist': 0.2,
    'signalName': 'clean_traces',
}
dataDir = '/home/julio/Documents/SP_project/old_version/spatial_navigation/JP_data/Castle_inscopix/processed_data/'

for mouse in miceList:
    print(f"Working on mouse {mouse}")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(dataDir, mouse)
    pdMouse = load_pickle(filePath,fileName)
    #signal
    signal = np.concatenate(pdMouse[params['signalName']].values, axis = 0)
    pos = np.concatenate(pdMouse['pos'].values, axis=0)
    trialMat = np.concatenate(pdMouse['trial_type_mat'].values, axis=0)
    pos_sec = np.concatenate(pdMouse['pos_sec'].values, axis=0)


    abids = dim_red.compute_abids(signal, 30)
    abidsDim = np.nanmean(abids)
    print(f"\tABIDS dim: {abidsDim:.2f}")


    trialVals = np.unique(trialMat)
    trialType = np.zeros((trialMat.shape))
    for p in range(trialMat.shape[0]):
        trialType[p] = np.where(trialVals==trialMat[p])[0][0]
    modelUmap = umap.UMAP(n_neighbors =params['nNeigh'], n_components =params['dim'], min_dist=params['minDist'])
    modelUmap.fit(signal)
    embUmap = modelUmap.transform(signal)

    fig = plt.figure(figsize = (12,10))  
    ax = plt.subplot(2,3,1)
    p = ax.scatter(*pos.T, c=np.sqrt(np.sum(pos**2, axis=1)), cmap = plt.cm.magma)
    ax.set_title(f"{mouse}")
    ax.set_xlabel('Dim 1', labelpad= -8)
    ax.set_ylabel('Dim 2', labelpad= -8)
    ax.set_xticks([])
    ax.set_yticks([])

    sections = ['H0', 'H3', 'H6', 'H9', 'A0', 'A3', 'A6', 'A9', 'CB']
    cols = np.array([[31,119,180],[255,127,14],[44,160,44],[214,39,40],
                    [94,181,242],[255,191,130],[94,200,94],[239,96,97],
                    [150,150,150]])/255
    ax = plt.subplot(2,3,2)
    for idx, typet in enumerate(sections):
        temp = pos[pos_sec==typet,:]
        p = ax.scatter(*temp.T,c = cols[idx,:],label = typet)
    ax.legend()
    ax.set_xlabel('Dim 1', labelpad= -8)
    ax.set_ylabel('Dim 2', labelpad= -8)
    ax.set_xticks([])
    ax.set_yticks([])

    types = ['E','W','R']
    cols = ['#325DF5', '#F56F49', '#8BF518']
    ax = plt.subplot(2,3,3)
    for idx, typet in enumerate(types):
        temp = pos[trialMat==typet,:]
        p = ax.scatter(*temp.T,color = cols[idx], label = typet)
    ax.legend()
    ax.set_title("Trial type")
    ax.set_xlabel('Dim 1', labelpad= -8)
    ax.set_ylabel('Dim 2', labelpad= -8)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2,3,4, projection='3d')
    p = ax.scatter(*embUmap[:,:3].T, c=np.sqrt(np.sum(pos**2, axis=1)), cmap = plt.cm.magma)
    # ax.set_title(f"{mouse}")
    # ax.view_init([30,45])
    ax.set_xlabel('Dim 1', labelpad= -8)
    ax.set_ylabel('Dim 2', labelpad= -8)
    ax.set_zlabel('Dim 3', labelpad= -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    sections = ['H0', 'H3', 'H6', 'H9', 'A0', 'A3', 'A6', 'A9', 'CB']
    cols = np.array([[31,119,180],[254,127,14],[44,160,44],[214,39,40],
                    [94,181,242],[254,191,130],[94,200,94],[239,96,97],
                    [150,150,150]])/255
    colType = np.zeros((pos_sec.shape[0],3))
    toPlot = np.zeros((pos_sec.shape[0])).astype(bool)*False
    for idx, typet in enumerate(sections):
        colType[pos_sec==typet,0] = cols[idx,0]
        colType[pos_sec==typet,1] = cols[idx,1]
        colType[pos_sec==typet,2] = cols[idx,2]
        toPlot[pos_sec==typet] = True

    ax = plt.subplot(2,3,5, projection='3d')
    ax.scatter(*embUmap[toPlot,:3].T, color = colType[toPlot])
    # ax.view_init([30,45])
    ax.set_title("Sections")
    ax.set_xlabel('Dim 1', labelpad= -8)
    ax.set_ylabel('Dim 2', labelpad= -8)
    ax.set_zlabel('Dim 3', labelpad= -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    types = ['E','W','R']
    colType = np.zeros((trialMat.shape[0],3))
    toPlot = np.zeros((trialMat.shape[0])).astype(bool)*False
    cols = np.array([[50,93,245],[245,111,73],[139,245,24]])/255
    for idx, typet in enumerate(types):
        colType[trialMat==typet,0] = cols[idx,0]
        colType[trialMat==typet,1] = cols[idx,1]
        colType[trialMat==typet,2] = cols[idx,2]
        toPlot[trialMat==typet] = True
    ax = plt.subplot(2,3,6, projection='3d')
    ax.scatter(*embUmap[toPlot,:3].T, color = colType[toPlot])
    # ax.view_init([30,45,0])
    ax.set_title("Trial type")
    ax.set_xlabel('Dim 1', labelpad= -8)
    ax.set_ylabel('Dim 2', labelpad= -8)
    ax.set_zlabel('Dim 3', labelpad= -8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig(os.path.join(saveDir,mouse+'_umap_emb.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(saveDir,mouse+'_umap_emb.svg'), dpi = 400,bbox_inches="tight")

params = {
    'maxDim':10,
    'nNeigh': 120,
    'minDist': 0.1,
    'nnDim': 30,
    'signalName': 'clean_traces',
}

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
model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
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






total_rot = np.logical_and(cellType<4,cellType>0)
total_rot = np.sum(total_rot)
rot_deep = 0
rot_deepm = 0
rot_sup = 0
rot_supm = 0

total_nrot = np.sum(cellType==0)
nrot_deep = 0
nrot_deepm = 0
nrot_sup = 0
nrot_supm = 0

total_remap = np.sum(cellType==4)
remap_deep = 0
remap_deepm = 0
remap_sup = 0
remap_supm = 0

for c in range(len(elena)):
    if cellType[c]<4 and cellType[c]>0:
        if elena[c]=='posit':
            rot_deep += 1
        elif elena[c]=='posit?':
            rot_deepm += 1
        elif elena[c]=='neg?':
            rot_supm += 1
        elif elena[c]=='neg':
            rot_sup += 1
    elif cellType[c]==0:
        if elena[c]=='posit':
            nrot_deep += 1
        elif elena[c]=='posit?':
            nrot_deepm += 1
        elif elena[c]=='neg?':
            nrot_supm += 1
        elif elena[c]=='neg':
            nrot_sup += 1
    elif cellType[c]==4:
        if elena[c]=='posit':
            remap_deep += 1
        elif elena[c]=='posit?':
            remap_deepm += 1
        elif elena[c]=='neg?':
            remap_supm += 1
        elif elena[c]=='neg':
            remap_sup += 1

total_sup = 0
total_deep = 0
for c in range(len(elena)):
    if 'posit' in elena[c]:
        total_deep += 1
    elif 'neg' in elena[c]:
        total_sup += 1



sup_rot = 0
sup_nrot = 0
deep_rot = 0
deep_nrot = 0

for c in range(len(elena)):
    if 'posit' in elena[c]:
        if cellType[c]<4 and cellType[c]>0:
            deep_rot += 1
        elif cellType[c]==0:
            deep_nrot += 1
    elif 'neg' in elena[c]:
        if cellType[c]<4 and cellType[c]>0:
            sup_rot += 1
        elif cellType[c]==0:
            sup_nrot += 1


deep_idx = [x for x in range(len(elena)) if 'posit' in elena[x]]

print(f"Sup: {total_sup} ({total_sup*100/len(elena)}) | Deep: {total_deep} ({total_deep*100/len(elena)})")
print(f"\tDe las superficiales un {(100*sup_rot/total_sup):.2f}% rotan, y un {(100*sup_nrot/total_sup):.2f}% no rotan.")
print(f"\tDe las deep un {(100*deep_rot/total_deep):.2f}% rotan, y un {(100*deep_nrot/total_deep):.2f}% no rotan.")


print('')
print(f"De las que rotan son deep: {rot_deep + rot_deepm}/{total_rot} ({(100*(rot_deep + rot_deepm)/total_rot):.2f})")
print(f"De las que rotan son sup: {rot_sup + rot_supm}/{total_rot} ({(100*(rot_sup + rot_supm)/total_rot):.2f})")

print(f"De las que no rotan son deep: {nrot_deep + nrot_deepm}/{total_nrot} ({(100*(nrot_deep + nrot_deepm)/total_nrot):.2f})")
print(f"De las que no rotan son sup: {nrot_sup + nrot_supm}/{total_nrot} ({(100*(nrot_sup + nrot_supm)/total_nrot):.2f})")

print(f"De las que hacen remap son deep: {remap_deep + remap_deepm}/{total_remap} ({(100*(remap_deep + remap_deepm)/total_remap):.2f})")
print(f"De las que hacen remap son sup: {remap_sup + remap_supm}/{total_remap} ({(100*(remap_sup + remap_supm)/total_remap):.2f})")


deep_rot = 0
deep_nrot = 0
deep_xmirr = 0
deep_ymirr = 0
deep_remap = 0
deep_na = 0
for c in range(len(elena)):
    if 'posit' in elena[c]:
        if cellType[c]==0:
            deep_nrot += 1
        elif cellType[c]==1:
            deep_rot += 1
        elif cellType[c]==2:
            deep_xmirr += 1
        elif cellType[c]==3:
            deep_ymirr += 1
        elif cellType[c]==4:
            deep_remap += 1
        elif cellType[c]==5:
            deep_na += 1

print(f"De las deep un {(100*deep_nrot/total_deep):.2f}% no rotan")
print(f"De las deep un {(100*deep_rot/total_deep):.2f}% rotan")
print(f"De las deep un {(100*deep_xmirr/total_deep):.2f}% hacen xmirr")
print(f"De las deep un {(100*deep_ymirr/total_deep):.2f}% hacen ymirr")
print(f"De las deep un {(100*deep_remap/total_deep):.2f}% hacen remapping")
print(f"De las deep un {(100*deep_na/total_deep):.2f}% son n/a")



for color in ['green', 'red']:
    signalPreOg = copy.deepcopy(np.concatenate(animalPre[color+'_raw_traces'].values, axis=0))
    signalRotOg = copy.deepcopy(np.concatenate(animalRot[color+'_raw_traces'].values, axis=0))

    signalGPre = clean_traces(copy.deepcopy(np.concatenate(animalPre['green_raw_traces'].values, axis=0)))
    signalGRot = clean_traces(copy.deepcopy(np.concatenate(animalRot['green_raw_traces'].values, axis=0)))

    signalRPre = clean_traces(copy.deepcopy(np.concatenate(animalPre['red_raw_traces'].values, axis=0)))
    signalRRot = clean_traces(copy.deepcopy(np.concatenate(animalRot['red_raw_traces'].values, axis=0)))
##############################################################################
##############################################################################


import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def clean_traces(ogSignal, sigma = 6, sig_up = 4, sig_down = 12, peak_th=0.1):
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
    return biSignal

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nnDist < np.percentile(nnDist, 20)
    return noiseIdx

######################
#       PARAMS       #
######################
nNeigh = 120
dim = 3
velTh = 6#3

greenNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_lt_green_raw.csv'
greenNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_green_raw.csv'
redNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_lt_red_raw.csv'
redNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_red_raw.csv'

posNamePre = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_lt_position.mat'
posNameRot = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_position.mat'


######################
#     LOAD SIGNAL    #
######################
signalGPre = pd.read_csv(greenNamePre).to_numpy()[1:,1:].astype(np.float64)
signalGRot = pd.read_csv(greenNameRot).to_numpy()[1:,1:].astype(np.float64)
signalRPre = pd.read_csv(redNamePre).to_numpy()[1:,1:].astype(np.float64)
signalRRot = pd.read_csv(redNameRot).to_numpy()[1:,1:].astype(np.float64)

######################
#      LOAD POS      #
######################
posPre = scipy.io.loadmat(posNamePre)['Position']
posPre = posPre[::2,:]/10
posRot = scipy.io.loadmat(posNameRot)['Position']
posRot = posRot[::2,:]/10

######################
#     DELETE NANs    #
######################
nanIdx = np.where(np.sum(np.isnan(signalGPre),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRPre),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posPre),axis=1)>0)[0]),axis=0)
signalGPre = np.delete(signalGPre,nanIdx, axis=0)
signalRPre = np.delete(signalRPre,nanIdx, axis=0)
posPre = np.delete(posPre,nanIdx, axis=0)

nanIdx = np.where(np.sum(np.isnan(signalGRot),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalRRot),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(posRot),axis=1)>0)[0]),axis=0)
signalGRot = np.delete(signalGRot,nanIdx, axis=0)
signalRRot = np.delete(signalRRot,nanIdx, axis=0)
posRot = np.delete(posRot,nanIdx, axis=0)

######################
#    MATCH LENGTH    #
######################
if signalGPre.shape[0]>signalRPre.shape[0]:
    signalGPre = signalGPre[:signalRPre.shape[0],:]
elif signalRPre.shape[0]>signalGPre.shape[0]:
    signalRPre = signalRPre[:signalGPre.shape[0],:]

if posPre.shape[0]>signalGPre.shape[0]:
    posPre = posPre[:signalGPre.shape[0],:]
else:
    signalGPre = signalGPre[:posPre.shape[0],:]
    signalRPre = signalRPre[:posPre.shape[0],:]

if signalGRot.shape[0]>signalRRot.shape[0]:
    signalGRot = signalGRot[:signalRRot.shape[0],:]
elif signalRRot.shape[0]>signalGRot.shape[0]:
    signalRRot = signalRRot[:signalGRot.shape[0],:]

if posRot.shape[0]>signalGRot.shape[0]:
    posRot = posRot[:signalGRot.shape[0],:]
else:
    signalGRot = signalGRot[:posRot.shape[0],:]
    signalRRot = signalRRot[:posRot.shape[0],:]

######################
#     COMPUTE VEL    #
######################
velPre = np.abs(np.diff(posPre[:,0]).reshape(-1,1)*10)
velPre = np.concatenate((velPre[0].reshape(-1,1), velPre), axis=0)
velPre = gaussian_filter1d(velPre, sigma = 5, axis = 0)


velRot = np.abs(np.diff(posRot[:,0]).reshape(-1,1)*10)
velRot = np.concatenate((velRot[0].reshape(-1,1), velRot), axis=0)
velRot = gaussian_filter1d(velRot, sigma = 5, axis = 0)

######################
#  DELETE LOW SPEED  #
######################
lowSpeedIdxPre = np.where(velPre<2.5)[0]
signalGPre = np.delete(signalGPre,lowSpeedIdxPre, axis=0)
signalRPre = np.delete(signalRPre,lowSpeedIdxPre, axis=0)
posPre = np.delete(posPre,lowSpeedIdxPre, axis=0)
velPre = np.delete(velPre,lowSpeedIdxPre, axis=0)


lowSpeedIdxRot = np.where(velRot<2.5)[0]
signalGRot = np.delete(signalGRot,lowSpeedIdxRot, axis=0)
signalRRot = np.delete(signalRRot,lowSpeedIdxRot, axis=0)
posRot = np.delete(posRot,lowSpeedIdxRot, axis=0)
velRot = np.delete(velRot,lowSpeedIdxRot, axis=0)

######################
#    CREATE TIME     #
######################
timePre = np.arange(posPre.shape[0])
timeRot = np.arange(posRot.shape[0])

######################
#    CLEAN TRACES    #
######################
signalGPre = clean_traces(signalGPre)
signalRPre = clean_traces(signalRPre)
signalGRot = clean_traces(signalGRot)
signalRRot = clean_traces(signalRRot)

############################
#  ALL CELLS INDIVIDUAL    #
############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)


plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'All Cells - Individual {velTh}')

############################
#     REGISTER SIGNALS     #
############################
preGreenCells = [1,2,3,6,7,12,13,15,16,17,19,20,21,22,23,25,26,27,29,30,32,33,34,35,37,39,41,43,44,46,47,48,50,51,52,55,66,67,68,69,73,75,76,77,78,79,80,82,84,86,87,89,91,92,93,94,97,98,99,100,101,102,103,106,108,110,111,112,114,115,117,119,120,122,123,124,125,126,127,128,129,130,131,133,134,136,137,138,139,141,145,146,147,148,149,151,152,153,155,156,157,158,165,166,167,170,171,172,173,174,175,176,177,178,179,180,182,183,185,189,190,191,192,193,194,195,196,197,200,202,206,208,209,210,211,214,216,218,219,224,225,226,227,228, 229,231,232,234,235,236,239,240,248,249,251,254,258,259,261,262,266,267,273,277,278,279,282,284,285,288,290,293]
preGreenCells = [x-1 for x in preGreenCells]
preRedCells = [2,3,4,6,7,10,12,13,18,22,23,24,25,27,28,29,30,31,32,34,36,37,40,41,42,43,45,46,47,49,50,51,53,54,55,56,57,58,61,62,63,64,65,67,68,69,70,72,74,75,79,81,82,83,84,85,86,87,88,89,94,95,97,98,99,100,101,102,103,105,107,109,112,113,114,117,118,119,120,121,122,124,125,126,127,128,129,130,131,133,135,137,138,139,140,141,142,143,144,145,146,149,152,153,154,155,157,158,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,182,184,185,186,187,188,191,192,196,197,198,199,200,201,203,204,206,207,208,210,211,212,214,215,216,217,218,219,220,221,225,226,227,232,233,234,235,236,237,238,239,240,241,242,244,245,251,252,257,258,259,260,261,262,263,264,266,269,270,271,275,276,277,278,279,280,281,282,284,285,286,287,290,291,292,293,294,295,296,297,299,302,303,304,305,306,307,308,309,310,312,313,314,315,316,319,320,321,324,325,326,329,330,331,332,334,336,337,343,344,346,347,348,350,351,353,354,355,357,361,364,366,367,368,369,375,376,378,379,381,382,383,384,387,389,390,393,395,402,405,406,407,409,410,412,413,415,416,418,422,426,427,432,433,434,435,436,437,438,439,440,441,442,443,445,447,449,450,454,455,457,460,461,462,463,464,465,466,468,471,475,484,486]
preRedCells = [x-1 for x in preRedCells]

rotRedCells = [6,7,0,1,2,3,4,8,9,10,17,24,16,28,12,11,15,13,18,22,23,29,32,21,27,30,5,34,36,37,40,39,38,41,42,48,43,20,51,47,158,46,44,57,56,54,170,63,65,64,59,60,61,55,58,62,53,68,71,69,73,72,207,86,67,89,74,77,83,85,82,80,76,88,91,75,94,96,95,92,93,97,125,115,101,100,90,99,103,236,111,113,109,116,105,106,104,108,121,114,122,119,107,120,117,123,126,130,128,133,259,159,146,127,135,141,150,154,138,151,152,134,147,144,131,137,140,160,149,153,142,129,156,136,161,157,145,165,163,164,162,277,167,174,173,172,186,176,168,171,182,181,184,179,169,178,180,177,188,187,183,192,185,166,190,367,301,195,200,198,197,196,199,307,201,202,204,205,219,209,206,208,214,212,215,217,216,210,220,228,242,221,247,223,224,244,230,249,315,226,229,234,245,232,235,241,240,227,238,237,246,368,231,233,252,251,263,250,276,274,270,264,254,256,269,258,261,273,260,266,255,262,268,265,222,267,275,253,279,281,283,282,280,278,291,290,284,288,293,287,300,285,286,298,294,299,303,304,311,302,305,309,363,306,364,310,314,313,329,322,318,317,323,327,319,326,320,328,324,321,312,316,351,330,350,331,332,344,337,343,339,335,333,338,345,342,346,349,336,355,356,360,361,352,357,362,354,353,365,132,257]
rotRedCells=[x-1 for x in rotRedCells]
rotGreenCells = [2,0,4,45,6,1,3,5,8,18,13,11,15,16,10,19,12,20,23,22,24,21,26,27,30,25,37,101,32,36,39,33,35,34,38,40,43,123,42,47,46,50,54,56,61,66,59,73,63,60,138,64,221,71,58,70,57,77,74,87,79,85,80,225,78,76,89,148,113,88,86,83,82,75,103,96,90,97,99,100,94,92,98,95,106,104,110,220,105,112,108,93,191,120,179,115,116,114,186,118,224,119,223,121,124,133,122,128,127,129,126,134,135,143,137,144,139,159,140,136,153,146,149,147,151,152,154,158,160,157,150,166,174,164,162,168,169,172,167,165,171,178,163,161,183,180,189,182,181,188,184,187,185,192,190,193,195,194,201,199,206,210,205,200,208,212,209,213,216,218,68,203]
rotGreenCells = [x-1 for x in rotGreenCells]

signalGPre = signalGPre[:,preGreenCells]
signalRPre = signalRPre[:, preRedCells]
signalGRot = signalGRot[:,rotGreenCells]
signalRRot = signalRRot[:, rotRedCells]

###############################
# REGISTERED CELLS INDIVIDUAL #
###############################
#%%PRE
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGPre)
embGPre = model.transform(signalGPre)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRPre)
embRPre = model.transform(signalRPre)

#%%ROT
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalGRot)
embGRot = model.transform(signalGRot)
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(signalRRot)
embRRot = model.transform(signalRRot)

plt.figure()
ax = plt.subplot(2,4,1, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax = plt.subplot(2,4,2, projection = '3d')
ax.scatter(*embGPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')
ax.set_title('Green Pre')
ax = plt.subplot(2,4,3, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
ax.set_title('Red Pre')
ax = plt.subplot(2,4,4, projection = '3d')
ax.scatter(*embRPre[:,:3].T, c = timePre, s=10, cmap = 'YlGn_r')

ax = plt.subplot(2,4,5, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Green Rot')
ax = plt.subplot(2,4,6, projection = '3d')
ax.scatter(*embGRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
ax = plt.subplot(2,4,7, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')
ax.set_title('Red Rot')
ax = plt.subplot(2,4,8, projection = '3d')
ax.scatter(*embRRot[:,:3].T, c = timeRot, s=10, cmap = 'YlGn_r')
plt.suptitle(f'Reg Cells - Individual {velTh}')

#############################
# REGISTERED CELLS TOGETHER #
#############################
#%%all data
index = np.vstack((np.zeros((signalGPre.shape[0],1)),np.ones((signalGRot.shape[0],1))))
concatSignalG = np.vstack((signalGPre, signalGRot))
model = umap.UMAP(n_neighbors =nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalG)
embBoth = model.transform(concatSignalG)
embGPre = embBoth[index[:,0]==0,:]
embGRot = embBoth[index[:,0]==1,:]

#%%all data
index = np.vstack((np.zeros((signalRPre.shape[0],1)),np.ones((signalRRot.shape[0],1))))
concatSignalR = np.vstack((signalRPre, signalRRot))
model = umap.UMAP(n_neighbors=nNeigh, n_components =dim, min_dist=0.1)
model.fit(concatSignalR)
embBoth = model.transform(concatSignalR)
embRPre = embBoth[index[:,0]==0,:]
embRRot = embBoth[index[:,0]==1,:]


D = pairwise_distances(embGPre)
noiseIdx = filter_noisy_outliers(embGPre,D=D)
cembGPre = embGPre[~noiseIdx,:]
cposGPre = posPre[~noiseIdx,:]

D = pairwise_distances(embRPre)
noiseIdx = filter_noisy_outliers(embRPre,D=D)
cembRPre = embRPre[~noiseIdx,:]
cposRPre = posPre[~noiseIdx,:]


D = pairwise_distances(embGRot)
noiseIdx = filter_noisy_outliers(embGRot,D=D)
cembGRot = embGRot[~noiseIdx,:]
cposGRot = posRot[~noiseIdx,:]

D = pairwise_distances(embRRot)
noiseIdx = filter_noisy_outliers(embRRot,D=D)
cembRRot = embRRot[~noiseIdx,:]
cposRRot = posRot[~noiseIdx,:]



plt.figure()
ax = plt.subplot(2,2,1, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembGRot[:,:3].T, color = 'r', s=10)
ax.set_title('Green')
ax = plt.subplot(2,2,2, projection = '3d')
ax.scatter(*cembGPre[:,:3].T, c = cposGPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembGRot[:,:3].T, c = cposGRot[:,0], s=10, cmap = 'magma')

ax = plt.subplot(2,2,3, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, color ='b', s=10)
ax.scatter(*cembRRot[:,:3].T, color = 'r', s=10)
ax.set_title('Red')
ax = plt.subplot(2,2,4, projection = '3d')
ax.scatter(*cembRPre[:,:3].T, c = cposRPre[:,0], s=10, cmap = 'magma')
ax.scatter(*cembRRot[:,:3].T, c = cposRRot[:,0], s=10, cmap = 'magma')
plt.suptitle(f'Reg Cells - Together {velTh}')


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



##############################################################################################################################3
import scipy
def clean_traces(ogSignal, sigma = 6, sig_up = 4, sig_down = 12, peak_th=0.1):
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
    return biSignal



# greenName = '/home/julio/Downloads/Thy1jRGECO_G21_LTtest_traces_green.csv'
# redName = '/home/julio/Downloads/Thy1jRGECO_G21_LTtest_traces_red.csv'
# redName = '/home/julio/Downloads/Thy1jRGECO_G21_RawTraces_red_ProjTest_055_75_05.csv'


greenName = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_green_raw.csv'
redName = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/Inscopix_data/ThyG21_rot_red_raw.csv'
posName = '/home/julio/Documents/SP_project/LT_DualColor/data/ThyG21/concatenated/Inscopix_data/pos_split/ThyG21_rot_position.mat'

signalG = pd.read_csv(greenName).to_numpy()[1:,1:].astype(np.float64)
signalR = pd.read_csv(redName).to_numpy()[1:,1:].astype(np.float64)
pos = scipy.io.loadmat(posName)['Position']
pos = pos[::2,:]


nanIdx = np.where(np.sum(np.isnan(signalG),axis=1)>0)[0]
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(signalR),axis=1)>0)[0]),axis=0)
nanIdx = np.concatenate((nanIdx,np.where(np.sum(np.isnan(pos),axis=1)>0)[0]),axis=0)

signalG = np.delete(signalG,nanIdx, axis=0)
signalR = np.delete(signalR,nanIdx, axis=0)
pos = np.delete(pos,nanIdx, axis=0)

signalG = clean_traces(signalG)
signalR = clean_traces(signalR)

if signalG.shape[0]>signalR.shape[0]:
    signalG = signalG[:signalR.shape[0],:]
elif signalR.shape[0]>signalG.shape[0]:
    signalR = signalR[:signalG.shape[0],:]

if pos.shape[0]>signalG.shape[0]:
    pos = pos[:signalG.shape[0],:]
else:
    signalG = signalG[:pos.shape[0],:]
    signalR = signalR[:pos.shape[0],:]

#GREEN
model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
model.fit(signalG)
embG = model.transform(signalG)

#RED
model = umap.UMAP(n_neighbors =nn_val, n_components =dim, min_dist=0.1)
model.fit(signalR)
embR = model.transform(signalR)

fig = plt.figure(figsize=(15,5))
ax = plt.subplot(1,2,1, projection = '3d')
b = ax.scatter(*embG[:,:3].T, c = pos[:,0],s =10, cmap = 'magma')
cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title('Green')
ax = plt.subplot(1,2,2, projection = '3d')
b = ax.scatter(*embR[:,:3].T, c = pos[:,0],s =10, cmap = 'magma')
cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
ax.set_xlabel('Dim 1', labelpad = -8)
ax.set_ylabel('Dim 2', labelpad = -8)
ax.set_zlabel('Dim 3', labelpad = -8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title('Red')


###############################################################################33
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
    del_cent_num = (ncentLabel_A<20) + (ncentLabel_B<20)
    del_cent = del_cent_nan + del_cent_num
    
    centLabel_A = np.delete(centLabel_A, del_cent, 0)
    centLabel_B = np.delete(centLabel_B, del_cent, 0)

    return centLabel_A, centLabel_B


miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dataDir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
saveDir = '/home/julio/Documents/SP_project/LT_DeepSup/ellipse/'
ellipseDict = dict()

for mouse in miceList:
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

    ellipseDict[mouse] = dict()

    posPre = np.concatenate(animalPre['pos'].values, axis = 0)
    dirMatPre = np.concatenate(animalPre['dir_mat'].values, axis=0)
    posRot = np.concatenate(animalRot['pos'].values, axis = 0)
    dirMatRot = np.concatenate(animalRot['dir_mat'].values, axis=0)

    embPre = np.concatenate(animalPre['umap'].values, axis = 0)[:,:3]
    embRot = np.concatenate(animalRot['umap'].values, axis = 0)[:,:3]

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
                                                    cdirMatPre, cdirMatRot, ndims = 3, nCentroids=40)  

    modelPCA = PCA(2)
    modelPCA.fit(centPre)
    centPre2D = modelPCA.transform(centPre)
    centPre2D = centPre2D - np.tile(np.mean(centPre2D,axis=0), (centPre2D.shape[0],1))

    modelPCA = PCA(2)
    modelPCA.fit(centPre)
    centRot2D = modelPCA.transform(centRot)
    centRot2D = centRot2D - np.tile(np.mean(centRot2D,axis=0), (centRot2D.shape[0],1))

    plt.figure()
    ax = plt.subplot(2,2,1, projection = '3d')
    ax.scatter(*embPre[:,:3].T, c = posPre[:,0], s=10, cmap = 'magma')
    ax.scatter(*embRot[:,:3].T, c = posRot[:,0], s=10, cmap = 'magma')

    ax = plt.subplot(2,2,2, projection = '3d')
    ax.scatter(*centPre[:,:3].T, color ='b', s=10)
    ax.scatter(*centRot[:,:3].T, color = 'r', s=10)

    ########################
    #         PRE          #
    ########################
    # Formulate and solve the least squares problem ||Ax - b ||^2
    X = centPre2D[:,0].reshape(-1,1)
    Y = centPre2D[:,1].reshape(-1,1)
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(centPre2D[:,0])
    x = np.linalg.lstsq(A, b)[0].squeeze()

    xLim = [np.min(centPre2D[:,0]), np.max(centPre2D[:,0])]
    yLim = [np.min(centPre2D[:,1]), np.max(centPre2D[:,1])]
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
    ax = plt.subplot(2,2,3)
    ax.scatter(*centPre2D[:,:2].T, color ='b', s=10)
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    plt.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, color ='m', s=10)
    ax.plot([center[0],pointLong[0]], [center[1], pointLong[1]], color = 'c')
    ax.plot([center[0],pointShort[0]], [center[1], pointShort[1]], color = 'c')
    ax.set_xlim(np.min(centPre2D[:,0])-0.2, np.max(centPre2D[:,0])+0.2)
    ax.set_ylim(np.min(centPre2D[:,1])-0.2, np.max(centPre2D[:,1])+0.2)
    ax.set_xlim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_ylim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{longAxis/shortAxis:4f}")

    ellipseDict[mouse][fnamePre] = {
        'pos':posPre,
        'dirMat': dirMatPre,
        'emb': embPre,
        'D': DPre,
        'noiseIdx': noiseIdxPre,
        'cpos': cposPre,
        'cdirMat': cdirMatPre,
        'cemb': cembPre,
        'cent': centPre,
        'cent2D': centPre2D,
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

    ########################
    #         ROT          #
    ########################
    # Formulate and solve the least squares problem ||Ax - b ||^2
    X = centRot2D[:,0].reshape(-1,1)
    Y = centRot2D[:,1].reshape(-1,1)
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(centRot2D[:,0])
    x = np.linalg.lstsq(A, b)[0].squeeze()

    xLim = [np.min(centRot2D[:,0]), np.max(centRot2D[:,0])]
    yLim = [np.min(centRot2D[:,1]), np.max(centRot2D[:,1])]
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
    ax = plt.subplot(2,2,4)
    ax.scatter(*centRot2D[:,:2].T, color ='b', s=10)
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
    plt.scatter(center[0], center[1], color = 'm', s=30)
    ax.scatter(xValid, yValid, color ='m', s=10)
    ax.plot([center[0],pointLong[0]], [center[1], pointLong[1]], color = 'c')
    ax.plot([center[0],pointShort[0]], [center[1], pointShort[1]], color = 'c')
    ax.set_xlim(np.min(centRot2D[:,0])-0.2, np.max(centRot2D[:,0])+0.2)
    ax.set_ylim(np.min(centRot2D[:,1])-0.2, np.max(centRot2D[:,1])+0.2)

    ax.set_xlim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_ylim([np.min([xLim[0], yLim[0]]),np.max([xLim[1], yLim[1]])])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{longAxis/shortAxis:4f}")
    plt.suptitle(mouse)


    ellipseDict[mouse][fnameRot] = {
        'pos':posRot,
        'dirMat': dirMatRot,
        'emb': embRot,
        'D': DRot,
        'noiseIdx': noiseIdxRot,
        'cpos': cposRot,
        'cdirMat': cdirMatRot,
        'cemb': cembRot,
        'cent': centRot,
        'cent2D': centRot2D,
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