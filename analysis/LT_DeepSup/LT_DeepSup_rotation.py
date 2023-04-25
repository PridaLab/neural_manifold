import umap, math
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
from datetime import datetime
import os
from neural_manifold import dimensionality_reduction as dim_red
from sklearn.metrics import pairwise_distances

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

#__________________________________________________________________________
#|                                                                        |#
#|                              COMPUTE UMAP                              |#
#|________________________________________________________________________|#

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

signal_name = 'clean_traces'
n_neigh = 120
dim = 3
min_dist = 0.1

data_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/processed_data'

for mouse in mice_list:
    print(f'Working on mouse: {mouse}')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)

    save_dir_fig = os.path.join(file_path, 'figures')
    animal_dict = load_pickle(file_path,file_name)
    fnames = list(animal_dict.keys())
    animal_p= copy.deepcopy(animal_dict[fnames[0]])
    animal_r= copy.deepcopy(animal_dict[fnames[1]])


    signal_p = copy.deepcopy(np.concatenate(animal_p[signal_name].values, axis=0))
    pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
    index_mat_p = copy.deepcopy(np.concatenate(animal_p['index_mat'].values, axis=0))

    signal_r = copy.deepcopy(np.concatenate(animal_r[signal_name].values, axis=0))
    pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
    index_mat_r = copy.deepcopy(np.concatenate(animal_p['index_mat'].values, axis=0))

    #%%all data
    index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
    concat_signal = np.vstack((signal_p, signal_r))
    model = umap.UMAP(n_neighbors =n_neigh, n_components =dim, min_dist=min_dist)
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

    ax = plt.subplot(1,2,2, projection = '3d')
    ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
    ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')
    plt.suptitle(f"{mouse}: {signal_name} - nn: {n_neigh} - dim: {dim}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_fig,mouse+'_saved_umap.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
    plt.savefig(os.path.join(save_dir_fig,'_saved_umap.svg'), dpi = 400,bbox_inches="tight",transparent=True)

    index_mat_p = np.concatenate(animal_p["index_mat"].values, axis=0)
    animal_p['umap'] = [emb_p[index_mat_p[:,0]==animal_p["trial_id"][idx] ,:] 
                                   for idx in animal_p.index]
    index_mat_r = np.concatenate(animal_r["index_mat"].values, axis=0)
    animal_r['umap'] = [emb_r[index_mat_r[:,0]==animal_r["trial_id"][idx] ,:] 
                                   for idx in animal_r.index]

    new_animal_dict = {
        fnames[0]: animal_p,
        fnames[1]: animal_r
    }
    with open(os.path.join(file_path, mouse+"_df_dict.pkl"), "wb") as file:
        pickle.dump(new_animal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(file_path, mouse+"_umap_object.pkl"), "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)



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

save_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/rotation'

deep_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
sup_list = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']


rot_error_deep_dict = dict()
rot_error_deep = np.zeros((100, len(deep_list)))
for m_idx, mouse in enumerate(deep_list):
    file_name = mouse + '_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    animal_dict = load_pickle(file_path,file_name)
    fnames = list(animal_dict.keys())
    animal_p= copy.deepcopy(animal_dict[fnames[0]])
    animal_r= copy.deepcopy(animal_dict[fnames[1]])

    emb_p = copy.deepcopy(np.concatenate(animal_p['umap'].values, axis=0))
    pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
    dir_mat_p = copy.deepcopy(np.concatenate(animal_p['dir_mat'].values, axis=0))

    emb_r = copy.deepcopy(np.concatenate(animal_r['umap'].values, axis=0))
    pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
    dir_mat_r = copy.deepcopy(np.concatenate(animal_r['dir_mat'].values, axis=0))

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
    rot_error_deep[:,m_idx] = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))

    rot_error_deep_dict[mouse] = {
        'emb_p': emb_p,
        'emb_r': emb_r,
        'pos_p': pos_p,
        'pos_r': pos_r,
        'dir_mat_p': dir_mat_p,
        'dir_mat_r': dir_mat_r,
        'noiseIdx_p': noiseIdx_p,
        'noiseIdx_r': noiseIdx_r,
        'cent_p': cent_p,
        'cent_r': cent_r,
        'mid_p': mid_p,
        'mid_r': mid_r,
        'norm_vector': norm_vector,
        'angles': angles,
        'error': error
    }
    with open(os.path.join(save_dir, "rot_error_deep_dict.pkl"), "wb") as file:
        pickle.dump(rot_error_deep_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    np.save('rot_error_deep.npy', rot_error_deep)
    # plt.figure()
    # ax = plt.subplot(121, projection='3d')
    # ax.scatter(*cemb_p[:,:3].T, color ='b', s= 30, cmap = 'magma')
    # ax.scatter(*cemb_r[:,:3].T, color = 'r', s= 30, cmap = 'magma')
    # ax.plot([mid_p[0,0], mid_r[0,0]], [mid_p[1,0], mid_r[1,0]], [mid_p[2,0],mid_r[2,0]], color = 'r')

    # ax = plt.subplot(122)
    # ax.plot(angles*180/np.pi, error)
    # plt.suptitle(f"{mouse}")
    # plt.tight_layout()

rot_error_sup_dict = dict()
rot_error_sup = np.zeros((100, len(sup_list)))
for m_idx, mouse in enumerate(sup_list):
    file_name = mouse + '_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    animal_dict = load_pickle(file_path,file_name)
    fnames = list(animal_dict.keys())
    animal_p= copy.deepcopy(animal_dict[fnames[0]])
    animal_r= copy.deepcopy(animal_dict[fnames[1]])

    emb_p = copy.deepcopy(np.concatenate(animal_p['umap'].values, axis=0))
    pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
    dir_mat_p = copy.deepcopy(np.concatenate(animal_p['dir_mat'].values, axis=0))

    emb_r = copy.deepcopy(np.concatenate(animal_r['umap'].values, axis=0))
    pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
    dir_mat_r = copy.deepcopy(np.concatenate(animal_r['dir_mat'].values, axis=0))

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
    rot_error_sup[:,m_idx] = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
    rot_error_sup_dict[mouse] = {
        'emb_p': emb_p,
        'emb_r': emb_r,
        'pos_p': pos_p,
        'pos_r': pos_r,
        'dir_mat_p': dir_mat_p,
        'dir_mat_r': dir_mat_r,
        'noiseIdx_p': noiseIdx_p,
        'noiseIdx_r': noiseIdx_r,
        'cent_p': cent_p,
        'cent_r': cent_r,
        'mid_p': mid_p,
        'mid_r': mid_r,
        'norm_vector': norm_vector,
        'angles': angles,
        'error': error
    }
    with open(os.path.join(save_dir, "rot_error_sup_dict.pkl"), "wb") as file:
        pickle.dump(rot_error_sup_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    np.save('rot_error_sup.npy', rot_error_sup)


angles_deg = angles*180/np.pi
plt.figure()
ax = plt.subplot(111)
m = np.mean(rot_error_deep,axis=1)
sd = np.std(rot_error_deep,axis=1)
ax.plot(angles_deg, m, label = 'deep')
ax.fill_between(angles_deg, m-sd, m+sd, alpha = 0.3)
m = np.mean(rot_error_sup,axis=1)
sd = np.std(rot_error_sup,axis=1)
ax.plot(angles_deg, m, label = 'sup')
ax.fill_between(angles_deg, m-sd, m+sd, alpha = 0.3)
ax.set_xlabel('Angle of rotation (º)')
ax.set_ylabel('Distance between sessions')
ax.legend()
plt.savefig(os.path.join(save_dir,'DeepSup_rotation.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
plt.savefig(os.path.join(save_dir,'DeepSup_rotation.svg'), dpi = 400,bbox_inches="tight",transparent=True)

max_deep = np.argmax(rot_error_deep[], axis= 0)
max_sup = np.argmax(rot_error_deep, axis= 0)


