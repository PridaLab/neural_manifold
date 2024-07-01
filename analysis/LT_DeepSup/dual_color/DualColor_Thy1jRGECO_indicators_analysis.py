import scipy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import copy
import os
import pickle
import seaborn as sns
from scipy import stats


def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

base_dir = '/home/julio/Documents/DeepSup_project/DualColor/Thy1jRGECO/'

#__________________________________________________________________________
#|                                                                        |#
#|                        COMPUTE ROTATION ANGLES                         |#
#|________________________________________________________________________|#

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nnDist < np.percentile(nnDist, 20)
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

for mouse in ['Thy1jRGECO22', 'Thy1jRGECO23']:
    mouse_dict = load_pickle(os.path.join(base_dir, 'processed_data', mouse), mouse+'_data_dict.pkl')
    rotation_dict = load_pickle(os.path.join(base_dir, 'rotation', mouse), mouse+'_rotation_dict.pkl')


    save_dir = os.path.join(base_dir, 'ca_indicators', mouse)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)


    indicator_rot_dict  = {}
    indicator_rot_dict['deep_green'] = copy.deepcopy(rotation_dict['deep'])


    #__________________________________________________________________________
    #|                                                                        |#
    #|                                 DEEP RED                               |#
    #|________________________________________________________________________|#

    
    sup_signal_pre = mouse_dict['registered_clean_traces']['signal_sup_pre']

    all_signal_pre = mouse_dict['registered_clean_traces']['signal_all_pre']
    all_signal_rot = mouse_dict['registered_clean_traces']['signal_all_rot']

    matched_indexes = np.zeros((sup_signal_pre.shape[1],))*np.nan
    cells_to_check = np.arange(all_signal_pre.shape[1]).astype(int)
    for cell_sup in range(sup_signal_pre.shape[1]):
        for cell_all in cells_to_check:
            corr_coeff = np.corrcoef(sup_signal_pre[:,cell_sup], all_signal_pre[:,cell_all])[0,1]
            if corr_coeff > 0.999:
                matched_indexes[cell_sup] = cell_all;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_all)[0])
                break;
    matched_indexes = matched_indexes.astype(int)


    cell_to_keep = np.zeros((all_signal_pre.shape[1],))==0
    cell_to_keep[matched_indexes] = False


    deep_signal_pre = all_signal_pre[:, cell_to_keep]
    deep_signal_rot = all_signal_rot[:, cell_to_keep]



    pos_pre = mouse_dict['speed_filtered_signals']['pos_pre']
    dir_pre = mouse_dict['speed_filtered_signals']['dir_pre']
    pos_rot = mouse_dict['speed_filtered_signals']['pos_rot']
    dir_rot = mouse_dict['speed_filtered_signals']['dir_rot']

    #%%all data
    index = np.vstack((np.zeros((deep_signal_pre.shape[0],1)),np.ones((deep_signal_rot.shape[0],1))))
    signal_concat = np.vstack((deep_signal_pre, deep_signal_rot))
    model = umap.UMAP(n_neighbors=120, n_components =3, min_dist=0.1)
    model.fit(signal_concat)
    emb_concat_deep = model.transform(signal_concat)
    emb_pre = emb_concat_deep[index[:,0]==0,:]
    emb_rot = emb_concat_deep[index[:,0]==1,:]

    D = pairwise_distances(emb_pre)
    noise_idx_pre = filter_noisy_outliers(emb_pre,D=D)
    deep_signal_pre = deep_signal_pre[~noise_idx_pre,:]
    deep_emb_pre = emb_pre[~noise_idx_pre,:]

    deep_pos_pre = pos_pre[~noise_idx_pre,:]
    deep_dir_pre = dir_pre[~noise_idx_pre][:,0]

    D = pairwise_distances(emb_rot)
    noise_idx_rot = filter_noisy_outliers(emb_rot,D=D)
    deep_signal_rot = deep_signal_rot[~noise_idx_rot,:]
    deep_emb_rot = emb_rot[~noise_idx_rot,:]

    deep_pos_rot = pos_rot[~noise_idx_rot,:]
    deep_dir_rot = dir_rot[~noise_idx_rot][:,0]


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
    print(f"\tDeep red: {signed_deep_rotation_angle:2f} degrees")

    rotated_aligned_deep_cent_rot = rotate_cloud_around_axis(aligned_deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_plane_deep_cent_rot = rotate_cloud_around_axis(aligned_plane_deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_deep_emb_rot = rotate_cloud_around_axis(aligned_deep_emb_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)
    rotated_aligned_plane_deep_emb_rot = rotate_cloud_around_axis(aligned_plane_deep_emb_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)

    rotated_deep_cent_rot = rotate_cloud_around_axis(deep_cent_rot, (np.pi/180)*signed_deep_rotation_angle,deep_norm_vec_pre)

    fig = plot_rotation(deep_emb_pre, deep_emb_rot, deep_pos_pre, deep_pos_rot, deep_dir_pre, deep_dir_rot, 
                deep_cent_pre, deep_cent_rot, deep_cent_pos, plane_deep_cent_pre, plane_deep_cent_rot, 
                aligned_plane_deep_cent_rot, rotated_aligned_plane_deep_cent_rot, deep_angles, deep_error, signed_deep_rotation_angle)
    plt.suptitle(f"{mouse} deep")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_red_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_red_rotation_plot.png'), dpi = 400,bbox_inches="tight")


    indicator_rot_dict['deep_red'] = {
        #initial data
        'deep_emb_pre': deep_emb_pre,
        'deep_pos_pre': deep_pos_pre,
        'deep_dir_pre': deep_dir_pre,

        'deep_emb_rot': deep_emb_rot,
        'deep_pos_rot': deep_pos_rot,
        'deep_dir_rot': deep_dir_rot,
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


    with open(os.path.join(save_dir, mouse+"_indicator_rot_dict.pkl"), "wb") as file:
        pickle.dump(indicator_rot_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                          PLOT ROTATION ANGLES                          |#
#|________________________________________________________________________|#

rot_angle = list()
mouse_list = list()
channel_list = list()
for mouse in ['Thy1jRGECO22', 'Thy1jRGECO23']:

    rotation_dir = os.path.join(base_dir, 'ca_indicators', mouse)
    rotation_dict = load_pickle(rotation_dir, mouse+'_indicator_rot_dict.pkl')
    rot_angle.append(rotation_dict['deep_green']['deep_rotation_angle'])
    channel_list.append('deep green')
    mouse_list.append(mouse)
    rot_angle.append(rotation_dict['deep_red']['deep_rotation_angle'])
    channel_list.append('deep red')
    mouse_list.append(mouse)

pd_angle = pd.DataFrame(data={'mouse': mouse_list,
                     'channel': channel_list,
                     'angle': rot_angle})    

fig, ax = plt.subplots(1, 1, figsize=(6,6))

b = sns.boxplot(x='channel', y='angle', data=pd_angle,
            linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='channel', y='angle', data=pd_angle,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylim([-2.5, 180.5])
ax.set_yticks([0,45,90,135,180]);

plt.savefig(os.path.join(os.path.join(base_dir, 'ca_indicators'),f'deep_ca_indicator_rotation.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(os.path.join(base_dir, 'ca_indicators'),f'deep_ca_indicator_rotation.png'), dpi = 400,bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|                            COMPUTE DISTANCE                            |#
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


for mouse in ['Thy1jRGECO22','Thy1jRGECO23']:
    print(f"Working on {mouse}:")

    save_dir = os.path.join(base_dir, 'ca_indicators', mouse)
    distance_dir = os.path.join(base_dir, 'distance', mouse)

    rotation_dict = load_pickle(save_dir, mouse+'_indicator_rot_dict.pkl')
    distance_dict = load_pickle(distance_dir, mouse+'_distance_dict.pkl')



    indicator_dist_dict  = {}
    indicator_dist_dict['deep_green'] = copy.deepcopy(distance_dict['deep'])

    #__________________________________________________________________________
    #|                                                                        |#
    #|                                   DEEP                                  |#
    #|________________________________________________________________________|#


    deep_cent_pre = rotation_dict['deep_red']['deep_cent_pre']
    deep_cent_rot = rotation_dict['deep_red']['deep_cent_rot']
    deep_cent_pos = rotation_dict['deep_red']['deep_cent_pos']
    deep_cent_dir = rotation_dict['deep_red']['deep_cent_dir']

    deep_inter_dist = np.linalg.norm(deep_cent_pre.mean(axis=0)-deep_cent_rot.mean(axis=0))
    deep_intra_dist_pre = np.percentile(pairwise_distances(deep_cent_pre),95)/2
    deep_intra_dist_rot = np.percentile(pairwise_distances(deep_cent_rot),95)/2
    deep_remap_dist = deep_inter_dist/np.mean((deep_intra_dist_pre, deep_intra_dist_rot))

    plane_deep_cent_pre = rotation_dict['deep_red']['plane_deep_cent_pre']
    plane_deep_cent_rot = rotation_dict['deep_red']['plane_deep_cent_rot']
    deep_norm_vector_pre = rotation_dict['deep_red']['deep_norm_vec_pre']
    plane_deep_cent_pos = rotation_dict['deep_red']['plane_deep_cent_pos']
    plane_deep_cent_dir = rotation_dict['deep_red']['plane_deep_cent_dir']
    deep_norm_vector_rot = rotation_dict['deep_red']['deep_norm_vec_rot']


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
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_red_distance_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_deep_red_distance_plot.png'), dpi = 400,bbox_inches="tight")

    indicator_dist_dict['deep_red'] = {

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



    with open(os.path.join(save_dir, mouse+"_indicator_dist_dict.pkl"), "wb") as file:
        pickle.dump(indicator_dist_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT DISTANCE                               |#
#|________________________________________________________________________|#

########################### mean

emb_distance_list = list()
mouse_list = list()
channel_list = list()

for mouse in mice_list:

    distance_dir = os.path.join(base_dir, 'ca_indicators', mouse)
    distance_dict = load_pickle(distance_dir, mouse+'_indicator_dist_dict.pkl')

    mouse_list += [mouse]*2
    emb_distance_list.append(distance_dict['deep_green']['deep_remap_dist'])
    channel_list.append('deep_green')
    emb_distance_list.append(distance_dict['deep_red']['deep_remap_dist'])
    channel_list.append('deep_red')

pd_distance = pd.DataFrame(data={'mouse': mouse_list,
                     'channel': channel_list,
                     'distance': emb_distance_list})    

fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='channel', y='distance', data=pd_distance,
            linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='channel', y='distance', data=pd_distance,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
plt.savefig(os.path.join(save_dir,f'dual_mean_distance.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dual_mean_distance.png'), dpi = 400,bbox_inches="tight")


