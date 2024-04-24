palette_deepsup = ["#cc9900ff", "#9900ffff"]
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
sup_mice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deep_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ12']

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



data_dir =  '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
save_dir = '/home/julio/Documents/DeepSup_project/DeepSup/rotation/'

for mouse in mice_list:
    print(f"Working on mouse {mouse}:")
    fileName =  mouse+'_df_dict.pkl'
    filePath = os.path.join(data_dir, mouse)
    save_dirFig = os.path.join(filePath, 'figures')
    animal = load_pickle(filePath,fileName)
    fnames = list(animal.keys())
    fnamePre = [fname for fname in fnames if 'lt' in fname][0]
    fnameRot = [fname for fname in fnames if 'rot' in fname][0]
    animalPre= copy.deepcopy(animal[fnamePre])
    animalRot= copy.deepcopy(animal[fnameRot])

    pos_pre = np.concatenate(animalPre['pos'].values, axis = 0)
    dir_pre = np.concatenate(animalPre['dir_mat'].values, axis=0)
    dir_pre[dir_pre==1] = -1
    dir_pre[dir_pre==2] = 1

    pos_rot = np.concatenate(animalRot['pos'].values, axis = 0)
    dir_rot = np.concatenate(animalRot['dir_mat'].values, axis=0)
    dir_rot[dir_rot==1] = -1
    dir_rot[dir_rot==2] = 1

    rotation_dict = dict()

    for emb_name in ['pca','isomap','umap']:
        emb_pre = np.concatenate(animalPre[emb_name].values, axis = 0)[:,:3]        
        pos_pre = np.concatenate(animalPre['pos'].values, axis = 0)
        dir_pre = np.concatenate(animalPre['dir_mat'].values, axis=0)[:,0]
        dir_pre[dir_pre==1] = -1
        dir_pre[dir_pre==2] = 1

        emb_rot = np.concatenate(animalRot[emb_name].values, axis = 0)[:,:3]
        pos_rot = np.concatenate(animalRot['pos'].values, axis = 0)
        dir_rot = np.concatenate(animalRot['dir_mat'].values, axis=0)[:,0]
        dir_rot[dir_rot==1] = -1
        dir_rot[dir_rot==2] = 1

        DPre = pairwise_distances(emb_pre)
        noise_pre = filter_noisy_outliers(emb_pre,DPre)
        max_dist = np.nanmax(DPre)
        emb_pre = emb_pre[~noise_pre,:]
        pos_pre = pos_pre[~noise_pre,:]
        dir_pre = dir_pre[~noise_pre]

        DRot = pairwise_distances(emb_rot)
        noise_rot = filter_noisy_outliers(emb_rot,DRot)
        max_dist = np.nanmax(DRot)
        emb_rot = emb_rot[~noise_rot,:]
        pos_rot = pos_rot[~noise_rot,:]
        dir_rot = dir_rot[~noise_rot]

        #compute centroids
        cent_pre, cent_rot, cent_pos, cent_dir = get_centroids(emb_pre, emb_rot, pos_pre[:,0], pos_rot[:,0], 
                                                        dir_pre, dir_rot, num_centroids=40) 

        #project into planes
        norm_vec_pre, cloud_center_pre = parametrize_plane(emb_pre)
        plane_emb_pre = project_onto_plane(emb_pre, norm_vec_pre, cloud_center_pre)

        norm_vec_rot, cloud_center_rot = parametrize_plane(emb_rot)
        plane_emb_rot = project_onto_plane(emb_rot, norm_vec_rot, cloud_center_rot)

        plane_cent_pre, plane_cent_rot, plane_cent_pos, plane_cent_dir = get_centroids(plane_emb_pre, plane_emb_rot, 
                                                                                            pos_pre[:,0], pos_rot[:,0], 
                                                                                            dir_pre, dir_rot, num_centroids=40) 
        #align them
        align_angle, align_mat = align_vectors(norm_vec_pre, cloud_center_pre, norm_vec_rot, cloud_center_rot)

        aligned_emb_rot =  apply_rotation_to_cloud(emb_rot, align_mat, cloud_center_rot)
        aligned_plane_emb_rot =  apply_rotation_to_cloud(plane_emb_rot, align_mat, cloud_center_rot)

        aligned_cent_rot =  apply_rotation_to_cloud(cent_rot, align_mat, cloud_center_rot)
        aligned_plane_cent_rot =  apply_rotation_to_cloud(plane_cent_rot, align_mat, cloud_center_rot)

        #compute angle of rotation
        angles = np.linspace(-np.pi,np.pi,200)
        error = find_rotation(plane_cent_pre, plane_cent_rot, -norm_vec_pre)
        norm_error = (np.array(error)-np.min(error))/(np.max(error)-np.min(error))
        signed_rotation_angle = angles[np.argmin(norm_error)]*180/np.pi
        rotation_angle = np.abs(signed_rotation_angle)
        print(f"\t{mouse} {emb_name}: {signed_rotation_angle:2f} degrees")

        rotated_aligned_cent_rot = rotate_cloud_around_axis(aligned_cent_rot, (np.pi/180)*signed_rotation_angle,norm_vec_pre)
        rotated_aligned_plane_cent_rot = rotate_cloud_around_axis(aligned_plane_cent_rot, (np.pi/180)*signed_rotation_angle,norm_vec_pre)
        rotated_aligned_emb_rot = rotate_cloud_around_axis(aligned_emb_rot, (np.pi/180)*signed_rotation_angle,norm_vec_pre)
        rotated_aligned_plane_emb_rot = rotate_cloud_around_axis(aligned_plane_emb_rot, (np.pi/180)*signed_rotation_angle,norm_vec_pre)

        rotated_cent_rot = rotate_cloud_around_axis(cent_rot, (np.pi/180)*signed_rotation_angle,norm_vec_pre)

        fig = plot_rotation(emb_pre, emb_rot, pos_pre, pos_rot, dir_pre, dir_rot, 
                    cent_pre, cent_rot, cent_pos, plane_cent_pre, plane_cent_rot, 
                    aligned_plane_cent_rot, rotated_aligned_plane_cent_rot, angles, error, signed_rotation_angle)
        plt.suptitle(f"{mouse} {emb_name}")
        plt.savefig(os.path.join(save_dir,f'{mouse}_{emb_name}_rotation_plot.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(save_dir,f'{mouse}_{emb_name}_rotation_plot.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)

        rotation_dict[emb_name] = {
            #initial data
            'emb_pre': emb_pre,
            'pos_pre': pos_pre,
            'dir_pre': dir_pre,
            'noise_pre': noise_pre,

            'emb_rot': emb_rot,
            'pos_rot': pos_rot,
            'dir_rot': dir_rot,
            'noise_rot': noise_rot,
            #centroids
            'cent_pre': cent_pre,
            'cent_rot': cent_rot,
            'cent_pos': cent_pos,
            'cent_dir': cent_dir,

            #project into plane
            'norm_vec_pre': norm_vec_pre,
            'cloud_center_pre': cloud_center_pre,
            'plane_emb_pre': plane_emb_pre,

            'norm_vec_rot': norm_vec_rot,
            'cloud_center_rot': cloud_center_rot,
            'plane_emb_rot': plane_emb_rot,

            #plane centroids
            'plane_cent_pre': plane_cent_pre,
            'plane_cent_rot': plane_cent_rot,
            'plane_cent_pos': plane_cent_pos,
            'plane_cent_dir': plane_cent_dir,

            #align planes
            'align_angle': align_angle,
            'align_mat': align_mat,

            'aligned_emb_rot': aligned_emb_rot,
            'aligned_plane_emb_rot': aligned_plane_emb_rot,
            'aligned_cent_rot': aligned_cent_rot,
            'aligned_plane_cent_rot': aligned_plane_cent_rot,

            #compute angle of rotation
            'angles': angles,
            'error': error,
            'norm_error': norm_error,
            'signed_rotation_angle': signed_rotation_angle,
            'rotation_angle': rotation_angle,

            #rotate post session
            'rotated_cent_rot': rotated_cent_rot,
            'rotated_aligned_cent_rot': rotated_aligned_cent_rot,
            'rotated_aligned_plane_cent_rot': rotated_aligned_plane_cent_rot,
            'rotated_aligned_emb_rot': rotated_aligned_emb_rot,
            'rotated_aligned_plane_emb_rot': rotated_aligned_plane_emb_rot,
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


data_dir =  '/home/julio/Documents/DeepSup_project/DeepSup/rotation/'
save_dir = '/home/julio/Documents/DeepSup_project/DeepSup/distance/'


for mouse in mice_list:
    print(f"Working on mouse {mouse}:")

    rotation_dict = load_pickle(data_dir, mouse+'_rotation_dict.pkl')
    distance_dict = dict()

    for emb_name in ['umap', 'isomap', 'pca']:

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

        print(f"\t{mouse} {emb_name}: {remap_dist:.2f} remap dist | {plane_remap_dist:.2f} remap dist plane")

        fig = plot_distance(cent_pre,cent_rot,cent_pos,cent_dir,
                plane_cent_pre,plane_cent_rot, plane_cent_pos, plane_cent_dir,
                ellipse_pre_fit_3D, ellipse_rot_fit_3D)
        plt.suptitle(f"{mouse} {emb_name}")
        plt.savefig(os.path.join(save_dir,f'{mouse}_{emb_name}_distance_plot.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(save_dir,f'{mouse}_{emb_name}_distance_plot.png'), dpi = 400,bbox_inches="tight")
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

    with open(os.path.join(save_dir,f'{mouse}_distance_dict.pkl'), 'wb') as f:
        pickle.dump(distance_dict, f)




#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT ROTATION                              |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/DeepSup_project/DeepSup/rotation/'
save_dir = '/home/julio/Documents/DeepSup_project/DeepSup/figures/'
fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, emb_name in enumerate(['umap','isomap','pca']):

    rotation_list = list()
    for mouse in mice_list:
        rotation_dict = load_pickle(data_dir, f'{mouse}_rotation_dict.pkl')
        rotation_list.append(rotation_dict[emb_name]['rotation_angle'])

    pd_dist = pd.DataFrame(data={'layer': ['deep']*5 + ['sup']*5,
                                     'dist': rotation_list})
    b = sns.boxplot(x='layer', y='dist', data=pd_dist,
                palette = palette_deepsup, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dist', data=pd_dist,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])


    if shapiro(rotation_list[:5]).pvalue<=0.05 or shapiro(rotation_list[5:]).pvalue<=0.05:
        ax[idx].set_title(f"{emb_name} p-value ks test: {stats.ks_2samp(rotation_list[:5], rotation_list[5:])[1]:.4f}")
    else:
        ax[idx].set_title(f"{emb_name} p-value t-test: {stats.ttest_ind(rotation_list[:5], rotation_list[5:], equal_var=True)[1]:.4f}")
    ax[idx].set_ylim([-2.5, 180.5])
    ax[idx].set_yticks([0,45,90,135,180]);
plt.tight_layout()

plt.savefig(os.path.join(save_dir,'rotation.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'rotation.png'), dpi = 400,bbox_inches="tight")



from scipy.stats import shapiro
from scipy import stats

for emb_name in ['umap','isomap','pca']:
    deep_rot = []
    sup_rot = []
    for mouse in mice_list:
        rotation_dict = load_pickle(data_dir, f'{mouse}_rotation_dict.pkl')
        if mouse in deep_mice:
            deep_rot.append(rotation_dict[emb_name]['rotation_angle'])
        if mouse in sup_mice:
            sup_rot.append(rotation_dict[emb_name]['rotation_angle'])

    deepShapiro = shapiro(deep_rot)
    supShapiro = shapiro(sup_rot)

    if deepShapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
        print(f'{emb_name} Rotation:',stats.ks_2samp(deep_rot, sup_rot))
    else:
        print(f'{emb_name} Rotation:', stats.ttest_ind(deep_rot, sup_rot))



#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT REMAP DIST                            |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/DeepSup_project/DeepSup/distance/'
save_dir = '/home/julio/Documents/DeepSup_project/DeepSup/figures/'

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, emb_name in enumerate(['umap','isomap','pca']):

    distance_list = list()
    for mouse in mice_list:
        distance_dict = load_pickle(data_dir, f'{mouse}_distance_dict.pkl')
        distance_list.append(distance_dict[emb_name]['plane_remap_dist'])

    pd_dist = pd.DataFrame(data={'layer': ['deep']*5 + ['sup']*5,
                                     'dist': distance_list})

    b = sns.barplot(x='layer', y='dist', data=pd_dist,
                palette = palette_deepsup, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dist', data=pd_dist,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])

    if shapiro(distance_list[:5]).pvalue<=0.05 or shapiro(distance_list[5:]).pvalue<=0.05:
        ax[idx].set_title(f"{emb_name} p-value ks test: {stats.ks_2samp(distance_list[:5], distance_list[5:])[1]:.4f}")
    else:
        ax[idx].set_title(f"{emb_name} p-value t-test: {stats.ttest_ind(distance_list[:5], distance_list[5:], equal_var=True)[1]:.4f}")

    ax[idx].set_ylim([0,0.45])
plt.tight_layout()

plt.savefig(os.path.join(save_dir,'distance_perimeter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'distance_perimeter.png'), dpi = 400,bbox_inches="tight")


fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, emb_name in enumerate(['umap','isomap','pca']):

    distance_list = list()
    for mouse in mice_list:
        distance_dict = load_pickle(data_dir, f'{mouse}_distance_dict.pkl')
        distance_list.append(distance_dict[emb_name]['remap_dist'])

    print(emb_name, ':', stats.ttest_ind(distance_list[:5], distance_list[5:], equal_var=True))
    pd_dist = pd.DataFrame(data={'layer': ['deep']*5 + ['sup']*5,
                                     'dist': distance_list})
    b = sns.barplot(x='layer', y='dist', data=pd_dist,
                palette = palette_deepsup, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dist', data=pd_dist,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])

    if shapiro(distance_list[:5]).pvalue<=0.05 or shapiro(distance_list[5:]).pvalue<=0.05:
        ax[idx].set_title(f"{emb_name} p-value ks test: {stats.ks_2samp(distance_list[:5], distance_list[5:])[1]:.4f}")
    else:
        ax[idx].set_title(f"{emb_name} p-value t-test: {stats.ttest_ind(distance_list[:5], distance_list[5:], equal_var=True)[1]:.4f}")

    ax[idx].set_ylim([0,3.5])

plt.tight_layout()

plt.savefig(os.path.join(save_dir,'distance_mean.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'distance_mean.png'), dpi = 400,bbox_inches="tight")



fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, emb_name in enumerate(['umap','isomap','pca']):

    distance_list = list()
    for mouse in mice_list:
        if ('GC5_nvista' in mouse or 'GC2' in mouse) and 'pca' in emb_name:
            continue;
        distance_dict = load_pickle(data_dir, f'{mouse}_distance_dict.pkl')
        distance_list.append(distance_dict[emb_name]['remap_dist'])

    if 'pca' in emb_name:
        print(emb_name, ':', stats.ttest_ind(distance_list[:3], distance_list[3:], equal_var=True))
        pd_dist = pd.DataFrame(data={'layer': ['deep']*3 + ['sup']*5,
                                         'dist': distance_list})
    else:
        print(emb_name, ':', stats.ttest_ind(distance_list[:5], distance_list[5:], equal_var=True))
        pd_dist = pd.DataFrame(data={'layer': ['deep']*5 + ['sup']*5,
                                         'dist': distance_list})
    b = sns.barplot(x='layer', y='dist', data=pd_dist,
                palette = palette_deepsup, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='dist', data=pd_dist,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])

    if 'pca' in emb_name:
        if shapiro(distance_list[:3]).pvalue<=0.05 or shapiro(distance_list[3:]).pvalue<=0.05:
            ax[idx].set_title(f"{emb_name} p-value ks test: {stats.ks_2samp(distance_list[:3], distance_list[3:])[1]:.4f}")
        else:
            ax[idx].set_title(f"{emb_name} p-value t-test: {stats.ttest_ind(distance_list[:3], distance_list[3:], equal_var=True)[1]:.4f}")
    else:
        if shapiro(distance_list[:5]).pvalue<=0.05 or shapiro(distance_list[5:]).pvalue<=0.05:
            ax[idx].set_title(f"{emb_name} p-value ks test: {stats.ks_2samp(distance_list[:5], distance_list[5:])[1]:.4f}")
        else:
            ax[idx].set_title(f"{emb_name} p-value t-test: {stats.ttest_ind(distance_list[:5], distance_list[5:], equal_var=True)[1]:.4f}")
    ax[idx].set_ylim([0,3.5])

plt.tight_layout()

from scipy.stats import shapiro
from scipy import stats

for emb_name in ['umap','isomap','pca']:
    deep_dist = []
    sup_dist = []
    for mouse in mice_list:
        distance_dict = load_pickle(data_dir, f'{mouse}_distance_dict.pkl')
        if mouse in deep_mice:
            deep_dist.append(distance_dict[emb_name]['remap_dist'])
        if mouse in sup_mice:
            sup_dist.append(distance_dict[emb_name]['remap_dist'])

    deepShapiro = shapiro(deep_dist)
    supShapiro = shapiro(sup_dist)

    if deepShapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
        print(f'{emb_name} Distance:',stats.ks_2samp(deep_dist, sup_dist))
    else:
        print(f'{emb_name} Distance:', stats.ttest_ind(deep_dist, sup_dist))
