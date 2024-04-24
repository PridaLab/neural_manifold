import numpy as np
import random, os, copy
import matplotlib.pyplot as plt
import umap
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_distances
import pickle
# from structure_index import compute_structure_index, draw_graph
# from sklearn.manifold import Isomap

class StatisticModel:
    def __init__(self,pos,dir_mov, rot_position= None, rot_dir_movement = None, num_cells=300):
        self.position = pos
        self.num_cells = num_cells
        self.dir_movement = dir_mov

        if not isinstance(rot_position, type(None)):
            self.rot_position = rot_position
        if not isinstance(rot_dir_movement, type(None)):
            self.rot_dir_movement = rot_dir_movement     

        self.min_pos = np.percentile(self.position[:,0],5)
        self.max_pos = np.percentile(self.position[:,0],95)

    def gaus(self,x,sig,x0):
        return np.exp(-(((x-x0)**2)/(2*sig**2)));

    def compute_one_field(self, mu_gauss = 8, sig_gauss = 3, same_prob= 0.2, xmirror_prob=0.3, remap_prob = 0.2, remap_no_field_prob= 0.25, no_field_prob= 0.05):
        min_pos = self.min_pos
        max_pos = self.max_pos
        #stats
        one_field_pd = pd.DataFrame(np.zeros((1,6)),
                            columns=['field_type','main_dir', 'sigma_left','x0_left','sigma_right', 'x0_right'])
            #cell type
        field_type_prob = random.uniform(0,1)
        if field_type_prob<no_field_prob: #no place cell
            field_type = 'no_field'
        elif field_type_prob<no_field_prob+same_prob: #same field on the other direction
            field_type = 'same_field'
        elif field_type_prob<no_field_prob+same_prob+xmirror_prob: #x-mirror
            field_type = 'xmirror_field'
        elif field_type_prob<no_field_prob+same_prob+xmirror_prob+remap_prob:
            field_type = 'remap_field'
        else:
            field_type = 'remap_no_field'
        one_field_pd.loc[0,'field_type'] = field_type
            #main direction
        main_dir_num = int(random.uniform(0,1)<0.5)
        weak_dir_num = np.abs(main_dir_num-1)
        if main_dir_num==0:
            main_dir = 'left'
            weak_dir = 'right'
        else:
            main_dir = 'right'
            weak_dir = 'left'
        one_field_pd.loc[0, 'main_dir'] = main_dir
        main_dir_idx = self.dir_movement==main_dir_num
        weak_dir_idx = self.dir_movement==weak_dir_num
            #sigma
        one_field_pd.loc[0, 'sigma_'+main_dir] = random.gauss(mu_gauss, sig_gauss)
        if (field_type == 'same_field') or (field_type == 'xmirror_field'):
            one_field_pd.loc[0, 'sigma_'+weak_dir] = one_field_pd.loc[0, 'sigma_'+main_dir]
        else:
            one_field_pd.loc[0, 'sigma_'+weak_dir] = random.gauss(mu_gauss, sig_gauss)

            #x0 place field
        if field_type == 'no_field':
            one_field_pd.loc[0, 'x0_'+main_dir] = min_pos-max_pos
            one_field_pd.loc[0, 'x0_'+weak_dir] = min_pos-max_pos
        else:
            one_field_pd.loc[0, 'x0_'+main_dir] = random.uniform(min_pos, max_pos)
            if field_type == 'same_field':
                one_field_pd.loc[0, 'x0_'+weak_dir] = one_field_pd.loc[0, 'x0_'+main_dir]
            elif field_type == 'xmirror_field':
                one_field_pd.loc[0, 'sigma_'+weak_dir] = max_pos-one_field_pd.loc[0, 'x0_'+main_dir]+min_pos
            elif field_type == 'remap_field':
                one_field_pd.loc[0, 'x0_'+weak_dir] = random.uniform(min_pos, max_pos)
            else:
                one_field_pd.loc[0, 'x0_'+weak_dir] = min_pos-max_pos

        return one_field_pd

    def compute_fields(self, mu_gauss= 8, sig_gauss= 3, same_prob= 0.2, xmirror_prob=0.3, remap_prob= 0.2, remap_no_field_prob= 0.25, no_field_prob= 0.05, num_bins = 100):

        assert np.abs(same_prob+xmirror_prob+remap_prob+remap_no_field_prob+no_field_prob-1)<19e-4, 'field probabilities must add 1'
        min_pos = self.min_pos
        max_pos = self.max_pos

        self.field_types = pd.DataFrame(np.zeros((self.num_cells,6)),
                                columns=['field_type','main_dir', 'sigma_left','x0_left','sigma_right', 'x0_right'])

        self.field_type_params = {
            'mu_gauss': mu_gauss,
            'sig_gauss': sig_gauss,
            'no_field_prob': no_field_prob,
            'same_prob': same_prob,
            'xmirror_prob': xmirror_prob,
            'remap_prob': remap_prob,
            'remap_no_field_prob': remap_no_field_prob
        }

        self.place_fields = np.zeros((num_bins,self.num_cells,2))
        self.space_bins = np.linspace(min_pos, max_pos, num_bins)

        for cell in range(self.num_cells):

            one_field_pd = self.compute_one_field(mu_gauss, sig_gauss, same_prob, xmirror_prob, remap_prob, remap_no_field_prob, no_field_prob)

            self.field_types.loc[cell,:] = copy.deepcopy(one_field_pd.loc[0,:])
            #compute place_field main direction
            self.place_fields[:, cell, 0] = self.gaus(self.space_bins,self.field_types.loc[cell, 'sigma_right'],
                                                                            self.field_types.loc[cell, 'x0_right'])
            #compute place_field weak direction
            self.place_fields[:, cell, 1] = self.gaus(self.space_bins,self.field_types.loc[cell, 'sigma_left'],
                                                                            self.field_types.loc[cell, 'x0_left'])

    def compute_traces(self,noise_sigma = 0.01):
        if not hasattr(self, 'field_types'):
            self.compute_fields()

        self.signal_params = {
            'noise_sigma': noise_sigma
        }

        self.signal = np.zeros((self.position.shape[0], self.num_cells))
        for cell in range(self.num_cells):

            main_dir = self.field_types.loc[cell,'main_dir']
            if main_dir == 'right': 
                weak_dir = 'left';
                main_dir_num = 0;
                weak_dir_num = 1;

            else: 
                weak_dir='right';
                main_dir_num = 1;
                weak_dir_num = 0;

            main_dir_idx = self.dir_movement==main_dir_num
            weak_dir_idx = self.dir_movement==weak_dir_num

            #compute signal main direction
            self.signal[main_dir_idx, cell] = self.gaus(self.position[main_dir_idx,0],
                                                                self.field_types.loc[cell, 'sigma_'+main_dir],
                                                                self.field_types.loc[cell, 'x0_'+main_dir])
            self.signal[main_dir_idx, cell] += np.random.normal(0, noise_sigma,self.position[main_dir_idx,0].shape[0])

            #compute signal weak direction
            self.signal[weak_dir_idx, cell] = self.gaus(self.position[weak_dir_idx,0],
                                                                self.field_types.loc[cell, 'sigma_'+weak_dir],
                                                                self.field_types.loc[cell, 'x0_'+weak_dir])
            self.signal[weak_dir_idx, cell] += np.random.normal(0, noise_sigma,self.position[weak_dir_idx,0].shape[0])

    def clean_traces(self,smooth_sigma = 5):
        self.signal_params['smooth_sigma'] = smooth_sigma
        if hasattr(self, 'signal'):
            self.clean_signal = gaussian_filter1d(self.signal, sigma = smooth_sigma, axis = 0)

        if hasattr(self, 'rot_signal'):
            self.rot_clean_signal = gaussian_filter1d(self.rot_signal, sigma = smooth_sigma, axis = 0)

    def compute_cell_types(self, allo_prob=0.2, local_anchored_prob=0.6, remap_prob = 0.15, remap_no_field_prob = 0.05):

        assert np.abs(allo_prob+local_anchored_prob+remap_prob+remap_no_field_prob-1)<10e-4, 'cell type probabilities must add 1'

        if not hasattr(self, 'field_types'):
            self.compute_fields()

        min_pos = self.min_pos
        max_pos = self.max_pos

        self.cell_types = pd.DataFrame(np.zeros((self.num_cells,1))*np.nan,
                                columns=['cell_type'])

        self.cell_type_params = {
            'allo_prob': allo_prob,
            'local_anchored_prob': local_anchored_prob,
            'remap_prob': remap_prob,
            'remap_no_field_prob': remap_no_field_prob
        }

        for cell in range(self.num_cells):
            cell_type_prob = random.uniform(0,1)


            old_field_type = self.field_types.loc[cell,'field_type']
            if old_field_type == 'no_field': #it was not a place field before
                if cell_type_prob<remap_prob: #create new place field
                    cell_type = 'remap_cell'
                else: #no field
                    cell_type = 'no_field'
            else:       
                if cell_type_prob<allo_prob: #same field on the other direction
                    cell_type = 'allocentric_cell'
                elif cell_type_prob<allo_prob+local_anchored_prob: #x-mirror
                    cell_type = 'local_anchored_cell'
                elif cell_type_prob<allo_prob+local_anchored_prob+remap_prob:
                    cell_type = 'remap_cell'
                else:
                    cell_type = 'remap_no_field_cell'

            self.cell_types.loc[cell,'cell_type'] = cell_type

    def compute_rotation_fields(self):

        if not hasattr(self, 'cell_types'):
            self.compute_cell_types()

        min_pos = self.min_pos
        max_pos = self.max_pos

        mu_gauss = self.field_type_params['mu_gauss']
        sig_gauss = self.field_type_params['sig_gauss']
        no_field_prob = self.field_type_params['no_field_prob']
        same_prob = self.field_type_params['same_prob']
        xmirror_prob = self.field_type_params['xmirror_prob']
        remap_prob = self.field_type_params['remap_prob']
        remap_no_field_prob = self.field_type_params['remap_no_field_prob']

        self.rotation_field_types = copy.deepcopy(self.field_types)
        self.rotation_place_fields = copy.deepcopy(self.place_fields)*0
        for cell in range(self.num_cells):
            cell_type = self.cell_types.loc[cell,'cell_type']

            if (cell_type == 'no_field') or (cell_type == 'remap_no_field_cell'):
                self.rotation_field_types.loc[cell, 'x0_left'] = min_pos-max_pos
                self.rotation_field_types.loc[cell, 'x0_right'] = min_pos-max_pos
                self.rotation_field_types.loc[cell, 'field_type'] = 'no_field'
                self.rotation_field_types.loc[cell, 'main_dir'] = 'left'


            elif cell_type == 'local_anchored_cell':
                old_main_dir = self.field_types.loc[cell, 'main_dir']
                if old_main_dir == 'right':
                    self.rotation_field_types.loc[cell, 'main_dir'] = 'left'
                else:
                    self.rotation_field_types.loc[cell, 'main_dir'] = 'right'

                old_sigma_left = self.field_types.loc[cell, 'sigma_left']
                old_x0_left = self.field_types.loc[cell, 'x0_left']
                old_sigma_right = self.field_types.loc[cell, 'sigma_right']
                old_x0_right = self.field_types.loc[cell, 'x0_right']

                self.rotation_field_types.loc[cell, 'sigma_left'] = old_sigma_right
                self.rotation_field_types.loc[cell, 'x0_left'] = max_pos-old_x0_right+min_pos
                self.rotation_field_types.loc[cell, 'sigma_right'] = old_sigma_left
                self.rotation_field_types.loc[cell, 'x0_right'] = max_pos-old_x0_left+min_pos

            elif cell_type == 'remap_cell':

                one_field_pd = self.compute_one_field(mu_gauss, sig_gauss, same_prob, xmirror_prob, remap_prob, remap_no_field_prob, no_field_prob)
                self.rotation_field_types.loc[cell,:] = copy.deepcopy(one_field_pd.loc[0,:])


            #compute place_field main direction
            self.rotation_place_fields[:, cell, 0] = self.gaus(self.space_bins,self.rotation_field_types.loc[cell, 'sigma_right'],
                                                                            self.rotation_field_types.loc[cell, 'x0_right'])
            #compute place_field weak direction
            self.rotation_place_fields[:, cell, 1] = self.gaus(self.space_bins,self.rotation_field_types.loc[cell, 'sigma_left'],
                                                                            self.rotation_field_types.loc[cell, 'x0_left'])

    def compute_rotation_traces(self,rot_position=None, rot_dir_movement=None, noise_sigma = None):
        if not hasattr(self, 'rotation_field_types'):
            self.compute_rotation_fields()

        assert not isinstance(rot_position, type(None)) or hasattr(self, 'rot_position'), 'you need to input rot_position'
        assert not isinstance(rot_dir_movement, type(None)) or hasattr(self, 'rot_dir_movement'), 'you need to input rot_position'

        if not isinstance(rot_position, type(None)):
            self.rot_position = rot_position
        if not isinstance(rot_dir_movement, type(None)):
            self.rot_dir_movement = rot_dir_movement     

        if isinstance(noise_sigma, type(None)):
            noise_sigma = self.signal_params['noise_sigma']

        self.rot_signal = np.zeros((self.rot_position.shape[0], self.num_cells))

        for cell in range(self.num_cells):

            main_dir = self.rotation_field_types.loc[cell,'main_dir']
            if main_dir == 'right': 
                weak_dir = 'left';
                main_dir_num = 0;
                weak_dir_num = 1;

            else: 
                weak_dir='right';
                main_dir_num = 1;
                weak_dir_num = 0;

            main_dir_idx = self.rot_dir_movement==main_dir_num
            weak_dir_idx = self.rot_dir_movement==weak_dir_num

            #compute signal main direction
            self.rot_signal[main_dir_idx, cell] = self.gaus(self.rot_position[main_dir_idx,0],
                                                                self.rotation_field_types.loc[cell, 'sigma_'+main_dir],
                                                                self.rotation_field_types.loc[cell, 'x0_'+main_dir])
            self.rot_signal[main_dir_idx, cell] += np.random.normal(0, noise_sigma,self.rot_position[main_dir_idx,0].shape[0])

            #compute signal weak direction
            self.rot_signal[weak_dir_idx, cell] = self.gaus(self.rot_position[weak_dir_idx,0],
                                                                self.rotation_field_types.loc[cell, 'sigma_'+weak_dir],
                                                                self.rotation_field_types.loc[cell, 'x0_'+weak_dir])
            self.rot_signal[weak_dir_idx, cell] += np.random.normal(0, noise_sigma,self.rot_position[weak_dir_idx,0].shape[0])

    def compute_umap_both(self, n_neigh= 120, dim = 3, min_dist = 0.1):

        assert hasattr(self, 'clean_signal'), 'first call self.compute_clean_traces()'
        assert hasattr(self, 'rot_clean_signal'), 'first call self.compute_rotation_clean_traces()'


        signal_both = np.vstack((self.clean_signal, self.rot_clean_signal))
        index_signal = np.zeros((signal_both.shape[0],1))
        index_signal[self.clean_signal.shape[0]:,0] += 1

        self.umap_model = umap.UMAP(n_neighbors=n_neigh, n_components=dim, min_dist=min_dist)
        self.umap_model.fit(signal_both)
        emb_both = self.umap_model.transform(signal_both)

        self.umap_emb = emb_both[index_signal[:,0]==0,:]
        self.rot_umap_emb = emb_both[index_signal[:,0]==1,:]

    def filter_noisy_outliers(self, data, D=None, prctile_th = 10, prctile_noise = 5):
        if isinstance(D, type(None)):
            D = pairwise_distances(data)
        np.fill_diagonal(D, np.nan)
        nn_dist = np.sum(D < np.nanpercentile(D,prctile_th), axis=1)
        noiseIdx = nn_dist < np.percentile(nn_dist, prctile_noise)
        return noiseIdx

    def clean_umap(self, prctile_th = 10, prctile_noise = 5):

        if hasattr(self, 'umap_emb'):
            D = pairwise_distances(self.umap_emb)
            noise_idx = self.filter_noisy_outliers(self.umap_emb,D, prctile_th,prctile_noise)
            self.nout_umap_emb = self.umap_emb[~noise_idx,:]
            self.nout_position = self.position[~noise_idx,:]
            self.nout_dir_movement = self.dir_movement[~noise_idx]

        if hasattr(self, 'rot_umap_emb'):
            D = pairwise_distances(self.rot_umap_emb)
            noise_idx = self.filter_noisy_outliers(self.rot_umap_emb,D, prctile_th,prctile_noise)
            self.nout_rot_umap_emb = self.rot_umap_emb[~noise_idx,:]
            self.nout_rot_position = self.rot_position[~noise_idx,:]
            self.nout_rot_dir_movement = self.rot_dir_movement[~noise_idx]

    def plot_umap(self):

        fig = plt.figure()
        ax = plt.subplot(1,3,1, projection = '3d')
        ax.scatter(*self.nout_umap_emb[:,:3].T, color ='b', s= 20)
        ax.scatter(*self.nout_rot_umap_emb[:,:3].T, color = 'r', s= 20)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(1,3,2, projection = '3d')
        ax.scatter(*self.nout_umap_emb[:,:3].T, c = self.nout_position[:,0], s= 20, cmap = 'magma')
        ax.scatter(*self.nout_rot_umap_emb[:,:3].T, c = self.nout_rot_position[:,0], s= 20, cmap = 'magma')
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        dir_color = np.zeros((self.nout_dir_movement.shape[0],3))
        for point in range(self.nout_dir_movement.shape[0]):
            if self.nout_dir_movement[point]==0:
                dir_color[point] = [12/255,136/255,249/255]
            elif self.nout_dir_movement[point]==1:
                dir_color[point] = [17/255,219/255,224/255]
            else:
                dir_color[point] = [14/255,14/255,143/255]

        rot_dir_color = np.zeros((self.nout_rot_dir_movement.shape[0],3))
        for point in range(self.nout_rot_dir_movement.shape[0]):
            if self.nout_rot_dir_movement[point]==0:
                rot_dir_color[point] = [12/255,136/255,249/255]
            elif self.nout_rot_dir_movement[point]==1:
                rot_dir_color[point] = [17/255,219/255,224/255]
            else:
                rot_dir_color[point] = [14/255,14/255,143/255]

        ax = plt.subplot(1,3,3, projection = '3d')
        ax.scatter(*self.nout_umap_emb[:,:3].T, color = dir_color, s= 20)
        ax.scatter(*self.nout_rot_umap_emb[:,:3].T, color = rot_dir_color, s= 20)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        return fig

    def get_centroids(self,input_A, input_B, label_A, label_B, dir_A = None, dir_B = None, num_dims = 2, num_centroids = 20):
        input_A = input_A[:,:num_dims]
        input_B = input_B[:,:num_dims]
        if label_A.ndim>1:
            label_A = label_A[:,0]
        if label_B.ndim>1:
            label_B = label_B[:,0]
        #compute label max and min to divide into centroids
        total_label = np.hstack((label_A, label_B))
        labelLimits = np.array([(np.percentile(total_label,5), np.percentile(total_label,95))]).T[:,0] 
        #find centroid size
        centSize = (labelLimits[1] - labelLimits[0]) / (num_centroids)
        #define centroid edges a snp.ndarray([lower_edge, upper_edge])
        centEdges = np.column_stack((np.linspace(labelLimits[0],labelLimits[0]+centSize*(num_centroids),num_centroids),
                                    np.linspace(labelLimits[0],labelLimits[0]+centSize*(num_centroids),num_centroids)+centSize))

        if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
            centLabel_A = np.zeros((num_centroids,num_dims))
            centLabel_B = np.zeros((num_centroids,num_dims))
            
            ncentLabel_A = np.zeros((num_centroids,))
            ncentLabel_B = np.zeros((num_centroids,))
            for c in range(num_centroids):
                points_A = input_A[np.logical_and(label_A >= centEdges[c,0], label_A<centEdges[c,1]),:]
                centLabel_A[c,:] = np.median(points_A, axis=0)
                ncentLabel_A[c] = points_A.shape[0]
                
                points_B = input_B[np.logical_and(label_B >= centEdges[c,0], label_B<centEdges[c,1]),:]
                centLabel_B[c,:] = np.median(points_B, axis=0)
                ncentLabel_B[c] = points_B.shape[0]
        else:
            input_A_left = copy.deepcopy(input_A[dir_A==0,:])
            label_A_left = copy.deepcopy(label_A[dir_A==0])
            input_A_right = copy.deepcopy(input_A[dir_A==1,:])
            label_A_right = copy.deepcopy(label_A[dir_A==1])
            
            input_B_left = copy.deepcopy(input_B[dir_B==0,:])
            label_B_left = copy.deepcopy(label_B[dir_B==0])
            input_B_right = copy.deepcopy(input_B[dir_B==1,:])
            label_B_right = copy.deepcopy(label_B[dir_B==1])
            
            centLabel_A = np.zeros((2*num_centroids,num_dims))
            centLabel_B = np.zeros((2*num_centroids,num_dims))
            ncentLabel_A = np.zeros((2*num_centroids,))
            ncentLabel_B = np.zeros((2*num_centroids,))
            
            for c in range(num_centroids):
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
        del_cent_num = (ncentLabel_A<40) + (ncentLabel_B<40)
        del_cent = del_cent_nan + del_cent_num
        
        centLabel_A = np.delete(centLabel_A, del_cent, 0)
        centLabel_B = np.delete(centLabel_B, del_cent, 0)

        centPos = np.nanmean(centEdges,axis=1).reshape(-1,1)
        centPos = np.hstack((centPos,centPos)).reshape(-1,1)
        centPos = np.delete(centPos, del_cent, 0)
        return centLabel_A, centLabel_B, centPos

    def find_rotation(self,data_A, data_B, v):
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

    def compute_rotation(self, num_centroids = 40):
        self.cent_pre, self.cent_rot, self.cent_pos = self.get_centroids(self.nout_umap_emb, self.nout_rot_umap_emb, 
                                                self.nout_position[:,0], self.nout_rot_position[:,0], 
                                                self.nout_dir_movement, self.nout_rot_dir_movement,
                                                num_dims = 3, num_centroids=num_centroids)   
        #find axis of rotatio                                                
        middle_pre = np.median(self.nout_umap_emb, axis=0).reshape(-1,1)
        middle_rot = np.median(self.nout_rot_umap_emb, axis=0).reshape(-1,1)
        self.norm_vector =  middle_pre - middle_rot
        self.norm_vector = self.norm_vector/np.linalg.norm(self.norm_vector)
        k = np.dot(np.median(self.nout_umap_emb, axis=0), self.norm_vector)

        self.angles = np.linspace(-np.pi,np.pi,100)
        self.error = self.find_rotation(self.cent_pre-middle_pre.T, self.cent_rot-middle_rot.T, self.norm_vector)
        self.norm_error = (np.array(self.error)-np.min(self.error))/(np.max(self.error)-np.min(self.error))
        self.rot_angle = np.abs(self.angles[np.argmin(self.norm_error)])*180/np.pi

    def plot_rotation(self):
        fig = plt.figure()
        ax = plt.subplot(2,3,1, projection = '3d')
        ax.scatter(*self.nout_umap_emb[:,:3].T, color ='b', s= 30)
        ax.scatter(*self.nout_rot_umap_emb[:,:3].T, color = 'r', s= 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax = plt.subplot(2,3,4, projection = '3d')
        ax.scatter(*self.nout_umap_emb[:,:3].T, c = self.nout_position[:,0], s= 30, cmap = 'magma')
        ax.scatter(*self.nout_rot_umap_emb[:,:3].T, c = self.nout_rot_position[:,0], s= 30, cmap = 'magma')
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(2,3,2, projection = '3d')
        ax.scatter(*self.cent_pre[:,:3].T, color ='b', s= 30)
        ax.scatter(*self.cent_rot[:,:3].T, color = 'r', s= 30)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(2,3,5, projection = '3d')
        ax.scatter(*self.cent_pre[:,:3].T, c = self.cent_pos[:,0], s= 30, cmap = 'magma')
        ax.scatter(*self.cent_rot[:,:3].T, c = self.cent_pos[:,0], s= 30, cmap = 'magma')
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(1,3,3)
        ax.plot(self.norm_error, self.angles*180/np.pi)
        ax.plot([0,1], [self.rot_angle]*2, '--r')
        ax.set_yticks([-180, -90, 0 , 90, 180])
        ax.set_xlabel('Norm Error')
        ax.set_ylabel('Angle')
        plt.tight_layout()
        plt.suptitle(f"num_cells: {self.num_cells} | allo: {self.cell_type_params['allo_prob']} " +
                    f"| local-anchored: {self.cell_type_params['local_anchored_prob']} " +
                    f"| remap: {self.cell_type_params['remap_prob']} " + 
                    f"| no-field: {self.cell_type_params['remap_no_field_prob']} " + 
                    f" -- rot angle: {self.rot_angle:.2f}")
        return fig

    def compute_distance(self, num_centroids = 40):
        if not hasattr(self, 'cent_pre'):
            self.cent_pre, self.cent_rot, self.cent_pos = self.get_centroids(self.nout_umap_emb, self.nout_rot_umap_emb, 
                                        self.nout_position[:,0], self.nout_rot_position[:,0], 
                                        self.nout_dir_movement, self.nout_rot_dir_movement,
                                        num_dims = 3, num_centroids=num_centroids)

        self.inter_dist = np.mean(pairwise_distances(self.cent_pre, self.cent_rot))
        self.intra_dist_pre = np.percentile(pairwise_distances(self.cent_pre),95)
        self.intra_dist_rot = np.percentile(pairwise_distances(self.cent_rot),95)

        self.remap_dist = self.inter_dist/np.max((self.intra_dist_pre, self.intra_dist_rot))

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

#__________________________________________________________________________
#|                                                                        |#
#|                         MODEL LOCAL CUE CELLS                          |#
#|________________________________________________________________________|#

#
data_dir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
save_dir = '/home/julio/Documents/SP_project/LT_DeepSup/model/model_v1'

mouse = 'GC2'
file_path = os.path.join(data_dir, mouse)
animal_dict = load_pickle(file_path,mouse+'_df_dict.pkl')

fnames = list(animal_dict.keys())
fname_pre = [fname for fname in fnames if 'lt' in fname][0]
fname_rot = [fname for fname in fnames if 'rot' in fname][0]

pos = copy.deepcopy(np.concatenate(animal_dict[fname_pre]['pos'].values, axis=0))
real_dir_mat = copy.deepcopy(np.concatenate(animal_dict[fname_pre]['dir_mat'].values, axis=0))[:,0]
direction_mat = np.zeros((pos.shape[0],))*np.nan
direction_mat[real_dir_mat==1] = 0
direction_mat[real_dir_mat==2] = 1

rot_pos = copy.deepcopy(np.concatenate(animal_dict[fname_rot]['pos'].values, axis=0))
rot_real_dir_mat = copy.deepcopy(np.concatenate(animal_dict[fname_rot]['dir_mat'].values, axis=0))[:,0]
rot_direction_mat = np.zeros((rot_pos.shape[0],))*np.nan
rot_direction_mat[rot_real_dir_mat==1] = 0
rot_direction_mat[rot_real_dir_mat==2] = 1


num_cells = 400
num_iters = 5
rot_remap_list = [0.1,0.15,0.2,0.3,0.4]
rot_remap_no_field_list = [0.05, 0.1, 0.15, 0.2]
save_dir = '/home/julio/Documents/SP_project/LT_DeepSup/model/hyperparameter_space_v2/'
for same_prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for xmirror_prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for remap_prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for remap_no_field_prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if same_prob+xmirror_prob+remap_prob+remap_no_field_prob>1: continue;

                no_field_prob = 1 - (same_prob+xmirror_prob+remap_prob+remap_no_field_prob)
                print(f"Working with: same_prob: {same_prob:.2f} | xmirror_prob: {xmirror_prob:.2f} | remap_prob: {remap_prob:.2f} | " +
                    f"remap_no_field_prob: {remap_no_field_prob:.2f} | no_field_prob: {no_field_prob:.2f}")
                #train model
                field_type_probs = {
                    'same_prob': same_prob,
                    'xmirror_prob': xmirror_prob,
                    'remap_prob': remap_prob,
                    'remap_no_field_prob': remap_no_field_prob,
                    'no_field_prob': no_field_prob
                }

                model_test = StatisticModel(pos, direction_mat, rot_pos,rot_direction_mat, num_cells = num_cells)


                rot_angles = np.zeros((len(rot_remap_list), len(rot_remap_no_field_list), 2, num_iters))
                rot_distances = np.zeros((len(rot_remap_list), len(rot_remap_no_field_list), 2, num_iters))


                folder_name = f"model_0{int(same_prob*10)}same_0{int(xmirror_prob*10)}xmirror_0{int(remap_prob*10)}remap_0{int(remap_no_field_prob*10)}remapnofield_0{int(no_field_prob*10)}nofield"
                iter_save = os.path.join(save_dir, folder_name)
                try:
                    os.mkdir(iter_save)
                except:
                    pass        

                for rot_remap_idx, rot_remap in enumerate(rot_remap_list):
                    for rot_remap_no_field_idx, rot_remap_no_field in enumerate(rot_remap_no_field_list):

                        model_test.compute_fields(**field_type_probs)
                        cell_type_probs = {
                            'local_anchored_prob': 1-rot_remap-rot_remap_no_field,
                            'allo_prob': 0,
                            'remap_prob': rot_remap,
                            'remap_no_field_prob': rot_remap_no_field
                        }

                        for it in range(num_iters):
                            model_test.compute_cell_types(**cell_type_probs)
                            model_test.compute_rotation_fields()

                            model_test.compute_traces(noise_sigma = 0.01)
                            model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
                            model_test.clean_traces()

                            model_test.compute_umap_both()
                            model_test.clean_umap()
                            model_test.compute_rotation()

                            rot_angles[rot_remap_idx, rot_remap_no_field_idx, 0, it] = model_test.rot_angle

                        fig = model_test.plot_rotation()
                        plt.savefig(os.path.join(iter_save,f'model_local_0{int(100*rot_remap)}remap_0{int(100*rot_remap_no_field)}remapnofield.png'), dpi = 400,bbox_inches="tight")
                        plt.close(fig)
                        print(f"\tremap_prob: {rot_remap} | rot_remap_no_field: {rot_remap_no_field} | " +
                            f"All local: {np.nanmedian(rot_angles[rot_remap_idx, rot_remap_no_field_idx, 0, :]):.2f} +- {np.std(rot_angles[rot_remap_idx, rot_remap_no_field_idx, 0, :]):.2f}º")

                        cell_type_probs = {
                            'local_anchored_prob': 0,
                            'allo_prob': 1-rot_remap-rot_remap_no_field,
                            'remap_prob': rot_remap,
                            'remap_no_field_prob': rot_remap_no_field
                        }

                        for it in range(num_iters):
                            model_test.compute_cell_types(**cell_type_probs)
                            model_test.compute_rotation_fields()

                            model_test.compute_traces(noise_sigma = 0.01)
                            model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
                            model_test.clean_traces()

                            model_test.compute_umap_both()
                            model_test.clean_umap()
                            model_test.compute_rotation()
                            rot_angles[rot_remap_idx, rot_remap_no_field_idx, 1, it] = model_test.rot_angle

                        fig = model_test.plot_rotation()
                        plt.savefig(os.path.join(iter_save,f'model_allo_0{int(100*rot_remap)}remap_0{int(100*rot_remap_no_field)}remapnofield.png'), dpi = 400,bbox_inches="tight")
                        plt.close(fig)
                        print(f"\tremap_prob: {rot_remap} | rot_remap_no_field: {rot_remap_no_field} | "+
                            f"All allo: {np.nanmedian(rot_angles[rot_remap_idx, rot_remap_no_field_idx, 1, :]):.2f} +- {np.std(rot_angles[rot_remap_idx, rot_remap_no_field_idx,1, :]):.2f}º")



                results = {'rot_angles': rot_angles, 
                            'rot_remap_list': rot_remap_list, 
                            'rot_remap_no_field_list': rot_remap_no_field_list,
                            'num_cells': num_cells, 
                            'field_type_probs': copy.deepcopy(field_type_probs)}

   
                with open(os.path.join(iter_save,'statistic_model_results.pkl'), 'wb') as f:
                    pickle.dump(results, f)

                save_params_file = open(os.path.join(iter_save, "probability_params.txt"), "w")
                with save_params_file as f:
                    [f.write("%s: %s\n" %(st, str(val))) for st, val in field_type_probs.items()]
                    [f.write("%s: %s\n" %(st, str(val))) for st, val in cell_type_probs.items()]
                save_params_file.close()


                fig = plt.figure(figsize=(10,8))

                ax = plt.subplot(2,2,1)
                b = ax.matshow(np.nanmedian(rot_angles[:,:,0,:], axis=-1),vmin=140, vmax=180, cmap='inferno')
                ax.set_yticks(ticks = np.arange(len(rot_remap_list)), labels = rot_remap_list)
                ax.set_xticks(ticks = np.arange(len(rot_remap_no_field_list)), labels = rot_remap_no_field_list)
                plt.colorbar(b)
                ax.set_title('Mean rotation when all is local-chue')

                ax = plt.subplot(2,2,3)
                b = ax.matshow(np.nanstd(rot_angles[:,:,0,:], axis=-1),vmin=0, vmax=10, cmap='inferno')
                ax.set_yticks(ticks = np.arange(len(rot_remap_list)), labels = rot_remap_list)
                ax.set_xticks(ticks = np.arange(len(rot_remap_no_field_list)), labels = rot_remap_no_field_list)
                plt.colorbar(b)
                ax.set_title('STD rotation when all is local-chue')


                ax = plt.subplot(2,2,2)
                b = ax.matshow(np.nanmedian(rot_angles[:,:,1,:], axis=-1),vmin=0, vmax=90, cmap='inferno')
                ax.set_yticks(ticks = np.arange(len(rot_remap_list)), labels = rot_remap_list)
                ax.set_xticks(ticks = np.arange(len(rot_remap_no_field_list)), labels = rot_remap_no_field_list)
                plt.colorbar(b)
                ax.set_title('Mean rotation when all is allocentric')

                ax = plt.subplot(2,2,4)
                b = ax.matshow(np.nanstd(rot_angles[:,:,1,:], axis=-1),vmin=0, vmax=10, cmap='inferno')
                ax.set_yticks(ticks = np.arange(len(rot_remap_list)), labels = rot_remap_list)
                ax.set_xticks(ticks = np.arange(len(rot_remap_no_field_list)), labels = rot_remap_no_field_list)
                plt.colorbar(b)
                ax.set_title('STD rotation when all is allocentric')
                plt.tight_layout()
                plt.savefig(os.path.join(iter_save,'model_main_plot.svg'), dpi = 400,bbox_inches="tight")
                plt.savefig(os.path.join(iter_save,'model_main_plot.png'), dpi = 400,bbox_inches="tight")
                plt.close(fig)


