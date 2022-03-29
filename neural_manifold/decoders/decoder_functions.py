# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:11:52 2022

@author: Usuario
"""
import copy
import numpy as np

#DIM RED LIBRARIES
import umap
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import median_absolute_error

#INNER PACKAGE IMPORTS
from neural_manifold import general_utils as gu
from neural_manifold.decoders.decoder_classes import DECODERS  #decoders classes

import warnings as warnings
warnings.filterwarnings(action='ignore', category=UserWarning) #supress slice-data warning for XGBoost: 
                                                               #https://stackoverflow.com/questions/67225016/warning-occuring-in-xgboost

@gu.check_inputs_for_pd
def decoders_1D(x_base_signal = None, y_signal_list=None, emb_list = [],
                    trial_signal = None, decoder_list = ["wf", "wc", "xgb", "svr"],
                    n_dims = 10, n_splits=10, verbose = False):  
    
    """Train decoders on x-base signal (and on projected one if indicated) to 
    predict a 1D y-signal.
    
    Parameters:
    ----------
    x_base_signal: Numpy array (TXN; rows are time)
        Array containing the base signal in which to train the decoders (rows
        are timestamps, columns are features). It will be also used to compute
        the embeddings and reduced. 
    
    y_signal_list: Numpy array (TXN; rows are time) or list (made of those)
        Array or list containing the arrays with the behavioral signal that we
        want the decoders to predict. 
        
    Optional parameters:
    --------------------
    emb_list: List of string
        List containing the name of the embeddings one wants to compute. It
        currently supports: ['pca','isomap','umap'].
        
    trial_signal: Numpy array (TX1; rows are time)
        Array containing the trial each timestamps of the x_base_signal belongs
        to. It provided, the train/test data set will be computed by dividing
        in half the number of trials (thus, the actual length of the two sets
        will not be identical, but no trial will be splitted in half). If not
        provided the kfold will split in half the data for the train/test set. 
    
    
    decoder_list: List of String (default: ['wf', 'wc', 'xgb', 'svr'])
        List containing the name of the decoders one wants to train/test. It
        currently supports the decoders: ['wf', 'wc', 'xgb', 'svr']
    
    n_dims: Integer (default: 10)
        Number of dimensions to project the data into.
        
    n_splits: Integer (default: 10)
        Number of kfolds to run the training/test. Repeated kfold is used to 
        randomly divide the signal (or trial_signal) into 50%train 50%test to 
        achieve the final number of kfolds.
        
    verbose: Boolean (default: False)
        Boolean specifing whether or not to print verbose relating the progress
        of the training (it mainly prints the kfold it is currently in).
    
    Returns:
    -------
    R2s: Dictionary
        Dictionary containing the training and test median absolute errors for 
        all kfold and all combinations of x_signal/y_signal/decoder.
        
    """
    #ensure y_signal_list and emb_list are list
    if isinstance(y_signal_list, np.ndarray): y_signal_list = list([y_signal_list])
    if isinstance(emb_list, str): emb_list = list([emb_list])
    #assert inputs
    assert isinstance(x_base_signal, np.ndarray), \
        f"'x_base_signal has to be a numpy array but it was a {type(x_base_signal)}"
    assert isinstance(y_signal_list, list), \
        f"'y_signal_list has to be a list of numpy.array but it was a {type(y_signal_list)}"
    assert isinstance(emb_list, list), \
        f"'emb_list' has to be a list of string but it was a {type(emb_list)}" 
    #reshape y_signal_list from column vectors to 1D-matrix
    for idx, y in enumerate(y_signal_list):
        if y.ndim == 1:
            y_signal_list[idx] = y.reshape(-1,1)
    #check if trial mat to use when spliting training/test for decoders
    rkf = RepeatedKFold(n_splits=2, n_repeats=np.ceil(n_splits/2).astype(int))
    if isinstance(trial_signal, np.ndarray):
        trial_list = np.unique(trial_signal)
        #train with half of trials
        kfold_signal = trial_list
        total_index = np.linspace(0, x_base_signal.shape[0]-1, x_base_signal.shape[0]).astype(int)
    else:
        kfold_signal = x_base_signal
        
    train_indexes = [];
    test_indexes = [];
    for train_index, test_index in rkf.split(kfold_signal, kfold_signal):
        train_indexes.append(train_index)
        test_indexes.append(test_index)
        
    if verbose:
        print("\t\tKfold: X/X",end='', sep='')
        pre_del = '\b\b\b'
    #initialize dictionary to save results
    R2s = dict()
    for emb in ['base_signal', *emb_list]:
        R2s[emb] = dict()
        for decoder_name in decoder_list:
            R2s[emb][decoder_name] = np.zeros((n_splits,len(y_signal_list),2))
            
    n_x_signals = len(['base_signal', *emb_list])
    predictions = [np.zeros((n_splits,y.shape[0],n_x_signals+2)) for y in y_signal_list]
    for y_idx, y in enumerate(y_signal_list):
        predictions[y_idx][:,:,1] = np.tile(y, (1, n_splits)).T
        
    for kfold_idx in range(n_splits):
        if verbose:
            print(f"{pre_del}{kfold_idx+1}/{n_splits}", sep = '', end='')
            pre_del = (len(str(kfold_idx+1))+len(str(n_splits))+1)*'\b'
            
        #split into train and test data
        if isinstance(trial_signal, np.ndarray):
            train_index = np.any(trial_signal.reshape(-1,1)==trial_list[train_indexes[kfold_idx]], axis=1)
            train_index = total_index[train_index]
            test_index = np.any(trial_signal.reshape(-1,1)==trial_list[test_indexes[kfold_idx]], axis=1)
            test_index = total_index[test_index]
        else:
            train_index = train_indexes[kfold_idx]
            test_index = test_indexes[kfold_idx]
            
        for y_idx, y in enumerate(y_signal_list):
            predictions[y_idx][kfold_idx,train_index,0] = 0
            predictions[y_idx][kfold_idx,test_index,0] = 1

        X_train = []
        X_base_train = x_base_signal[train_index,:]
        X_train.append(X_base_train)
        X_test = []
        X_base_test = x_base_signal[test_index,:]
        X_test.append(X_base_test)
        Y_train = [y[train_index,:] for y in y_signal_list]
        Y_test = [y[test_index,:] for y in y_signal_list]
        #compute embeddings
        for emb in emb_list:
            if 'umap' in emb:
                n_neighbours = np.round(X_base_train.shape[0]*0.01).astype(int)
                model = umap.UMAP(n_neighbors = n_neighbours, n_components =n_dims, min_dist=0.75)
            elif 'iso' in emb:
                model = Isomap(n_neighbors = 15,n_components = n_dims)
            elif 'pca' in emb:
                model = PCA(n_dims)
            X_signal_train = model.fit_transform(X_base_train)
            X_signal_test = model.transform(X_base_test)
            X_train.append(X_signal_train)
            X_test.append(X_signal_test)
        #train and test decoders 
        for emb_idx, emb in enumerate(['base_signal', *emb_list]):
            for y_idx in range(len(y_signal_list)):
                for decoder_name in decoder_list:
                    #train decoder
                    model_decoder = DECODERS[decoder_name]()
                    model_decoder.fit(X_train[emb_idx], Y_train[y_idx])
                    #make predictions
                    train_pred = model_decoder.predict(X_train[emb_idx])[:,0]
                    test_pred = model_decoder.predict(X_test[emb_idx])[:,0]
                    #check errors
                    test_error = median_absolute_error(Y_test[y_idx][:,0], test_pred)
                    train_error = median_absolute_error(Y_train[y_idx][:,0], train_pred)
                    #store results
                    R2s[emb][decoder_name][kfold_idx,y_idx,0] = test_error
                    R2s[emb][decoder_name][kfold_idx,y_idx,1] = train_error

                    total_pred = np.hstack((train_pred, test_pred)).T
                    total_pred = total_pred[np.argsort(np.hstack((train_index, test_index)).T)]
                    predictions[y_idx][kfold_idx,:,emb_idx+2] = total_pred
                    
    if verbose:
        print("")            
    return R2s , predictions

def cross_session_decoders_LT(input_signal_A, input_signal_B, field_signal = None, field_signal_A = None, field_signal_B = None, 
                              input_label = None, input_label_A = None, input_label_B = None, 
                              input_trial = None, input_trial_A = None, input_trial_B= None, 
                              input_direction = None, input_direction_A = None, input_direction_B = None,
                              n_splits=10, n_neigh = 0.01, n_dims = 3, decoder_list = ["wf", "wc", "xgb", "svr"],verbose = True):
    #check signal input A
    assert isinstance(field_signal_A, str)^isinstance(field_signal, str), "Both field_signal and field_signal_A defined. Only specify one."
    signal_A = gu.handle_input_dataframe_array(input_signal_A, field_signal, field_signal_A, first_array = True)
    #check signal input B
    assert isinstance(field_signal_B, str)^isinstance(field_signal, str), "Both field_signal and field_signal_B defined. Only specify one."
    signal_B = gu.handle_input_dataframe_array(input_signal_B, field_signal, field_signal_B, first_array = True)
    #check input label
    assert isinstance(input_label, str)^(isinstance(input_label_A, str) and isinstance(input_label_B, str)), "Both input_label, and input_label_A/input_label_B defined. Only specify one"
    label_A = gu.handle_input_dataframe_array(input_signal_A, input_label, input_label_A, first_array = False)
    label_B = gu.handle_input_dataframe_array(input_signal_B, input_label, input_label_B, first_array = False)
    #Check if trial mat to use when spliting training/test
    assert isinstance(input_trial, str)^(isinstance(input_trial_A, str) and isinstance(input_trial_B, str)), "Both input_trial, and input_trial_A/input_trial_B defined. Only specify one"
    if not isinstance(input_trial, type(None)) or not (isinstance(input_trial_A, type(None)) 
                                                       and isinstance(input_trial_B, type(None))):
        trial_signal_A = gu.handle_input_dataframe_array(input_signal_A, input_trial, input_trial_A, first_array = False)
        trial_signal_B = gu.handle_input_dataframe_array(input_signal_B, input_trial, input_trial_B, first_array = False)
    else:
        trial_signal_A = None
        trial_signal_B = None
    #Program future train/test split accordingly
    if isinstance(trial_signal_A , np.ndarray):
        if trial_signal_A.ndim>1:
            trial_signal_A = trial_signal_A[:,0]
        trial_list_A = np.unique(trial_signal_A)
        #train with half of trials
        train_test_division_index_A = int(trial_list_A.shape[0]//1.25)
        train_indexes_A = None
        test_indexes_A = None
        
    else:
        from sklearn.model_selection import RepeatedKFold
        rkf = RepeatedKFold(n_splits=5, n_repeats=n_splits)
        train_indexes_A = [];
        test_indexes_A = [];
        for train_index, test_index in rkf.split(signal_A):
            train_indexes_A.append(train_index)
            test_indexes_A.append(test_index)
    if isinstance(trial_signal_B , np.ndarray):
        if trial_signal_B.ndim>1:
            trial_signal_B = trial_signal_B[:,0]
        trial_list_B = np.unique(trial_signal_B)
        #train with half of trials
        train_test_division_index_B = int(trial_list_B.shape[0]//1.25)
        train_indexes_B = None
        test_indexes_B = None
    else:
        from sklearn.model_selection import RepeatedKFold
        rkf = RepeatedKFold(n_splits=5, n_repeats=n_splits)
        train_indexes_B = [];
        test_indexes_B = [];
        for train_index, test_index in rkf.split(signal_B):
            train_indexes_B.append(train_index)
            test_indexes_B.append(test_index)
    #Check if dirmat specified:
    assert isinstance(input_direction, str)^(isinstance(input_direction_A, str) and isinstance(input_direction_B, str)),"Both input_direction, and input_direction_A/input_direction_B defined. Only specify one"
    if not isinstance(input_direction, type(None)) or not (isinstance(input_direction_A, type(None)) and 
                                                             isinstance(input_direction_B, type(None))):
        dir_A = gu.handle_input_dataframe_array(input_signal_A, input_direction, input_direction_A, first_array = False)
        dir_B = gu.handle_input_dataframe_array(input_signal_B, input_direction, input_direction_B, first_array = False)
    else:
        dir_A = None
        dir_B = None 
        
    if verbose:
        print('\tKfold: X/X',end='', sep='')
        pre_del = '\b\b\b'
    R2s = dict()
    for decoder_name in decoder_list:
        R2s[decoder_name] = np.zeros((5, n_splits))*np.nan
    
    def split_signals(signal, label, trial_signal = None, trial_list = None, train_test_division_index = None, dir_mat= None,
                      train_indexes = None, test_indexes = None):
        if isinstance(trial_signal, np.ndarray):
            fold_split = np.copy(trial_list)
            np.random.shuffle(fold_split)
            train_index = np.any(trial_signal.reshape(-1,1)==fold_split[:train_test_division_index], axis=1)
            test_index = np.any(trial_signal.reshape(-1,1)==fold_split[train_test_division_index:], axis=1)
        else:
            train_index = train_indexes[kfold_index]
            test_index = test_indexes[kfold_index]
        
        signal_train = signal[train_index]
        signal_test = signal[test_index]
        label_train = label[train_index]
        label_test = label[test_index]
        if isinstance(dir_mat, np.ndarray):
            dir_train = dir_mat[train_index]
        else:
            dir_train = None
        return signal_train, signal_test, label_train, label_test, dir_train
     
    for kfold_index in range(n_splits):
        if verbose:
            print(pre_del,"%d/%d" %(kfold_index+1,n_splits), sep = '', end='')
            pre_del = (len(str(kfold_index+1))+len(str(n_splits)))*'\b'
        #split A signals
        signal_A_train, signal_A_test, label_A_train, label_A_test, dir_A_train = split_signals(signal_A, 
                                                    label_A, trial_signal_A,trial_list_A, train_test_division_index_A, 
                                                    dir_A, train_indexes_A, test_indexes_A)
        #create embeddings and project A
        if n_neigh<1:
            n_neighbours_A = np.round(signal_A_train.shape[0]*n_neigh).astype(int)
        else:
            n_neighbours_A = n_neigh
        model_A = umap.UMAP(n_neighbors = n_neighbours_A, n_components =n_dims, min_dist=0.75)
        emb_A_train = model_A.fit_transform(signal_A_train)
        emb_A_test = model_A.transform(signal_A_test)
        #split B signals
        signal_B_train, signal_B_test, label_B_train, label_B_test, dir_B_train = split_signals(signal_B, 
                                                    label_B, trial_signal_B, trial_list_B, train_test_division_index_B, 
                                                    dir_B, train_indexes_B, test_indexes_B)
        #create embeddings and project B
        if n_neigh<1:
            n_neighbours_B = np.round(signal_B_train.shape[0]*n_neigh).astype(int)
        else:
            n_neighbours_B = n_neigh
        model_B = umap.UMAP(n_neighbors = n_neighbours_B, n_components =n_dims, min_dist=0.75)
        emb_B_train = model_B.fit_transform(signal_B_train)
        emb_B_test = model_B.transform(signal_B_test)
        
        ###########################################################
        #                       DO A TO B                         #
        ###########################################################
        #3.Find algiment for train data
        TAB, RAB = align_manifolds_1D(emb_A_train, emb_B_train, label_A_train, label_B_train,
                                              dir_A_train, dir_B_train, ndims = n_dims, nCentroids = 20)
        #4.1 train decoder A
        emb_AB_test = np.transpose(TAB + np.matmul(RAB, emb_A_test.T))
        #5. Get errors
        for decoder_name in decoder_list:
            model_decoder_A = DECODERS[decoder_name]()
            model_decoder_A.fit(emb_A_train, label_A_train)
            label_A_pred_A = model_decoder_A.predict(emb_A_test)
            
            model_decoder_B = DECODERS[decoder_name]()
            model_decoder_B.fit(emb_B_train, label_B_train)
            label_B_pred_B = model_decoder_B.predict(emb_B_test)
            
            label_A_pred_B = model_decoder_B.predict(emb_A_test)
            label_AB_pred_B = model_decoder_B.predict(emb_AB_test)
            
            TAB_label, RAB_label = get_point_registration(np.c_[label_A_pred_B, np.zeros(label_A_pred_B.shape[0])], 
                                                  np.c_[label_A_test, np.zeros(label_A_test.shape[0])])
            label_A_pred_B_rot = np.transpose(TAB_label + np.matmul(RAB_label, np.c_[label_A_pred_B, 
                                                                                     np.zeros(label_A_pred_B.shape[0])].T))
            
            R2s[decoder_name][0,kfold_index] = median_absolute_error(label_A_test[:,0], label_A_pred_A[:,0])
            R2s[decoder_name][1,kfold_index] = median_absolute_error(label_B_test[:,0], label_B_pred_B[:,0])
            R2s[decoder_name][2,kfold_index] = median_absolute_error(label_A_test[:,0], label_A_pred_B[:,0])
            R2s[decoder_name][3,kfold_index] = median_absolute_error(label_A_test[:,0], label_AB_pred_B[:,0])
            R2s[decoder_name][4,kfold_index] = median_absolute_error(label_A_test[:,0], label_A_pred_B_rot[:,0])

    return R2s


def align_manifolds_1D(input_A, input_B, label_A, label_B, dir_A = None, dir_B = None, ndims = 2, nCentroids = 20):
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
        for c in range(nCentroids):
            points_A = input_A[np.logical_and(label_A >= centEdges[c,0], label_A<centEdges[c,1]),:]
            centLabel_A[c,:] = np.median(points_A, axis=0)
            
            
            points_B = input_B[np.logical_and(label_B >= centEdges[c,0], label_B<centEdges[c,1]),:]
            centLabel_B[c,:] = np.median(points_B, axis=0)
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
        for c in range(nCentroids):
            points_A_left = input_A_left[np.logical_and(label_A_left >= centEdges[c,0], label_A_left<centEdges[c,1]),:]
            centLabel_A[2*c,:] = np.median(points_A_left, axis=0)
            points_A_right = input_A_right[np.logical_and(label_A_right >= centEdges[c,0], label_A_right<centEdges[c,1]),:]
            centLabel_A[2*c+1,:] = np.median(points_A_right, axis=0)
            
            points_B_left = input_B_left[np.logical_and(label_B_left >= centEdges[c,0], label_B_left<centEdges[c,1]),:]
            centLabel_B[2*c,:] = np.median(points_B_left, axis=0)
            points_B_right = input_B_right[np.logical_and(label_B_right >= centEdges[c,0], label_B_right<centEdges[c,1]),:]
            centLabel_B[2*c+1,:] = np.median(points_B_right, axis=0)
            
    TAB,RAB = get_point_registration(centLabel_A, centLabel_B)
    return TAB, RAB


def get_point_registration(p1, p2, verbose=False):
    #from https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
    if p1.shape[0]>p1.shape[1]:
        p1 = p1.transpose()
        p2 = p2.transpose()
    #Calculate centroids
    p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 1).reshape((-1,1))
    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c
    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())
    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt
    #Calculate rotation matrix
    R = np.matmul(V_t.transpose(),U.transpose())
    if not np.allclose(np.linalg.det(R), 1.0) and verbose:
        print("Rotation matrix of N-point registration not 1, see paper Arun et al.")
    #Calculate translation matrix
    T = p2_c - np.matmul(R,p1_c)
    return T,R

def cross_session_decoders_LT_dict(dict_df_A, dict_df_B, x_base = "ML_rates", label_signal = "posx", n_dims = 3, n_splits=10, 
                              decoder_list = ["wf", "wc", "xgb", "svr"], verbose = False):  
    
    for file_A, pd_struct_A in dict_df_A.items():
        if 'index_mat' not in pd_struct_A.columns:
            pd_struct_A["index_mat"] = [np.zeros((pd_struct_A["pos"][idx].shape[0],1)).astype(int)+pd_struct_A["trial_id"][idx] 
                                      for idx in range(pd_struct_A.shape[0])]
        if 'dir_mat' not in pd_struct_A.columns:
            pd_struct_A["dir_mat"] = [np.zeros((pd_struct_A["pos"][idx].shape[0],1)).astype(int)+
                                      ('L' in pd_struct_A["dir"][idx])+ 2*('R' in pd_struct_A["dir"][idx])
                                          for idx in range(pd_struct_A.shape[0])]

    for file_B, pd_struct_B in dict_df_B.items():
        if 'index_mat' not in pd_struct_B.columns:
            pd_struct_B["index_mat"] = [np.zeros((pd_struct_B["pos"][idx].shape[0],1)).astype(int)+pd_struct_B["trial_id"][idx] 
                                      for idx in range(pd_struct_B.shape[0])]
        if 'dir_mat' not in pd_struct_B.columns:
            pd_struct_B["dir_mat"] = [np.zeros((pd_struct_B["pos"][idx].shape[0],1)).astype(int)+
                                      ('L' in pd_struct_B["dir"][idx])+ 2*('R' in pd_struct_B["dir"][idx])
                                          for idx in range(pd_struct_B.shape[0])]
    R2s = dict()
    for decoder_name in decoder_list:
        R2s[decoder_name] = np.zeros((len(dict_df_A),len(dict_df_B),5, n_splits))*np.nan
    count_A = -1
    for file_A, pd_struct_A in dict_df_A.items():
        count_A += 1
        count_B = -1
        if verbose:
            print('\nWorking on file: %s'%file_A, end = '')   
        prefix_file_A = file_A[:file_A.find('_LT')]
        
        for file_B, pd_struct_B in dict_df_B.items():
            count_B +=1
            prefix_file_B = file_B[:file_B.find('_LT')]
            if prefix_file_A != prefix_file_B:
                if verbose:
                    print('\n\tComparing it to: %s'%file_B, end = '')
                    
                temp_R2s = cross_session_decoders_LT(pd_struct_A, pd_struct_B, field_signal = x_base, input_label = label_signal, 
                                                    input_trial ="index_mat", input_direction = "dir_mat", n_splits=n_splits, 
                                                    n_neigh = 0.01, n_dims = n_dims, decoder_list =decoder_list,verbose = verbose)
                for decoder_name in decoder_list:
                    R2s[decoder_name][count_A,count_B,:,:] = temp_R2s[decoder_name] 
    return R2s


def decoders_delay_1D(input_signal,field_signal = None, input_label=None, emb_list = ["umap"], input_trial = None,
                       n_dims = 10, n_splits=10, decoder_list = ["wf", "wc", "xgb", "svr"], 
                       time_shift = [-5, 5], verbose = False):  
    '''Train decoders on base signal and/or embedded one.
    
    Parameters:
    ----------
        input_signal (DataFrame or numpy Array): object containing the base signal in which to train the decoders.
                        If it is a Dataframe, input 'field_signal' is needed. 
        
    Optional parameters:
    --------------------
        field_signal (string): if 'input_signal' is a dataframe, name of column with the signal (otherwise set it 
                        to None, as per default).
        
        input_label (numpy Array, str, or list made of those): signals one wants to predict with the decoders. They 
                        can either be a np.dnarray (or a list of those), or a string (or a list of strings) with 
                        the name of the columns of 'input_signal'. 
                        
        emb_list (string, list of string): name of the embeddings once want to also use to project the base signal
                        and train the decoders. Currently supported: 'pca', 'isomap', and 'umap'.
        
        input_trial (numpy Array or string): array contaning the trial to which each time stamp belongs. It can also
                        be a string with the name of the column of the 'input_signal' dataframe where the array is 
                        stored.
                        
        n_dims (int): number of dimensions in which to project with the different embeddings.
        
        n_splits (int): number of iterations done (the data is always splitted in half for test and for train).
        
        decoder_list (list of str): name of the decoders once wants to train and test. Currently supported:
                        ['wf', 'wc', 'xgb', 'svr']
        
        delay (list): list containing the minimum and maximum delay to be applied to all 
                        
        verbose (boolean)

                           
    Returns:
    -------
        R2s (dict): dictionary containing the training and test median absolute errors for all combinations. 
        
    '''
    #check signal input
    signal = gu.handle_input_dataframe_array(input_signal, field_signal, first_array = True)
    #check label list input
    if isinstance(input_label, np.ndarray): #user inputs an independent array for sI
            label_list = list([copy.deepcopy(input_label)])
    elif isinstance(input_label, str): #user specifies field_signal
        label_list = list([gu.dataframe_to_1array_translator(input_signal,input_label)])
    elif isinstance(input_label, list):
        if isinstance(input_label[0], str):
            label_list = gu.dataframe_to_manyarray_translator(input_signal,input_label)
        elif isinstance(input_label[0], np.ndarray):
            label_list = copy.deepcopy(input_label)
        else:
            raise ValueError("If 'input_label' is a list, it must be composed of either strings " +
                             "(referring to the columns of the input_signal dataframe), or arrays. " +
                             "However it was %s" %type(input_label[0]))
    else:
        raise ValueError(" 'input_label' must be either an array or a string (referring to the columns "+
                         "of the input_signal dataframe) (or a list composed of those). However it was %s"
                         %type(input_label))
    #Reshape label_list from column vectors to 1D-matrix
    for idx, y in enumerate(label_list):
        if y.ndim == 1:
            label_list[idx] = y.reshape(-1,1)
    #Check delay input
    if isinstance(time_shift, list):
        time_shift_array = np.linspace(time_shift[0], time_shift[1],20).astype(int)
    elif isinstance(time_shift, np.ndarray):
        time_shift_array = copy.deepcopy(time_shift).astype(int)
    time_shift_array = np.unique(time_shift_array)
    #Check if trial mat to use when spliting training/test for decoders
    if not isinstance(input_trial, type(None)):
        trial_signal = gu.handle_input_dataframe_array(input_signal, input_trial, first_array = False)
    else:
        trial_signal = None 
    if isinstance(trial_signal, np.ndarray):
        if trial_signal.ndim>1:
            trial_signal = trial_signal[:,0]
        trial_list = np.unique(trial_signal)
        #train with half of trials
        train_test_division_index = trial_list.shape[0]//2
    else:
        from sklearn.model_selection import RepeatedKFold
        rkf = RepeatedKFold(n_splits=2, n_repeats=n_splits)
        train_indexes = [];
        test_indexes = [];
        for train_index, test_index in rkf.split(signal):
            train_indexes.append(train_index)
            test_indexes.append(test_index)
    if verbose:
        print('\t\tKfold: X/X',end='', sep='')
        pre_del = '\b\b\b'
    #initialize dictionary of this session (then it will be appended into the global dictionary)
    if not field_signal:
        field_signal = 'base_signal'
    R2s = dict()
    for emb in [field_signal, *emb_list]:
        R2s[emb] = dict()
        for decoder_name in decoder_list:
            R2s[emb][decoder_name] = np.zeros((len(time_shift_array),n_splits,len(label_list),2))
    for kfold_index in range(n_splits):
        if verbose:
            print(pre_del,"%d/%d" %(kfold_index+1,n_splits), sep = '', end='')
            pre_del = (len(str(kfold_index+1))+len(str(n_splits))+1)*'\b'
        if isinstance(trial_signal, np.ndarray):
            #split into train and test data
            fold_split = np.copy(trial_list)
            np.random.shuffle(fold_split)
            train_index = np.any(trial_signal.reshape(-1,1)==fold_split[:train_test_division_index], axis=1)
            test_index = np.any(trial_signal.reshape(-1,1)==fold_split[train_test_division_index:], axis=1)
        else:
            train_index = train_indexes[kfold_index]
            test_index = test_indexes[kfold_index]
            
        X_train = []
        X_base_train = signal[train_index]
        X_train.append(X_base_train)
        X_test = []
        X_base_test = signal[test_index]
        X_test.append(X_base_test)
        #compute embeddings
        for emb in emb_list:
            if 'umap' in emb:
                n_neighbours = np.round(X_base_train.shape[0]*0.01).astype(int)
                model = umap.UMAP(n_neighbors = n_neighbours, n_components =n_dims, min_dist=0.75)
            elif 'iso' in emb:
                model = Isomap(n_neighbors = 15,n_components = n_dims)
            elif 'pca' in emb:
                model = PCA(n_dims)
            X_signal_train = model.fit_transform(X_base_train)
            X_signal_test = model.transform(X_base_test)
            X_train.append(X_signal_train)
            X_test.append(X_signal_test)
        for shift_idx, shift_val in enumerate(time_shift_array):
            label_list_shifted = [np.concatenate((y[shift_val:,:],y[:shift_val,:]), axis=0) for y in label_list]
            Y_train = [y[train_index] for y in label_list_shifted]
            Y_test = [y[test_index] for y in label_list_shifted] 
            #delete extremes of shift (if past delete first entries, if future delete last)
            keep_index = np.linspace(0, signal.shape[0]-1, signal.shape[0]).astype(int)*np.float64(1)
            if shift_val >0: #future delay
                keep_index[-shift_val:] *= np.nan
            elif shift_val<0: #past delay
                keep_index[:-shift_val] *= np.nan
                
    
            keep_index_train = keep_index[train_index] >=0
            keep_index_test = keep_index[test_index] >= 0
            
            Y_train = [y[keep_index_train] for y in Y_train]
            Y_test = [y[keep_index_test] for y in Y_test]
            X_train_temp = [x[keep_index_train] for x in X_train]
            X_test_temp = [x[keep_index_test] for x in X_test]
            #train and test decoders 
            for emb_idx, emb in enumerate([field_signal, *emb_list]):
                for y_idx in range(len(label_list)):
                    for decoder_name in decoder_list:
                        model_decoder = DECODERS[decoder_name]()
                        model_decoder.fit(X_train_temp[emb_idx], Y_train[y_idx])
                        R2s[emb][decoder_name][shift_idx,kfold_index,y_idx,0] = median_absolute_error(Y_test[y_idx][:,0], 
                                                                                    model_decoder.predict(X_test_temp[emb_idx])[:,0])
                        R2s[emb][decoder_name][shift_idx,kfold_index,y_idx,1] = median_absolute_error(Y_train[y_idx][:,0], 
                                                                                    model_decoder.predict(X_train_temp[emb_idx])[:,0])
    if verbose:
        print("")
    return R2s 