# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:11:52 2022

@author: Usuario
"""
import copy
import numpy as np
from sklearn.metrics import median_absolute_error

#DIM RED LIBRARIES
import umap
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import pandas as pd

#INNER PACKAGE IMPORTS
from neural_manifold import general_utils as gu
from neural_manifold.decoders.decoder_classes import DECODERS  #decoders classes
from neural_manifold.dimensionality_reduction import reduce_dimensionality as rd #reduce dimensionality of DataFrame 

import warnings as warnings
warnings.filterwarnings(action='ignore', category=UserWarning) #supress slice-data warning for XGBoost: https://stackoverflow.com/questions/67225016/warning-occuring-in-xgboost


def decoders_1D(input_signal,field_signal = None, input_label=None, emb_list = ["umap"], input_trial = None,
                       n_dims = 10, n_splits=10, decoder_list = ["wf", "wc", "xgb", "svr"], verbose = False):  
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
                        
        verbose (boolean)

                           
    Returns:
    -------
        R2s (dict): dictionary containing the training and test median absolute errors for all combinations. 
        
    '''
    #check signal input
    if isinstance(input_signal, pd.DataFrame):
        signal = gu.dataframe_to_1array_translator(input_signal,field_signal)
    elif isinstance(input_signal,np.ndarray):
        signal = copy.deepcopy(input_signal)
    else:
        raise ValueError("Input object has to be a dataframe or a numpy array. However it as %s"
                         %type(input_signal))
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
    #Reshape label_list from column vectors to matrix
    for idx, y in enumerate(label_list):
        if y.ndim == 1:
            label_list[idx] = y.reshape(-1,1)
    #Check if trial mat to use when spliting training/test for decoders
    if isinstance(input_trial, np.ndarray): #user inputs an independent array for sI
        trial_signal = copy.deepcopy(input_trial)
    elif isinstance(input_trial, str): #user specifies field_signal for sI but not a dataframe (then use input_signal)
        trial_signal = gu.dataframe_to_1array_translator(input_signal,input_trial)
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
    #initialize dictionary of this session (then it will be appended into the global dictionary)
    if not field_signal:
        field_signal = 'base_signal'
    R2s = dict()
    for emb in [field_signal, *emb_list]:
        R2s[emb] = dict()
        for decoder_name in decoder_list:
            R2s[emb][decoder_name] = np.zeros((n_splits,len(label_list),2))
    
    for kfold_index in range(n_splits):
        if verbose:
            if kfold_index<10:
                print('\b\b \b\b\b',kfold_index+1, '/',n_splits,sep='', end='')
            else:
                print('\b\b\b \b\b\b',kfold_index+1, '/',n_splits,sep='', end='')
        if isinstance(trial_signal, np.ndarray):
            #split into train and test data
            fold_split = np.copy(trial_list)
            np.random.shuffle(fold_split)
            train_index = np.any(trial_signal.reshape(-1,1)==fold_split[:train_test_division_index], axis=1)
            test_index = np.any(trial_signal.reshape(-1,1)==fold_split[train_test_division_index:], axis=1)
            
            X_train = []
            X_base_train = signal[train_index,:]
            X_train.append(X_base_train)
            X_test = []
            X_base_test = signal[test_index,:]
            X_test.append(X_base_test)
            
            Y_train = [y[train_index,:] for y in label_list]
            Y_test = [y[test_index,:] for y in label_list]
        else:
            X_train = []
            X_base_train = signal[train_indexes[kfold_index]]
            X_train.append(X_base_train)
            X_test = []
            X_base_test = signal[test_indexes[kfold_index]]
            X_test.append(X_base_test)
    
            Y_train = [y[train_indexes[kfold_index]] for y in label_list]
            Y_test = [y[test_indexes[kfold_index]] for y in label_list]
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
        for emb_idx, emb in enumerate([field_signal, *emb_list]):
            for y_idx in range(len(label_list)):
                for decoder_name in decoder_list:
                    model_decoder = DECODERS[decoder_name]()
                    model_decoder.fit(X_train[emb_idx], Y_train[y_idx])
                    R2s[emb][decoder_name][kfold_index,y_idx,0] = median_absolute_error(Y_test[y_idx][:,0], model_decoder.predict(X_test[emb_idx])[:,0])
                    R2s[emb][decoder_name][kfold_index,y_idx,1] = median_absolute_error(Y_train[y_idx][:,0], model_decoder.predict(X_train[emb_idx])[:,0])
                
    return R2s 


def decoders_1D_dict(dict_df, field_signal = "ML_rates", emb_list=["ML_umap"], input_label = ["posx"],
                     n_dims = 10, n_splits=10, decoder_list = ["wf", "wc", "xgb", "svr"],verbose = False):  
    
    R2s_dict = dict()
    if verbose:
        fnames = list(dict_df.keys())
    count = 0
    for file, pd_struct in dict_df.items():
        count +=1
        if verbose:
            print('\tWorking on entry %i/' %count, '%i: ' %len(fnames), file, sep='')   
        #add trial signal if not present in session
        if 'index_mat' not in pd_struct.columns:
            pd_struct["index_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+pd_struct["trial_id"][idx] 
                                      for idx in range(pd_struct.shape[0])]
        #train decoders
        R2s_dict[file] = decoders_1D(pd_struct,field_signal = field_signal, input_label=input_label, emb_list = emb_list, 
                             input_trial = 'index_mat', n_dims = 10, n_splits=10, decoder_list = decoder_list, verbose = False)
        #print
        if verbose:
            print("")
    return R2s_dict 



def cross_session_decoders_LT(dict_df_A, dict_df_B, x_base = "ML_rates", x_emb="ML_umap", y_signal = "pos", n_dims = 3, n_splits=10, 
                              decoder_list = ["wf", "wc", "xgb", "svr"], verbose = False):  
    
    for file_A, pd_struct_A in dict_df_A.items():
        if 'index_mat' not in pd_struct_A.columns:
            pd_struct_A["index_mat"] = [np.zeros((pd_struct_A["pos"][idx].shape[0],1)).astype(int)+pd_struct_A["trial_id"][idx] 
                                      for idx in range(pd_struct_A.shape[0])]
    
    for file_B, pd_struct_B in dict_df_B.items():
        if 'index_mat' not in pd_struct_B.columns:
            pd_struct_B["index_mat"] = [np.zeros((pd_struct_B["pos"][idx].shape[0],1)).astype(int)+pd_struct_B["trial_id"][idx] 
                                      for idx in range(pd_struct_B.shape[0])]
    
    R2s = dict()
    R2s[x_emb] = dict()
    for decoder_name in decoder_list:
        R2s[x_emb][decoder_name] = np.zeros((len(dict_df_A),len(dict_df_B),5, n_splits))*np.nan
    
    query = lambda trial, fold_split: np.any(trial['trial_id']==fold_split)
    count_A = -1
    for file_A, pd_struct_A in dict_df_A.items():
        count_A += 1
        count_B = -1
        if verbose:
            print('\nWorking on file: %s'%file_A, end = '')   
        prefix_file_A = file_A[:file_A.find('_LT')]
        
        index_mat_A = copy.deepcopy(np.concatenate(pd_struct_A["index_mat"], axis = 0))
        index_list_A = np.unique(index_mat_A)
        train_index_A = int(index_list_A.shape[0]//1.25)

        for file_B, pd_struct_B in dict_df_B.items():
            count_B +=1
            prefix_file_B = file_B[:file_B.find('_LT')]
            if prefix_file_A != prefix_file_B:
                if verbose:
                    print('\n\tComparing it to: %s'%file_B, end = '')
                index_mat_B = copy.deepcopy(np.concatenate(pd_struct_B["index_mat"], axis = 0))
                index_list_B = np.unique(index_mat_B)
                train_index_B = int(index_list_B.shape[0]//1.25)
                if verbose:
                    print('\tKfold: X/X',end='', sep='')
                    pre_del = '\b\b\b'
                for kfold in range(n_splits):
                    if verbose:
                        print(pre_del,"%d/%d" %(kfold+1,n_splits), sep = '', end='')
                        pre_del = (len(str(kfold+1))+len(str(n_splits))+1)*'\b'
                    #1.1divide session A into train and test
                    fold_split_A = np.copy(index_list_A)
                    np.random.shuffle(fold_split_A)
                    pd_struct_A_train = copy.deepcopy(pd_struct_A.loc[[query(trial, fold_split_A[:train_index_A]) for (_, trial) in
                                                                  pd_struct_A.iterrows()],:])
                    pd_struct_A_test = copy.deepcopy(pd_struct_A.loc[[query(trial, fold_split_A[train_index_A:]) for (_, trial) in
                                                                  pd_struct_A.iterrows()],:])
                    pd_struct_A_train.reset_index()
                    pd_struct_A_test.reset_index()
                    #1.2divide session B into train and test
                    fold_split_B = np.copy(index_list_B)
                    np.random.shuffle(fold_split_B)
                    pd_struct_B_train = copy.deepcopy(pd_struct_B.loc[[query(trial, fold_split_B[:train_index_B]) for (_, trial) in
                                                                  pd_struct_B.iterrows()],:])
                    pd_struct_B_test = copy.deepcopy(pd_struct_B.loc[[query(trial, fold_split_B[train_index_B:]) for (_, trial) in
                                                                  pd_struct_B.iterrows()],:])
                    pd_struct_B_train.reset_index()
                    pd_struct_B_test.reset_index()
                    
                    #2.1compute umap embedding on train A and project test A
                    n_neighbours_A = np.round(np.concatenate(pd_struct_A_train[x_base].values, axis=0).shape[0]*0.01).astype(int)
                    pd_struct_A_train, model_umap_A = rd.dim_reduce(pd_struct_A_train, umap.UMAP(n_neighbors = n_neighbours_A, 
                                                                             n_components = n_dims, min_dist=0.75), 
                                                                              x_base, x_emb, return_model = True)
                    pd_struct_A_test = rd.apply_dim_reduce_model(pd_struct_A_test, model_umap_A, x_base, x_emb)
                    
                    #2.2compute umap embedding on train B and project test B
                    n_neighbours_B = np.round(np.concatenate(pd_struct_B_train[x_base].values, axis=0).shape[0]*0.01).astype(int)
                    pd_struct_B_train, model_umap_B = rd.dim_reduce(pd_struct_B_train, umap.UMAP(n_neighbors = n_neighbours_B, 
                                                                             n_components = n_dims, min_dist=0.75), 
                                                                              x_base, x_emb, return_model = True)
                    pd_struct_B_test = rd.apply_dim_reduce_model(pd_struct_B_test, model_umap_B, x_base, x_emb)
                    
                    
                    ###########################################################
                    #                       DO A TO B                         #
                    ###########################################################
                    #3.Find algiment for train data
                    TAB, RAB = align_manifolds_LT(pd_struct_A_train, pd_struct_B_train, ndims = n_dims, nCentroids = 20,
                                                   align_field = y_signal, emb_field = x_emb)
                    
                    #4.1 train decoder A
                    emb_A_train = np.concatenate(pd_struct_A_train[x_emb].values, axis=0)
                    for dim in range(emb_A_train.shape[1]):
                        emb_A_train[:,dim] -= np.mean(emb_A_train[:,dim], axis=0)
                    y_A_train = np.concatenate(pd_struct_A_train[y_signal].values, axis=0)
                    emb_A_test = np.concatenate(pd_struct_A_test[x_emb].values, axis=0)
                    for dim in range(emb_A_test.shape[1]):
                        emb_A_test[:,dim] -= np.mean(emb_A_test[:,dim], axis=0)
                    y_A_test = np.concatenate(pd_struct_A_test[y_signal].values, axis=0)
                    emb_AB_test = np.transpose(TAB + np.matmul(RAB, emb_A_test.T))
                    #4.2 train decoder B
                    emb_B_train = np.concatenate(pd_struct_B_train[x_emb].values, axis=0)
                    y_B_train = np.concatenate(pd_struct_B_train[y_signal].values, axis=0)
                    for dim in range(emb_B_train.shape[1]):
                        emb_B_train[:,dim] -= np.mean(emb_B_train[:,dim], axis=0)
                    emb_B_test = np.concatenate(pd_struct_B_test[x_emb].values, axis=0)
                    for dim in range(emb_B_test.shape[1]):
                        emb_B_test[:,dim] -= np.mean(emb_B_test[:,dim], axis=0)
                    y_B_test = np.concatenate(pd_struct_B_test[y_signal].values, axis=0)
                    #5. Get errors
                    for decoder_name in decoder_list:
                        model_decoder_A = DECODERS[decoder_name]()
                        model_decoder_A.fit(emb_A_train, y_A_train)
                        y_A_pred_A = model_decoder_A.predict(emb_A_test)
                        
                        model_decoder_B = DECODERS[decoder_name]()
                        model_decoder_B.fit(emb_B_train, y_B_train)
                        y_B_pred_B = model_decoder_B.predict(emb_B_test)
                        
                        y_A_pred_B = model_decoder_B.predict(emb_A_test)
                        y_AB_pred_B = model_decoder_B.predict(emb_AB_test)
                        
                        TAB_y, RAB_y = get_point_registration(np.c_[y_A_pred_B, np.zeros(y_A_pred_B.shape[0])], 
                                                              np.c_[y_A_test, np.zeros(y_A_test.shape[0])])
                        y_A_pred_B_rot = np.transpose(TAB_y + np.matmul(RAB_y, np.c_[y_A_pred_B, np.zeros(y_A_pred_B.shape[0])].T))
                        
                        R2s[x_emb][decoder_name][count_A,count_B,0,kfold] = median_absolute_error(y_A_test[:,0], y_A_pred_A[:,0])
                        R2s[x_emb][decoder_name][count_A,count_B,1,kfold] = median_absolute_error(y_B_test[:,0], y_B_pred_B[:,0])
                        R2s[x_emb][decoder_name][count_A,count_B,2,kfold] = median_absolute_error(y_A_test[:,0], y_A_pred_B[:,0])
                        R2s[x_emb][decoder_name][count_A,count_B,3,kfold] = median_absolute_error(y_A_test[:,0], y_AB_pred_B[:,0])
                        R2s[x_emb][decoder_name][count_A,count_B,4,kfold] = median_absolute_error(y_A_test[:,0], y_A_pred_B_rot[:,0])

    return R2s

def align_manifolds_LT(pd_struct_1, pd_struct_2, align_field = 'pos', emb_field = "ML_umap", ndims = 2, nCentroids = 20):

    emb_1 = copy.deepcopy(np.concatenate(pd_struct_1[emb_field].values, axis = 0))[:,:ndims]
    for dim in range(emb_1.shape[1]):
        emb_1[:,dim] -= np.mean(emb_1[:,dim], axis=0)
    pos_1 = copy.deepcopy(np.concatenate(pd_struct_1[align_field].values, axis = 0))[:,0].reshape(-1,1)
    if "dir_mat" not in pd_struct_1:
        pd_struct_1["dir_mat"] = [np.zeros((pd_struct_1[align_field][idx].shape[0],1)).astype(int)+ ('L' in pd_struct_1["dir"][idx])
                                + 2*('R' in pd_struct_1["dir"][idx]) for idx in pd_struct_1.index]
    dirmat_1 = copy.deepcopy(np.concatenate(pd_struct_1["dir_mat"].values, axis = 0))[:,0].reshape(-1,1)
    
    emb_1_left = copy.deepcopy(emb_1[dirmat_1[:,0]==1,:])
    emb_1_right = copy.deepcopy(emb_1[dirmat_1[:,0]==2,:])
    pos_1_left = copy.deepcopy(pos_1[dirmat_1==1])
    pos_1_right = copy.deepcopy(pos_1[dirmat_1==2])

    emb_2 = copy.deepcopy(np.concatenate(pd_struct_2[emb_field].values, axis = 0))[:,:ndims]
    for dim in range(emb_2.shape[1]):
        emb_2[:,dim] -= np.mean(emb_2[:,dim], axis=0)
    pos_2 = copy.deepcopy(np.concatenate(pd_struct_2[align_field].values, axis = 0))[:,0].reshape(-1,1)
    if "dir_mat" not in pd_struct_2:
        pd_struct_2["dir_mat"] = [np.zeros((pd_struct_2[align_field][idx].shape[0],1)).astype(int)+ ('L' in pd_struct_2["dir"][idx])
                                + 2*('R' in pd_struct_2["dir"][idx]) for idx in pd_struct_2.index]
    dirmat_2 = copy.deepcopy(np.concatenate(pd_struct_2["dir_mat"].values, axis = 0))[:,0].reshape(-1,1)
    
    emb_2_left = copy.deepcopy(emb_2[dirmat_2[:,0]==1,:])
    emb_2_right = copy.deepcopy(emb_2[dirmat_2[:,0]==2,:])
    
    pos_2_left = copy.deepcopy(pos_2[dirmat_2==1])
    pos_2_right = copy.deepcopy(pos_2[dirmat_2==2])

    #find pos edges to start dividing into intervals to later compute their centroids
    try:
        posLimits = np.array([(np.percentile(pos,1), np.percentile(pos,99)) for pos in [pos_1_left, pos_2_left, pos_1_right, pos_2_right]])
        posLimits = [np.max(posLimits[:,0]), np.min(posLimits[:,1])]
    except:
        total_pos = np.hstack((pos_1_left, pos_2_left, pos_1_right, pos_2_right))
        posLimits = np.array([(np.percentile(total_pos,1), np.percentile(total_pos,99))]).T[:,0]
        
    centSize = (posLimits[1] - posLimits[0]) / (nCentroids)
    centEdges = np.column_stack((np.linspace(posLimits[0],posLimits[0]+centSize*(nCentroids),nCentroids),
                                np.linspace(posLimits[0],posLimits[0]+centSize*(nCentroids),nCentroids)+centSize))
    centPos_1 = np.zeros((2*nCentroids,ndims))
    centPos_2 = np.zeros((2*nCentroids,ndims))
    for c in range(nCentroids):
        points_left = emb_1_left[np.logical_and(pos_1_left >= centEdges[c,0], pos_1_left<centEdges[c,1]),:]
        centPos_1[2*c,:] = np.median(points_left, axis=0)
        
        points_right = emb_1_right[np.logical_and(pos_1_right >= centEdges[c,0], pos_1_right<centEdges[c,1]),:]
        centPos_1[2*c+1,:] = np.median(points_right, axis=0)
        
        points_left = emb_2_left[np.logical_and(pos_2_left >= centEdges[c,0], pos_2_left<centEdges[c,1]),:]
        centPos_2[2*c,:] = np.median(points_left, axis=0)
        
        points_right = emb_2_right[np.logical_and(pos_2_right >= centEdges[c,0], pos_2_right<centEdges[c,1]),:]
        centPos_2[2*c+1,:] = np.median(points_right, axis=0)
    
    T12,R12 = get_point_registration(centPos_1, centPos_2)
    return T12, R12


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