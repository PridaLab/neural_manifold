# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:28:58 2022

@author: JulioEI
"""
import os, pickle
from neural_manifold import general_utils as gu
import timeit
import numpy as np
from neural_manifold import structure_index as sI 

def compute_sI(data_dir, mouse, fieldList, varList, save_file = True,move_dict = None, **kwargs):
    ###########################################################################
    #                           VERBOSE PARAMETERS
    ###########################################################################
    #display verbose
    if 'verbose' not in kwargs:
        kwargs["verbose"] = True
    if 'time_verbose' not in kwargs:
        kwargs["time_verbose"] = True
    #save verbose to external txt
    if 'save_verbose' not in kwargs:
        kwargs["save_verbose"] = True
    ###########################################################################
    #                           GENERAL PARAMETERS
    ###########################################################################
    if 'save_dir' not in kwargs:
        kwargs["save_dir"] = data_dir
    if 'n_dims' not in kwargs:
        kwargs["n_dims"] = 10
    if 'nRep' not in kwargs:
        kwargs["nRep"] = 100    
    if 'load_old_dict' not in kwargs:
        kwargs["load_old_dict"] = None
    if 'comp_method' not in kwargs:
        kwargs["comp_method"] = 'pairwise'
    if 'nBins' not in kwargs:
        kwargs["nBins"] = 10
        
    global_starttime = timeit.default_timer()
    local_starttime = timeit.default_timer()
    if not move_dict:
        if kwargs["verbose"]:
            print("Loading data: ")
        move_dict = gu.load_files(data_dir, '*_move_data_dict.pkl', verbose = kwargs["verbose"], struct_type = "pickle")
        if not move_dict:
            move_dict = gu.load_files(data_dir, '*_data_dict.pkl', verbose = kwargs["verbose"], struct_type = "pickle")
    fnames = list(move_dict.keys())
    if kwargs["time_verbose"]:
        gu.print_time_verbose(local_starttime, global_starttime)
        
    if kwargs["load_old_dict"]:
        local_starttime = timeit.default_timer()
        if kwargs["verbose"]:
            print("Loading old dictionary: ")
        if"old_dict_dir" not in kwargs:
            kwargs["old_dict_dir"] = data_dir
        if "old_dict_name" not in kwargs:
            kwargs["old_dict_name"] = '*_sI_dict.pkl'
        sI_dict = gu.load_files(kwargs["old_dict_dir"], kwargs["old_dict_name"], verbose = kwargs["verbose"], struct_type = "pickle")
        if kwargs["time_verbose"]:
            gu.print_time_verbose(local_starttime, global_starttime)
        for file, _ in move_dict.items():
            if file not in sI_dict.keys():
                sI_dict[file] = dict()
            for field in fieldList:
                if field not in sI_dict[file]:
                    sI_dict[file][field] = dict()
    else:
        sI_dict = dict()
        for file, _ in move_dict.items():
            sI_dict[file] = dict()
            for field in fieldList:
                sI_dict[file][field] = dict()    
    ###########################################################################
    #                              MAIN FUNCTION
    ###########################################################################
    count_file = 0
    for file, pd_struct in move_dict.items():
        #For each session
        if isinstance(kwargs["n_dims"], str):
            if 'adapt_to_umap' in kwargs["n_dims"]:
                umap_field = [field for field in fieldList if "umap" in field]
                umap_dims = np.concatenate(pd_struct[umap_field[0]].values, axis=0).shape[1]
        local_starttime_file = timeit.default_timer()
        if kwargs["verbose"]:
            count_file +=1
            print("Working on entry %i/" %count_file, "%i: " %len(fnames), file, sep='')
        
        count_field = 0
        #For each embedding (e.g. PCA, Isomap, Umap)
        for field in fieldList:
            if kwargs["verbose"]:
                count_field  +=1
                print("\tWorking on field %i/" %count_field, "%i: " %len(fieldList), field, sep='')
            local_starttime_field = timeit.default_timer()
            emb = np.concatenate(pd_struct[field].values, axis=0)
            labelList = []
            for v in range(len(varList)):
               if 'pos' in varList[v]:
                   label = np.concatenate(pd_struct["pos"].values, axis=0)
               elif 'index_mat' in varList[v] and 'index_mat' not in pd_struct.columns:
                   pd_struct["index_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+pd_struct["trial_id"][idx] 
                                                 for idx in range(pd_struct.shape[0])]
                   label = np.concatenate(pd_struct[varList[v]].values, axis=0)
               elif 'dir_mat' in varList[v] and 'dir_mat' not in pd_struct.columns:
                   pd_struct["dir_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+
                                             ('L' in pd_struct["dir"][idx])+ 2*('R' in pd_struct["dir"][idx])
                                                 for idx in range(pd_struct.shape[0])]
                   label = np.concatenate(pd_struct[varList[v]].values, axis=0)
               else:
                   label = np.concatenate(pd_struct[varList[v]].values, axis=0)

               if label.shape[1]>0:
                   if 'posx' in varList[v]:
                       label = label[:,0].reshape(-1,1)
                   elif 'posy' in varList[v]:
                       label = label[:,1].reshape(-1,1)
               labelList.append(label.T[0])
                
            valLimits = [(np.percentile(label,5), np.percentile(label,95)) for label in labelList]
            if isinstance(kwargs["n_dims"], str):
                if 'adapt_to_umap' in kwargs["n_dims"]:
                    if emb.shape[1]<umap_dims or 'rates' in field:
                        n_dims = emb.shape[1]
                    else:
                        n_dims = umap_dims                        
                elif 'adapt' in kwargs["n_dims"]:
                    n_dims = emb.shape[1]
            else:
                n_dims = kwargs["n_dims"]
            
            if 'pairwise' in kwargs["comp_method"]:
                sI_labels = np.empty((n_dims, n_dims, len(varList)))
            elif 'triplets' in kwargs["comp_method"]:
                first = np.linspace(0,n_dims-1,n_dims).astype(int)
                second = np.linspace(1,n_dims-1,n_dims-1).astype(int)
                third = np.linspace(2,n_dims-1,n_dims-2).astype(int)
                #all non-repeating combinations
                combined = [(f,s,t) for f in first for s in second if s>f for t in third if t>s]
                
                combined_labels =[(f+1,s+1,t+1) for f in first for s in second if s>f for t in third if t>s]
                sI_labels = np.empty((len(combined), len(varList)))
            elif 'all' in kwargs["comp_method"]:
                sI_labels = np.empty((len(varList)))
            
            pval_labels = []
            #For each variable (e.g. posx, posy, velocity)
            for v in range(len(varList)):
                if kwargs["verbose"]:
                    print('\t\tComputing for variable %s' %varList[v])
                label = labelList[v]
                minVal, maxVal = valLimits[v]
                if 'pairwise' in kwargs["comp_method"]:
                    sI_mat = np.ones((n_dims, n_dims))*np.nan
                    pval_mat = np.ones((n_dims, n_dims, kwargs["nRep"]))*np.nan
                    print("\t\t\tComputing dimensions (%d,%d)" %(0,0), sep= '', end ='')
                    pre_del = '\b\b\b\b\b'
                    for ii in range(n_dims):
                        for jj in range(ii+1,n_dims):
                            if kwargs["verbose"]:
                                print(pre_del,"(%d,%d)" %(ii+1,jj+1), sep = '', end='')
                                pre_del = (len(str(ii+1))+len(str(jj+1))+3)*'\b'
    
                            sI_val, bLab, _ = sI.compute_structure_index(emb, label, kwargs["nBins"], [ii,jj], 0, vmin= minVal, vmax = maxVal)
                            sI_mat[ii,jj], sI_mat[jj,ii] = np.repeat(sI_val,2)
                            sI_shuffle = np.empty((kwargs["nRep"], 1))
                            for n in range(kwargs["nRep"]):
                                randLabel = label[np.random.permutation(range(len(label)))]
                                sI_shuffle[n],_,_ = sI.compute_structure_index(emb, randLabel, kwargs["nBins"], [ii,jj], 0, vmin= minVal, vmax = maxVal)
                            pval_mat[ii,jj,:], pval_mat[jj,ii,:] = sI_shuffle.ravel(), sI_shuffle.ravel()
                    print('')
                    sI_labels[:,:,v] = sI_mat
                    pval_labels.append(pval_mat)
                    
                elif 'triplets' in kwargs["comp_method"]:
                    sI_mat = np.ones((len(combined), 1))*np.nan
                    pval_mat = np.ones((len(combined), 1, kwargs["nRep"]))*np.nan
                    print("\t\t\tComputing dimensions (%d,%d,%d)" %(0,0,0), sep= '', end ='')
                    pre_del = '\b\b\b\b\b\b\b'
                    for ii, comb in enumerate(combined):
                        if kwargs["verbose"]:
                            print(pre_del, '%s' %(combined_labels[ii], ), sep="", end="")
                            pre_del = (len(str(combined_labels[ii])))*'\b'
                        
                        sI_mat[ii], bLab, _ = sI.compute_structure_index(emb, label, kwargs["nBins"], np.array(comb), 0, vmin= minVal, vmax = maxVal)
                        sI_shuffle = np.empty((kwargs["nRep"], 1))
                        for n in range(kwargs["nRep"]):
                            randLabel = label[np.random.permutation(range(len(label)))]
                            sI_shuffle[n],_,_ = sI.compute_structure_index(emb, randLabel, kwargs["nBins"], np.array(comb), 0, vmin= minVal, vmax = maxVal)
                        pval_mat[ii,:] = sI_shuffle.ravel()
                        
                    print('')
                    sI_labels[:,v] = sI_mat[:,0]
                    pval_labels.append(pval_mat)
                elif 'all' in kwargs["comp_method"]:
                    pval_mat = np.ones((kwargs["nRep"]))*np.nan
                    emb_space = np.linspace(0,n_dims-1, n_dims).astype(int)                                
                    sI_val, bLab, _ = sI.compute_structure_index(emb, label, kwargs["nBins"], emb_space,
                                                                      0, vmin= minVal, vmax = maxVal)
                    sI_shuffle = np.empty((kwargs["nRep"], 1))
                    for n in range(kwargs["nRep"]):
                        randLabel = label[np.random.permutation(range(len(label)))]
                        sI_shuffle[n],_,_ = sI.compute_structure_index(emb, randLabel, kwargs["nBins"], emb_space,
                                                                       0, vmin= minVal, vmax = maxVal)
                    sI_labels[v] = sI_val
                    pval_labels.append(sI_shuffle.ravel())
                        
            if 'pairwise' in kwargs["comp_method"]:  
                
                sI_dict[file][field].update({'sI_pw':sI_labels, 'pval_pw':pval_labels, 
                                        'label_pw': np.array(varList, dtype='object'), 
                                        'file_pw': file, 'field_pw':field})
            elif 'triplets' in kwargs["comp_method"]:
                sI_dict[file][field].update({'sI_trip':sI_labels, 'pval_trip':pval_labels, 
                                        'comb_trip': combined,
                                        'label_trip': np.array(varList, dtype='object'), 
                                        'file_trip': file, 'field_trip':field})
            elif 'all' in kwargs["comp_method"]:
                sI_dict[file][field].update({'sI_all':sI_labels, 'pval_all':pval_labels, 
                                        'label_all': np.array(varList, dtype='object'), 
                                        'file_all': file, 'field_all':field})
            if save_file:
                save_file = open(os.path.join(kwargs["save_dir"], mouse+ "_sI_dict.pkl"), "wb")
                pickle.dump(sI_dict, save_file)
                save_file.close()        
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime_field, global_starttime)
        if kwargs["time_verbose"]:
            gu.print_time_verbose(local_starttime_file, global_starttime)
            
    return sI_dict
