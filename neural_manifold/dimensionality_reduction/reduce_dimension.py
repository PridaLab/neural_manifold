# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:29:18 2022

@author: Usuario
"""

import numpy as np


def dim_reduce(pd_struct, model,signal_field, new_signal_field, return_model = True):
    signal = np.concatenate(pd_struct[signal_field].values, axis=0)
    
    if "trial_id_mat" not in pd_struct.columns:
        pd_struct["trial_id_mat"] = [np.zeros((pd_struct[signal_field][idx].shape[0],1))+pd_struct["trial_id"][idx] 
                                  for idx in pd_struct.index]
    
    trial_id_mat = np.concatenate(pd_struct["trial_id_mat"].values, axis=0)
    
    signal_emb = model.fit_transform(signal)
    pd_struct[new_signal_field] = [signal_emb[trial_id_mat[:,0]==pd_struct["trial_id"][idx] ,:] 
                                   for idx in pd_struct.index]
    if return_model:
        return pd_struct, model
    else:
        return pd_struct
       
def apply_dim_reduce_model(pd_struct, model,signal_field, new_signal_field):
    
    signal = np.concatenate(pd_struct[signal_field].values, axis=0)
    if 'trial_id_mat' not in pd_struct:
        pd_struct["trial_id_mat"] = [np.zeros((pd_struct[signal_field][idx].shape[0],1))+pd_struct["trial_id"][idx] 
                                  for idx in pd_struct.index]
    trial_id_mat = np.concatenate(pd_struct["trial_id_mat"].values, axis=0)
    
    signal_emb = model.transform(signal)
    pd_struct[new_signal_field] = [signal_emb[trial_id_mat[:,0]==pd_struct["trial_id"][idx] ,:] 
                                   for idx in pd_struct.index]
    return pd_struct