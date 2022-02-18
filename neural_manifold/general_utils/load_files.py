# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:13:36 2022

@author: Usuario
"""

import os, fnmatch
import pyaldata as pyd
import numpy as np
import pickle

# %%
###########################################################################
#                                   LOAD FILES
###########################################################################    

def get_files(data_dir, pattern, struct_type = "PyalData", verbose=False):
    """
    Load all files in the given directory whose name/type match the pattern specified.
    
    Parameters
    ----------
    data_dir: string with the directory in which to perform the query
    
    pattern: string with the pattern to test the files against before loading 
            (Shortcuts: '*pattern' matches everything, '?pattern' matches any single character, 
             '[seq]pattern' matches any character in seq, '[!seq]pattern' matches any character not in seq)
        
    verbose: False/True
    
    Returns
    -------
    dictionary containing all the files loaded
    """
    fnames = [f for f in os.listdir(data_dir) if fnmatch.fnmatch(os.path.join(data_dir,f), pattern)]
    if verbose:
        print('\tFound ',len(fnames),' files.')
        [print('\t',idx,fname, sep='. ') for idx, fname in enumerate(fnames)];
        print('\tLoading file: 0/', len(fnames), sep = '',end='')
        
    data_dict = dict()
    for idx, file in enumerate(fnames):
        if "PyalData" in struct_type:
            try:
                name_dict = file[:file.find('_PyalData')]
            except:
                name_dict = file[:-3]
            data_dict[name_dict] = pyd.mat2dataframe(os.path.join(data_dir,file), shift_idx_fields = True)
        elif "pickle" in struct_type:
            load_file = open(os.path.join(data_dir,file),"rb")
            data_dict[file] = pickle.load(load_file)
            load_file.close()
        else:
            data_dict[file] = np.load(os.path.join(data_dir,file))
        if verbose:
            print('\b\b\b', idx+1, '/', len(fnames), sep = '', end = '')
    if verbose:
        print('')
    if len(data_dict)==1:
        data_dict = data_dict[file]
    return data_dict
