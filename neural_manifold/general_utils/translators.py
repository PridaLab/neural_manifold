# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:22:42 2022

@author: Usuario
"""

import pandas as pd
import numpy as np
import warnings

def apply_to_dict(input_function, input_dict, *args, **kwargs):
    try:
        if 'verbose' in kwargs and kwargs['verbose']:
            print(f"Applying function '{input_function.__name__}' to dict: ", end='')
            dict_new = dict()
            for name, pd_struct in input_dict.items():
                print(f'\n{name}: ')
                dict_new[name] = input_function(pd_struct, *args, **kwargs) 
        else:
            dict_new = {name: input_function(pd_struct, *args, **kwargs) for name, pd_struct in input_dict.items()}
        return dict_new
    except:
        raise ValueError("Could not apply function to dictionary")

def dataframe_to_1array_translator(pd_struct,field):
    '''Translator used to convert a specific panda DataFrame atribute into a numpy array by concatenating
    along axis=0.
    
    Parameters:
        pd_struct (DataFrame): pandas dataframe
        field (string): name of column with the signal one wants to extract. Note that it will be concatenated
                           
    Returns:
        signal (array): concatenated dataframe column.
    '''
    if isinstance(pd_struct, pd.DataFrame):
        if field is not None:
            if isinstance(field, str):
                if ('pos' in field) and (field not in pd_struct.columns):
                    signal= np.concatenate(pd_struct['pos'], axis=0)
                    if 'posx' in field:
                        return signal[:,0].reshape(-1,1)
                    elif 'posy' in field:
                        return signal[:,1].reshape(-1,1)
                    
                elif field in pd_struct.columns:
                    signal = np.concatenate(pd_struct[field], axis=0)
                    return signal
                else:
                    raise ValueError("Field is not in columns of dataframe.")
        else:
            raise ValueError("If input is a Dataframe, you must indicate the name of the column to use as signal (e.g. 'field'='ML_rates').")

    else:
        warnings.warn("Called dataframe translator function but input is not a dataframe." +
                      "Returning original input.",SyntaxWarning)
        return pd_struct
    
def dataframe_to_manyarray_translator(pd_struct,field_list):
    '''Translator used to convert a specific panda DataFrame atribute into a numpy array by concatenating
    along axis=0.
    
    Parameters:
        pd_struct (DataFrame): pandas dataframe
        field (string): name of column with the signal one wants to extract. Note that it will be concatenated
                           
    Returns:
        signal (array): concatenated dataframe column.
    '''
    if isinstance(pd_struct, pd.DataFrame):
        if field_list is not None:
            if isinstance(field_list, list):
                signal = list()
                for field in field_list:
                    if ('pos' in field) and (field not in pd_struct.columns):
                        signal.append(dataframe_to_1array_translator(pd_struct,'pos'))
                    else:
                        signal.append(dataframe_to_1array_translator(pd_struct,field))
                        
                    if 'posx' in field:
                        signal[-1] = signal[-1][:,0]
                    elif 'posy' in field:
                        signal[-1] = signal[-1][:,1]
                return signal
                
        else:
            raise ValueError("If input is a Dataframe, you must indicate the name of the column to use as signal (e.g. 'field'='ML_rates').")

    else:
        warnings.warn("Called dataframe translator function but input is not a dataframe." +
                      "Returning original input.",SyntaxWarning)
        return pd_struct