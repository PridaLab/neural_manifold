# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:22:42 2022

@author: Usuario
"""

import pandas as pd
import numpy as np
import warnings
import copy
import functools

def check_inputs_for_pd(func):
    """Decorator to check if inputs are inside a dataframe. 
    
    It will check if an kwarg with the name 'pd_object' has been specified when
    calling the func. If so, it will search for any other kwarg with the name
    '*signal'. Any '*signal' kwarg whose type is 'str' (or a list containing 
    'str') will be pass through the 'pd_to_array_translator' translator to
    extract a numpy array (or list of numpy arrays) before calling the final
    function. 
    
    Note that args (without a key) will be ignored and passed to the final
    function without any change. 

    Parameters
    ----------
    func : function
        Function being decorated. 
    
    *args: arguments
        Arguments to be passed down to the decored function w/o any modification
    **kwargs: keyword arguments
        Key arguments for the decorated function. If the kwarg 'pd_object' is 
        passed down, then any kwargs whose name has the sequence '*_signal' that
        is a string or a list of string will be interpreted as columns of the 
        'pd_object' where the actual signal resides and thus will be transformed
        into numpy array (or a list of the former) by calling the translator 
        'pd_to_array_translator' before passing them down to the decorated 
        function.
        
    Returns
    -------
    wrapper_check_inputs_for_pd: function
        Decorated function. Same function as the one prior to be decorated but
        the inputs have been changed if a 'pd_object' has been specified.
    """
    @functools.wraps(func)
    def wrapper_check_inputs_for_pd(*args, **kwargs):
        if 'pd_object' in kwargs and isinstance(kwargs['pd_object'], pd.DataFrame):
            #if there is a 'pd_object' check if any input has to be extract from 
            #the columns of the panda dataframe before running the function.
            new_kwargs = copy.deepcopy(kwargs)
            del new_kwargs['pd_object'] #delete pd_object from kwargs of final function
            #get list of kwargs with the key word 'signal'
            signal_keys = [key for key in list(kwargs.keys()) if '_signal' in key] 
            for signal in signal_keys:
                if isinstance(kwargs[signal], str):
                    #extract signal from pd_struct into array
                    new_kwargs[signal] = pd_to_array_translator(kwargs['pd_object'], kwargs[signal])
                elif isinstance(kwargs[signal], list):
                    #extract list of signals from pd_struct into list of arrays
                    for sub_signal_idx, sub_signal in enumerate(kwargs[signal]):
                        if isinstance(sub_signal, str):
                            new_kwargs[signal][sub_signal_idx] = pd_to_array_translator(kwargs['pd_object'], sub_signal)          
        else:
            #If there is not a 'pd_object' then simply run the function
            new_kwargs = copy.deepcopy(kwargs)
        return func(*args, **new_kwargs)
    
    return wrapper_check_inputs_for_pd


def pd_to_array_translator(pd_struct,field):
    """Translator used to convert a DataFrame column into a numpy array by 
    concatenating along axis=0. Note it also accepts 'posx' & 'posy' subquery 
    if the Dataframe has a 'pos' column.
    
    Parameters
    ----------
    pd_struct: DataFrame
        Pandas DataFrame from which to extract the signal.
        
    field: String
        Name of column in the pd_struct with the signal one wants to 
        extract. Note that it will be concatenated along axis=0.
                           
    Returns
    -------
    signal: numpy array
        Concatenated dataframe column into numpy array along axis=0.
    """
    if isinstance(pd_struct, pd.DataFrame):
        if isinstance(field, str):
            if ('pos' in field) and (field not in pd_struct.columns):
                signal= np.concatenate(pd_struct['pos'].values, axis=0)
                if 'posx' in field:
                    return signal[:,0].reshape(-1,1)
                elif 'posy' in field:
                    return signal[:,1].reshape(-1,1)
                
            elif field in pd_struct.columns:
                signal = np.concatenate(pd_struct[field].values, axis=0)
                return signal
            else:
                raise ValueError(f"Indicated field {field} is not in the "+
                                 "columns of the dataframe.")
        else:
            raise ValueError("Input 'signal' must be a string with the name "+
                             "of the column of the dataframe from which to "+
                             f"extract the signal. But it was a {type(signal)}")
    else:
        raise ValueError("Called dataframe translator function but input is "+
                         f"not a dataframe but a {type(pd_struct)}. Returning"+
                         " original input.")


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




#-----------------------------------------------------------------------------#
#old functions to be removed when all is transitioned
def dataframe_to_1array_translator(pd_struct,field):
    
    '''Translator used to convert a specific panda DataFrame atribute into a 
        numpy array by concatenating along axis=0.
        
    Parameters:
        pd_struct (DataFrame): pandas dataframe
        field (string): name of column with the signal one wants to extract. 
                        Note that it will be concatenated along axis=0.
                        
    Returns:
        signal (array): concatenated dataframe column.
    '''
    
    if isinstance(pd_struct, pd.DataFrame):
        if isinstance(field, str):
            if ('pos' in field) and (field not in pd_struct.columns):
                signal= np.concatenate(pd_struct['pos'].values, axis=0)
                if 'posx' in field:
                    return signal[:,0].reshape(-1,1)
                elif 'posy' in field:
                    return signal[:,1].reshape(-1,1)
                
            elif field in pd_struct.columns:
                signal = np.concatenate(pd_struct[field].values, axis=0)
                return signal
            else:
                raise ValueError(f"Indicated field {field} is not in the "+
                                 "columns of the dataframe.")
        else:
            raise ValueError("Input 'signal' must be a string with the name "+
                             "of the column of the dataframe from which to "+
                             f"extract the signal. But it was a {type(signal)}")
    else:
        raise ValueError("Called dataframe translator function but input is "+
                         f"not a dataframe but a {type(pd_struct)}. Returning"+
                         " original input.")
    
def dataframe_to_manyarray_translator(pd_struct,field_list):
    '''Translator used to convert a specific panda DataFrame atribute into a numpy array by concatenating
    along axis=0.
    
    Parameters
    ----------
    pd_struct: DataFrame
        Pandas DataFrame from which to extract the signal.
        
    field: String
        Name of column in the pd_struct with the signal one wants to 
        extract. Note that it will be concatenated along axis=0.
                           
    Returns
    -------
        signal: (array): concatenated dataframe column.
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