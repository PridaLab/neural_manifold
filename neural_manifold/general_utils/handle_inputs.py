# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:06:27 2022

@author: Usuario
"""
import pandas as pd
import numpy as np
import copy
from neural_manifold.general_utils import translators as trns


def handle_input_dataframe_array(input_object, input_signal_1 = None, input_signal_2 = None, first_array = True): 
    '''General function to handle inputs that can either be a dataframe+column_string or a numpy array.
    Priority is as follow:
    input_object[input_signal_1]>input_object[input_signal_2]>input_signal_1>input_signal_2>input_object
    '''
    if isinstance(input_signal_1, str) and isinstance(input_object, pd.DataFrame): 
        return trns.dataframe_to_1array_translator(input_object,input_signal_1)
    elif isinstance(input_signal_2, str) and isinstance(input_object, pd.DataFrame):
        return trns.dataframe_to_1array_translator(input_object,input_signal_2)
    elif isinstance(input_signal_1, np.ndarray):
        return copy.deepcopy(input_signal_1)
    elif isinstance(input_signal_2, np.ndarray):
        return copy.deepcopy(input_signal_2)
    elif isinstance(input_object,np.ndarray) and first_array:
        return copy.deepcopy(input_object)
    else:
        raise ValueError("Could not handle input. Input object was %s. Signal_1 was %s. Signal 2 was %s."
                         %(type(input_object), type(input_signal_1), type(input_signal_2)))