
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:54:46 2022

@author: julio
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime
import numpy as np
import sys, os, pickle, copy, timeit
from neural_manifold import general_utils as gu

def LT_session_length(data_dir,mouse, save_dir, **kwargs):

    if 'vel_th' in kwargs:
        vel_th = kwargs['vel_th']
    else:
        vel_th = 3
        kwargs['vel_th'] = vel_th

    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = True
        kwargs["verbose"] = verbose

    assert 'kernel_std' in kwargs, "provide kernel_std (s) to compute rates/revents"
    kernel_std = kwargs['kernel_std']
    if 'kernel_num_std' in kwargs:
        kernel_num_std = kwargs['kernel_num_std']
    else:
        kernel_num_std = 5
        kwargs['kernel_num_std'] = kernel_num_std

    if 'equalize_session_len' in kwargs:
        equalize_session_len = kwargs['equalize_session_len']
    else:
        equalize_session_len = True
        kwargs['equalize_session_len'] = equalize_session_len

    if 'min_session_len' in kwargs:
        min_session_len = kwargs['min_session_len']
    else:
        min_session_len = 6000
        kwargs['min_session_len'] = min_session_len

    #initialize time
    global_starttime = timeit.default_timer()
    #create save folder data by adding time suffix
    save_dir = _create_save_folders(save_dir, mouse)
    kwargs['save_dir'] = save_dir
    #check if verbose has to be saved into txt file
    if verbose:
        f = open(os.path.join(save_dir,mouse + '_logFile.txt'), 'w')
        original = sys.stdout
        sys.stdout = gu.Tee(sys.stdout, f)

    # %% 1.LOAD DATA
    local_starttime = timeit.default_timer()
    print('\n### 1. LOAD DATA ###')
    print('1 Searching & loading data in directory:\n', data_dir)
    df_og = gu.load_files(data_dir, '*_PyalData_struct*.mat', verbose=verbose)
    fnames = list(df_og.keys())
    kwargs['fnames'] = fnames
    if verbose:
        gu.print_time_verbose(local_starttime, global_starttime)

    #%% 2. PROCESS DATA
    print('\n### 2. PROCESS DATA ###')

    #2.1 compute firing rate
    print(f"2.1 Computing rates/revents with window size = {1e3*kernel_std:.2f} ms", sep='')
    df_og = gu.apply_to_dict(gu.add_firing_rates, df_og, 'smooth', std=kwargs["kernel_std"],
                            num_std = kernel_num_std, continuous = True, assymetry = True)
    if verbose:
        gu.print_time_verbose(local_starttime, global_starttime)

    #2.2 Check if only keep moving periods
    if vel_th>0:
        #2.2 keep only moving epochs
        print(f'2.2 Dividing into moving/still data w/ vel_th= {vel_th:.2f}.')
        local_starttime = timeit.default_timer()

        df = dict()
        still_df = dict()
        for name, pd_struct in df_og.items():
            df[name], still_df[name] = gu.keep_only_moving(pd_struct, vel_th)
       
        if verbose:
            print('\tDuration change:')
            og_dur = [np.concatenate(pd_struct["pos"].values, axis=0).shape[0] for _, pd_struct in df_og.items()]
            move_dur = [np.concatenate(pd_struct["pos"].values, axis=0).shape[0] for _, pd_struct in df.items()]

            for idx, name in enumerate(fnames):
                print(f"\t\t{name}: Og={og_dur[idx]} ({og_dur[idx]/20}s) Move= {move_dur[idx]} ({move_dur[idx]/20}s)")

    else:
        print('2.2 Keeping all data (not limited to moving periods).')
        df = copy.deepcopy(df_og)
        #create dummy still dictionaries
        still_df = dict()

    #2.3 Fix uneven session length
    if equalize_session_len:
        print('2.3 Fixing uneven session length')
        local_starttime = timeit.default_timer()
        df = _fix_cross_session_length(df, min_session_len, verbose = verbose)
        fnames = list(df.keys())
        kwargs["fnames"] = fnames
        if verbose:
            gu.print_time_verbose(local_starttime, global_starttime)

    save_df = open(os.path.join(save_dir, mouse+ "_df_dict.pkl"), "wb")
    pickle.dump(df, save_df)
    save_df.close()

    save_still = open(os.path.join(save_dir, mouse+ "_still_df_dict.pkl"), "wb")
    pickle.dump(still_df, save_still)
    save_still.close()


    kwargs['data_dir'] = data_dir
    kwargs['mouse'] = mouse

    save_params = open(os.path.join(save_dir, mouse+ "_params.pkl"), "wb")
    pickle.dump(kwargs, save_params)
    save_params.close()

    # create list of strings
    list_of_strings = [ f'{key} : {kwargs[key]}' for key in kwargs]
    # write string one by one adding newline
    save_params_file = open(os.path.join(save_dir, mouse+ "_params.txt"), "w")
    with save_params_file as my_file:
        [my_file.write("%s\n" %st) for st in list_of_strings]
    save_params_file.close()
    
    sys.stdout = original
    f.close()

    return

# %% 
###############################################################################
#                              GENERAL FUNCTIONS                              #
###############################################################################
def _create_save_folders(save_dir, mouse):
    #try creating general folder
    try:
        os.mkdir(save_dir)
    except:
        pass
    #add new folder with mouse name + current date-time
    save_dir = os.path.join(save_dir, mouse)
    #create this new folder
    try:
        os.mkdir(save_dir)
    except:
        pass
    #create figure subfolder inside save folder
    '''
    save_plot_dir =os.path.join(save_dir,'Figures')
    try:
        os.mkdir(save_plot_dir)
    except:
        pass
    return save_dir, save_plot_dir

    '''
    return save_dir


def _fix_cross_session_length(df_dict, min_session_len, verbose = False):
    #get length of each session
    def recursive_len(item):
        try:
           iter(item)
           return sum(recursive_len(subitem) for subitem in item)
        except TypeError:
           return 1

    session_len = {file: np.sum(pd_struct["pos"].apply(recursive_len), axis=0)/2 for file, pd_struct in df_dict.items()}
    bin_size = [pd_struct["bin_size"][0] for _, pd_struct in df_dict.items()]
    if verbose:
        print(f'\tSetting session duration to the shortest one (or to {min_session_len*bin_size[0]:.2f}s): ', sep ='', end= '')

    final_session_len = np.max([min_session_len,  np.min([dur for _, dur in session_len.items()])])
    bin_size = bin_size[np.argmin([dur for _, dur in session_len.items()])]
    if verbose:
        print(f" {int(final_session_len)} samples ({final_session_len*bin_size}s)")

        print('\tOriginal session duration: ')
        [print(f"\t\t{file[:21]}: {int(session_len[file])} samples ({session_len[file]*pd_struct['bin_size'][0]:.2f}s)")
                                                                        for file, pd_struct in df_dict.items()];    
    df_new_dict = dict()
    for file, pd_struct in df_dict.items():

        temporal_fields = gu.get_temporal_fields(pd_struct)
        bin_size = pd_struct["bin_size"][0]
        relative_duration = round(session_len[file]/final_session_len,2)
        
        if relative_duration<0.9: #session last less than 90% of max length
            if verbose:
                print(f"\tSession {file[:21]} last only {100*relative_duration:.2f} of the desired one. Take it into account")
                df_new_dict[file] = pd_struct

        elif 0.9<=relative_duration<=1:
            df_new_dict[file] = pd_struct

        else: 
            num_div = np.ceil(relative_duration).astype(int)
            if verbose:
                print(f"\tSession {file[:21]} lasts {100*relative_duration:.2f} of the desired one. Diving it into {num_div} sections")
            for div in range(num_div-1):
                limit_index = 0
                trial = 0
                consec_length = 0
                stp = 0

                while trial < pd_struct.shape[0] and stp == 0:
                    consec_length += pd_struct["pos"][trial].shape[0]
                    if consec_length<final_session_len:
                        trial +=1
                    else:
                        if pd_struct["pos"][trial].shape[0]/(consec_length-final_session_len)>0.5:
                            limit_index = trial
                        else:
                            limit_index = trial+1
                        stp = 1
                if stp==1:
                    df_new_dict[file+'_'+str(div+1)] = copy.deepcopy(pd_struct.iloc[:limit_index,:].reset_index(drop = True))
                    pd_struct = copy.deepcopy(pd_struct.iloc[limit_index+1:, :].reset_index(drop = True))
                else:
                    df_new_dict[file+'_'+str(div+1)] = pd_struct.reset_index(drop = True)
                    pd_struct = []

            try:    
                new_relative_duration = 0.5*np.sum(pd_struct["pos"].apply(recursive_len), axis=0)/final_session_len
            except:
                new_relative_duration = 0
                
            if new_relative_duration<0.8:
                if verbose:
                    print(f"\t\tPart {div+2} lasts only {100*new_relative_duration:.2f} of the desired one. Discarding it")
            elif new_relative_duration<1:
                if verbose:
                    print(f"\t\tPart {div+2} lasts {100*new_relative_duration:.2f} of the desired one. Keeping it")
                df_new_dict[file+'_'+str(div+2)] = copy.deepcopy(pd_struct.reset_index(drop = True))
                
    if verbose:
        new_session_length = {file: np.sum(pd_struct["pos"].apply(recursive_len), axis=0)/2 for file, pd_struct in df_new_dict.items()}
        print('\tNew session duration: ')
        [print(f"\t\t{file}: {int(new_session_length[file])} samples ({new_session_length[file]*pd_struct['bin_size'][0]:.2f}s)")
                                                                        for file, pd_struct in df_new_dict.items()];
    return df_new_dict