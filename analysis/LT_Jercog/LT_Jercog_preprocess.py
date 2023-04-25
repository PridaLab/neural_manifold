import umap
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import neural_manifold.general_utils as gu
import copy
import matplotlib.pyplot as plt
import sys, os, copy, pickle, timeit
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks
import timeit
#__________________________________________________________________________
#|                                                                        |#
#|                       MAKE SESSIONS SAME LENGTH                        |#
#|________________________________________________________________________|#

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

def add_dir_mat_field(pd_struct):
    out_pd = copy.deepcopy(pd_struct)
    if 'dir_mat' not in out_pd.columns:
        out_pd["dir_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            ('L' == out_pd["dir"][idx])+ 2*('R' == out_pd["dir"][idx])+
                            for idx in out_pd.index]
    return out_pd

def preprocess_traces(pd_struct, signal_field, sigma = 5, sig_up = 4, sig_down = 12, peak_th=0.1):
    out_pd = copy.deepcopy(pd_struct)

    out_pd["index_mat"] = [np.zeros((out_pd[signal_field][idx].shape[0],1))+out_pd["trial_id"][idx] 
                                  for idx in range(out_pd.shape[0])]                     
    index_mat = np.concatenate(out_pd["index_mat"].values, axis=0)

    signal_og = copy.deepcopy(np.concatenate(pd_struct[signal_field].values, axis=0))
    lowpass_signal = uniform_filter1d(signal_og, size = 4000, axis = 0)
    signal = gaussian_filter1d(signal_og, sigma = sigma, axis = 0)

    for nn in range(signal.shape[1]):
        base_signal = np.histogram(signal_og[:,nn], 100)
        base_signal = base_signal[1][np.argmax(base_signal[0])]
        base_signal = base_signal + lowpass_signal[:,nn] - np.min(lowpass_signal[:,nn]) 

        clean_signal = signal[:,nn]-base_signal
        clean_signal = clean_signal/np.max(clean_signal,axis = 0)
        clean_signal[clean_signal<0] = 0
        signal[:,nn] = clean_signal

    bi_signal = np.zeros(signal.shape)

    gaus = lambda x,sig,amp,vo: amp*np.exp(-(((x)**2)/(2*sig**2)))+vo;
    x = np.arange(-5*sig_down, 5*sig_down,1);
    gaus_up = gaus(x,sig_up, 1, 0); 
    gaus_up[5*sig_down+1:] = 0
    gaus_down = gaus(x,sig_down, 1, 0); 
    gaus_down[:5*sig_down+1] = 0
    gaus_final = gaus_down + gaus_up;

    for nn in range(signal.shape[1]):
        peaks_signal,_ =find_peaks(signal[:,nn],height=peak_th)
        bi_signal[peaks_signal, nn] = signal[peaks_signal, nn]
        if gaus_final.shape[0]<signal.shape[0]:
            bi_signal[:, nn] = np.convolve(bi_signal[:, nn],gaus_final, 'same')

    out_pd['clean_traces'] = [bi_signal[index_mat[:,0]==out_pd["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd.shape[0])]

    return out_pd


mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
#%% PARAMS
kernel_std = 0.3
kernel_num_std = 5
sigma = 6
sig_up = 4
sig_down = 12
signal_field = 'rawProb'
peak_th = 0.1
vel_th = 3
min_session_len = 6000
equalize_session_len = True
verbose = True
og_save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/processed_data/'
og_data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/'

for mouse in mice_list:
    #initialize time
    global_starttime = timeit.default_timer()
    #create save folder data by adding time suffix
    save_dir = _create_save_folders(og_save_dir, mouse)
    data_dir = os.path.join(og_data_dir, mouse)
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
    if verbose:
        gu.print_time_verbose(local_starttime, global_starttime)

    #%% 2. PROCESS DATA
    print('\n### 2. PROCESS DATA ###')

    #2.1 compute firing rate
    print(f"2.1 Computing rates/revents with window size = {1e3*kernel_std:.2f} ms", sep='')
    df_og = gu.apply_to_dict(gu.add_firing_rates, df_og, 'smooth', std=kernel_std,
                            num_std = kernel_num_std, continuous = True, assymetry = True)

    df_og = gu.apply_to_dict(add_dir_mat_field,df_og)

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

    print(f"2.2 Computing clean-traces from {signal_field} with sigma = {sigma}," +
        f"sigma_up = {sig_up}, sigma_down = {sig_down}", sep='')
    df = gu.apply_to_dict(preprocess_traces, df, signal_field, sigma = sigma, sig_up = sig_up,
                            sig_down = sig_down, peak_th = peak_th)
    if vel_th>0:
        still_df = gu.apply_to_dict(preprocess_traces, still_df, signal_field, sigma = sigma, sig_up = sig_up,
                                sig_down = sig_down, peak_th = peak_th)
    #2.3 Fix uneven session length
    if equalize_session_len:
        print('2.3 Fixing uneven session length')
        local_starttime = timeit.default_timer()
        df = _fix_cross_session_length(df, min_session_len, verbose = verbose)
        fnames = list(df.keys())
        if verbose:
            gu.print_time_verbose(local_starttime, global_starttime)

    save_df = open(os.path.join(save_dir, mouse+ "_df_dict.pkl"), "wb")
    pickle.dump(df, save_df)
    save_df.close()

    save_still = open(os.path.join(save_dir, mouse+ "_still_df_dict.pkl"), "wb")
    pickle.dump(still_df, save_still)
    save_still.close()
    
    params = {
        'kernel_std': kernel_std,
        'kernel_num_std': kernel_num_std,
        'sigma': sigma,
        'sig_up': sig_up,
        'sig_down': sig_down,
        'signal_field': signal_field,
        'peak_th': peak_th,
        'vel_th': vel_th,
        'min_session_len': min_session_len,
        'data_dir': data_dir,
        'save_dir': save_dir,
        'mouse': mouse
    }
    save_params = open(os.path.join(save_dir, mouse+ "_params.pkl"), "wb")
    pickle.dump(params, save_params)
    save_params.close()

    # create list of strings
    list_of_strings = [ f'{key} : {params[key]}' for key in params]
    # write string one by one adding newline
    save_params_file = open(os.path.join(save_dir, mouse+ "_params.txt"), "w")
    with save_params_file as my_file:
        [my_file.write("%s\n" %st) for st in list_of_strings]
    save_params_file.close()
    sys.stdout = original
    f.close()