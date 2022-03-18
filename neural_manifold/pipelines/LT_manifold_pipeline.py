# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:12:15 2022

@author: JulioEI
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import copy
import timeit
from datetime import datetime
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from neural_manifold import general_utils as gu
from neural_manifold import dimensionality_reduction as dim_red
# %%
###############################################################################
#                                 MAIN FUNCTION                               #
###############################################################################
def LT_manifold_pipeline(data_dir,mouse, save_dir, **kwargs):
    #initialize time
    global_starttime = timeit.default_timer()
    #validate inputs
    kwargs = _validate_inputs(kwargs)
    #create save folder data by adding suffix
    save_dir, save_plot_dir = _create_save_folders(save_dir, mouse)
    #check if verbose has to be saved into txt file
    if kwargs["save_verbose"]:
        f = open(os.path.join(save_dir,'logFile.txt'), 'w')
        original = sys.stdout
        sys.stdout = gu.Tee(sys.stdout, f)
        
    # %% 1.LOAD DATA
    local_starttime = timeit.default_timer()
    print('\n### 1. LOAD DATA ###')
    print('1 Searching & loading data in directory:\n', data_dir)
    df_og = gu.load_files(data_dir, '*_PyalData_struct*.mat', verbose=kwargs["verbose"])
    kwargs["fnames"] = list(df_og.keys())
    if kwargs["time_verbose"]:
        gu.print_time_verbose(local_starttime, global_starttime)
    # %% 2. PREPROCESS DATA
    local_starttime = timeit.default_timer()
    print('\n### 2.PREPROCESS DATA ###')
    #display number of neurons
    if kwargs["verbose"]:
        if kwargs["spikes_field"]:
            print('2.0 Original Number of neurons: ')
            num_cells = {fname: pd_struct[kwargs["spikes_field"]][0].shape[1]
                              for fname , pd_struct in df_og.items()}
            [print('\t',entry_name[:21],':',
                   '\n\t\tOriginal # neurons: ', num_cells[entry_name], 
                    sep='') for entry_name, _ in df_og.items()];
        elif kwargs["traces_field"]:
            print('2.0 Original Number of neurons: ')
            num_cells = {fname: pd_struct[kwargs["traces_field"]][0].shape[1]
                              for fname , pd_struct in df_og.items()}
            [print('\t',entry_name[:21],':',
                   '\n\t\tOriginal # neurons: ', num_cells[entry_name], 
                    sep='') for entry_name, _ in df_og.items()]; 
    
    #2.1 Check neuron minimum activity (both for rates and traces if applicable)
    if kwargs["rates_field"]:
        #2.1 compute firing rate
        print('2.1 Computing firing rates with window size = %.2f' %(1e3*kwargs["rates_kernel_std"]), 'ms', sep='')
        df_og = gu.apply_to_dict(gu.add_firing_rates, df_og, 'smooth', std=kwargs["rates_kernel_std"], 
                                                                 continuous = True, assymetry = True)
        if kwargs["time_verbose"]:
            gu.print_time_verbose(local_starttime, global_starttime)
            
    #2.2 keep only moving epochs
    if kwargs["keep_only_moving"]:
        print('2.2 Dividing into moving/still/fail trials.')
        local_starttime = timeit.default_timer()
        
        df = gu.apply_to_dict(gu.select_trials, df_og,"dir == ['L', 'R']")
        still_df = gu.apply_to_dict(gu.select_trials, df_og,"dir == 'N'")
        fail_df = gu.apply_to_dict(gu.select_trials, df_og,"dir != ['L','R','N']")

        if kwargs["time_verbose"]:
            [print(f"\t{entry_name[:21]}:\n\t\tMoving trials: {df[entry_name].shape[0]}",
                   f"(L:{sum(df[entry_name]['dir']=='L')}, R:{sum(df[entry_name]['dir']=='R')})",
                   f"\n\t\tStill trials: {still_df[entry_name].shape[0]}",
                   f"\n\t\tFail trials: {fail_df[entry_name].shape[0]}") for entry_name, _ in df.items()];
            gu.print_time_verbose(local_starttime, global_starttime)
        #TODO: implement minimum trial duration and time warping
        '''
        #This should go inside the keep_only_moving and before fixing uneven length
        #2.3 keep only long enough trials to do time warping
        if kwargs["min_bins"]>0:
            print('2.4 keeping only trials longer than %i bins.' %kwargs["min_bins"])
            local_starttime = timeit.default_timer()
            df = _remove_short_trials(df, kwargs)
            if kwargs["time_verbose"]:
                _print_time_verbose(local_starttime, global_starttime)
        #Warping can be done after proyecting the data. that is, you timwarp the 
        #projected data to find the mean trajectory!
        #2.4 do time warping
        if kwargs["do_time_warping"]:
            print('2.5 Doing time warping.')
            local_starttime = timeit.default_timer()
            df = _do_timewarping(df, kwargs)
            if kwargs["time_verbose"]:
                _print_time_verbose(local_starttime, global_starttime)
        '''
    else:
        print('2.2 Keeping all data (not limited to moving trials).')
        df = copy.deepcopy(df_og)
        #create dummy fail and still dictionaries
        fail_df = dict()
        still_df = dict()
        
    #2.3 Fix uneven session length
    if kwargs["max_session_length"]:
        print('2.3 Fixing uneven session length')
        local_starttime = timeit.default_timer()
        df = _fix_cross_session_length(df, kwargs)
        kwargs["fnames"] = list(df.keys())
        if kwargs["time_verbose"]:
            gu.print_time_verbose(local_starttime, global_starttime)
            
    #2.4 remove low firing neurons
    if kwargs["th_rates_freq"]>0:
        local_starttime = timeit.default_timer()
        print('\t2.4.1 Removing low firing neurons (th: ', kwargs["th_rates_freq"], 'Hz):',sep='')
        #TODO: note that now cells are removed from all data (not only form spiking activity but also
        #from traces. Fix this when implementing traces_events_th)
        if kwargs["verbose"]: 
            old_num_cells = {name: pd_struct[kwargs["rates_field"]][0].shape[1]
                              for name , pd_struct in df.items()}
        if not kwargs["apply_same_model"]:
            df = gu.apply_to_dict(gu.remove_low_firing_neurons, df, kwargs["rates_field"], 
                                         kwargs["th_rates_freq"], divide_by_bin_size = False)
        else:
            print('same model th')
            av_rates = [np.mean(np.concatenate(pd_struct[kwargs["rates_field"]].values, axis=0), 
                                           axis=0)< kwargs["th_rates_freq"] for _,pd_struct in df.items()]
            mask_rates = np.invert(np.multiply(av_rates[0], av_rates[1])) 
            df = gu.apply_to_dict(gu.remove_low_firing_neurons, df, kwargs["rates_field"], mask = mask_rates)
            
        if kwargs["verbose"]:
            [print('\t\t',entry_name[:21],':',
                   '\n\t\t\tOriginal # neurons: ', old_num_cells[entry_name], 
                   '\n\t\t\tRemaining # neurons: ', df[entry_name][kwargs["rates_field"]][0].shape[1],
                   '\n\t\t\t%.2f' % (100*df[entry_name][kwargs["rates_field"]][0].shape[1]/ old_num_cells[entry_name]), 
                   '%', sep='') for entry_name, _ in df.items()];   

        if kwargs["time_verbose"]:
            gu.print_time_verbose(local_starttime, global_starttime) 
    if kwargs["traces_field"] and kwargs["th_traces_freq"]>0:
        #2.4 Calculate number of events
        #TODO: implement threshold by rates of events in traces
        print('\t2.4.2 Removing neurons with low transient events (th: ', kwargs["th_traces_freq"], 'Hz):',sep='')
        print('\t\tFunction to detect events still needs to be coded. Keeping all neurons')
        
    # %% 3. CHECK INNER DIMENSIONALITY
    if kwargs["check_inner_dim"]:
        print('\n### 3.INTERNAL DIMENSIONALITY ###')
        if kwargs["rates_field"]:
            local_starttime = timeit.default_timer()
            print('3.1 Internal dimensionality rates: ')
            internal_dim_rates = _compute_inner_dim(df, kwargs["rates_field"], save_plot_dir, mouse,kwargs)
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime, global_starttime)
                
        if kwargs["traces_field"]:
            local_starttime = timeit.default_timer()
            print('3.2 Internal dimensionality traces: ')
            internal_dim_traces = _compute_inner_dim(df, kwargs["traces_field"], save_plot_dir, mouse, kwargs)
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime, global_starttime)
                
    # %% 4.PCA
    if kwargs["compute_pca"]:
        print('\n### 4.PCA ###')
        if kwargs["rates_field"]:
            local_starttime = timeit.default_timer()
            print("4.1 Computing PCA on rates.")
            df, still_df, fail_df, models_pca_rates = _compute_pca_LT(df, kwargs["rates_field"],
                                                            kwargs["pca_rates_field"], still_df, fail_df,
                                                            save_plot_dir, mouse, kwargs)
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime, global_starttime)
                
        if kwargs["traces_field"]:
            local_starttime = timeit.default_timer()
            print("4.2 Computing PCA on traces.")
            df, still_df, fail_df, models_pca_traces = _compute_pca_LT(df, kwargs["traces_field"],
                                                            kwargs["pca_traces_field"], still_df, fail_df,
                                                            save_plot_dir, mouse, kwargs)
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime, global_starttime)
                
    # %% 5.Isomap
    if kwargs["compute_iso"]:
        #define isomap kernel to compute reconstruction error
        #K = lambda D: -0.5*((np.eye(D.shape[0])-(1/D.shape[0])).dot(np.square(D))).dot(np.eye(D.shape[0])-(1/D.shape[0])) 
        print('\n### 5. Isomap ###')
        if kwargs["rates_field"]:
            local_starttime = timeit.default_timer()
            print("5.1 Computing Isomap on spikes.")
            df, still_df, fail_df, models_iso_rates, kwargs = _compute_isomap_LT(df, kwargs["rates_field"],
                                                            kwargs["iso_rates_field"], still_df, fail_df,
                                                            save_plot_dir, mouse, kwargs)
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime, global_starttime)

        if kwargs["traces_field"]:
            local_starttime = timeit.default_timer()
            print("5.2 Computing isomap on traces.")
            df, still_df, fail_df, models_iso_traces, kwargs = _compute_isomap_LT(df, kwargs["traces_field"],
                                                            kwargs["iso_traces_field"], still_df, fail_df,
                                                            save_plot_dir, mouse, kwargs)
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime, global_starttime)
                
    # %% 6. Umap
    if kwargs["compute_umap"]:
        print('\n### 6. Umap ###')
        if kwargs["rates_field"]:
            local_starttime = timeit.default_timer()
            print("6.1 Computing Umap on spikes.")
            df, still_df, fail_df, models_umap_rates = _compute_umap_LT(df, kwargs["rates_field"],
                                                            kwargs["umap_rates_field"], still_df, fail_df,
                                                            kwargs["neighbours_umap_rates"], kwargs["min_dist_umap_rates"],
                                                            save_plot_dir, mouse, kwargs)
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime, global_starttime)
                
        if kwargs["traces_field"]:
            local_starttime = timeit.default_timer()
            print("6.2 Computing Umap on traces.")
            df, still_df, fail_df, models_umap_traces = _compute_umap_LT(df, kwargs["traces_field"],
                                                            kwargs["umap_traces_field"], still_df, fail_df,
                                                            kwargs["neighbours_umap_traces"], kwargs["min_dist_umap_traces"],
                                                            save_plot_dir, mouse, kwargs)
            if kwargs["time_verbose"]:
                gu.print_time_verbose(local_starttime, global_starttime)
                
    # %% 7. PLACE CELLS
    if kwargs["compute_place_cells"]:
        import utils_place_cells as upc
        local_starttime = timeit.default_timer()
        print('\n### 7. Place Cells ###')
        place_cells_dict = dict()
        print("7 Computing place cells according to ",kwargs["place_method"]," :", sep='')
        count = 0
        for file, pd_struct in df.items():
            if kwargs["verbose"]:
                count += 1
                print('\tWorking on entry %i/' %count, '%i: ' %len(kwargs["fnames"]), file, sep='')  
                
            temp_dict = dict()
            save_plot_dir_cells = os.path.join(save_plot_dir,"Cells_"+file[:21])
            try:
                os.mkdir(save_plot_dir_cells)
            except:
                pass
            place_cells, metric_val, th_metric_val, rate_map, pdf_map  = upc.get_place_cells_LT(pd_struct, 
                                                                "pos", kwargs["rates_field"], kwargs["spikes_field"],
                                                                save_plot_dir_cells, method = kwargs["place_method"])
            temp_dict["place_cells"] = place_cells
            temp_dict[kwargs["place_method"]] = metric_val
            temp_dict["th_"+kwargs["place_method"]] = th_metric_val
            temp_dict["rate_map"] = rate_map
            temp_dict["pdf_map"] = pdf_map
            place_cells_dict[file] = temp_dict
        if kwargs["time_verbose"]:
            gu.print_time_verbose(local_starttime, global_starttime)
            
    # %% 8. SAVE FILES
    print('\n### 8. Saving files ###')
    import pickle
    if kwargs["keep_only_moving"]:
        save_move_file = open(os.path.join(save_dir, mouse+ "_move_data_dict.pkl"), "wb")
        pickle.dump(df, save_move_file)
        save_move_file.close()
        
        save_still_file = open(os.path.join(save_dir, mouse+ "_still_data_dict.pkl"), "wb")
        pickle.dump(still_df, save_still_file)
        save_still_file.close()
        
        save_fail_file = open(os.path.join(save_dir, mouse+ "_fail_data_dict.pkl"), "wb")
        pickle.dump(fail_df, save_fail_file)
        save_fail_file.close()
    else:
        save_df_file = open(os.path.join(save_dir, mouse+ "_data_dict.pkl"), "wb")
        pickle.dump(df, save_df_file)
        save_df_file.close()
        
    if 'internal_dim_rates' in locals():
        save_internal_dim_file = open(os.path.join(save_dir, mouse+ "_internal_dim_rates_dict.pkl"), "wb")
        pickle.dump(internal_dim_rates, save_internal_dim_file)
        save_internal_dim_file.close()
    if 'internal_dim_traces' in locals():
        save_internal_dim_file = open(os.path.join(save_dir, mouse+ "_internal_dim_traces_dict.pkl"), "wb")
        pickle.dump(internal_dim_traces, save_internal_dim_file)
        save_internal_dim_file.close()
        
    if 'models_pca_rates' in locals():
        save_move_file = open(os.path.join(save_dir, mouse+ "_models_pca_rates_dict.pkl"), "wb")
        pickle.dump(models_pca_rates, save_move_file)
        save_move_file.close()
    if 'models_pca_traces' in locals():
        save_move_file = open(os.path.join(save_dir, mouse+ "_models_pca_traces_dict.pkl"), "wb")
        pickle.dump(models_pca_traces, save_move_file)
        save_move_file.close()    
        
    if 'models_iso_rates' in locals():
        save_move_file = open(os.path.join(save_dir, mouse+ "_models_iso_rates_dict.pkl"), "wb")
        pickle.dump(models_pca_rates, save_move_file)
        save_move_file.close()
    if 'models_iso_traces' in locals():
        save_move_file = open(os.path.join(save_dir, mouse+ "_models_iso_traces_dict.pkl"), "wb")
        pickle.dump(models_pca_traces, save_move_file)
        save_move_file.close()  
        
    if 'models_umap_rates' in locals():
        save_move_file = open(os.path.join(save_dir, mouse+ "_models_umap_rates_dict.pkl"), "wb")
        pickle.dump(models_umap_rates, save_move_file)
        save_move_file.close()
    if 'models_umap_traces' in locals():
        save_move_file = open(os.path.join(save_dir, mouse+ "_models_umap_traces_dict.pkl"), "wb")
        pickle.dump(models_umap_traces, save_move_file)
        save_move_file.close()  
    
    if 'place_cells_dict' in locals():
        save_move_file = open(os.path.join(save_dir, mouse+ "_place_cells_dict.pkl"), "wb")
        pickle.dump(place_cells_dict, save_move_file)
        save_move_file.close()  
    
    save_params_file = open(os.path.join(save_dir, mouse+ "_pipeline_params.pkl"), "wb")
    pickle.dump(kwargs, save_params_file)
    save_params_file.close()
    
    # create list of strings
    list_of_strings = [ f'{key} : {kwargs[key]}' for key in kwargs]
    # write string one by one adding newline
    save_params_file = open(os.path.join(save_dir, mouse+ "_pipeline_params.txt"), "w")
    with save_params_file as my_file:
        [my_file.write("%s\n" %st) for st in list_of_strings]
    save_params_file.close()
    
    sys.stdout = original
    f.close()

    return save_dir, df

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
    save_dir = os.path.join(save_dir, mouse +'_' + datetime.now().strftime("%d%m%y_%H%M%S"))
    #create this new folder
    try:
        os.mkdir(save_dir)
    except:
        pass
    #create figure subfolder inside save folder
    save_plot_dir =os.path.join(save_dir,'Figures')
    try:
        os.mkdir(save_plot_dir)
    except:
        pass
    return save_dir, save_plot_dir

def _validate_inputs(kwargs):
    '''
    Validate pipeline inputs and complete missing ones with default values. The order in which inputs are checked are as follow:
        
    Verbose Parameters
    Plot Parameters
    Rates Parameters
    Traces Parameters
    Preprocessing Parameters
    Inner Dimensionalitty Parameters
    Manifold General Parameters
    PCA Parameters
    Isomap Parameters
    UMAP Parameters
    Place Cells Parameters
    
    '''
    ###########################################################################
    #                           VERBOSE PARAMETERS
    ###########################################################################
    #display verbose
    if 'verbose' not in kwargs:
        kwargs["verbose"] = True
    #save verbose to external txt
    if 'save_verbose' not in kwargs:
        kwargs["save_verbose"] = True
    #display time verbose
    if 'time_verbose' not in kwargs:
        kwargs["time_verbose"] = True
    ########################################################################### 
    #                               PLOT PARAMETERS
    ###########################################################################
    #display plots
    if 'display_plots' not in kwargs:
        kwargs["display_plots"] = True
    #plot trajectory on manifolds (note time warping needed)
    if 'plot_trajectory' not in kwargs:
        kwargs["plot_trajectory"] = False
    ###########################################################################
    #                            RATES PARAMETERS  
    ###########################################################################                          
    #name of field containing the firing rates
    if 'spikes_field' not in kwargs:
        kwargs["spikes_field"] = "ML_spikes" #check that field exists or if similar with spikes
    #add name of future rates field
    if 'rates_field' not in kwargs:
        kwargs["rates_field"] = kwargs["spikes_field"][:kwargs["spikes_field"].rfind('_')] + "_rates"
    #std in seconds of of gaussian window to compute firing rates
    if 'rates_kernel_std' not in kwargs:
        kwargs["rates_kernel_std"] = 0.05 #s 
    #minimum firing rate below which a neuron is deleted (in Hz)
    if 'th_rates_freq' not in kwargs:
        kwargs["th_rates_freq"] = 0.05 #Hz
    ###########################################################################        
    #                            TRACES PARAMETERS  
    ###########################################################################                         
    #name of field containing the Ca traces
    if 'traces_field' not in kwargs:    
        kwargs["traces_field"] = "deconvProb" #check that field exists
    if 'th_traces_freq' not in kwargs:
        kwargs["th_traces_freq"] = 0.01 #Hz
    ###########################################################################
    #                           PREPROCESS PARAMETERS               
    ###########################################################################    
    if 'keep_only_moving' not in kwargs:
        kwargs["keep_only_moving"] = False
    #maximum session length
    if 'max_session_length' not in kwargs:
        kwargs["max_session_length"] = 'adapt_to_min' #sec
    #minimum number of time bins on each trial (left&right) (aka minimum trial duration)
    #below which a trial is discarded (maybe include max duration?)
    if 'min_bins' not in kwargs:
        kwargs["min_bins"] = 0
    #whether or not to do time warping. Note if it set to True it will take the min_bin
    #parameter as grid to perform. 
    if 'do_time_warping' not in kwargs:
        kwargs["do_time_warping"] = False
    ###########################################################################
    #                      INNER DIMENSIONALITY PARAMETERS      
    ###########################################################################              
    if 'check_inner_dim' not in kwargs:
        kwargs["check_inner_dim"] =True
    ###########################################################################    
    #                              MANIFOLD                   
    ###########################################################################  
    # %% MANIFOLD
    #compute model only on first session and apply it to the following ones. (Requires CellReg)
    if 'apply_same_model' not in kwargs:
        kwargs["apply_same_model"] = False
    ###########################################################################    
    #                              PCA PARAMETERS                   
    ###########################################################################              
    #boolean indicating whether or not to compute PCA projection
    if 'compute_pca' not in kwargs:
        kwargs["compute_pca"] = True
    if kwargs["compute_pca"]:
        #number of dimensions for pca
        if 'pca_dims' not in kwargs:
            if 'manifold_dims' in kwargs: #if specified general manifold dimension
                kwargs["pca_dims"] = kwargs["manifold_dims"]
            else:
                kwargs["pca_dims"] = 10
        #field name to asign to the new pca embedding using firing rates
        if ('pca_rates_field' not in kwargs) and (kwargs["rates_field"]):
            kwargs["pca_rates_field"] = "ML_pca"
        #field name to asign to the new pca embedding using Ca traces
        if ('pca_traces_field' not in kwargs) and (kwargs["traces_field"]):
            kwargs["pca_traces_field"] = kwargs["traces_field"] + "_pca"
    ###########################################################################
    #                             ISOMAP PARAMETERS    
    ###########################################################################                         
    if 'compute_iso' not in kwargs:
        kwargs["compute_iso"] = True
    if kwargs["compute_iso"]:
        #number of dimensions for pca
        if 'iso_dims' not in kwargs:
            if 'manifold_dims' in kwargs: #if specified general manifold dimension
                kwargs["iso_dims"] = kwargs["manifold_dims"]
            else:
                kwargs["iso_dims"] = 10
        #field name to asign to the new iso embedding using firing rates
        if ('iso_rates_field' not in kwargs) and (kwargs["rates_field"]):
            kwargs["iso_rates_field"] = "ML_iso"     
        #number of neighbours to construct the rates embedding
        if ('neighbours_iso_rates' not in kwargs) and (kwargs["rates_field"]):
            kwargs["neighbours_iso_rates"] = 15
        #field name to asign to the new iso embedding using firing rates
        if ('iso_traces_field' not in kwargs) and (kwargs["traces_field"]):
            kwargs["iso_traces_field"] = kwargs["traces_field"] + "_iso"
        #number of neighbours to construct the traces embedding            
        if ('neighbours_iso_traces' not in kwargs) and (kwargs["traces_field"]):
            kwargs["neighbours_iso_traces"] = 45 
        if 'compute_iso_resvar' not in kwargs:
            kwargs["compute_iso_resvar"] = True
    ###########################################################################
    #                              UMAP PARAMETERS       
    ###########################################################################                      
    if 'compute_umap' not in kwargs:
        kwargs["compute_umap"] = True
    if kwargs["compute_umap"]:
        #number of dimensions for pca
        if 'umap_dims' not in kwargs:
            if 'manifold_dims' in kwargs: #if specified general manifold dimension
                kwargs["umap_dims"] = kwargs["manifold_dims"]
            else:
                kwargs["umap_dims"] = 10
        #rand state when computing umap embedding (if None, random state on which is faster)
        if 'rand_state_umap' not in kwargs:
            kwargs["rand_state_umap"] = None
        #field name to asign to the new iso embedding using firing rates
        if ('umap_rates_field' not in kwargs) and (kwargs["rates_field"]):
            kwargs["umap_rates_field"] = "ML_umap"     
        #number of neighbours to construct the rates embedding
        if ('neighbours_umap_rates' not in kwargs) and (kwargs["rates_field"]):
            kwargs["neighbours_umap_rates"] = 'adapt'
        #min dist for umap embedding on traces
        if ('min_dist_umap_rates' not in kwargs) and (kwargs["rates_field"]):
            kwargs["min_dist_umap_rates"] = 0.5  
        #field name to asign to the new iso embedding using firing rates
        if ('umap_traces_field' not in kwargs) and (kwargs["traces_field"]):
            kwargs["umap_traces_field"] = kwargs["traces_field"] + "_umap"
        #number of neighbours to construct the traces embedding            
        if ('neighbours_umap_traces' not in kwargs) and (kwargs["traces_field"]):
            kwargs["neighbours_umap_traces"] = 'adapt' 
        #min dist for umap embedding on traces
        if ('min_dist_umap_traces' not in kwargs) and (kwargs["traces_field"]):
            kwargs["min_dist_umap_traces"] = 0.5
        if 'check_dim_to_cells_umap' not in kwargs:
            kwargs["check_dim_to_cells_umap"] = False
    ###########################################################################
    #                           PLACE CELLS PARAMETERS                        #
    ###########################################################################      
    if 'compute_place_cells' not in kwargs:
        kwargs["compute_place_cells"] = True
    if kwargs["compute_place_cells"]:
        if 'place_method' not in kwargs:
            kwargs["place_method"] = "response_profile"
    return kwargs
# %%
###############################################################################
#                            PREPROCESSING FUNCTIONS                          #
###############################################################################
#TODO: implement delete short trials
'''
def _remove_short_trials(df_dict, kwargs):
    old_num_trials = {fname: [sum(pd_struct["dir"]=='L'),sum(pd_struct["dir"]=='R')]
                      for fname , pd_struct in df_dict.items()}
                      
    df_dict = {name: pyd.select_trials(pd_struct, lambda trial: trial.pos.shape[0]>=kwargs["min_bins"])
                       for name, pd_struct in df_dict.items()}
    if kwargs["verbose"]:
        [print('\t',entry_name[:21],': ',
               '\n\t\tOld moving trials- L:', old_num_trials[entry_name][0],
               ' R:', old_num_trials[entry_name][1],
               
               '\n\t\tNew moving trials- L:', sum(pd_struct["dir"]=='L'),
               ', R:', sum(pd_struct["dir"]=='R'),
               '\n\t\t(L: %.2f' %(100*sum(pd_struct["dir"]=='R')/old_num_trials[entry_name][1]),
               '%)', sep='') for entry_name, pd_struct in df_dict.items()];
    return df_dict
'''
#TODO: implement timewarping 
'''
def _do_timewarping(df_dict, kwargs):
    count = 0
    for name, pd_struct in df_dict.items():
        count +=1
        df_dict.update({name: tj.time_warp(pd_struct, bin_num = kwargs["min_bins"])})
        print('\tWorking on entry %i/' %count, '%i: ' %len(kwargs["fnames"]), name, sep='')   
    if kwargs["verbose"]:
        print('\tBin size after time warping: ')
        [print('\t\t', entry_name[:21],': %.2f' %(1000*np.mean(pd_struct["bin_size"])), 
              u" \u00B1 %.2f" %(1000*np.std(pd_struct["bin_size"])), 'ms', sep ='')
             for entry_name, pd_struct in df_dict.items()];
    return df_dict
'''
def _fix_cross_session_length(df_dict, kwargs):
    #get length of each session
    def recursive_len(item):
        try:
           iter(item)
           return sum(recursive_len(subitem) for subitem in item)
        except TypeError:
           return 1
    session_length = {file: np.sum(pd_struct["pos"].apply(recursive_len), axis=0)/2 for file, pd_struct in df_dict.items()}
    if isinstance(kwargs["max_session_length"], str):
        if 'adapt_to_min' in kwargs["max_session_length"]:
            bin_size = [pd_struct["bin_size"][0] for _, pd_struct in df_dict.items()]
            print('\tSetting session duration to the shortest one: ', sep ='', end= '')
            kwargs["max_session_length"] = np.min([dur for _, dur in session_length.items()])
            bin_size = bin_size[np.argmin([dur for _, dur in session_length.items()])]
            if kwargs["max_session_length"]*bin_size<180:
                kwargs["max_session_length"] = 180/bin_size
            print(kwargs["max_session_length"].astype(int), 'timestamps (%.2f s)' %(kwargs["max_session_length"]*bin_size))
            kwargs["max_session_length"] *= bin_size
            
    if kwargs["verbose"]:
        print('\tOriginal session duration: ')
        [print('\t\t', file[:21],': %.2f' %(session_length[file]*pd_struct["bin_size"][0]), 
             's', sep ='') for file, pd_struct in df_dict.items()];    
        
    df_new_dict = dict()
    for file, pd_struct in df_dict.items():
        bin_size = pd_struct["bin_size"][0]
        duration_in_sec = session_length[file]*bin_size
        relative_duration = round(duration_in_sec/kwargs["max_session_length"],2)
        
        if relative_duration<0.9: #session last less than 60% of max length
            if kwargs["verbose"]:
                print('\tSession ' , file[:21], ' lasts only %.2f' %(100*relative_duration),
                      '%% of the desired duration. Take it into account.', sep='' )
                df_new_dict[file] = pd_struct
        elif 0.9<=relative_duration<=1:
            df_new_dict[file] = pd_struct
        else: 
            num_div = np.ceil(relative_duration).astype(int)
            if kwargs["verbose"]:
                print('\tSession ' , file[:21], ' lasts %.2f' %(100*relative_duration),
                      '%% of the desired duration. Diving it into %d.' %num_div, sep='' )
            for div in range(num_div-1):
                limit_index = 0
                trial = 0
                consec_length = 0
                stp = 0
                while trial < pd_struct.shape[0] and stp == 0:
                    consec_length += pd_struct["pos"][trial].shape[0]
                    if consec_length*bin_size<kwargs["max_session_length"]:
                        trial +=1
                    else:
                        if pd_struct["pos"][trial].shape[0]*bin_size/(consec_length-kwargs["max_session_length"])>0.5:
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
                new_relative_duration = 0.5*np.sum(pd_struct["pos"].apply(recursive_len), axis=0)*\
                                            bin_size/kwargs["max_session_length"]
            except:
                new_relative_duration = 0
                
            if new_relative_duration<0.8:
                if kwargs["verbose"]:
                    print('\t\tPart %d'%(div+2), ' lasts only %.2f' %(100*new_relative_duration),
                          '%% of the desired duration. Discarding it.', sep='' )
            elif new_relative_duration<1:
                if kwargs["verbose"]:
                    print('\t\tPart %d'%(div+2), ' lasts %.2f' %(100*new_relative_duration),
                          '%% of the desired duration. Take it into account.', sep='' )
                df_new_dict[file+'_'+str(div+2)] = copy.deepcopy(pd_struct.reset_index(drop = True))
                
    if kwargs["verbose"]:
        new_session_length = {file: np.sum(pd_struct["pos"].apply(recursive_len), axis=0)/2 for file, pd_struct in df_new_dict.items()}
        print('\tNew session duration: ')
        [print('\t\t', file,': %.2f' %(new_session_length[file]*pd_struct["bin_size"][0]), 
             's', sep ='') for file, pd_struct in df_new_dict.items()];   
    return df_new_dict
# %%
###############################################################################
#                            INTERNAL DIMENSIONALITY                          #
###############################################################################
def _compute_inner_dim(df_dict, field, save_plot_dir, mouse, kwargs):
    internal_dim = dict()
    count = 0
    for file, pd_struct in df_dict.items():
        if kwargs["verbose"]:
            count += 1
            print('\tWorking on entry %i/' %count, '%i: ' %len(kwargs["fnames"]), file, sep='')   
        m, radius, neigh = dim_red.compute_inner_dim(pd_struct, field)
        internal_dim[file + "_internal_dim"] = m
        internal_dim[file + "_radii_vs_nn"] = np.hstack((radius, neigh))
        if kwargs["verbose"]:
            if np.isnan(m):
                print('\t\t Could not compute internal dimension.')
            else:
                print('\t\t Dim: %.2f' %internal_dim[file + "_internal_dim"])
                
        if isinstance(kwargs["umap_dims"], str):
            if 'adapt_to_inner_dim' in kwargs["umap_dims"]:
                if count==1:
                    dim_list = list();
                dim_list.append(np.round(internal_dim[file + "_internal_dim"]+0.1))
                
    if isinstance(kwargs["umap_dims"], str):
        if 'adapt_to_inner_dim' in kwargs["umap_dims"]:
            kwargs["umap_dims"] = np.array(dim_list)
            if  kwargs["apply_same_model"]:
                kwargs["umap_dims"] = np.max(dim_list)
                
            if kwargs["verbose"]:
                print('\tAdapting Umap dimensions to those found by inner_dimensionality:' ,kwargs["umap_dims"])
                kwargs["umap_dims"] = np.max(kwargs["umap_dims"])

                print('\t\tKeeping largest dimension: ', kwargs["umap_dims"])
                
    if kwargs["display_plots"]:  
        gu.plot_internal_dim(internal_dim, kwargs["fnames"], mouse + "_"+ field)
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  '_internal_dim_'+ field))
        plt.close()
    return internal_dim    
#%%
###############################################################################
#                                   PCA                                       #
###############################################################################
from sklearn.decomposition import PCA
def _compute_pca_LT(df_dict, field, pca_field, still_dict, fail_dict, save_plot_dir,mouse, kwargs):
    models_pca = dict()
    if kwargs["display_plots"]:
        #to infer axis limits
        max_pc = np.NINF
        max_cumu_pc = np.NINF
    count = 0
    for file, pd_struct in df_dict.items():
        count += 1
        if kwargs["verbose"]:
            print('\tWorking on entry %i/' %count, '%i: ' %len(kwargs["fnames"]), file, sep='')   
        #compute model based on moving activity and reduce 
        if count==1 or not kwargs["apply_same_model"]:
            out_struct, model_pca = dim_red.dim_reduce(pd_struct, PCA(pd_struct[field][0].shape[1]), 
                                                   field,pca_field, return_model = True)
            #find number of dim to explain 80% of variance
            dim_80 = np.where(np.cumsum(model_pca.explained_variance_ratio_)>=0.80)[0][0]
        else:
            out_struct = dim_red.apply_dim_reduce_model(pd_struct, model_pca, field, pca_field)
            
        df_dict.update({file: out_struct})
        #save model
        models_pca[file] = model_pca
        #apply model to still and fail data
        if still_dict and fail_dict:
            try:
                still_dict.update({file: dim_red.apply_dim_reduce_model(still_dict[file], models_pca[file],
                                                                 field, pca_field)})
                fail_dict.update({file: dim_red.apply_dim_reduce_model(fail_dict[file], models_pca[file],
                                                                 field, pca_field)})
            except:
                pass
        if kwargs["verbose"]:
            print('\t\tExplained variance: 0: %.4f' %models_pca[file].explained_variance_ratio_[0],
                 '\n\t\tExplained variance 10: %.2f' %(100*np.sum(model_pca.explained_variance_ratio_[:10])),
                 '\n\t\tDims needed to explain 80%%: %d' %dim_80,sep='')
            
        if kwargs["display_plots"]:
            max_pc = np.max((max_pc,100*models_pca[file].explained_variance_ratio_[0]))
            max_cumu_pc = np.max((max_cumu_pc, 100*np.sum(models_pca[file].explained_variance_ratio_)))
            
    if kwargs["display_plots"]:
        #plot variance explained
        max_pc *= 1.1
        if max_cumu_pc<100:
            max_cumu_pc *=1.1
        gu.plot_PCA_variance(models_pca ,max_pc, max_cumu_pc, kwargs["pca_dims"], 'PCA: ' +pca_field)
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  pca_field + '_variance'))
        plt.close()
        #plot 3D points
        gu.plot_3D_embedding_LT(df_dict, pca_field, 'PCA: '+ pca_field) 
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  pca_field + '_3D'))
        plt.close()
        gu.plot_3D_embedding_LT_v2(df_dict, pca_field,  'PCA: '+pca_field)
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+   pca_field +'_3D_time'))
        plt.close()
        #plot trajectory
        if kwargs["plot_trajectory"]:
            gu.plot_trajectory_embedding_LT(df_dict, pca_field, 'PCA: '+ pca_field)
            plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  pca_field +'_trajectory'))
    return df_dict, still_dict, fail_dict, models_pca

###############################################################################
#                                   ISOMAP                                    #
###############################################################################
from sklearn.manifold import Isomap
from scipy.stats import pearsonr
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances

def _compute_isomap_LT(df_dict, field, iso_field, still_dict, fail_dict, save_plot_dir,mouse, kwargs):
    models_iso = dict()
    count = 0
    for file, pd_struct in df_dict.items():
        count += 1
        if kwargs["verbose"]:
            print('\tWorking on entry %i/' %count, '%i: ' %len(kwargs["fnames"]), file, sep='')  
        #check if new model or apply previous one
        if count==1 or not kwargs["apply_same_model"]:
            #compute model and reduce
            out_struct, model_iso = dim_red.dim_reduce(pd_struct, Isomap(n_neighbors = kwargs["neighbours_iso_rates"], 
                                                                     n_components = kwargs["iso_dims"]), 
                                                                       field, iso_field, return_model = True)
            if kwargs["compute_iso_resvar"] or ('adapt_to_isomap' in kwargs["umap_dims"]):
                #compute data dimensionality
                if kwargs["verbose"]:
                    print('\t\tComputing res-var dimensionality: ', sep='', end = '')  
                res_var = np.zeros((kwargs["iso_dims"], 1))
                for dim in range(kwargs["iso_dims"]):
                    D_emb = pairwise_distances(np.concatenate(out_struct[iso_field].values, axis=0)[:,:dim+1], metric = 'minkowski') 
                    
                    res_var[dim,0] = 1 -pearsonr(np.matrix.flatten(model_iso.dist_matrix_),
                                                             np.matrix.flatten(D_emb.astype('float32')))[0]**2
                models_iso[file+"_res_var"] = res_var
                kl = KneeLocator(np.linspace(0, kwargs["iso_dims"]-1, kwargs["iso_dims"]),
                                res_var[:,0], curve = "convex", direction = "decreasing")
                if kl.knee:
                    models_iso[file+"dimension"] = int(kl.knee+1)
                else:
                    models_iso[file+"dimension"] = np.nan
                if kwargs["verbose"]:
                    print( models_iso[file+"dimension"])
                    
                if isinstance(kwargs["umap_dims"], str):
                    if 'adapt_to_isomap' in kwargs["umap_dims"]:
                        if count==1:
                            dim_list = list();
                        dim_list.append(int(kl.knee+1))
        else:
            out_struct = dim_red.apply_dim_reduce_model(pd_struct, model_iso, field, iso_field)
        df_dict.update({file: out_struct})
        #save model
        models_iso[file] = model_iso       
        #apply model to still and fail data
        if still_dict and fail_dict:
            try:
                still_dict.update({file: dim_red.apply_dim_reduce_model(still_dict[file], models_iso[file], 
                                                                 field, iso_field)})
                fail_dict.update({file: dim_red.apply_dim_reduce_model(fail_dict[file], models_iso[file],
                                                                 field, iso_field)})
            except:
                pass
    if isinstance(kwargs["umap_dims"], str):
        if 'adapt_to_isomap' in kwargs["umap_dims"]:
            kwargs["umap_dims"] = np.max(np.array(dim_list))
            if kwargs["verbose"]:
                print('\tAdapting Umap dimensions to those found by Isomap')
    if kwargs["display_plots"]:
        #plot 3D points
        gu.plot_3D_embedding_LT(df_dict, iso_field, 'Isomap: '+iso_field) 
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  iso_field + '_3D'))
        
        gu.plot_3D_embedding_LT_v2(df_dict, iso_field, 'Isomap: '+iso_field) 
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  iso_field+ '_3D_time'))
        plt.close()
        #plot trajectory
        if kwargs["plot_trajectory"]:
            gu.plot_trajectory_embedding_LT(df_dict, iso_field, 'Isomap: '+iso_field)
            plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  iso_field + '_trajectory'))
    return df_dict, still_dict, fail_dict, models_iso, kwargs

###############################################################################
#                                     UMAP                                    #
###############################################################################
import umap
from umap import validation
def _compute_umap_LT(df_dict, field, umap_field, still_dict, fail_dict, neighbours, min_dist, save_plot_dir,mouse, kwargs):
    models_umap = dict()
    if not kwargs["apply_same_model"]: 
        count = 0
        for file, pd_struct in df_dict.items():
            models_umap[file] = dict()
            count += 1
            if kwargs["verbose"]:
                print('\tWorking on entry %i/' %count, '%i: ' %len(kwargs["fnames"]), file, sep='')  
            #TODO: optimize number of neighbours
            if neighbours<1:
                length = np.concatenate(pd_struct[field].values, axis=0).shape[0]
                nn = np.round(length*neighbours).astype(int)
                if kwargs["verbose"]:
                    print('\t\tInterpreting umap neighbours (%.4f) as fraction of points. Resulting in %d neighbours.'
                          %(neighbours, nn))
            else:
                nn = neighbours
            #check number of dimensions value
            if isinstance(kwargs["umap_dims"],str):
                if 'optimize_to_umap_trust' in kwargs["umap_dims"]:
                    umap_dim, num_trust =dim_red.compute_umap_trust_dim(pd_struct,field,n_neigh=nn, 
                                                                    min_dist=min_dist, verbose=kwargs["verbose"])  
                    if umap_dim<3:
                        umap_dim = 3
                    models_umap[file]["dim_num_trust"] = num_trust
            else:
                umap_dim = kwargs["umap_dims"]
            #compute model and project
            out_struct, model_umap = dim_red.dim_reduce(pd_struct, umap.UMAP(n_neighbors = nn, 
                                                                     n_components = umap_dim,
                                                                     random_state=kwargs["rand_state_umap"], 
                                                                     min_dist=min_dist), 
                                                   field, umap_field, return_model = True)
            #TODO: consider including here dim_red.study_dim_to_cells but it takes for ever
            df_dict.update({file: out_struct})
            #save model
            models_umap[file]["model"] = model_umap
            num_trust = validation.trustworthiness_vector(source=np.concatenate(pd_struct[field].values, axis=0),
                                              embedding=np.concatenate(pd_struct[umap_field].values, axis=0), max_k=30)
            models_umap[file]["num_trust"] = num_trust[1:]
            if kwargs["verbose"]:
                print('\t\tNumerical trustworthiness: ', np.nanmean(num_trust), sep='')
            #apply model to still and fail data
            if still_dict and fail_dict:
                try:
                    still_dict.update({file: dim_red.apply_dim_reduce_model(still_dict[file], models_umap[file]["model"], 
                                                                     field, umap_field)})
                    fail_dict.update({file: dim_red.apply_dim_reduce_model(fail_dict[file], models_umap[file]["model"],
                                                                     field, umap_field)})
                except:
                    pass
    else:
        #concat data of all sessions
        data_list = [np.concatenate(pd_struct[field].values, axis=0) for _, pd_struct in df_dict.items()]
        data_array = np.concatenate(data_list, axis=0)
        if kwargs["verbose"]:
            print('\tProjecting all data together to find a common model.')  
        #TODO: optimize number of neighbours
        if neighbours<1:
            nn_list = [np.round(data.shape[0]*neighbours).astype(int) for data in data_list]
            nn = np.round(np.mean(nn_list)).astype(int)
            if kwargs["verbose"]:
                print('\t\tInterpreting umap neighbours (%.4f) as fraction of points. Resulting in %d neighbours.'
                      %(neighbours, nn))
        else:
            nn = neighbours
        if isinstance(kwargs["umap_dims"],str):
            if 'optimize_to_umap_trust' in kwargs["umap_dims"]:
                umap_dim, num_trust =dim_red.compute_umap_trust_dim(data_array,n_neigh=nn, 
                                                                min_dist=min_dist, verbose=kwargs["verbose"])  
                if umap_dim<3:
                    umap_dim = 3
                models_umap["dim_num_trust"] = num_trust
        else:
            umap_dim = kwargs["umap_dims"]
        model_umap = umap.UMAP(n_neighbors = nn, n_components = umap_dim, 
                               random_state=kwargs["rand_state_umap"], min_dist=min_dist)
        model_umap.fit(data_array)
        count = 0
        for file, pd_struct in df_dict.items():
            models_umap[file] = dict()
            count += 1
            if kwargs["verbose"]:
                print('\tWorking on entry %i/' %count, '%i: ' %len(kwargs["fnames"]), file, sep='')  
            
            #compute model and project
            out_struct = dim_red.apply_dim_reduce_model(pd_struct, model_umap, field, umap_field)
            df_dict.update({file: out_struct})
            #save model
            models_umap[file]["model"] = model_umap
            num_trust = validation.trustworthiness_vector(source=np.concatenate(pd_struct[field].values, axis=0),
                                              embedding=np.concatenate(pd_struct[umap_field].values, axis=0), max_k=30)
            models_umap[file]["num_trust"] = num_trust[1:]
            if kwargs["verbose"]:
                print('\t\tNumerical trustworthiness: ', np.nanmean(num_trust), sep='')
            #apply model to still and fail data
            if still_dict and fail_dict:
                try:
                    still_dict.update({file: dim_red.apply_dim_reduce_model(still_dict[file], models_umap[file]["model"], 
                                                                     field, umap_field)})
                    fail_dict.update({file: dim_red.apply_dim_reduce_model(fail_dict[file], models_umap[file]["model"],
                                                                     field, umap_field)})
                except:
                    pass
    if kwargs["display_plots"]:
        #plot 3D points
        gu.plot_3D_embedding_LT(df_dict, umap_field, 'Umap: '+umap_field) 
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  umap_field + '_3D'))
        
        gu.plot_3D_embedding_LT_v2(df_dict, umap_field, 'Umap: '+umap_field) 
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  umap_field+ '_3D_time'))
        #plot 2D projections
        gu.plot_2D_embedding_LT(df_dict, umap_field, mouse=mouse, save = True, save_dir = save_plot_dir)
        plt.close()
        if kwargs["apply_same_model"] and len(kwargs["fnames"])==2:
            gu.plot_2D_embedding_DS(df_dict, umap_field, mouse=mouse, save = True, save_dir = save_plot_dir)
            plt.close()
        #plot trajectory
        if kwargs["plot_trajectory"]:
            gu.plot_trajectory_embedding_LT(df_dict, umap_field, 'Umap: '+umap_field)
            plt.savefig(os.path.join(save_plot_dir, mouse+'_'+ umap_field + '_trajectory'))
    return df_dict, still_dict, fail_dict, models_umap

'''
def _compute_umap_LT(df_dict, field, umap_field, still_dict, fail_dict, neighbours, min_dist, save_plot_dir,mouse, kwargs):
    models_umap = dict()
    count = 0
    for file, pd_struct in df_dict.items():
        models_umap[file] = dict()
        count += 1
        if kwargs["verbose"]:
            print('\tWorking on entry %i/' %count, '%i: ' %len(kwargs["fnames"]), file, sep='')  
        #check if apply same model
        if count==1 or not kwargs["apply_same_model"]:            #compute model and reduce 
            #check number of neighbours value
            if isinstance(neighbours, str):
                #get dimension on which to optimize nn
                if isinstance(kwargs["umap_dims"],np.ndarray):
                    umap_dim = kwargs["umap_dims"][count-1]
                elif isinstance(kwargs["umap_dims"],str):
                    if 'optimize_to_umap_trust' in kwargs["umap_dims"]:
                        umap_dim = 12
                else:
                    umap_dim = kwargs["umap_dims"]
                    
                data = np.concatenate(pd_struct[field].values, axis=0)
                nn, num_trust = __optimize_nn_umap(data, min_dist, umap_dim, kwargs)
                models_umap[file]["nn_numtrust"] = num_trust

            else:
                if neighbours<1:
                    length = np.concatenate(pd_struct[field].values, axis=0).shape[0]
                    nn = np.round(length*neighbours).astype(int)
                    if kwargs["verbose"]:
                        print('\t\tInterpreting umap neighbours (%.4f) as fraction of points. Resulting in %d neighbours.'
                              %(neighbours, nn))
                else:
                    nn = neighbours
            #check number of dimensions value
            if isinstance(kwargs["umap_dims"],str):
                if 'optimize_to_umap_trust' in kwargs["umap_dims"]:
                    umap_dim, num_trust =dim_red.compute_umap_trust_dim(pd_struct,field,n_neigh=nn, 
                                                                    min_dist=min_dist, verbose=kwargs["verbose"])  
                    if umap_dim<3:
                        umap_dim = 3
                    models_umap[file]["dim_num_trust"] = num_trust
            else:
                umap_dim = kwargs["umap_dims"]
            #compute model and project
            out_struct, model_umap = dim_red.dim_reduce(pd_struct, umap.UMAP(n_neighbors = nn, 
                                                                     n_components = umap_dim,
                                                                     random_state=kwargs["rand_state_umap"], 
                                                                     min_dist=min_dist), 
                                                   field, umap_field, return_model = True)
            #if kwargs["check_dim_to_cells_umap"]:
                #TODO: consider including here dim_red.study_dim_to_cells but it takes for ever
                #num_cells_to_include, dimen_to_number = dim_red.check_dim_to_cells_umap(copy.deepcopy(data),nn, min_dist, verbose = kwargs["verbose"])
                #models_umap[file]["num_cells_to_include"] = num_cells_to_include
                #models_umap[file]["dimen_to_number"] = dimen_to_number
                
        #if apply same model just project data
        else:
            out_struct = dim_red.apply_dim_reduce_model(pd_struct, model_umap, field, umap_field)
            
        df_dict.update({file: out_struct})
        #save model
        
        models_umap[file]["model"] = model_umap
        num_trust = validation.trustworthiness_vector(source=np.concatenate(pd_struct[field].values, axis=0),
                                          embedding=np.concatenate(pd_struct[umap_field].values, axis=0), max_k=30)
        models_umap[file]["num_trust"] = num_trust[1:]
        if kwargs["verbose"]:
            print('\t\tNumerical trustworthiness: ', np.nanmean(num_trust), sep='')
        #apply model to still and fail data
        if still_dict and fail_dict:
            try:
                still_dict.update({file: dim_red.apply_dim_reduce_model(still_dict[file], models_umap[file]["model"], 
                                                                 field, umap_field)})
                fail_dict.update({file: dim_red.apply_dim_reduce_model(fail_dict[file], models_umap[file]["model"],
                                                                 field, umap_field)})
            except:
                pass
    if kwargs["display_plots"]:
        #plot 3D points
        gu.plot_3D_embedding_LT(df_dict, umap_field, 'Umap: '+umap_field) 
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  umap_field + '_3D'))
        
        gu.plot_3D_embedding_LT_v2(df_dict, umap_field, 'Umap: '+umap_field) 
        plt.savefig(os.path.join(save_plot_dir, mouse+'_'+  umap_field+ '_3D_time'))
        #plot 2D projections
        gu.plot_2D_embedding_LT(df_dict, umap_field, mouse=mouse, save = True, save_dir = save_plot_dir)
        plt.close()
        if kwargs["apply_same_model"] and len(kwargs["fnames"])==2:
            gu.plot_2D_embedding_DS(df_dict, umap_field, mouse=mouse, save = True, save_dir = save_plot_dir)
            plt.close()
        #plot trajectory
        if kwargs["plot_trajectory"]:
            gu.plot_trajectory_embedding_LT(df_dict, umap_field, 'Umap: '+umap_field)
            plt.savefig(os.path.join(save_plot_dir, mouse+'_'+ umap_field + '_trajectory'))
    return df_dict, still_dict, fail_dict, models_umap

def __optimize_nn_umap(data, min_dist, umap_dim, kwargs):
    if kwargs["verbose"]:
        print('\t\tFinding best number of neighbours combination: ', sep='')
    n_neighbours = np.array([0.1,0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,1,2,3,4,5,10])/100
    length = data.shape[0]
    nn_real = np.round(n_neighbours*length).astype(int)
    nn_real[nn_real<2] = 2
    nn_real = np.unique(nn_real)
    k_trust = 30
    num_trust = np.zeros((n_neighbours.shape[0],k_trust))*np.nan()
    for nn_index, nn in enumerate(nn_real):
        emb = umap.UMAP(n_neighbors = nn, n_components = umap_dim, min_dist=min_dist).fit_transform(data)
        num_trust[nn_index, :] = validation.trustworthiness_vector(source=data,
                                                  embedding=emb ,max_k=k_trust)[1:]
    idx_max = np.argmax(np.nanmean(num_trust,axis=1))
    if kwargs["verbose"]:
        print('\t\t\tBest combination is %d (%.2f%%) neighbours' %(nn_real[idx_max], n_neighbours[idx_max]*100)) 
    nn = nn_real[idx_max]
    return nn, num_trust
'''

'''
def __optimize_dim_umap(data,nn, min_dist, max_dim = 10, verbose=True):
    dims = 1
    k_trust = np.round(0.5*data.shape[0]/100).astype(int)
    if verbose:
        print('\t\tFinding dimensionality through trustworthiness (max_k=%d): ' %k_trust, sep='')
    num_trust = np.zeros((max_dim,1))*np.nan
    dims= np.min([max_dim,data.shape[1]])
    for dim in range(dims):
        if verbose:
            print('\t\t\tChecking dimension %d: ' %(dim+1), sep='', end = '')
        emb = umap.UMAP(n_neighbors = nn, n_components = dim+1, min_dist=min_dist).fit_transform(data)
        #breakpoint()
        num_trust[dim,0] = validation.trustworthiness_vector(data, emb ,k_trust)[-1]
        if dim>0:
            red_error = abs(num_trust[dim,0]-num_trust[dim-1,0])
            if verbose:
                print('Error improvement of %.4f' %red_error, sep='')
        else:
            if verbose:
                print('')
    x_space = np.linspace(1,dim+1, dim+1)        
    kl = KneeLocator(x_space, num_trust[:dim+1,0], curve = "concave", direction = "increasing")
    if kl.knee:
        umap_dim = kl.knee
    else:
        umap_dim = None
    if verbose:
        print('\t\t\tFinal dimension: %d - Final error of %.4f' %(umap_dim, 1-np.nanmean(num_trust[umap_dim-1])), sep='')
    return umap_dim, num_trust


def __optimize_dim_umap(data,nn, min_dist, verbose=True):
    dims = 1
    k_trust = np.round(0.5*data.shape[0]/100).astype(int)
    if verbose:
        print('\t\tFinding dimensionality through trustworthiness (max_k=%d): ' %k_trust, sep='')
    num_trust = []
    stp = 0
    while stp == 0:
        dims +=1
        if verbose:
            print('\t\t\tChecking dimension %d: ' %dims, sep='', end = '')
        emb = umap.UMAP(n_neighbors = nn, n_components = dims, min_dist=min_dist).fit_transform(data)
        #breakpoint()
        trust = validation.trustworthiness_vector(data, emb ,k_trust)[1:]
        num_trust.append(trust)
        if dims>2:
            red_error = abs(np.nanmean(num_trust[-2])-np.nanmean(num_trust[-1]))
            if verbose:
                print('Error improvement of %.4f' %red_error, sep='')
            if red_error<5e-4:
                stp = 1
                umap_dim = dims-1;
            elif dims>data.shape[1]:
                stp = 1
                umap_dim = dims
        else:
            print('')
    if verbose:
        print('\t\t\tFinal dimension: %d - Final error of %.4f' %(umap_dim, 1-np.nanmean(num_trust[-2])), sep='')
    return umap_dim, num_trust

import random
def __check_dim_to_cells_umap(data, nn, min_dist, verbose = True):
    n_splits = 5
    num_cells_to_include =np.unique(np.logspace(np.log10(5), np.log10(data.shape[1]),15,dtype=int))
    dimen_to_number = np.zeros((num_cells_to_include.shape[0], n_splits))
    num_trust = np.zeros((num_cells_to_include.shape[0],10,n_splits))
    if verbose:
        print('Checking number of cells idx X/X',end='', sep='')
        pre_del = '\b\b\b'
    for num_cells_idx, num_cells_val in enumerate(num_cells_to_include):
        if verbose:
            print(pre_del,"%d/%d" %(num_cells_idx+1,num_cells_to_include.shape[0]), sep = '', end='')
            pre_del = (len(str(num_cells_idx+1))+len(str(num_cells_to_include.shape[0]))+1)*'\b'
        for n_idx in range(n_splits):
            cells_picked = random.sample(range(0, data.shape[1]), num_cells_val)
            data_new = data[:,cells_picked]
            dimen_to_number[num_cells_idx, n_idx], num_trust[num_cells_idx, :, n_idx] = df.compute_umap_trust_dim(data_new,n_neigh=nn, 
                                                                                                      min_dist=min_dist, verbose=False)  
    return num_cells_to_include, dimen_to_number
            

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score, median_absolute_error
def _evaluate_umap(og_data,neighbours, min_dist, max_dim=4, n_splits = 5,random_state = None):
    print('Evaluating recontruction of UMAP as a function of dimensions:')
    #intialize kfold function 
    kfold_index = -1
    kf = KFold(n_splits= n_splits,random_state = random_state)
    #allocate save variables
    corr_vals = np.zeros((n_splits, max_dim-1))
    trust_vals = np.zeros((n_splits, max_dim-1))
    maerr_vals = np.zeros((n_splits, max_dim-1))
    r2s_vals = np.zeros((n_splits, max_dim-1))
    test_data_memory = np.zeros((max_dim+1, n_splits, 100, og_data.shape[1]))
    #start computing each fold
    for train_index, test_index in kf.split(og_data):
        kfold_index +=1
        print("\tKfold: %d/%d " %(kfold_index+1,n_splits), sep ='', end='')
        #split into train and test data
        train_data, test_data = og_data[train_index], og_data[test_index]
        #limit test data to 5s 
        test_data = test_data[:100,:]
        print(' - dim: X/X',end='', sep='')
        pre_del = '\b\b\b'
        test_data_memory[0,kfold_index,:,:] = test_data
        for dims in range(2,max_dim+1):
            print(pre_del,"%d/%d" %(dims,max_dim), sep = '', end='')
            pre_del = (len(str(dims))+len(str(max_dim))+1)*'\b'
            #compute UMAP on train data
            model = umap.UMAP(n_neighbors = neighbours, n_components = dims, min_dist=min_dist).fit(train_data)
            #project test data
            emb_test_data = model.transform(test_data)
            #resconstruct test data from projection
            rec_test_data = model.inverse_transform(emb_test_data)
            #compute metrics of reconstruction quality
            corr_vals[kfold_index,dims-2] = np.corrcoef(np.ndarray.flatten(test_data), 
                                                np.ndarray.flatten(rec_test_data))[1,0]
            num_trust = validation.trustworthiness_vector(source=test_data,
                                              embedding=emb_test_data,
                                              max_k=30)
            trust_vals[kfold_index, dims-2] = np.nanmean(num_trust[1:])
            maerr_vals[kfold_index, dims-2] = median_absolute_error(np.ndarray.flatten(test_data),
                                                                   np.ndarray.flatten(rec_test_data))
            r2s_vals[kfold_index, dims-2] = r2_score(np.ndarray.flatten(test_data),
                                                                   np.ndarray.flatten(rec_test_data))
            #save reconstructed data
            test_data_memory[dims-1,kfold_index,:,:] = rec_test_data
        
    return corr_vals, trust_vals, maerr_vals, r2s_vals, test_data_memory
'''   
    
    
    