import sys, os, copy, pickle, timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neural_manifold.general_utils as gu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from neural_manifold import dimensionality_reduction as dim_red
import seaborn as sns
from neural_manifold.dimensionality_reduction import validation as dim_validation 
import umap
from kneed import KneeLocator
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from structure_index import compute_structure_index, draw_graph
from neural_manifold import decoders as dec
from scipy.stats import pearsonr
import random
import skdim

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data


import time
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


#__________________________________________________________________________
#|                                                                        |#
#|                                  START                                 |#
#|________________________________________________________________________|#

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig2/ncells/'

min_cells = 10
max_cells = 300
n_steps = 8
n_splits = 10
max_dim = 10


for mouse in miceList:
    tic()
    print(f"Working on mouse {mouse}: ",  end = '')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)

    results = {'params': {
        'min_cells' : min_cells,
        'max_cells' : max_cells,
        'n_steps' : n_steps,
        'n_splits' : max_dim
    }}

    umap_embds = {}

    #signal
    signal = np.concatenate(pd_mouse['clean_traces'].values, axis = 0)
    pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pd_mouse['vel'].values, axis=0))
    trial = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))

    num_cells = signal.shape[1]
    cell_list = np.unique(np.logspace(np.log10(min_cells), np.log10(max_cells),n_steps,dtype=int))
    results['params']["og_num_cells"] = cell_list
    cell_list = cell_list[cell_list<=num_cells]
    results['params']["cell_list"] = cell_list
    cells_picked_list = np.zeros((n_steps,n_splits,num_cells)).astype(bool)


    inner_dim = {
        'abids': np.zeros((n_steps, n_splits))*np.nan,
        # 'mom': np.zeros((n_steps, n_splits))*np.nan,
        # 'tle': np.zeros((n_steps, n_splits))*np.nan,
        'params': {
            'signalName': 'clean_traces',
            'nNeigh': 30,
            'verbose': False
        }
    }

    # umap_dim = {
    #     'trustNum': np.zeros((n_steps, n_splits,max_dim))*np.nan,
    #     'contNum': np.zeros((n_steps, n_splits,max_dim))*np.nan,
    #     'trustDim': np.zeros((n_steps, n_splits))*np.nan,
    #     'contDim': np.zeros((n_steps, n_splits))*np.nan,
    #     'hmeanDim': np.zeros((n_steps, n_splits))*np.nan,
    #     'params': {
    #         'maxDim':10,
    #         'nNeigh': 120,
    #         'minDist': 0.1,
    #         'nnDim': 30,
    #         'signalName': 'clean_traces'
    #     }
    # }

    si_temp = {
        'pos': {
            'sI': np.zeros((n_steps, n_splits))*np.nan,
            'binLabel': {},
            'overlapMat': {},
            },
        'dir_mat': {
            'sI': np.zeros((n_steps, n_splits))*np.nan,
            'binLabel': {},
            'overlapMat': {},
        },
        # 'time': {
        #     'sI': np.zeros((n_steps, n_splits))*np.nan,
        #     'binLabel': {},
        #     'overlapMat': {},
        # },
        'vel': {
            'sI': np.zeros((n_steps, n_splits))*np.nan,
            'binLabel': {},
            'overlapMat': {},
        }
    }

    SI_dict = {
        'clean_traces': copy.deepcopy(si_temp),
        'umap': copy.deepcopy(si_temp),
        'params': {
            'nNeigh_perc': 0.005,
            'numShuffles': 0
        }
    }
    SI_dict['params']['nNeigh'] = np.round(SI_dict['params']['nNeigh_perc']*signal.shape[0]).astype(int)
    dec_params = {
        'x_base_signal': 'clean_traces',
        'y_signal_list': ['posx', 'vel'],
        'verbose': False,
        'trial_signal': 'index_mat',
        'nn': 120,
        'min_dist':0.1,
        'n_splits': 5,
        'n_dims': 3,
        'emb_list': ['umap'],
        'decoder_list': ['xgb']
    }   

    dec_dict = {
        'clean_traces': np.zeros((n_steps, n_splits, dec_params['n_splits'], len(dec_params['y_signal_list']),2))*np.nan,
        'umap': np.zeros((n_steps, n_splits, dec_params['n_splits'], len(dec_params['y_signal_list']),2))*np.nan,
        'params': dec_params
    }

    for ncell_idx, ncell_val in enumerate(cell_list):
        print(f"\nChecking number of cells {ncell_val} ({ncell_idx+1}/{n_steps}):")
        print("\tIteration X/X", sep= '', end = '')
        pre_del = '\b\b\b'
        for split_idx in range(n_splits):
            print(pre_del, f"{split_idx+1}/{n_splits}", sep = '', end = '')
            pre_del = (len(str(split_idx+1))+len(str(n_splits))+1)*"\b"

            cells_picked = random.sample(list(np.arange(num_cells).astype(int)), ncell_val)
            cells_picked_list[ncell_idx, split_idx, cells_picked] = True 
            it_signal = copy.deepcopy(signal[:, cells_picked])

            #compute inner dim
            inner_dim['abids'][ncell_idx, split_idx] = np.nanmean(dim_red.compute_abids(it_signal, inner_dim['params']['nNeigh'],verbose=False))
            # inner_dim['mom'][ncell_idx, split_idx] = skdim.id.MOM().fit_transform(it_signal,n_neighbors = inner_dim['params']['nNeigh'])
            # try:
            #     inner_dim['tle'][ncell_idx, split_idx] = skdim.id.TLE().fit_transform(it_signal,n_neighbors = inner_dim['params']['nNeigh'])
            # except:
            #     inner_dim['tle'][ncell_idx, split_idx] = np.nan

            #compute umap dim
            # rank_idxs = dim_validation.compute_rank_indices(it_signal)

            # for dim in range(umap_dim['params']['maxDim']):
            #     emb_space = np.arange(dim+1)
            #     model = umap.UMAP(n_neighbors = umap_dim['params']['nNeigh'], n_components =dim+1, min_dist=umap_dim['params']['minDist'])
            #     emb = model.fit_transform(it_signal)
            #     if dim==2:
            #         it_umap = copy.deepcopy(emb)
            #     #1. Compute trustworthiness
            #     temp = dim_validation.trustworthiness_vector(it_signal, emb, umap_dim['params']['nnDim'], indices_source = rank_idxs)
            #     umap_dim['trustNum'][ncell_idx,split_idx][dim] = temp[-1]

            #     #2. Compute continuity
            #     temp = dim_validation.continuity_vector(it_signal, emb ,umap_dim['params']['nnDim'])
            #     umap_dim['contNum'][ncell_idx,split_idx][dim] = temp[-1]

            # dimSpace = np.linspace(1,umap_dim['params']['maxDim'], umap_dim['params']['maxDim']).astype(int)   
            model = umap.UMAP(n_neighbors = 120, n_components =3, min_dist=0.1)
            it_umap = model.fit_transform(it_signal)

            # kl = KneeLocator(dimSpace, umap_dim['trustNum'][ncell_idx,split_idx], curve = "concave", direction = "increasing")
            # if kl.knee:
            #     umap_dim['trustDim'][ncell_idx, split_idx] = kl.knee
            # else:
            #     umap_dim['trustDim'][ncell_idx, split_idx] = np.nan
            # kl = KneeLocator(dimSpace, umap_dim['contNum'][ncell_idx,split_idx], curve = "concave", direction = "increasing")
            # if kl.knee:
            #     umap_dim['contDim'][ncell_idx, split_idx] = kl.knee
            # else:
            #     umap_dim['contDim'][ncell_idx, split_idx] = np.nan
            # umap_dim['hmeanDim'][ncell_idx, split_idx] = (2*umap_dim['trustDim'][ncell_idx, split_idx]*umap_dim['contDim'][ncell_idx, split_idx])/(umap_dim['trustDim'][ncell_idx, split_idx]+umap_dim['contDim'][ncell_idx, split_idx])


            #compute SI
            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, pos[:,0], n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['pos']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['clean_traces']['pos']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['clean_traces']['pos']['overlapMat'][ncell_idx, split_idx] = toverlapMat


            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, dir_mat, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['dir_mat']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['clean_traces']['dir_mat']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['clean_traces']['dir_mat']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            # tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, trial, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            # SI_dict['clean_traces']['time']['sI'][ncell_idx, split_idx] = tSI
            # SI_dict['clean_traces']['time']['binLabel'][ncell_idx, split_idx] = tbinLabel
            # SI_dict['clean_traces']['time']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, vel, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['vel']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['clean_traces']['vel']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['clean_traces']['vel']['overlapMat'][ncell_idx, split_idx] = toverlapMat


            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, pos[:,0], n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['pos']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['umap']['pos']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['umap']['pos']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, dir_mat, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['dir_mat']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['umap']['dir_mat']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['umap']['dir_mat']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            # tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, trial, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            # SI_dict['umap']['time']['sI'][ncell_idx, split_idx] = tSI
            # SI_dict['umap']['time']['binLabel'][ncell_idx, split_idx] = tbinLabel
            # SI_dict['umap']['time']['overlapMat'][ncell_idx, split_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, vel, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['vel']['sI'][ncell_idx, split_idx] = tSI
            SI_dict['umap']['vel']['binLabel'][ncell_idx, split_idx] = tbinLabel
            SI_dict['umap']['vel']['overlapMat'][ncell_idx, split_idx] = toverlapMat


            #compute decoders
            new_pd_mouse = copy.deepcopy(pd_mouse)
            for idx in range(pd_mouse.shape[0]):
                new_pd_mouse['clean_traces'][idx] = pd_mouse['clean_traces'][idx][:, cells_picked]

            dec_R2s, _ = dec.decoders_1D(pd_object = copy.deepcopy(new_pd_mouse), **dec_params)
            dec_dict['clean_traces'][ncell_idx, split_idx,:, :,:] = dec_R2s['base_signal']['xgb']
            dec_dict['umap'][ncell_idx, split_idx,:, :,:] = dec_R2s['umap']['xgb']

            umap_embds[ncell_idx, split_idx] = it_umap


        results['inner_dim'] = copy.deepcopy(inner_dim)
        # results['umap_dim'] = copy.deepcopy(umap_dim)
        results['SI_dict'] = copy.deepcopy(SI_dict)
        results['dec_dict'] = copy.deepcopy(dec_dict)
        results['umap_embds'] = copy.deepcopy(umap_embds)
        results['params']['cells_picked_list'] = copy.deepcopy(cells_picked_list)
        save_file = open(os.path.join(save_dir, f'{mouse}_ncells_results_dict.pkl'), "wb")
        pickle.dump(results, save_file)
        save_file.close()
    print('\n')
    toc()

###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################            PLOT            #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################

miceList = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir = '/home/julio/Documents/SP_project/Fig2/ncells/'


umap_dec = list()
og_dec = list()
inner_dim_abids = list()
inner_dim_mom = list()
inner_dim_tle = list()

umap_dim_trust = list()
umap_dim_cont = list()
umap_dim_hmean = list()

umap_SI_pos = list()
umap_SI_dirmat = list()
umap_SI_time = list()
umap_SI_vel = list()

og_SI_pos = list()
og_SI_dirmat = list()
og_SI_time = list()
og_SI_vel = list()

mouse_name_list = list()
layers_list = list()
strain_list = list()
for mouse in miceList:

    ncells_dict = load_pickle(data_dir, f'{mouse}_ncells_results_dict.pkl')
    cell_list = ncells_dict['params']['cell_list']

    umap_dec.append(np.nanmean(ncells_dict['dec_dict']['umap'][:,:,:,0,0],axis=(1,2)))
    og_dec.append(np.nanmean(ncells_dict['dec_dict']['clean_traces'][:,:,:,0,0],axis=(1,2)))

    inner_dim_abids.append(np.nanmean(ncells_dict['inner_dim']['abids'],axis=1))
    # inner_dim_tle.append(np.nanmean(ncells_dict['inner_dim']['tle'],axis=1))
    # inner_dim_mom.append(np.nanmean(ncells_dict['inner_dim']['mom'],axis=1))

    # umap_dim_trust.append(np.nanmean(ncells_dict['umap_dim']['trustDim'],axis=1))
    # umap_dim_cont.append(np.nanmean(ncells_dict['umap_dim']['contDim'],axis=1))
    # umap_dim_hmean.append(np.nanmean(ncells_dict['umap_dim']['hmeanDim'],axis=1))

    umap_SI_pos.append(np.nanmean(ncells_dict['SI_dict']['umap']['pos']['sI'][:,:],axis=1))
    umap_SI_dirmat.append(np.nanmean(ncells_dict['SI_dict']['umap']['dir_mat']['sI'][:,:],axis=1))
    # umap_SI_time.append(np.nanmean(ncells_dict['SI_dict']['umap']['time']['sI'][:,:],axis=1))
    umap_SI_vel.append(np.nanmean(ncells_dict['SI_dict']['umap']['vel']['sI'][:,:],axis=1))

    og_SI_pos.append(np.nanmean(ncells_dict['SI_dict']['clean_traces']['pos']['sI'][:,:],axis=1))
    og_SI_dirmat.append(np.nanmean(ncells_dict['SI_dict']['clean_traces']['dir_mat']['sI'][:,:],axis=1))
    # og_SI_time.append(np.nanmean(ncells_dict['SI_dict']['clean_traces']['time']['sI'][:,:],axis=1))
    og_SI_vel.append(np.nanmean(ncells_dict['SI_dict']['clean_traces']['vel']['sI'][:,:],axis=1))


umap_dec = np.stack(umap_dec,axis=1)
og_dec = np.stack(og_dec,axis=1)

inner_dim_abids = np.stack(inner_dim_abids,axis=1)
# inner_dim_tle = np.stack(inner_dim_tle,axis=1)
# inner_dim_mom = np.stack(inner_dim_mom,axis=1)

# umap_dim_trust = np.stack(umap_dim_trust,axis=1)
# umap_dim_cont = np.stack(umap_dim_cont,axis=1)
# umap_dim_hmean = np.stack(umap_dim_hmean,axis=1)

umap_SI_pos = np.stack(umap_SI_pos,axis=1)
umap_SI_dirmat = np.stack(umap_SI_dirmat,axis=1)
# umap_SI_time = np.stack(umap_SI_time,axis=1)
umap_SI_vel = np.stack(umap_SI_vel,axis=1)

og_SI_pos = np.stack(og_SI_pos,axis=1)
og_SI_dirmat = np.stack(og_SI_dirmat,axis=1)
# og_SI_time = np.stack(og_SI_time,axis=1)
og_SI_vel = np.stack(og_SI_vel,axis=1)

palette_deepsup = ["#cc9900ff", "#9900ffff"]

#DECODERS 
plt.figure()
ax = plt.subplot(1,1,1)
m = np.nanmean(umap_dec[:,:8],axis=1)
sd = np.nanstd(umap_dec[:,:8],axis=1)
ax.plot(cell_list, m, label='deep', color=palette_deepsup[0])
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=palette_deepsup[0])
m = np.nanmean(umap_dec[:,8:],axis=1)
sd = np.nanstd(umap_dec[:,8:],axis=1)
ax.plot(cell_list, m, label='sup', color=palette_deepsup[1])
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=palette_deepsup[1])
ax.set_xlabel('Num cells')
ax.set_ylabel('Med Abs Error (cm)')
ax.set_ylim([0, 18])
ax.legend()
ax.set_title('Umap')
plt.savefig(os.path.join(data_dir,'nCells_decoders_umap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_decoders_umap.png'), dpi = 400,bbox_inches="tight")


#DECODERS 
plt.figure()
ax = plt.subplot(1,1,1)
m = np.nanmean(og_dec[:,:8],axis=1)
sd = np.nanstd(og_dec[:,:8],axis=1)
ax.plot(cell_list, m, label='deep', color=palette_deepsup[0])
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=palette_deepsup[0])
m = np.nanmean(og_dec[:,8:],axis=1)
sd = np.nanstd(og_dec[:,8:],axis=1)
ax.plot(cell_list, m, label='sup', color=palette_deepsup[1])
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=palette_deepsup[1])
ax.set_xlabel('Num cells')
ax.set_ylabel('Med Abs Error (cm)')
ax.set_ylim([0, 18])
ax.legend()
ax.set_title('traces')
plt.savefig(os.path.join(data_dir,'nCells_decoders_traces.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_decoders_traces.png'), dpi = 400,bbox_inches="tight")



#INNER DIM
plt.figure()
ax = plt.subplot(1,1,1)
m = np.nanmean(inner_dim_abids,axis=1)
sd = np.nanstd(inner_dim_abids,axis=1)
ax.plot(cell_list, m, label='abids')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
# m = np.nanmean(inner_dim_tle,axis=1)
# sd = np.nanstd(inner_dim_tle,axis=1)
# ax.plot(cell_list, m, label='tle')
# ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
# m = np.nanmean(inner_dim_mom,axis=1)
# sd = np.nanstd(inner_dim_mom,axis=1)
# ax.plot(cell_list, m, label='mom')
# ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
ax.set_xlabel('Num cells')
ax.set_ylabel('inner dim')
ax.set_ylim([-0.1, 4.1])
ax.legend()
plt.savefig(os.path.join(data_dir,'nCells_inner_dim.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_inner_dim.png'), dpi = 400,bbox_inches="tight")

# #UMAP DIM
# plt.figure()
# ax = plt.subplot(1,1,1)
# m = np.nanmean(umap_dim_trust,axis=1)
# sd = np.nanstd(umap_dim_trust,axis=1)
# ax.plot(cell_list, m, label='trust')
# ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
# m = np.nanmean(umap_dim_cont,axis=1)
# sd = np.nanstd(umap_dim_cont,axis=1)
# ax.plot(cell_list, m, label='cont')
# ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
# m = np.nanmean(umap_dim_hmean,axis=1)
# sd = np.nanstd(umap_dim_hmean,axis=1)
# ax.plot(cell_list, m, label='hmean')
# ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
# ax.set_xlabel('Num cells')
# ax.set_ylabel('umap dim')
# ax.set_ylim([-0.1, 4.1])
# ax.legend()
# plt.savefig(os.path.join(data_dir,'nCells_umap_dim.svg'), dpi = 400,bbox_inches="tight")
# plt.savefig(os.path.join(data_dir,'nCells_umap_dim.png'), dpi = 400,bbox_inches="tight")

#UMAP SI
plt.figure()
ax = plt.subplot(1,1,1)
m = np.nanmean(umap_SI_pos,axis=1)
sd = np.nanstd(umap_SI_pos,axis=1)
ax.plot(cell_list, m, label='pos')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(umap_SI_dirmat,axis=1)
sd = np.nanstd(umap_SI_dirmat,axis=1)
ax.plot(cell_list, m, label='direction')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
# m = np.nanmean(umap_SI_time,axis=1)
# sd = np.nanstd(umap_SI_time,axis=1)
# ax.plot(cell_list, m, label='time')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(umap_SI_vel,axis=1)
sd = np.nanstd(umap_SI_vel,axis=1)
ax.plot(cell_list, m, label='vel')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
ax.set_xlabel('Num cells')
ax.set_ylabel('SI umap')
ax.set_ylim([-0.05, 1.05])
ax.legend()
plt.savefig(os.path.join(data_dir,'nCells_umap_SI.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_umap_SI.png'), dpi = 400,bbox_inches="tight")

#OG SI
plt.figure()
ax = plt.subplot(1,1,1)
m = np.nanmean(og_SI_pos,axis=1)
sd = np.nanstd(og_SI_pos,axis=1)
ax.plot(cell_list, m, label='pos')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(og_SI_dirmat,axis=1)
sd = np.nanstd(og_SI_dirmat,axis=1)
ax.plot(cell_list, m, label='direction')
# ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
# m = np.nanmean(og_SI_time,axis=1)
# sd = np.nanstd(og_SI_time,axis=1)
# ax.plot(cell_list, m, label='time')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(og_SI_vel,axis=1)
sd = np.nanstd(og_SI_vel,axis=1)
ax.plot(cell_list, m, label='vel')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
ax.set_xlabel('Num cells')
ax.set_ylabel('SI og')
ax.set_ylim([-0.05, 1.05])
ax.legend()
plt.savefig(os.path.join(data_dir,'nCells_og_SI.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_og_SI.png'), dpi = 400,bbox_inches="tight")

#BOTH SI
plt.figure()
ax = plt.subplot(1,1,1)
m = np.nanmean(og_SI_pos,axis=1)
sd = np.nanstd(og_SI_pos,axis=1)
ax.plot(cell_list, m, label='pos-og')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(og_SI_dirmat,axis=1)
sd = np.nanstd(og_SI_dirmat,axis=1)
ax.plot(cell_list, m, label='direction-og')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(og_SI_time,axis=1)
sd = np.nanstd(og_SI_time,axis=1)
ax.plot(cell_list, m, label='time-og')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(og_SI_vel,axis=1)
sd = np.nanstd(og_SI_vel,axis=1)
ax.plot(cell_list, m, label='vel-og')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(umap_SI_pos,axis=1)
sd = np.nanstd(umap_SI_pos,axis=1)
ax.plot(cell_list, m, linestyle='--', label='pos-umap')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(umap_SI_dirmat,axis=1)
sd = np.nanstd(umap_SI_dirmat,axis=1)
ax.plot(cell_list, m, linestyle='--', label='direction-umap')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(umap_SI_time,axis=1)
sd = np.nanstd(umap_SI_time,axis=1)
ax.plot(cell_list, m, linestyle='--', label='time-umap')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
m = np.nanmean(umap_SI_vel,axis=1)
sd = np.nanstd(umap_SI_vel,axis=1)
ax.plot(cell_list, m, linestyle='--', label='vel-umap')
ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
ax.set_xlabel('Num cells')
ax.set_ylabel('SI')
ax.set_ylim([-0.05, 1.05])
ax.legend()
plt.savefig(os.path.join(data_dir,'nCells_both_SI.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'nCells_both_SI.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                                 UMAP EMB                               |#
#|________________________________________________________________________|#
miceList = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
data_dir =  '/home/julio/Documents/SP_project/Fig1/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig1/ncells/'
ncells_dict = load_pickle(save_dir, 'ncells_results_dict.pkl')

save_dir = '/home/julio/Documents/SP_project/Fig1/ncells/emb_examples/'
for mouse in miceList:
    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
    pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
    trial = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))
    time = np.arange(pos.shape[0])
    dir_color = np.zeros((dir_mat.shape[0],3))
    for point in range(dir_mat.shape[0]):
        if dir_mat[point]==0:
            dir_color[point] = [14/255,14/255,143/255]
        elif dir_mat[point]==1:
            dir_color[point] = [12/255,136/255,249/255]
        else:
            dir_color[point] = [17/255,219/255,224/255]
    for ncell_idx in range(8):
        for split_idx in range(10):
            umap_emb = copy.deepcopy(ncells_dict[mouse]['umap_embds'][(ncell_idx,split_idx)])

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1, projection = '3d')
            b = ax.scatter(*umap_emb[:,:3].T, color = dir_color,s = 20)
            # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
            ax.view_init(45, -45)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_zlabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,:2].T, color = dir_color,s = 20)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir_xy.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir_xy.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,(0,2)].T, color = dir_color,s = 20)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir_xz.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_dir_xz.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1, projection = '3d')
            b = ax.scatter(*umap_emb[:,:3].T, c = pos[:,0],s = 20, cmap = 'inferno', vmin= 0, vmax = 120)
            # cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
            ax.view_init(45, -45)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_zlabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,:2].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos_xy.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos_xy.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,(0,2)].T, c = pos[:,0], cmap = 'magma',s = 20, vmin= 0, vmax = 120)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos_xz.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_pos_xz.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1, projection = '3d')
            b = ax.scatter(*umap_emb[:,:3].T, c = time,s = 20, cmap = 'YlGn_r', vmax = 13000)
            ax.view_init(45, -45)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_zlabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,:2].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time_xy.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time_xy.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            b = ax.scatter(*umap_emb[:,(0,2)].T, c = time, cmap = 'YlGn_r',s = 20, vmax = 13000)
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 3', labelpad = -8)
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time_xz.svg'), dpi = 400,bbox_inches="tight")
            plt.savefig(os.path.join(save_dir,f'{mouse}_umap_emb_{ncell_idx}_{split_idx}_time_xz.png'), dpi = 400,bbox_inches="tight")
            plt.close(fig)

#__________________________________________________________________________
#|                                                                        |#
#|                          COMPUTE INFORMATION                           |#
#|________________________________________________________________________|#
# def compute_information(x,y):

#     nbins = 20
#     #Create grid along each dimensions
#     min_x = np.percentile(x,1)
#     max_x = np.percentile(x,99)
#     obs_length = max_x - min_x

#     bin_width = np.round(obs_length/nbins,4)
#     mapAxis = np.linspace(min_x, max_x, nbins+1)[:-1].reshape(-1,1);

#     x_pdf = np.zeros((len(mapAxis),))
#     y_pdf = np.zeros((len(mapAxis),y.shape[1]))

#     for sample in range(x.shape[0]):
#         entry_idx = np.where(x[sample]>=mapAxis)
#         if np.any(entry_idx):
#             entry_idx = entry_idx[0][-1]
#         else:
#             entry_idx = 0
        
#         x_pdf[entry_idx] += 1
#         y_pdf[entry_idx] += (1/y[entry_idx])* \
#                                    (y[sample,:]-y[entry_idx]) #online average
#     x_pdf = x_pdf/np.sum(x_pdf) #normalize probability density function


#     mean_y_pdf = np.nanmean(y_pdf,axis=0)
#     mean_y_pdf = np.tile(mean_y_pdf.reshape(1,-1),(len(mapAxis),1))

#     norm_y_pdf = np.divide(y_pdf, mean_y_pdf) #element wise division (r(x,y)/mean(r))
#     norm_y_pdf[norm_y_pdf==0] = np.nan #set no activity to 0 to avoid log2(0)
#     tile_x_pdf = np.tile(x_pdf.reshape(-1,1), (1,y.shape[1]))
    
#     return np.nansum(np.multiply(tile_x_pdf, np.multiply(norm_y_pdf, 
#                   np.log2(norm_y_pdf))), axis=0)

from sklearn.feature_selection import mutual_info_regression

miceList = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
data_dir =  '/home/julio/Documents/SP_project/Fig1/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig1/ncells/'
mi_scores = {}
for mouse in miceList:

    mi_scores[mouse] = {
        'label_order': ['posx','posy','dir','vel','time']
    }
    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
    #signal
    signal = np.concatenate(pd_mouse['clean_traces'].values, axis = 0)
    pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
    dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
    vel = copy.deepcopy(np.concatenate(pd_mouse['vel'].values, axis=0)).reshape(-1,1)
    trial = copy.deepcopy(np.concatenate(pd_mouse['index_mat'].values, axis=0))
    time = np.arange(pos.shape[0]).reshape(-1,1)
    x = np.concatenate((pos,dir_mat, vel, time),axis=1)
    mi_regression = np.zeros((x.shape[1], signal.shape[1]))*np.nan
    for cell in range(signal.shape[1]):
        mi_regression[:, cell] = mutual_info_regression(x, signal[:,cell], n_neighbors = 50, random_state = 16)
    mi_scores[mouse]['mi_scores'] = copy.deepcopy(mi_regression)
    saveFile = open(os.path.join(save_dir, 'mi_scores_dict.pkl'), "wb")
    pickle.dump(mi_scores, saveFile)
    saveFile.close()

#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT INFORMATION                            |#
#|________________________________________________________________________|#

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
save_dir = '/home/julio/Documents/SP_project/Fig1/ncells/'
mi_scores_dict = load_pickle(save_dir, 'mi_scores_dict.pkl')

mi_scores = list()
for mouse in miceList:
    mi_scores.append(mi_scores_dict[mouse]['mi_scores'])

mi_scores = np.hstack(mi_scores)

pd_mi_scores = pd.DataFrame(data={'posx': mi_scores[0,:],
                     'dir': mi_scores[2,:],
                     'vel': mi_scores[3,:],
                     'time': mi_scores[4,:]}) 

plt.figure(figsize=(8,10))
labels = list(pd_mi_scores.columns)
for idx, label_name in enumerate(labels):
    ax = plt.subplot(len(labels),1,idx+1)
    sns.kdeplot(pd_mi_scores, x=label_name, ax = ax, fill=True)
    ax.set_xlim([-0.05, 0.75])
    ax.set_ylim([0,16])
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'mutual_info_kdes.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_kdes.png'), dpi = 400,bbox_inches="tight")

plt.figure(figsize=(8,8))
sns.kdeplot(pd_mi_scores, x='posx', y='dir',fill=True, alpha=0.7, label = 'posx-dir')
sns.kdeplot(pd_mi_scores, x='posx', y='time',fill=True, alpha=0.7, label = 'posx-time')
sns.kdeplot(pd_mi_scores, x='posx', y='vel',fill=True, alpha=0.7, label = 'posx-vel')
plt.ylim([-0.1,0.85])
plt.xlim([-0.1,0.85])
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes.png'), dpi = 400,bbox_inches="tight")


kwargs = {'cut': 0, 'cbar': False, 'fill': True}
plt.figure(figsize=(14,8))
ax = plt.subplot(1,3,1)
sns.kdeplot(pd_mi_scores, x='posx', y='dir', ax=ax, **kwargs)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.65])
ax.set_xlim([0,0.65])
ax.set_aspect('equal')
ax = plt.subplot(1,3,2)
sns.kdeplot(pd_mi_scores, x='posx', y='time', ax=ax, **kwargs)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.65])
ax.set_xlim([0,0.65])
ax.set_aspect('equal')
ax = plt.subplot(1,3,3)
sns.kdeplot(pd_mi_scores, x='posx', y='vel', ax=ax, **kwargs)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.65])
ax.set_xlim([0,0.65])
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes_separated.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_cross_kdes_separated.png'), dpi = 400,bbox_inches="tight")


plt.figure(figsize=(14,8))
ax = plt.subplot(1,3,1)
sns.scatterplot(pd_mi_scores, x='posx', y='dir', ax=ax)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.82])
ax.set_xlim([0,0.82])
ax.set_aspect('equal')
ax = plt.subplot(1,3,2)
sns.scatterplot(pd_mi_scores, x='posx', y='time', ax=ax)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.82])
ax.set_xlim([0,0.82])
ax.set_aspect('equal')
ax = plt.subplot(1,3,3)
sns.scatterplot(pd_mi_scores, x='posx', y='vel', ax=ax)
ax.plot([-0.1,0.85],[-0.1,0.85],'k--')
ax.set_ylim([0,0.82])
ax.set_xlim([0,0.82])
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'mutual_info_cross_scatter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_cross_scatter.png'), dpi = 400,bbox_inches="tight")

pd_mi_scores2 = pd.DataFrame(data={'MI': mi_scores[(0,2,3,4),:].reshape(-1,1)[:,0],
                     'label': ['posx']*mi_scores.shape[1] + ['dir']*mi_scores.shape[1] + ['vel']*mi_scores.shape[1] +['time']*mi_scores.shape[1]}) 

from scipy import stats
for idx1 in range(5):
    shaphiroResults  = stats.shapiro(mi_scores[idx1,:])
    print(f"{mi_scores_dict[mouse]['label_order'][idx1]}: {shaphiroResults.pvalue}")
for idx1 in range(5):
    label_name1 = mi_scores_dict[mouse]['label_order'][idx1]
    for idx2 in range(idx1+1,5):
        label_name2 = mi_scores_dict[mouse]['label_order'][idx2]
        print(f"{label_name1} vs {label_name2}:",stats.ks_2samp(mi_scores[idx1,:], mi_scores[idx2,:]))

plt.figure()
sns.violinplot(data=pd_mi_scores2, x="label", y="MI", split=True, inner="quart", cut=0)
plt.savefig(os.path.join(save_dir,'mutual_info_violinplots.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'mutual_info_violinplots.png'), dpi = 400,bbox_inches="tight")

modelPCA = PCA(mi_scores.shape[0])
modelPCA = modelPCA.fit(mi_scores.T)
embPCA = modelPCA.transform(mi_scores.T)

plt.figure(); plt.plot(modelPCA.explained_variance_)
plt.figure(); ax = plt.subplot(111,projection='3d'); ax.scatter(*embPCA[:,:3].T)

#__________________________________________________________________________
#|                                                                        |#
#|                        PLOT TRACES INFORMATION                         |#
#|________________________________________________________________________|#

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
data_dir =  '/home/julio/Documents/SP_project/Fig1/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig1/ncells/'
mi_scores_dict = load_pickle(save_dir, 'mi_scores_dict.pkl')
save_dir = '/home/julio/Documents/SP_project/Fig1/ncells/traces_examples/'
num_cells = 8
for mouse in miceList:
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
    clean_traces = np.concatenate(pd_mouse['clean_traces'].values, axis=0)
    label_traces = {
        'posx': np.concatenate(pd_mouse['pos'].values, axis=0)[:,0],
        'dir': np.concatenate(pd_mouse['dir_mat'].values, axis=0),
        'vel': np.concatenate(pd_mouse['vel'].values, axis=0),
        'time': np.arange(clean_traces.shape[0])/20
    }

    for label_idx, label_name in enumerate(mi_scores_dict[mouse]['label_order']):
        if label_name == 'posy':
            continue;
        label_mi_scores = mi_scores_dict[mouse]['mi_scores'][label_idx]
        mi_order = np.argsort(label_mi_scores)[::-1]

        fig = plt.figure(figsize=(10,12))    
        gridspan = fig.add_gridspec(10, 1)
        ax = fig.add_subplot(gridspan[0, 0])
        ax.plot(label_traces['time'], label_traces[label_name])
        ax = fig.add_subplot(gridspan[1:, 0])
        ax.plot(label_traces['time'], clean_traces[:,mi_order[:num_cells]] - 2*np.arange(num_cells))
        ax.set_xlabel('Time (s)')
        plt.text(0, -0.1, f"{mi_order[:num_cells]}: {label_mi_scores[mi_order[:num_cells]]}",transform=ax.transAxes)
        plt.suptitle(label_name)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir,f'nCells_{mouse}_{label_name}_example_traces.svg'), dpi = 400,bbox_inches="tight")
        plt.savefig(os.path.join(data_dir,f'nCells_{mouse}_{label_name}_example_traces.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                 COMPUTE NCELLS WITH ORDER INFORMATION                  |#
#|________________________________________________________________________|#


miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
data_dir =  '/home/julio/Documents/SP_project/Fig1/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig1/ncells/'

min_cells = 10
max_cells = 200
n_steps = 8
max_dim = 10

mi_scores_dict = load_pickle(save_dir, 'mi_scores_dict.pkl')
for label_idx, label_name in enumerate(['posx','posy','dir','vel','time']):
    print(f"\n---Checking cells according to {label_name} ({label_idx+1}/{len(['posx','posy','dir','vel','time'])}):---")
    results = {}
    for mouse in miceList:
        tic()
        print(f"\nWorking on mouse {mouse}: ")
        file_name =  mouse+'_df_dict.pkl'
        file_path = os.path.join(data_dir, mouse)
        pd_mouse = load_pickle(file_path,file_name)
        results[mouse] = {'params': {
        'min_cells' : 10,
        'max_cells' : 200,
        'n_steps' : 8,
        }}
        umap_embds = {}

        #signal
        signal = np.concatenate(pd_mouse['clean_traces'].values, axis = 0)
        pos = copy.deepcopy(np.concatenate(pd_mouse['pos'].values, axis=0))
        dir_mat = copy.deepcopy(np.concatenate(pd_mouse['dir_mat'].values, axis=0))
        vel = copy.deepcopy(np.concatenate(pd_mouse['vel'].values, axis=0))
        timeT = np.arange(pos.shape[0])

        num_cells = signal.shape[1]
        cell_list = np.unique(np.logspace(np.log10(min_cells), np.log10(max_cells),n_steps,dtype=int))
        results[mouse]['params']["og_num_cells"] = cell_list
        cell_list = cell_list[cell_list<=num_cells]
        results[mouse]['params']["cell_list"] = cell_list
        cells_picked_list = np.zeros((n_steps,num_cells)).astype(bool)

        inner_dim = {
            'abids': np.zeros((n_steps,))*np.nan,
            'mom': np.zeros((n_steps,))*np.nan,
            'tle': np.zeros((n_steps,))*np.nan,
            'params': {
                'signalName': 'clean_traces',
                'nNeigh': 30,
                'verbose': False
            }
        }

        umap_dim = {
            'trustNum': np.zeros((n_steps,max_dim))*np.nan,
            'contNum': np.zeros((n_steps,max_dim))*np.nan,
            'trustDim': np.zeros((n_steps))*np.nan,
            'contDim': np.zeros((n_steps))*np.nan,
            'hmeanDim': np.zeros((n_steps))*np.nan,
            'params': {
                'maxDim':10,
                'nNeigh': 120,
                'minDist': 0.1,
                'nnDim': 30,
                'signalName': 'clean_traces'
            }
        }

        si_temp = {
            'pos': {
                'sI': np.zeros((n_steps,))*np.nan,
                'binLabel': {},
                'overlapMat': {},
                },
            'dir_mat': {
                'sI': np.zeros((n_steps,))*np.nan,
                'binLabel': {},
                'overlapMat': {},
            },
            'time': {
                'sI': np.zeros((n_steps,))*np.nan,
                'binLabel': {},
                'overlapMat': {},
            },
            'vel': {
                'sI': np.zeros((n_steps,))*np.nan,
                'binLabel': {},
                'overlapMat': {},
            }
        }

        SI_dict = {
            'clean_traces': copy.deepcopy(si_temp),
            'umap': copy.deepcopy(si_temp),
            'params': {
                'nNeigh': 20,
                'numShuffles': 0
            }
        }

        dec_params = {
            'x_base_signal': 'clean_traces',
            'y_signal_list': ['posx', 'vel', 'index_mat', 'dir_mat'],
            'verbose': False,
            'trial_signal': 'index_mat',
            'nn': 120,
            'min_dist':0.1,
            'n_splits': 5,
            'n_dims': 3,
            'emb_list': ['umap'],
            'decoder_list': ['xgb']
        }   

        dec_dict = {
            'clean_traces': np.zeros((n_steps, dec_params['n_splits'], len(dec_params['y_signal_list']),2))*np.nan,
            'umap': np.zeros((n_steps, dec_params['n_splits'], len(dec_params['y_signal_list']),2))*np.nan,
            'params': dec_params
        }

        mi_scores = mi_scores_dict[mouse]['mi_scores'][label_idx]
        mi_order = np.argsort(mi_scores)[::-1]

        for ncell_idx, ncell_val in enumerate(cell_list):
            print(f"\nChecking number of cells: {ncell_val} ({ncell_idx+1}/{n_steps}):")
            cells_picked = mi_order[:ncell_val]
            cells_picked_list[ncell_idx, cells_picked] = True 
            it_signal = copy.deepcopy(signal[:, cells_picked])

            #compute inner dim
            inner_dim['abids'][ncell_idx] = np.nanmean(dim_red.compute_abids(it_signal, inner_dim['params']['nNeigh'],verbose=False))
            inner_dim['mom'][ncell_idx] = skdim.id.MOM().fit_transform(it_signal,n_neighbors = inner_dim['params']['nNeigh'])
            try:
                inner_dim['tle'][ncell_idx] = skdim.id.TLE().fit_transform(it_signal,n_neighbors = inner_dim['params']['nNeigh'])
            except:
                inner_dim['tle'][ncell_idx] = np.nan
            #compute umap dim
            rank_idxs = dim_validation.compute_rank_indices(it_signal)

            for dim in range(umap_dim['params']['maxDim']):
                emb_space = np.arange(dim+1)
                model = umap.UMAP(n_neighbors = umap_dim['params']['nNeigh'], n_components =dim+1, min_dist=umap_dim['params']['minDist'])
                emb = model.fit_transform(it_signal)
                if dim==2:
                    it_umap = copy.deepcopy(emb)
                #1. Compute trustworthiness
                temp = dim_validation.trustworthiness_vector(it_signal, emb, umap_dim['params']['nnDim'], indices_source = rank_idxs)
                umap_dim['trustNum'][ncell_idx][dim] = temp[-1]

                #2. Compute continuity
                temp = dim_validation.continuity_vector(it_signal, emb ,umap_dim['params']['nnDim'])
                umap_dim['contNum'][ncell_idx][dim] = temp[-1]

            dimSpace = np.linspace(1,umap_dim['params']['maxDim'], umap_dim['params']['maxDim']).astype(int)   

            kl = KneeLocator(dimSpace, umap_dim['trustNum'][ncell_idx], curve = "concave", direction = "increasing")
            if kl.knee:
                umap_dim['trustDim'][ncell_idx] = kl.knee
            else:
                umap_dim['trustDim'][ncell_idx] = np.nan
            kl = KneeLocator(dimSpace, umap_dim['contNum'][ncell_idx], curve = "concave", direction = "increasing")
            if kl.knee:
                umap_dim['contDim'][ncell_idx] = kl.knee
            else:
                umap_dim['contDim'][ncell_idx] = np.nan
            umap_dim['hmeanDim'][ncell_idx] = (2*umap_dim['trustDim'][ncell_idx]*umap_dim['contDim'][ncell_idx])/(umap_dim['trustDim'][ncell_idx]+umap_dim['contDim'][ncell_idx])

            #compute SI
            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, pos[:,0], n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['pos']['sI'][ncell_idx] = tSI
            SI_dict['clean_traces']['pos']['binLabel'][ncell_idx] = tbinLabel
            SI_dict['clean_traces']['pos']['overlapMat'][ncell_idx] = toverlapMat


            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, dir_mat, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['dir_mat']['sI'][ncell_idx] = tSI
            SI_dict['clean_traces']['dir_mat']['binLabel'][ncell_idx] = tbinLabel
            SI_dict['clean_traces']['dir_mat']['overlapMat'][ncell_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, timeT, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['time']['sI'][ncell_idx] = tSI
            SI_dict['clean_traces']['time']['binLabel'][ncell_idx] = tbinLabel
            SI_dict['clean_traces']['time']['overlapMat'][ncell_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_signal, vel, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['clean_traces']['vel']['sI'][ncell_idx] = tSI
            SI_dict['clean_traces']['vel']['binLabel'][ncell_idx] = tbinLabel
            SI_dict['clean_traces']['vel']['overlapMat'][ncell_idx] = toverlapMat


            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, pos[:,0], n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['pos']['sI'][ncell_idx] = tSI
            SI_dict['umap']['pos']['binLabel'][ncell_idx] = tbinLabel
            SI_dict['umap']['pos']['overlapMat'][ncell_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, dir_mat, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['dir_mat']['sI'][ncell_idx] = tSI
            SI_dict['umap']['dir_mat']['binLabel'][ncell_idx] = tbinLabel
            SI_dict['umap']['dir_mat']['overlapMat'][ncell_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, timeT, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['time']['sI'][ncell_idx] = tSI
            SI_dict['umap']['time']['binLabel'][ncell_idx] = tbinLabel
            SI_dict['umap']['time']['overlapMat'][ncell_idx] = toverlapMat

            tSI, tbinLabel, toverlapMat, _ = compute_structure_index(it_umap, vel, n_neighbors=SI_dict['params']['nNeigh'], num_shuffles=SI_dict['params']['numShuffles'], verbose=False)
            SI_dict['umap']['vel']['sI'][ncell_idx] = tSI
            SI_dict['umap']['vel']['binLabel'][ncell_idx] = tbinLabel
            SI_dict['umap']['vel']['overlapMat'][ncell_idx] = toverlapMat


            #compute decoders
            new_pd_mouse = copy.deepcopy(pd_mouse)
            for idx in range(pd_mouse.shape[0]):
                new_pd_mouse['clean_traces'][idx] = pd_mouse['clean_traces'][idx][:, cells_picked]

            dec_R2s, _ = dec.decoders_1D(pd_object = copy.deepcopy(new_pd_mouse), **dec_params)
            dec_dict['clean_traces'][ncell_idx,:, :,:] = dec_R2s['base_signal']['xgb']
            dec_dict['umap'][ncell_idx,:, :,:] = dec_R2s['umap']['xgb']

            umap_embds[ncell_idx] = it_umap

            results[mouse]['inner_dim'] = copy.deepcopy(inner_dim)
            results[mouse]['umap_dim'] = copy.deepcopy(umap_dim)
            results[mouse]['SI_dict'] = copy.deepcopy(SI_dict)
            results[mouse]['dec_dict'] = copy.deepcopy(dec_dict)
            results[mouse]['umap_embds'] = copy.deepcopy(umap_embds)
            results[mouse]['params']['cells_picked_list'] = copy.deepcopy(cells_picked_list)
            saveFile = open(os.path.join(save_dir, f'ncells_{label_name}_results_dict.pkl'), "wb")
            pickle.dump(results, saveFile)
            saveFile.close()

        toc()


###############################################################################################################################
###############################################################################################################################
##############################################                            #####################################################
##############################################            PLOT            #####################################################
##############################################                            #####################################################
###############################################################################################################################
###############################################################################################################################

miceList = ['M2019','M2023', 'M2024', 'M2025', 'M2026']
data_dir = '/home/julio/Documents/SP_project/Fig1/ncells/'

ncells_dict = {}

ncells_random_dict = load_pickle(data_dir, 'ncells_results_dict.pkl')
cell_list = ncells_random_dict['params']['cell_list']
for order_label in ['posx','dir','vel','time']:
    ncells_dict[order_label]= load_pickle(data_dir, f'ncells_{order_label}_results_dict.pkl')

#__________________________________________________________________________
#|                                                                        |#
#|                               DECODERS                                 |#
#|________________________________________________________________________|#
label_dict = {
    'posx': list(),
    'dir': list(),
    'vel': list(),
    'time': list()
}
order_dict = {
    'random': copy.deepcopy(label_dict),
    'posx': copy.deepcopy(label_dict),
    'dir': copy.deepcopy(label_dict),
    'vel': copy.deepcopy(label_dict),
    'time': copy.deepcopy(label_dict)
}
dec = {
    'og': copy.deepcopy(order_dict),
    'umap': copy.deepcopy(order_dict)
}
for mouse in miceList:
    for signal in ['og', 'umap']:
        if signal == 'og':
            key_name = 'clean_traces'
        else:
            key_name = signal
        temp = np.nanmean(ncells_random_dict[mouse]['dec_dict'][key_name][:,:,:,0,0],axis=(1,2))
        dec[signal]['random']['posx'].append(temp)
        temp = np.nanmean(ncells_random_dict[mouse]['dec_dict'][key_name][:,:,:,3,0],axis=(1,2))
        dec[signal]['random']['dir'].append(temp)
        temp = np.nanmean(ncells_random_dict[mouse]['dec_dict'][key_name][:,:,:,1,0],axis=(1,2))
        dec[signal]['random']['vel'].append(temp)
        temp = np.nanmean(ncells_random_dict[mouse]['dec_dict'][key_name][:,:,:,2,0],axis=(1,2))
        dec[signal]['random']['time'].append(temp)
        for order_label in ['posx','dir','vel','time']:
            temp = np.nanmean(ncells_dict[order_label][mouse]['dec_dict'][key_name][:,:,0,0],axis=1)
            dec[signal][order_label]['posx'].append(temp)
            temp = np.nanmean(ncells_dict[order_label][mouse]['dec_dict'][key_name][:,:,3,0],axis=1)
            dec[signal][order_label]['dir'].append(temp)
            temp = np.nanmean(ncells_dict[order_label][mouse]['dec_dict'][key_name][:,:,1,0],axis=1)
            dec[signal][order_label]['vel'].append(temp)
            temp = np.nanmean(ncells_dict[order_label][mouse]['dec_dict'][key_name][:,:,2,0],axis=1)
            dec[signal][order_label]['time'].append(temp)

for signal in ['og','umap']:
    for order_label in ['random','posx','dir','vel','time']:
        for analyzed_label in ['posx','dir','vel','time']:
            dec[signal][order_label][analyzed_label] = np.stack(dec[signal][order_label][analyzed_label],axis=1)

#DECODERS 
limits = {
    'posx': [-0.1,18.1],
    'dir': [-0.05,0.55],
    'vel': [-0.1,12.5],
    'time': [15,110]
}

color_dict = {
    'random': [.5,.5,.5],
    'posx': '#3274a1ff',
    'dir': '#e1812cff',
    'time': '#3a923aff',
    'vel': '#c03d3eff'
}

for feature in ['posx','dir','vel','time']:
    plt.figure(figsize=(12,5))
    ax = plt.subplot(1,2,1)
    for order_label in ['random','posx','dir', 'vel', 'time']:
        val = dec['og'][order_label][feature]
        m = np.nanmean(val,axis=1)
        sd = np.nanstd(val,axis=1)
        ax.plot(cell_list, m, label=order_label, color=color_dict[order_label])
        ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=color_dict[order_label])
    ax.set_xlabel('Num cells')
    ax.set_ylabel('Med Abs Error')
    ax.set_title('Og')
    ax.legend()
    ax.set_ylim(limits[feature])
    ax = plt.subplot(1,2,2)
    for order_label in ['random','posx','dir', 'vel', 'time']:
        val = dec['umap'][order_label][feature]
        m = np.nanmean(val,axis=1)
        sd = np.nanstd(val,axis=1)
        ax.plot(cell_list, m, label=order_label, color=color_dict[order_label])
        ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=color_dict[order_label])
    ax.set_xlabel('Num cells')
    ax.set_ylabel('Med Abs Error')
    ax.set_title('Umap')
    ax.set_ylim(limits[feature])
    ax.legend()
    plt.tight_layout()
    plt.suptitle(feature)
    plt.savefig(os.path.join(data_dir,f'nCells_{feature}_decoder_ordered.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(data_dir,f'nCells_{feature}_decoder_ordered.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                              INNER DIM                                 |#
#|________________________________________________________________________|#
label_dict = {
    'abids': list(),
    'tle': list(),
    'mom': list()
}

inner_dim = {
    'random': copy.deepcopy(label_dict),
    'posx': copy.deepcopy(label_dict),
    'dir': copy.deepcopy(label_dict),
    'vel': copy.deepcopy(label_dict),
    'time': copy.deepcopy(label_dict)
}

for mouse in miceList:
    val = np.nanmean(ncells_random_dict[mouse]['inner_dim']['abids'],axis=1)
    inner_dim['random']['abids'].append(val)
    val = np.nanmean(ncells_random_dict[mouse]['inner_dim']['tle'],axis=1)
    inner_dim['random']['tle'].append(val)
    val = np.nanmean(ncells_random_dict[mouse]['inner_dim']['mom'],axis=1)
    inner_dim['random']['mom'].append(val)
    for order_label in ['posx','dir','vel','time']:
        val = ncells_dict[order_label][mouse]['inner_dim']['abids']
        inner_dim[order_label]['abids'].append(val)
        val = ncells_dict[order_label][mouse]['inner_dim']['tle']
        inner_dim[order_label]['tle'].append(val)
        val = ncells_dict[order_label][mouse]['inner_dim']['mom']
        inner_dim[order_label]['mom'].append(val)

for order_label in ['random','posx','dir','vel','time']:
    for dim_method in ['abids','tle','mom']:
        inner_dim[order_label][dim_method] = np.stack(inner_dim[order_label][dim_method],axis=1)

color_dict = {
    'random': [.5,.5,.5],
    'posx': '#3274a1ff',
    'dir': '#e1812cff',
    'time': '#3a923aff',
    'vel': '#c03d3eff'
}

plt.figure()
ax = plt.subplot(1,2,1)
for order_label in ['random','posx','dir','vel','time']:
    val = inner_dim[order_label]['abids']
    m = np.nanmean(val,axis=1)
    sd = np.nanstd(val,axis=1)
    ax.plot(cell_list, m, label=order_label)
    ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
ax.set_xlabel('Num cells')
ax.set_ylabel('Dim')
ax.set_title('Abids')
ax.legend()
ax.set_ylim([-0.1,6])
ax = plt.subplot(1,2,2)
for order_label in ['random','posx','dir','vel','time']:
    val = inner_dim[order_label]['tle']
    m = np.nanmean(val,axis=1)
    sd = np.nanstd(val,axis=1)
    ax.plot(cell_list, m, label=order_label)
    ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3)
ax.set_xlabel('Num cells')
ax.set_ylabel('Dim')
ax.set_title('TLE')
ax.set_ylim([-0.1,6])
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir,f'nCells_inner_dim_ordered.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,f'nCells_inner_dim_ordered.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                               UMAP DIM                                 |#
#|________________________________________________________________________|#

label_dict = {
    'trustDim': list(),
    'contDim': list(),
    'hmeanDim': list()
}

umap_dim = {
    'random': copy.deepcopy(label_dict),
    'posx': copy.deepcopy(label_dict),
    'dir': copy.deepcopy(label_dict),
    'vel': copy.deepcopy(label_dict),
    'time': copy.deepcopy(label_dict)
}

for mouse in miceList:
    val = np.nanmean(ncells_random_dict[mouse]['umap_dim']['trustDim'],axis=1)
    umap_dim['random']['trustDim'].append(val)
    val = np.nanmean(ncells_random_dict[mouse]['umap_dim']['contDim'],axis=1)
    umap_dim['random']['contDim'].append(val)
    val = np.nanmean(ncells_random_dict[mouse]['umap_dim']['hmeanDim'],axis=1)
    umap_dim['random']['hmeanDim'].append(val)
    for order_label in ['posx','dir','vel','time']:
        val = ncells_dict[order_label][mouse]['umap_dim']['trustDim']
        umap_dim[order_label]['trustDim'].append(val)
        val = ncells_dict[order_label][mouse]['umap_dim']['contDim']
        umap_dim[order_label]['contDim'].append(val)
        val = ncells_dict[order_label][mouse]['umap_dim']['hmeanDim']
        umap_dim[order_label]['hmeanDim'].append(val)

for order_label in ['random','posx','dir','vel','time']:
    for dim_method in ['trustDim','contDim','hmeanDim']:
        umap_dim[order_label][dim_method] = np.stack(umap_dim[order_label][dim_method],axis=1)


color_dict = {
    'random': [.5,.5,.5],
    'posx': '#3274a1ff',
    'dir': '#e1812cff',
    'time': '#3a923aff',
    'vel': '#c03d3eff'
}


plt.figure(figsize=(14,6))
ax = plt.subplot(1,3,1)
for order_label in ['random','posx','dir','vel','time']:
    val = umap_dim[order_label]['trustDim']
    m = np.nanmean(val,axis=1)
    sd = np.nanstd(val,axis=1)
    ax.plot(cell_list, m, label=order_label, color=color_dict[order_label])
    ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=color_dict[order_label])
ax.set_xlabel('Num cells')
ax.set_ylabel('Dim')
ax.set_title('TrustDim')
ax.legend()
ax.set_ylim([-0.1,6])
ax = plt.subplot(1,3,2)
for order_label in ['random','posx','dir','vel','time']:
    val = umap_dim[order_label]['contDim']
    m = np.nanmean(val,axis=1)
    sd = np.nanstd(val,axis=1)
    ax.plot(cell_list, m, label=order_label, color=color_dict[order_label])
    ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=color_dict[order_label])
ax.set_xlabel('Num cells')
ax.set_ylabel('Dim')
ax.set_title('ContDim')
ax.set_ylim([-0.1,6])
ax.legend()
ax = plt.subplot(1,3,3)
for order_label in ['random','posx','dir','vel','time']:
    val = umap_dim[order_label]['hmeanDim']
    m = np.nanmean(val,axis=1)
    sd = np.nanstd(val,axis=1)
    ax.plot(cell_list, m, label=order_label, color=color_dict[order_label])
    ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color=color_dict[order_label])
ax.set_xlabel('Num cells')
ax.set_ylabel('Dim')
ax.set_title('hmeanDim')
ax.set_ylim([-0.1,6])
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir,f'nCells_umap_dim_ordered.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,f'nCells_umap_dim_ordered.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                                   SI                                   |#
#|________________________________________________________________________|#
label_dict = {
    'posx': list(),
    'dir': list(),
    'vel': list(),
    'time': list()
}
order_dict = {
    'random': copy.deepcopy(label_dict),
    'posx': copy.deepcopy(label_dict),
    'dir': copy.deepcopy(label_dict),
    'vel': copy.deepcopy(label_dict),
    'time': copy.deepcopy(label_dict)
}
SI = {
    'og': copy.deepcopy(order_dict),
    'umap': copy.deepcopy(order_dict)
}

for mouse in miceList:
    for signal in ['og', 'umap']:
        if signal == 'og':
            key_name = 'clean_traces'
        else:
            key_name = signal
        temp = np.nanmean(ncells_random_dict[mouse]['SI_dict'][key_name]['pos']['sI'],axis=1)
        SI[signal]['random']['posx'].append(temp)
        temp = np.nanmean(ncells_random_dict[mouse]['SI_dict'][key_name]['dir_mat']['sI'],axis=1)
        SI[signal]['random']['dir'].append(temp)
        temp = np.nanmean(ncells_random_dict[mouse]['SI_dict'][key_name]['vel']['sI'],axis=1)
        SI[signal]['random']['vel'].append(temp)
        temp = np.nanmean(ncells_random_dict[mouse]['SI_dict'][key_name]['time']['sI'],axis=1)
        SI[signal]['random']['time'].append(temp)
        for order_label in ['posx','dir','vel','time']:
            temp = ncells_dict[order_label][mouse]['SI_dict'][key_name]['pos']['sI']
            SI[signal][order_label]['posx'].append(temp)
            temp = ncells_dict[order_label][mouse]['SI_dict'][key_name]['dir_mat']['sI']
            SI[signal][order_label]['dir'].append(temp)
            temp = ncells_dict[order_label][mouse]['SI_dict'][key_name]['vel']['sI']
            SI[signal][order_label]['vel'].append(temp)
            temp = ncells_dict[order_label][mouse]['SI_dict'][key_name]['time']['sI']
            SI[signal][order_label]['time'].append(temp)

for signal in ['og','umap']:
    for order_label in ['random','posx','dir','vel','time']:
        for analyzed_label in ['posx','dir','vel','time']:
            SI[signal][order_label][analyzed_label] = np.stack(SI[signal][order_label][analyzed_label],axis=1)

limits = {
    'posx': [-0.05,1.05],
    'dir': [-0.05,1.05],
    'vel': [-0.05,1.05],
    'time': [-0.05,1.05]
}

color_dict = {
    'random': [.5,.5,.5],
    'posx': '#3274a1ff',
    'dir': '#e1812cff',
    'time': '#3a923aff',
    'vel': '#c03d3eff'
}

for feature in ['posx','dir','vel','time']:
    plt.figure(figsize=(12,5))
    ax = plt.subplot(1,2,1)
    for order_label in ['random','posx','dir', 'vel', 'time']:
        val = SI['og'][order_label][feature]
        m = np.nanmean(val,axis=1)
        sd = np.nanstd(val,axis=1)
        ax.plot(cell_list, m, label=order_label, color = color_dict[order_label])
        ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color = color_dict[order_label])
    ax.set_xlabel('Num cells')
    ax.set_ylabel('SI')
    ax.set_title('Og')
    ax.legend()
    ax.set_ylim(limits[feature])
    ax = plt.subplot(1,2,2)
    for order_label in ['random','posx','dir', 'vel', 'time']:
        val = SI['umap'][order_label][feature]
        m = np.nanmean(val,axis=1)
        sd = np.nanstd(val,axis=1)
        ax.plot(cell_list, m, label=order_label, color = color_dict[order_label])
        ax.fill_between(cell_list, m-sd, m+sd, alpha = 0.3, color = color_dict[order_label])
    ax.set_xlabel('Num cells')
    ax.set_ylabel('SI')
    ax.set_title('Umap')
    ax.set_ylim(limits[feature])
    ax.legend()
    plt.tight_layout()
    plt.suptitle(feature)
    plt.savefig(os.path.join(data_dir,f'nCells_{feature}_SI_ordered.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(data_dir,f'nCells_{feature}_SI_ordered.png'), dpi = 400,bbox_inches="tight")


