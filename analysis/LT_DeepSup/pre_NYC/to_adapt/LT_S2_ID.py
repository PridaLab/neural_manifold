from neural_manifold import general_utils as gu
from neural_manifold.dimensionality_reduction import compute_dimensionality as cd

import sys, os, timeit
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


mouse = sys.argv[1]
data_dir = sys.argv[2]
save_dir = sys.argv[3]
if len(sys.argv)>4:
    signal = sys.argv[4]
else:
    signal = 'revents_SNR3'

run_abid_here = True
run_umap_here = True
#__________________________________________________________________________
#|                                                                        |#
#|                              1. LOAD DATA                              |#
#|________________________________________________________________________|#
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

figures_dir = os.path.join(save_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

f = open(os.path.join(save_dir,mouse + '_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

global_starttime = timeit.default_timer()

print(f"Working on mouse {mouse}:")
print(f"\tdata_dir: {data_dir}")
print(f"\tsave_dir: {save_dir}")
print(f"\tDate: {datetime.now():%Y-%m-%d %H:%M}")

#1. Load data
local_starttime = timeit.default_timer()
print('### 1. LOAD DATA ###')
print('1 Searching & loading data in directory:\n', data_dir)
mouse_pd = gu.load_files(data_dir, '*'+mouse+'_rates_dict*.pkl', verbose=True, 
                                                    struct_type = "pickle")

fnames = list(mouse_pd.keys())
gu.print_time_verbose(local_starttime, global_starttime)

#__________________________________________________________________________
#|                                                                        |#
#|                                2. ABIDS                                |#
#|________________________________________________________________________|#
if run_abid_here:
    local_starttime = timeit.default_timer()
    print('### 2. ABIDS ###')
    params = {
        'n_neigh': 50
    }
    print(f'Params: {params}')
    abids_dict = dict()

    print('\tPre:')
    cloud_pre = np.concatenate(mouse_pd[fnames[0]][signal].values, axis=0)
    abids = cd.compute_abids(cloud_pre,**params)
    print(f'\tdim: {np.nanmean(abids)}')
    abids_dict[fnames[0]] = {
        'dim': np.nanmean(abids),
        'abids': abids,
        'params': params
    }

    print('\tRot:')
    cloud_rot = np.concatenate(mouse_pd[fnames[1]][signal].values, axis=0)
    abids = cd.compute_abids(cloud_rot,**params)
    print(f'\tdim: {np.nanmean(abids)}')
    abids_dict[fnames[1]] = {
        'dim': np.nanmean(abids),
        'abids': abids,
        'params': params
    }

    print('\tBoth:')
    cloud_both = np.concatenate((cloud_pre, cloud_rot), axis= 0)
    abids = cd.compute_abids(cloud_both,**params)
    print(f'\tdim: {np.nanmean(abids)}')
    abids_dict['both'] = {
        'dim': np.nanmean(abids),
        'abids': abids,
        'params': params
    }

    print(f'Saving data on {save_dir}')
    #save abids dict
    file_name = os.path.join(save_dir, mouse+ "_abids_dict.pkl")
    save_df = open(file_name, "wb")
    pickle.dump(abids_dict, save_df)
    save_df.close()

    #Plot abids dim
    abids_pre = abids_dict[fnames[0]]['abids']
    abids_rot = abids_dict[fnames[1]]['abids']
    abids_both = abids_dict['both']['abids']

    pre_label = ['pre']*abids_pre.shape[0]
    rot_label = ['rot']*abids_rot.shape[0]
    both_label = ['both']*abids_both.shape[0]

    abids_struct = pd.DataFrame(data={'abids':np.hstack((abids_pre, abids_rot,abids_both)), 
                                  'session_type':np.hstack((pre_label, rot_label,both_label))})

    colors = ['#62C370', '#C360B4', '#6083C3', '#C3A060']
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.set(palette=colors)

    sns.kdeplot(x='abids', data=abids_struct, hue='session_type', shade=True, 
        common_norm=False, clip=[0, None], common_grid=True, ax=ax)  

    ax.axvline(np.nanmean(abids_pre), color = colors[0], linestyle='--')
    ax.axvline(np.nanmean(abids_rot), color = colors[1], linestyle='--')
    ax.axvline(np.nanmean(abids_both), color = colors[2], linestyle='--')

    ax.set_title(signal)
    plt.savefig(os.path.join(figures_dir, mouse + '_abids_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir,mouse + '_abids_plot.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)


#__________________________________________________________________________
#|                                                                        |#
#|                                3. UMAP                                |#
#|________________________________________________________________________|#
if run_umap_here:
    local_starttime = timeit.default_timer()
    print('### 3. UMAP ###')
    params = {
        'n_neigh': 50,
        'max_dim' : 8,
        'min_dist': 0.1,
        'verbose': True
    }

    print(f'Params: {params}')
    umap_dict = dict()

    print('\tPre:')
    cloud_pre = np.concatenate(mouse_pd[fnames[0]][signal].values, axis=0)
    dim_trust, num_trust = cd.compute_umap_trust_dim(cloud_pre,**params)
    dim_cont, num_cont = cd.compute_umap_continuity_dim(cloud_pre,**params)
    dim_HM = (2*dim_cont*dim_trust)/(dim_cont+dim_trust)
    print(f'\tdim_trust: {dim_trust} - dim_cont: {dim_cont} (HM: {dim_HM:.2f})')
    umap_dict[fnames[0]] = {
        'dim_trust': dim_trust,
        'num_trust': num_trust,
        'dim_cont': dim_cont,
        'num_cont': num_cont,
        'dim_HM': dim_HM,
        'params': params
    }

    print('\tRot:')
    cloud_rot = np.concatenate(mouse_pd[fnames[1]][signal].values, axis=0)
    dim_trust, num_trust = cd.compute_umap_trust_dim(cloud_rot,**params)
    dim_cont, num_cont = cd.compute_umap_continuity_dim(cloud_rot,**params)
    dim_HM = (2*dim_cont*dim_trust)/(dim_cont+dim_trust)
    print(f'\tdim_trust: {dim_trust} - dim_cont: {dim_cont} (HM: {dim_HM:.2f})')
    umap_dict[fnames[1]] = {
        'dim_trust': dim_trust,
        'num_trust': num_trust,
        'dim_cont': dim_cont,
        'num_cont': num_cont,
        'dim_HM': dim_HM,
        'params': params
    }

    print('\tBoth:')
    cloud_both = np.concatenate((cloud_pre, cloud_rot), axis= 0)
    dim_trust, num_trust = cd.compute_umap_trust_dim(cloud_both,**params)
    dim_cont, num_cont = cd.compute_umap_continuity_dim(cloud_both,**params)
    dim_HM = (2*dim_cont*dim_trust)/(dim_cont+dim_trust)
    print(f'\tdim_trust: {dim_trust} - dim_cont: {dim_cont} (HM: {dim_HM:.2f})')
    umap_dict['both'] = {
        'dim_trust': dim_trust,
        'num_trust': num_trust,
        'dim_cont': dim_cont,
        'num_cont': num_cont,
        'dim_HM': dim_HM,
        'params': params
    }

    print(f'Saving data on {save_dir}')
    #save abids dict
    file_name = os.path.join(save_dir, mouse+ "_umap_dims_dict.pkl")
    save_df = open(file_name, "wb")
    pickle.dump(umap_dict, save_df)
    save_df.close()

else:
    print("Loading umap dict")
    umap_dict = gu.load_files(save_dir, '*'+mouse+'_umap_dims_dict*.pkl', verbose=True, 
                                                    struct_type = "pickle")
    params = umap_dict['both']['params']
#Plot dim
colors = ['#62C370', '#C360B4', '#6083C3', '#C3A060']
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(2,3,1)
ax.plot(np.arange(params['max_dim'])+1,umap_dict[fnames[0]]['num_trust'], label = 'trust', color = colors[0])
ax.axvline(umap_dict[fnames[0]]['dim_trust'],color = colors[0],linestyle='--')
ax.plot(np.arange(params['max_dim'])+1,umap_dict[fnames[0]]['num_cont'], label = 'cont', color = colors[1])
ax.axvline(umap_dict[fnames[0]]['dim_cont'],color = colors[1],linestyle='--')
ax.axvline(umap_dict[fnames[0]]['dim_HM'],color = colors[2],linestyle='--')
ax.set_title(fnames[0])
ax.set_xticks([1,2,3,4,5,6,7,8])
ax.set_xlabel('Dim')
ax.set_ylabel('Score')


ax = plt.subplot(2,3,2)
ax.plot(np.arange(params['max_dim'])+1,umap_dict[fnames[1]]['num_trust'], label = 'trust', color = colors[0])
ax.axvline(umap_dict[fnames[1]]['dim_trust'],color = colors[0],linestyle='--')
ax.plot(np.arange(params['max_dim'])+1,umap_dict[fnames[1]]['num_cont'], label = 'cont', color = colors[1])
ax.axvline(umap_dict[fnames[1]]['dim_cont'],color = colors[1],linestyle='--')
ax.axvline(umap_dict[fnames[1]]['dim_HM'],color = colors[2],linestyle='--')
ax.set_title(fnames[1])
ax.set_xlabel('Dim')
ax.set_ylabel('Score')
ax.set_xticks([1,2,3,4,5,6,7,8])

ax = plt.subplot(2,3,3)
ax.plot(np.arange(params['max_dim'])+1,umap_dict['both']['num_trust'], label = 'trust', color = colors[0])
ax.axvline(umap_dict['both']['dim_trust'],color = colors[0],linestyle='--')
ax.plot(np.arange(params['max_dim'])+1,umap_dict['both']['num_cont'], label = 'cont', color = colors[1])
ax.axvline(umap_dict['both']['dim_cont'],color = colors[1],linestyle='--')
ax.axvline(umap_dict['both']['dim_HM'],color = colors[2],linestyle='--')
ax.set_title('both')
ax.set_xlabel('Dim')
ax.set_ylabel('Score')
ax.set_xticks([1,2,3,4,5,6,7,8])

ax = plt.subplot(2,1,2)
dims = [umap_dict[fnames[0]]['dim_trust'],umap_dict[fnames[1]]['dim_trust'],umap_dict['both']['dim_trust'],
        umap_dict[fnames[0]]['dim_cont'],umap_dict[fnames[1]]['dim_cont'],umap_dict['both']['dim_cont'],
        umap_dict[fnames[0]]['dim_HM'],umap_dict[fnames[1]]['dim_HM'],umap_dict['both']['dim_HM']]

dims = np.array(dims)
metric = np.hstack((['trust']*3,['cont']*3,['HM']*3))
session = np.array([fnames[0], fnames[1], 'both']*3)

dim_struct = pd.DataFrame(data={'dims':dims,
                                'metric':metric, 
                                'session_type':session})


sns.set(palette=['#62C370', '#C360B4', '#6083C3', '#C3A060'])
sns.barplot(data=dim_struct, x='session_type', y='dims', hue='metric', ax=ax)  

plt.savefig(os.path.join(figures_dir, mouse + '_umap_dims_plot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(figures_dir,mouse + '_umap_dims_plot.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)

#__________________________________________________________________________
#|                                                                        |#
#|                          3. ISOMAP RESVAR                              |#
#|________________________________________________________________________|#
local_starttime = timeit.default_timer()
print('### 4. ISOMAP ###')
params = {
    'n_neigh': 50,
    'max_dim' :10,
    'verbose': True
}

print(f'Params: {params}')
isomap_dict = dict()

print('\tPre:')
cloud_pre = np.concatenate(mouse_pd[fnames[0]][signal].values, axis=0)
dim, res_var = cd.compute_isomap_resvar_dim(cloud_pre,**params)
print(f'\tdim: {dim}')
isomap_dict[fnames[0]] = {
    'dim_res_var': dim,
    'res_var': res_var,
    'params': params
}

print('\tRot:')
cloud_rot = np.concatenate(mouse_pd[fnames[1]][signal].values, axis=0)
dim, res_var = cd.compute_isomap_resvar_dim(cloud_rot,**params)
print(f'\tdim: {dim}')
isomap_dict[fnames[1]] = {
    'dim_res_var': dim,
    'res_var': res_var,
    'params': params
}

print('\tBoth:')
cloud_both = np.concatenate((cloud_pre, cloud_rot), axis= 0)
dim, res_var = cd.compute_isomap_resvar_dim(cloud_both,**params)
print(f'\tdim: {dim}')
isomap_dict['both'] = {
    'dim_res_var': dim,
    'res_var': res_var,
    'params': params
}

print(f'Saving data on {save_dir}')
#save abids dict
file_name = os.path.join(save_dir, mouse+ "_isomap_dims_dict.pkl")
save_df = open(file_name, "wb")
pickle.dump(isomap_dict, save_df)
save_df.close()

#Plot dim
colors = ['#62C370', '#C360B4', '#6083C3', '#C3A060']
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(1,1,1)

ax.plot(np.arange(params['max_dim'])+1,isomap_dict[fnames[0]]['res_var'], label = fnames[0], color = colors[0])
ax.axvline(isomap_dict[fnames[0]]['dim_res_var'],color = colors[0],linestyle='--')

ax.plot(np.arange(params['max_dim'])+1,isomap_dict[fnames[1]]['res_var'], label = fnames[1], color = colors[1])
ax.axvline(isomap_dict[fnames[1]]['dim_res_var'],color = colors[1],linestyle='--')

ax.plot(np.arange(params['max_dim'])+1,isomap_dict['both']['res_var'], label = 'both', color = colors[2])
ax.axvline(isomap_dict['both']['dim_res_var'],color = colors[2],linestyle='--')
ax.set_xlabel('Dim')
ax.set_ylabel('Res-var')

plt.savefig(os.path.join(figures_dir, mouse + '_isomap_resvar_plot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(figures_dir, mouse + '_isomap_resvar_plot.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)
