from os import path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import os, pickle, copy
from sklearn.metrics import median_absolute_error
import scipy.stats as stats

def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

palette_deepsup = ["#cc9900ff", "#9900ffff"]
palette_deepsup_strain = ["#f5b800ff", "#b48700ff", "#9900ffff"]


#__________________________________________________________________________
#|                                                                        |#
#|                         CUES SIZE IN MANIFOLD                          |#
#|________________________________________________________________________|#


data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
cues_dir =  '/home/julio/Documents/SP_project/LT_DeepSup/data/'
mouse_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

in_cue_list = list()
out_cue_list = list()
diff_cue_list = list()
strain_list = list()
layer_list = list()

for mouse in mouse_list:

    if ('CZ' in mouse) or ('CG' in mouse):
        strain_list.append('Calb')        
        layer_list.append('Sup')
    elif ('GC' in mouse) or ('TG' in mouse):
        strain_list.append('Thy1')
        layer_list.append('Deep')
    elif 'Ch' in mouse:
        strain_list.append('ChRNA7')
        layer_list.append('Deep')

    file_path = path.join(data_dir, mouse)
    mouse_pd = load_pickle(file_path,mouse+'_df_dict.pkl')

    cues_files = [f for f in os.listdir(path.join(cues_dir, mouse)) if 'cues_info.csv' in f]
    cues_file = [fname for fname in cues_files if 'lt' in fname][0]
    cues_info = pd.read_csv(path.join(cues_dir, mouse, cues_file))

    traces = get_signal(mouse_pd, 'clean_traces')
    umap = get_signal(mouse_pd, 'umap')

    pos = get_signal(mouse_pd, 'pos')
    min_pos = np.percentile(pos[:,0], 1)
    max_pos = np.percentile(pos[:,0], 99)

    speed = get_signal(mouse_pd , 'vel')

    mov_dir = get_signal(mouse_pd, 'dir_mat')
    left_idx = np.where(mov_dir[:,0]==1)[0]
    right_idx = np.where(mov_dir[:,0]==2)[0]

    st_cue = cues_info['x_start_cm'][0]
    en_cue = cues_info['x_end_cm'][0]

    st_fake_cue = max_pos - en_cue + min_pos;
    en_fake_cue = max_pos - st_cue + min_pos;

    idx_in_cue = np.where(np.logical_and(pos[:,0]>st_cue, pos[:,0]<en_cue))[0]
    idx_in_fake_cue = np.where(np.logical_and(pos[:,0]>st_fake_cue, pos[:,0]<en_fake_cue))[0]
    idx_none = [x for x in range(pos.shape[0]) if x not in idx_in_cue and x not in idx_in_fake_cue]


    overlap = np.intersect1d(idx_in_cue, left_idx)
    left_cue_umap = umap[np.intersect1d(idx_in_cue, left_idx),:]
    right_cue_umap = umap[np.intersect1d(idx_in_cue, right_idx),:]

    left_fake_cue_umap = umap[np.intersect1d(idx_in_fake_cue, left_idx),:]
    right_fake_cue_umap = umap[np.intersect1d(idx_in_fake_cue, right_idx),:]

    in_cue_list.append(np.mean([np.median(pairwise_distances(left_cue_umap)), np.median(pairwise_distances(right_cue_umap))]))
    out_cue_list.append(np.mean([np.median(pairwise_distances(left_fake_cue_umap)), np.median(pairwise_distances(right_fake_cue_umap))]))
    diff_cue_list.append((in_cue_list[-1]-out_cue_list[-1])/out_cue_list[-1])



fig = plt.figure()
ax = plt.subplot(1,2,1,projection = '3d')
ax.scatter(*umap.T, c = pos[:,0])

ax = plt.subplot(1,2,2,projection = '3d')
ax.scatter(*umap[idx_none,:].T, color = 'gray', alpha=0.5)
ax.scatter(*umap[idx_in_cue,:].T, color = 'b')
ax.scatter(*umap[idx_in_fake_cue,:].T, color = 'r')

cues_pd = pd.DataFrame(data={'layer': layer_list,
                     'strain': strain_list,
                     'mouse': mouse_list,
                     'in_cue': in_cue_list,
                     'out_cue': out_cue_list,
                     'diff_cue': diff_cue_list
                     })


fig, ax = plt.subplots(1, 1, figsize=(6,6))
sns.boxplot(x='layer', y='diff_cue', data=cues_pd,
            palette = palette_deepsup, linewidth = 1, width= .5, ax = ax)
sns.scatterplot(x='layer', y='diff_cue', data=cues_pd,
            palette = palette_deepsup, ax = ax)
ax.set_ylim([-0.25, 0.25])

x = cues_pd['layer'].to_list()*2
y = cues_pd['in_cue'].to_list()+cues_pd['out_cue'].to_list()
hue = ['in_cue']*int(len(x)/2) + ['out_cue']*int(len(x)/2)
fig, ax = plt.subplots(1, 1, figsize=(6,6))
sns.boxplot(x=x, y=y, hue=hue, linewidth = 1, width= .5, ax = ax)
sns.scatterplot(x=x, y=y, hue=hue, ax = ax)


#__________________________________________________________________________
#|                                                                        |#
#|                                DECODER                                 |#
#|________________________________________________________________________|#

from neural_manifold import decoders as dec
from sklearn.metrics import median_absolute_error


data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
cues_dir =  '/home/julio/Documents/SP_project/LT_DeepSup/data/'
save_dir = '/home/julio/Pictures/'
mouse_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

in_cue_list = list()
out_cue_list = list()
diff_cue_list = list()
strain_list = list()
layer_list = list()


params = {
    'x_base_signal': 'clean_traces',
    'y_signal_list': ['posx'],
    'verbose': True,
    'trial_signal': 'index_mat',
    'nn': 120,
    'min_dist':0.1,
    'n_splits': 10,
    'n_dims': 3,
    'emb_list': ['umap']
}

params = {

    'verbose': True,
    'nn': 120,
    'min_dist':0.1,
    'n_splits': 10,
    'n_dims': 3,
    'emb_list': ['umap']
}



dec_R2s = {}
dec_pred = {}
for mouse in mouse_list:
    file_path = path.join(data_dir, mouse)
    mouse_pd = load_pickle(file_path,mouse+'_df_dict.pkl')

    traces = get_signal(mouse_pd, 'clean_traces')
    pos = get_signal(mouse_pd, 'pos')[:,0]
    index_mat = get_signal(mouse_pd, 'index_mat')

    #cut extremes
    min_pos = np.percentile(pos, 1)
    max_pos = np.percentile(pos, 99)

    low_th = min_pos + 0.05*(max_pos-min_pos)
    high_th = max_pos - 0.05*(max_pos-min_pos)
    idx_in_maze = np.where(np.logical_and(pos>low_th, pos<high_th))[0]

    traces = traces[idx_in_maze,:]
    pos = pos[idx_in_maze].reshape(-1,1)
    index_mat = index_mat[idx_in_maze]


    dec_R2s[mouse], dec_pred[mouse] = dec.decoders_1D(x_base_signal = copy.deepcopy(traces),
                                                    y_signal_list = copy.deepcopy(pos), 
                                                    trial_signal = copy.deepcopy(index_mat), **params)

    with open(os.path.join(save_dir,'dec_R2s_no_edges_dict.pkl'), 'wb') as f:
        pickle.dump(dec_R2s, f)
    with open(os.path.join(save_dir,'dec_pred_no_edges_dict.pkl'), 'wb') as f:
        pickle.dump(dec_pred, f)
    with open(os.path.join(save_dir,'dec_no_edges_params.pkl'), 'wb') as f:
        pickle.dump(params, f)


in_cue_traces_list = list()
out_cue_traces_list = list()
diff_cue_traces_list = list()
all_traces_list = list()

in_cue_umap_list = list()
out_cue_umap_list = list()
diff_cue_umap_list = list()
all_umap_list = list()

strain_list = list()
layer_list = list()

decoder = 'xgb'
decoder_idx = 2
for mouse in mouse_list:

    if ('CZ' in mouse) or ('CG' in mouse):
        strain_list.append('Calb')        
        layer_list.append('Sup')
    elif ('GC' in mouse) or ('TG' in mouse):
        strain_list.append('Thy1')
        layer_list.append('Deep')
    elif 'Ch' in mouse:
        strain_list.append('ChRNA7')
        layer_list.append('Deep')

    #mouse/y_signal/decoder/
    pred_array = dec_pred[mouse][0][decoder_idx].reshape(-1,4)

    train_idx = np.where(pred_array[:,0] == 0)[0]
    test_idx = np.where(pred_array[:,0] == 1)[0]

    real_pred = pred_array[test_idx,1]
    traces_pred = pred_array[test_idx,2]
    umap_pred = pred_array[test_idx,3]


    cues_files = [f for f in os.listdir(path.join(cues_dir, mouse)) if 'cues_info.csv' in f]
    cues_file = [fname for fname in cues_files if 'lt' in fname][0]
    cues_info = pd.read_csv(path.join(cues_dir, mouse, cues_file))

    st_cue = cues_info['x_start_cm'][0]
    en_cue = cues_info['x_end_cm'][0]
    min_pos = np.percentile(pred_array[:,1], 1)
    max_pos = np.percentile(pred_array[:,1], 99)
    st_fake_cue = max_pos - en_cue + min_pos;
    en_fake_cue = max_pos - st_cue + min_pos;

    idx_in_cue = np.where(np.logical_and(real_pred>st_cue, real_pred<en_cue))[0]
    idx_in_fake_cue = np.where(np.logical_and(real_pred>st_fake_cue, real_pred<en_fake_cue))[0]

    in_cue_traces_list.append(median_absolute_error(real_pred[idx_in_cue], traces_pred[idx_in_cue]))
    out_cue_traces_list.append(median_absolute_error(real_pred[idx_in_fake_cue], traces_pred[idx_in_fake_cue]))
    diff_cue_traces_list.append((in_cue_traces_list[-1]-out_cue_traces_list[-1])/out_cue_traces_list[-1])
    all_traces_list.append(np.mean(dec_R2s[mouse]['base_signal'][decoder][:,0,0]))

    in_cue_umap_list.append(median_absolute_error(real_pred[idx_in_cue], umap_pred[idx_in_cue]))
    out_cue_umap_list.append(median_absolute_error(real_pred[idx_in_fake_cue], umap_pred[idx_in_fake_cue]))
    diff_cue_umap_list.append((in_cue_umap_list[-1]-out_cue_umap_list[-1])/out_cue_umap_list[-1])
    all_umap_list.append(np.mean(dec_R2s[mouse]['umap'][decoder][:,0,0]))

cues_pd = pd.DataFrame(data={'layer': layer_list,
                     'strain': strain_list,
                     'mouse': mouse_list,
                     'in_cue_traces': in_cue_traces_list,
                     'out_cue_traces': out_cue_traces_list,
                     'diff_cue_traces': out_cue_traces_list,
                     'all_traces': all_traces_list,

                     'in_cue_umap': in_cue_umap_list,
                     'out_cue_umap': out_cue_umap_list,
                     'diff_cue_umap': out_cue_umap_list,
                     'all_umap': all_umap_list
                     })

cues_pd['diff_cue_traces'] = cues_pd['in_cue_traces']-cues_pd['out_cue_umap']
cues_pd['diff_cue_umap'] = cues_pd['in_cue_umap']-cues_pd['out_cue_umap']

fig, ax = plt.subplots(1, 2, figsize=(6,6))
sns.boxplot(x='layer', y='diff_cue_traces', data=cues_pd,
            palette = palette_deepsup, linewidth = 1, width= .5, ax = ax[0])
sns.scatterplot(x='layer', y='diff_cue_umap', data=cues_pd,
            palette = palette_deepsup, ax = ax[0])


sns.boxplot(x='layer', y='diff_cue_umap', data=cues_pd,
            palette = palette_deepsup, linewidth = 1, width= .5, ax = ax[1])
sns.scatterplot(x='layer', y='diff_cue_umap', data=cues_pd,
            palette = palette_deepsup, ax = ax[1])


fig, ax = plt.subplots(1, 2, figsize=(6,6))

x = ['in_cue_traces']*int(len(x)/2) + ['all_traces']*int(len(x)/2)
y = cues_pd['in_cue_traces'].to_list()+cues_pd['all_traces'].to_list()
hue = cues_pd['layer'].to_list()*2
sns.boxplot(x=x, y=y, hue=hue, linewidth = 1, width= .5, ax = ax[0])
sns.scatterplot(x=x, y=y, hue=hue, ax = ax[0])

x = ['in_cue_umap']*int(len(x)/2) + ['all_umap']*int(len(x)/2) 
y = cues_pd['in_cue_umap'].to_list()+cues_pd['all_umap'].to_list()
hue = cues_pd['layer'].to_list()*2
sns.boxplot(x=x, y=y, hue=hue, linewidth = 1, width= .5, ax = ax[1])
sns.scatterplot(x=x, y=y, hue=hue, ax = ax[1])


deep_error = cues_pd[cues_pd['layer']=='Deep']['in_cue_traces'].to_list()
sup_error = cues_pd[cues_pd['layer']=='Sup']['in_cue_traces'].to_list()
deep_shapiro = stats.shapiro(deep_error)
sup_shapiro = stats.shapiro(sup_error)
if deep_shapiro.pvalue<=0.05 or sup_shapiro.pvalue<=0.05:
    print(f'traces ks_2samp p: {stats.ks_2samp(deep_error, sup_error)[1]:.4f}')
else:
    print(f'traces ttest_ind p: {stats.ttest_ind(deep_error, sup_error)[1]:.4f}')

deep_error = cues_pd[cues_pd['layer']=='Deep']['in_cue_umap'].to_list()
sup_error = cues_pd[cues_pd['layer']=='Sup']['in_cue_umap'].to_list()
deep_shapiro = stats.shapiro(deep_error)
sup_shapiro = stats.shapiro(sup_error)
if deep_shapiro.pvalue<=0.05 or sup_shapiro.pvalue<=0.05:
    print(f'umap ks_2samp p: {stats.ks_2samp(deep_error, sup_error)[1]:.4f}')
else:
    print(f'umap ttest_ind p: {stats.ttest_ind(deep_error, sup_error)[1]:.4f}')



#__________________________________________________________________________
#|                                                                        |#
#|                           DECODER SCATTERS                             |#
#|________________________________________________________________________|#
mouse_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']


data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
cues_dir =  '/home/julio/Documents/SP_project/LT_DeepSup/data/'
betti_dir = '/home/julio/Documents/SP_project/Fig2/betti_numbers/og_mi_cells'

decoders_dir = '/home/julio/Documents/SP_project/Fig2/decoders'
dec_R2s = load_pickle(decoders_dir, 'dec_R2s_dict.pkl')
dec_pred = load_pickle(decoders_dir, 'dec_pred_dict.pkl')

mi_dir = '/home/julio/Documents/SP_project/Fig2/mutual_info/'
mi_scores_dict = load_pickle(mi_dir, 'mi_scores_sep_dict.pkl')

eccentricity_dir =  '/home/julio/Documents/SP_project/Fig2/eccentricity/'
eccentricity_dict = load_pickle(eccentricity_dir, 'ellipse_fit_dict.pkl')


mae_umap_list = list()
mae_traces_list = list()
mae_no_edges_umap_list = list()
mae_no_edges_traces_list = list()
mi_list = list()
eccentricity_list = list()
h1_lifetime_list = list()

strain_list = list()
layer_list = list()
num_cells_list = list()
decoder = 'xgb'

decoder_idx = 2
label_idx = 0


for mouse in mouse_list:

    #general info of mouse
    if ('CZ' in mouse) or ('CG' in mouse):
        strain_list.append('Calb')        
        layer_list.append('Sup')
    elif ('GC' in mouse) or ('TG' in mouse):
        strain_list.append('Thy1')
        layer_list.append('Deep')
    elif 'Ch' in mouse:
        strain_list.append('ChRNA7')
        layer_list.append('Deep')

    #decoder data
    mae_traces_list.append(np.mean(dec_R2s[mouse]['base_signal'][decoder][:,0,0], axis=0))
    mae_umap_list.append(np.mean(dec_R2s[mouse]['umap'][decoder][:,0,0], axis=0))

    #decoder data no edges
    test = dec_pred[mouse][label_idx][decoder_idx][:,:,0].reshape(-1,1) == 1
    ground_truth = dec_pred[mouse][label_idx][decoder_idx][:,:,1].reshape(-1,1)[test]
    og_pred = dec_pred[mouse][label_idx][decoder_idx][:,:,2].reshape(-1,1)[test]
    umap_pred = dec_pred[mouse][label_idx][decoder_idx][:,:,-1].reshape(-1,1)[test]

    track_length = np.max(ground_truth) - np.min(ground_truth)
    min_x = 0.1*track_length+np.min(ground_truth)
    max_x = 0.9*track_length+np.min(ground_truth)
    within_edges = np.logical_and(ground_truth>=min_x,ground_truth<=max_x)
    mae_no_edges_traces_list.append(median_absolute_error(ground_truth[within_edges], og_pred[within_edges]))
    mae_no_edges_umap_list.append(median_absolute_error(ground_truth[within_edges], umap_pred[within_edges]))

    #num cells
    mouse_pd = load_pickle(path.join(data_dir, mouse),mouse+'_df_dict.pkl')
    num_cells_list.append(get_signal(mouse_pd, 'clean_traces').shape[1])

    #lifetime
    betti_dict = load_pickle(path.join(betti_dir,mouse), mouse+'_betti_dict_og.pkl')
    try:
        dense_conf_interval1 = betti_dict['dense_conf_interval'][1]
        dense_conf_interval2 = betti_dict['dense_conf_interval'][2]
    except:
        dense_conf_interval1 = 0
        dense_conf_interval2 = 0

    h1_diagrams = np.array(betti_dict['dense_diagrams'][1])
    h1_length = np.sort(np.diff(h1_diagrams, axis=1)[:,0])
    second_length = np.max([dense_conf_interval1, h1_length[-2]])
    h1_lifetime_list.append((h1_length[-1]-second_length))

    #mutual info
    mi_list.append(np.mean(mi_scores_dict[mouse]['mi_scores'][0,:]))


    #eccentricity
    pos_length = eccentricity_dict[mouse]['posLength']
    dir_length = eccentricity_dict[mouse]['dirLength']
    eccentricity_list.append(100*(pos_length-dir_length)/(pos_length))




decoder_pd = pd.DataFrame(data={'layer': layer_list,
                     'strain': strain_list,
                     'mouse': mouse_list,
                     'mae_umap': mae_umap_list,
                     'mae_traces': mae_traces_list,

                     'mae_no_edges_umap': mae_no_edges_umap_list,
                     'mae_no_edges_traces': mae_no_edges_traces_list,

                     'num_cells': num_cells_list,
                     'h1_lifetime': h1_lifetime_list,
                     'mi': mi_list,
                     'eccentricity': eccentricity_list,
                     })


palette= ['grey', '#249aefff']
label_list = ['posx']
signal_list = ['base_signal', 'umap']
decoder_list = ['xgb']
miceList = list(dec_R2s.keys())

palette_deepsup = ["#cc9900ff", "#9900ffff"]

fig, ax = plt.subplots(1,4,figsize=(15,5))
sns.scatterplot(data=decoder_pd, y='mae_umap', x='num_cells', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[0])
slope, intercept, r_value, p_value, std_err = stats.linregress(decoder_pd["num_cells"].to_list(),decoder_pd["mae_umap"].to_list())
s = stats.pearsonr(decoder_pd["num_cells"].to_list(),decoder_pd["mae_umap"].to_list())
ax[0].plot([decoder_pd["num_cells"].min(),decoder_pd["num_cells"].max()], 
    [slope*decoder_pd["num_cells"].min()+intercept, slope*decoder_pd["num_cells"].max()+intercept], 'k--')
ax[0].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

sns.scatterplot(data=decoder_pd, y='mae_umap', x='mi', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(decoder_pd["mi"].to_list(),decoder_pd["mae_umap"].to_list())
s = stats.pearsonr(decoder_pd["mi"].to_list(),decoder_pd["mae_umap"].to_list())
ax[1].plot([decoder_pd["mi"].min(),decoder_pd["mi"].max()], 
    [slope*decoder_pd["mi"].min()+intercept, slope*decoder_pd["mi"].max()+intercept], 'k--')
ax[1].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

sns.scatterplot(data=decoder_pd, y='mae_umap', x='h1_lifetime', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[2])
slope, intercept, r_value, p_value, std_err = stats.linregress(decoder_pd["h1_lifetime"].to_list(),decoder_pd["mae_umap"].to_list())
s = stats.pearsonr(decoder_pd["h1_lifetime"].to_list(),decoder_pd["mae_umap"].to_list())
ax[2].plot([decoder_pd["h1_lifetime"].min(),decoder_pd["h1_lifetime"].max()], 
    [slope*decoder_pd["h1_lifetime"].min()+intercept, slope*decoder_pd["h1_lifetime"].max()+intercept], 'k--')
ax[2].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

sns.scatterplot(data=decoder_pd, y='mae_umap', x='eccentricity', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[3])
slope, intercept, r_value, p_value, std_err = stats.linregress(decoder_pd["eccentricity"].to_list(),decoder_pd["mae_umap"].to_list())
s = stats.pearsonr(decoder_pd["eccentricity"].to_list(),decoder_pd["mae_umap"].to_list())
ax[3].plot([decoder_pd["eccentricity"].min(),decoder_pd["eccentricity"].max()], 
    [slope*decoder_pd["eccentricity"].min()+intercept, slope*decoder_pd["eccentricity"].max()+intercept], 'k--')
ax[3].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

plt.savefig(os.path.join(decoders_dir,'deepsup_decoderx_scatter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(decoders_dir,'deepsup_decoderx_scatter.png'), dpi = 400,bbox_inches="tight")



fig, ax = plt.subplots(1,4,figsize=(15,5))
sns.scatterplot(data=decoder_pd, y='mae_no_edges_umap', x='num_cells', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[0])
slope, intercept, r_value, p_value, std_err = stats.linregress(decoder_pd["num_cells"].to_list(),decoder_pd["mae_no_edges_umap"].to_list())
s = stats.pearsonr(decoder_pd["num_cells"].to_list(),decoder_pd["mae_no_edges_umap"].to_list())
ax[0].plot([decoder_pd["num_cells"].min(),decoder_pd["num_cells"].max()], 
    [slope*decoder_pd["num_cells"].min()+intercept, slope*decoder_pd["num_cells"].max()+intercept], 'k--')
ax[0].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

sns.scatterplot(data=decoder_pd, y='mae_no_edges_umap', x='mi', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(decoder_pd["mi"].to_list(),decoder_pd["mae_no_edges_umap"].to_list())
s = stats.pearsonr(decoder_pd["mi"].to_list(),decoder_pd["mae_no_edges_umap"].to_list())
ax[1].plot([decoder_pd["mi"].min(),decoder_pd["mi"].max()], 
    [slope*decoder_pd["mi"].min()+intercept, slope*decoder_pd["mi"].max()+intercept], 'k--')
ax[1].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

sns.scatterplot(data=decoder_pd, y='mae_no_edges_umap', x='h1_lifetime', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[2])
slope, intercept, r_value, p_value, std_err = stats.linregress(decoder_pd["h1_lifetime"].to_list(),decoder_pd["mae_no_edges_umap"].to_list())
s = stats.pearsonr(decoder_pd["h1_lifetime"].to_list(),decoder_pd["mae_no_edges_umap"].to_list())
ax[2].plot([decoder_pd["h1_lifetime"].min(),decoder_pd["h1_lifetime"].max()], 
    [slope*decoder_pd["h1_lifetime"].min()+intercept, slope*decoder_pd["h1_lifetime"].max()+intercept], 'k--')
ax[2].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

sns.scatterplot(data=decoder_pd, y='mae_no_edges_umap', x='eccentricity', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[3])
slope, intercept, r_value, p_value, std_err = stats.linregress(decoder_pd["eccentricity"].to_list(),decoder_pd["mae_no_edges_umap"].to_list())
s = stats.pearsonr(decoder_pd["eccentricity"].to_list(),decoder_pd["mae_no_edges_umap"].to_list())
ax[3].plot([decoder_pd["eccentricity"].min(),decoder_pd["eccentricity"].max()], 
    [slope*decoder_pd["eccentricity"].min()+intercept, slope*decoder_pd["eccentricity"].max()+intercept], 'k--')
ax[3].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

plt.savefig(os.path.join(decoders_dir,'deepsup_decoderx_scatter_no_edges.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(decoders_dir,'deepsup_decoderx_scatter_no_edges.png'), dpi = 400,bbox_inches="tight")








for label_idx, label_name in enumerate(label_list):

    fig, ax = plt.subplots(1,4,figsize=(15,5))
    for dec_idx, dec_name in enumerate(decoder_list):
        R2s_list = list()
        pd_mice_list = list()
        pd_sig_list = list()
        pd_layer_list = list()
        for mouse in miceList:
            for sig in signal_list:
                R2s_list.append(np.mean(dec_R2s[mouse][sig][dec_name][:,label_idx,0], axis=0))
                pd_mice_list.append(mouse)
                pd_sig_list.append(sig)
                if mouse in deepMice:
                    pd_layer_list.append('deep')
                elif mouse in supMice:
                    pd_layer_list.append('sup')

        pd_R2s = pd.DataFrame(data={'mouse': pd_mice_list,
                                     'R2s': R2s_list,
                                     'signal': pd_sig_list,
                                     'layer': pd_layer_list})



temp_pd = decoder_pd[decoder_pd["mouse"]!="GC5_nvista"]
temp_pd = temp_pd[temp_pd["mouse"]!="TGrin1"]
temp_pd = temp_pd[temp_pd["mouse"]!="ChZ4"]
temp_pd = temp_pd[temp_pd["mouse"]!="CZ3"]
temp_pd = temp_pd[temp_pd["mouse"]!="CZ9"]


fig, ax = plt.subplots(1,4,figsize=(15,5))
sns.scatterplot(data=temp_pd, y='mae_umap', x='h1_lifetime', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[0])
slope, intercept, r_value, p_value, std_err = stats.linregress(temp_pd["h1_lifetime"].to_list(),temp_pd["mae_umap"].to_list())
s = stats.pearsonr(temp_pd["h1_lifetime"].to_list(),temp_pd["mae_umap"].to_list())
ax[0].plot([temp_pd["h1_lifetime"].min(),temp_pd["h1_lifetime"].max()], 
    [slope*temp_pd["h1_lifetime"].min()+intercept, slope*temp_pd["h1_lifetime"].max()+intercept], 'k--')
ax[0].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')

sns.scatterplot(data=temp_pd, y='mae_umap', x='h1_lifetime', hue='layer',
    palette = palette_deepsup, style='strain', ax= ax[0])
slope, intercept, r_value, p_value, std_err = stats.linregress(temp_pd["h1_lifetime"].to_list(),temp_pd["mae_umap"].to_list())
s = stats.pearsonr(temp_pd["h1_lifetime"].to_list(),temp_pd["mae_umap"].to_list())
ax[0].plot([temp_pd["h1_lifetime"].min(),temp_pd["h1_lifetime"].max()], 
    [slope*temp_pd["h1_lifetime"].min()+intercept, slope*temp_pd["h1_lifetime"].max()+intercept], 'k--')
ax[0].set_title(f'pearsonr: stat={s[0]:.4f} p={s[1]:.4f}')