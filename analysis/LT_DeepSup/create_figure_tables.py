

from os.path import join
import pandas as pd
import numpy as np


def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

save_dir = '/home/julio/Documents/DeepSup_project/'


#############################################################################
#FIG 1: PRE SESSION Calb/Thy1/Chrna7
    #num cells / mutual information (posx, dir, vel)/ event rate / place cells
#############################################################################

base_dir = '/home/julio/Documents/SP_project/Fig2/'
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
mi_scores_dict = load_pickle(join(base_dir,'mutual_info'), 'mi_scores_sep_dict.pkl')
firing_rates_dict =load_pickle(join(base_dir,'firing_rates'), 'firing_rates_dict.pkl')


mouse_name_list = []
strain_list = []
layer_list = []
num_cells_list = []
mi_posx_list = []
mi_dir_list = []
mi_vel_list = []
event_rate_list = []
place_cells_list = []

for mouse in mice_list:

    #general info
    mouse_name_list.append(mouse)
    if ('CZ' in mouse) or ('CGrin' in mouse):
        strain_list.append('Calb')        
        layer_list.append('Sup')
    elif ('GC' in mouse) or ('TGrin' in mouse):
        strain_list.append('Thy1')
        layer_list.append('Deep')
    elif 'ChZ' in mouse:
        strain_list.append('Chrna7')
        layer_list.append('Deep')

    #num cells
    mouse_pd = load_pickle(join(base_dir, 'processed_data', mouse), mouse+'_df_dict.pkl')
    num_cells_list.append(mouse_pd['clean_traces'][0].shape[1])

    #mutual info
    mouse_mi_scores = mi_scores_dict[mouse]['mi_scores']

    mi_posx_list.append(np.nanmean(mouse_mi_scores[0,:]))
    mi_dir_list.append(np.nanmean(mouse_mi_scores[2,:]))
    mi_vel_list.append(np.nanmean(mouse_mi_scores[3,:]))

    #event rate
    event_rate_list.append(np.nanmean(firing_rates_dict[mouse]))

    #place cells (if applicable)
    if 'GC5_nvista' in mouse:
        place_cells_list.append(np.nan)
    else:
        place_cells = np.load(join(base_dir, 'functional_cells', mouse, mouse+'_cellType.npy'))
        place_cells_list.append(np.sum(place_cells<4)/place_cells.shape[0])


fig1_table = pd.DataFrame(data={
                            'mouse': mouse_name_list,
                            'strain': strain_list,
                            'layer': layer_list,

                            'num_cells': num_cells_list,

                            'mi_posx': mi_posx_list,
                            'mi_dir': mi_dir_list,
                            'mi_vel': mi_vel_list,

                            'event_rate': event_rate_list,

                            'place_cells': place_cells_list,
                             })


#############################################################################
#FIG 2: PRE SESSION Calb/Thy1/Chrna7
    #num cells / inner dim ABIDS (50, 100, 150)/ SI (og dir, umap dir, og posx, 
    #umap posx) / life time (H1, H2) / eccentricity
#############################################################################

base_dir = '/home/julio/Documents/SP_project/Fig2/'
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

id_dict = load_pickle(join(base_dir, 'bin_size'), 'inner_dim_dict.pkl')
si_dict = load_pickle(join(base_dir, 'SI'), 'sI_perc_dict.pkl')
ellipse_dict = load_pickle(join(base_dir, 'eccentricity'), 'ellipse_fit_dict.pkl')


mouse_name_list = []
strain_list = []
layer_list = []
num_cells_list = []
id_50ms_list = []
id_100ms_list = []
id_150ms_list = []
si_og_dir_list = []
si_umap_dir_list = []
si_og_posx_list = []
si_umap_posx_list = []
lifetime_h1_list = []
lifetime_h2_list = []
eccentricity_list = []



for mouse in mice_list:
    #general info
    mouse_name_list.append(mouse)
    if ('CZ' in mouse) or ('CGrin' in mouse):
        strain_list.append('Calb')        
        layer_list.append('Sup')
    elif ('GC' in mouse) or ('TGrin' in mouse):
        strain_list.append('Thy1')
        layer_list.append('Deep')
    elif 'ChZ' in mouse:
        strain_list.append('Chrna7')
        layer_list.append('Deep')

    #num cells
    mouse_pd = load_pickle(join(base_dir, 'processed_data', mouse), mouse+'_df_dict.pkl')
    num_cells_list.append(mouse_pd['clean_traces'][0].shape[1])

    #inner dim
    id_50ms_list.append(id_dict[mouse][1]['abidsDim'])
    id_100ms_list.append(id_dict[mouse][2]['abidsDim'])
    id_150ms_list.append(id_dict[mouse][3]['abidsDim'])

    #SI
    si_og_posx_list.append(si_dict[mouse]['clean_traces']['results']['pos']['sI'][2])
    si_og_dir_list.append(si_dict[mouse]['clean_traces']['results']['dir']['sI'][2])

    si_umap_posx_list.append(si_dict[mouse]['umap']['results']['pos']['sI'][2])
    si_umap_dir_list.append(si_dict[mouse]['umap']['results']['dir']['sI'][2])



    #LIFE TIME
    betti_dict = load_pickle(join(base_dir,'betti_numbers', 'og_mi_cells', mouse), mouse+'_betti_dict_og.pkl')
    try:
        dense_conf_interval1 = betti_dict['dense_conf_interval'][1]
        dense_conf_interval2 = betti_dict['dense_conf_interval'][2]
    except:
        dense_conf_interval1 = 0
        dense_conf_interval2 = 0
    h1_diagrams = np.array(betti_dict['dense_diagrams'][1])
    h1_length = np.sort(np.diff(h1_diagrams, axis=1)[:,0])
    second_length = np.max([dense_conf_interval1, h1_length[-2]])
    lifetime_h1_list.append(h1_length[-1]-second_length)


    h2_diagrams = np.array(betti_dict['dense_diagrams'][2])
    h2_length = np.sort(np.diff(h2_diagrams, axis=1)[:,0])
    second_length = np.max([dense_conf_interval2, h2_length[-2]])
    lifetime_h2_list.append(h2_length[-1]-second_length)

    #ECCENTRICITY
    pos_length = ellipse_dict[mouse]['posLength']
    dir_length = ellipse_dict[mouse]['dirLength']
    eccentricity_list.append(100*(pos_length-dir_length)/(pos_length))



fig2_table = pd.DataFrame(data={
                            'mouse': mouse_name_list,
                            'strain': strain_list,
                            'layer': layer_list,

                            'num_cells': num_cells_list,

                            'id_50ms': id_50ms_list,
                            'id_100ms': id_100ms_list,
                            'id_150ms': id_150ms_list,

                            'si_og_posx': si_og_posx_list,
                            'si_umap_posx': si_umap_posx_list,
                            'si_og_dir': si_og_dir_list,
                            'si_umap_dir': si_umap_dir_list,


                            'lifetime_h1': lifetime_h1_list,
                            'lifetime_h2': lifetime_h2_list,

                            'eccentricity':eccentricity_list
                             })




#############################################################################
#FIG 3: PRE/ROT SESSION Calb/Thy1/Chrna7
    #num cells / inner dim ABIDS (50, 100, 150)/ SI (og dir, umap dir, og posx, 
    #umap posx) / life time (H1, H2) / eccentricity
#############################################################################
base_dir = '/home/julio/Documents/SP_project/LT_DeepSup/'
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']


def process_axis(ax):
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_aspect('equal', adjustable='box')


for mouse in mice_list:
    #general info
    mouse_name_list.append(mouse)
    if ('CZ' in mouse) or ('CGrin' in mouse):
        strain_list.append('Calb')        
        layer_list.append('Sup')
    elif ('GC' in mouse) or ('TGrin' in mouse):
        strain_list.append('Thy1')
        layer_list.append('Deep')
    elif 'ChZ' in mouse:
        strain_list.append('Chrna7')
        layer_list.append('Deep')

    mouse_pd = load_pickle(join(base_dir, 'processed_data', mouse), mouse+'_df_dict.pkl')
    fnames = list(mouse_pd.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]
    pre_session= copy.deepcopy(mouse_pd[fname_pre])
    rot_session= copy.deepcopy(mouse_pd[fname_rot])

    #time
    pos_pre = np.concatenate(pre_session['pos'].values, axis = 0)
    dir_pre = np.concatenate(pre_session['dir_mat'].values, axis=0)
    dir_pre[dir_pre==1] = -1
    dir_pre[dir_pre==2] = 1
    umap_pre = np.concatenate(pre_session['isomap'].values, axis=0)
    trial_pre = np.concatenate(pre_session['index_mat'].values, axis=0)

    pos_rot = np.concatenate(rot_session['pos'].values, axis = 0)
    dir_rot = np.concatenate(rot_session['dir_mat'].values, axis=0)
    dir_rot[dir_rot==1] = -1
    dir_rot[dir_rot==2] = 1
    umap_rot = np.concatenate(rot_session['isomap'].values, axis=0)
    trial_rot = np.concatenate(rot_session['index_mat'].values, axis=0)


    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(1,3,1, projection = '3d')
    ax.scatter(*umap_pre[:,:3].T, color ='b', s= 10)
    ax.scatter(*umap_rot[:,:3].T, color = 'r', s= 10)
    process_axis(ax)

    ax = plt.subplot(1,3,2, projection = '3d')
    ax.scatter(*umap_pre[:,:3].T, c = pos_pre[:,0], s= 10, cmap = 'viridis')
    ax.scatter(*umap_rot[:,:3].T, c = pos_rot[:,0], s= 10, cmap = 'magma')
    process_axis(ax)

    ax = plt.subplot(1,3,3, projection = '3d')
    ax.scatter(*umap_pre[:,:3].T, c = trial_pre, s= 10, cmap = 'YlGn_r')
    ax.scatter(*umap_rot[:,:3].T, c = trial_rot, s= 10, cmap = 'YlGn_r')
    process_axis(ax)

    plt.suptitle(mouse)

base_dir = '/home/julio/Documents/DeepSup_project/'
dreadds_mice = ['CalbCharly2', 'CalbCharly11_concat', 'CalbV23', 'DD2','ChRNA7Charly1', 'ChRNA7Charly2']

dreadds_dir = join(base_dir, 'DREADDs')


for mouse in dreadds_mice:

    for condition in ['veh', 'CNO']:
        if mouse in Calb_mice:
            mouse_dir = join(dreadds_dir, 'Calb', 'processed_data', mouse+'_'+condition)
        elif mouse in ChRNA7_mice:
            mouse_dir = join(dreadds_dir, 'ChRNA7', 'processed_data', mouse+'_'+condition)


        mouse_pd = load_pickle(mouse_dir, mouse+'_'+condition+'_df_dict.pkl')
        session = list(mouse_pd.keys())[0]
        num_cells = mouse_pd[session]['clean_traces'][0].shape[1]
        print(f"{mouse} {condition} - {num_cells}")


    for condition in ['veh', 'CNO']: