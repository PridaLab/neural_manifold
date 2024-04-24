
supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']
palette_deepsup = ["#cc9900ff", "#9900ffff"]
palette_dual = ["gray"]+palette_deepsup

model_dir = '/home/julio/Documents/SP_project/LT_DeepSup/model/exploration/010same_010xmirror_010remap_040remapnofield_030nofield_010noise'
model_dict = load_pickle(model_dir, 'statistic_model_results.pkl')
mode_params = (0.2, 0.1)
rot_angles = model_dict[mode_params]['rot_angles']
local_prob_list  = model_dict[mode_params]['local_prob_list']

no_field_prob = model_dict[mode_params]['field_type_probs']['no_field_prob']*(1-model_dict[mode_params]['cell_type_probs']['remap_prob'])

model_types = {
    'allocentric': local_prob_list[::-1]*(1-no_field_prob),
    'local-cue-anchored': local_prob_list*(1-no_field_prob),
    'remap': local_prob_list*0 + (model_dict[mode_params]['cell_type_probs']['remap_prob'] +model_dict[mode_params]['cell_type_probs']['remap_no_field_prob'])*(1-no_field_prob),
    'N/A': local_prob_list*0 + no_field_prob
}

model_types_place_cells = {
    'allocentric': local_prob_list[::-1],
    'local-cue-anchored': local_prob_list,
    'remap': local_prob_list*0 + (model_dict[mode_params]['cell_type_probs']['remap_prob'] +model_dict[mode_params]['cell_type_probs']['remap_no_field_prob']),
}


mean_rot_model = np.nanmean(rot_angles, axis=1)
std_rot_model = np.nanstd(rot_angles, axis=1)
#__________________________________________________________________________
#|                                                                        |#
#|                        FUNCTIONAL CLASSIFICATION                       |#
#|________________________________________________________________________|#

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

func_cells_dir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
rotation_dir = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'


remap_dist_dict = load_pickle(rotation_dir, 'remap_distance_dict.pkl')
rotation_dict = load_pickle(rotation_dir, 'rot_error_dict.pkl')


perc_list = list()
type_list = list()
layer_list = list()
name_list = list()
angle_list = list()
dist_list = list()
for mouse in mice_list:

    file_name =  mouse+'_cellType.npy'
    file_path = os.path.join(func_cells_dir, mouse)
    cell_type = np.load(os.path.join(file_path, file_name))
    num_cells = cell_type.shape[0]
    static_cells = np.where(cell_type==0)[0].shape[0]
    perc_list.append(static_cells/num_cells)
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    perc_list.append(rot_cells/num_cells)
    remap_cells = np.where(cell_type==4)[0].shape[0]
    perc_list.append(remap_cells/num_cells)
    na_cells = np.where(cell_type==5)[0].shape[0]
    perc_list.append(na_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap', 'N/A']
    if mouse in deepMice:
        layer_list += ['deep']*4
    elif mouse in supMice:
        layer_list += ['sup']*4
    name_list += [mouse]*4

    angle_rot = rotation_dict[mouse]['umap']['rotAngle']
    angle_list += [angle_rot]*4

    remap_dist = remap_dist_dict[mouse]['umap']['remapDist']
    dist_list += [remap_dist]*4


cells_pd = pd.DataFrame(data={'type': type_list,
                            'layer': layer_list,
                            'mouse': name_list,
                            'percentage': perc_list,
                            'rotation': angle_list,
                            'distance': dist_list})


fig, ax = plt.subplots(2,2, figsize=(10,6))
for idx, cell_type in enumerate(cells_pd['type'].unique()):

    c = idx%2
    r = idx//2
    sub_pd = cells_pd.loc[cells_pd['type']==cell_type]
    sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
        palette= palette_deepsup, ax= ax[r,c])

    ax[r,c].plot(model_types[cell_type], mean_rot_model, color = 'k')
    ax[r,c].fill_between(model_types[cell_type], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
    ax[r,c].set_title(cell_type)
    ax[r,c].set_ylim([0, 180])
    ax[r,c].set_xlim([0,0.65])
plt.suptitle('Deep/Sup % of all cells')
plt.tight_layout()
plt.savefig(os.path.join(func_cells_dir,'func_cells_rot_scatter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(func_cells_dir,'func_cells_rot_scatter.png'), dpi = 400,bbox_inches="tight")

fig, ax = plt.subplots(2,2, figsize=(10,6))
for idx, cell_type in enumerate(cells_pd['type'].unique()):

    c = idx%2
    r = idx//2
    sub_pd = cells_pd.loc[cells_pd['type']==cell_type]
    sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
        palette= palette_deepsup, ax= ax[r,c])

    # ax[r,c].plot(model_types[cell_type], m, color = 'k')
    # ax[r,c].fill_between(model_types[cell_type], m-sd, m+sd, color = 'k', alpha = .1)
    ax[r,c].set_title(cell_type)
    ax[r,c].set_ylim([0, 1.8])
    ax[r,c].set_xlim([0,0.65])
plt.tight_layout()
plt.suptitle('Deep/Sup % of all cells')

plt.savefig(os.path.join(func_cells_dir,'func_cells_dist_scatter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(func_cells_dir,'func_cells_dist_scatter.png'), dpi = 400,bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|              FUNCTIONAL CLASSIFICATION ONLY PLACE CELLLS               |#
#|________________________________________________________________________|#


func_cells_dir = '/home/julio/Documents/SP_project/LT_DeepSup/functional_cells/'
rotation_dir = '/home/julio/Documents/SP_project/LT_DeepSup/rotation/'
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']



model_dir = '/home/julio/Documents/SP_project/LT_DeepSup/model/exploration/010same_010xmirror_010remap_040remapnofield_030nofield_010noise'
model_dict = load_pickle(model_dir, 'statistic_model_results.pkl')
mode_params = (0.2, 0.1)
rot_angles = model_dict[mode_params]['rot_angles']
local_prob_list  = model_dict[mode_params]['local_prob_list']



remap_dist_dict = load_pickle(rotation_dir, 'remap_distance_dict.pkl')
rotation_dict = load_pickle(rotation_dir, 'rot_error_dict.pkl')


perc_list = list()
type_list = list()
layer_list = list()
name_list = list()
angle_list = list()
dist_list = list()
for mouse in mice_list:

    file_name =  mouse+'_cellType.npy'
    file_path = os.path.join(func_cells_dir, mouse)
    cell_type = np.load(os.path.join(file_path, file_name))

    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]

    num_cells = static_cells+rot_cells+remap_cells
    perc_list.append(static_cells/num_cells)
    perc_list.append(rot_cells/num_cells)
    perc_list.append(remap_cells/num_cells)


    type_list += ['allocentric', 'local-cue-anchored','remap']
    if mouse in deepMice:
        layer_list += ['deep']*3
    elif mouse in supMice:
        layer_list += ['sup']*3
    name_list += [mouse]*3

    angle_rot = rotation_dict[mouse]['umap']['rotAngle']
    angle_list += [angle_rot]*3

    remap_dist = remap_dist_dict[mouse]['umap']['remapDist']
    dist_list += [remap_dist]*3


cells_pd = pd.DataFrame(data={'type': type_list,
                            'layer': layer_list,
                            'mouse': name_list,
                            'percentage': perc_list,
                            'rotation': angle_list,
                            'distance': dist_list})


fig, ax = plt.subplots(2,2, figsize=(10,6))
for idx, cell_type in enumerate(cells_pd['type'].unique()):
    c = idx%2
    r = idx//2
    sub_pd = cells_pd.loc[cells_pd['type']==cell_type]
    sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
        palette= palette_deepsup, ax= ax[r,c])

    ax[r,c].plot(model_types_place_cells[cell_type], mean_rot_model, color = 'k')
    ax[r,c].fill_between(model_types_place_cells[cell_type], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)

    ax[r,c].set_title(cell_type)
    ax[r,c].set_ylim([0, 180])
    ax[r,c].set_xlim([0,0.65])
plt.tight_layout()
plt.suptitle('Deep/Sup % of place cells')

plt.savefig(os.path.join(func_cells_dir,'func_cells_rot_scatter_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(func_cells_dir,'func_cells_rot_scatter_place_cells.png'), dpi = 400,bbox_inches="tight")



fig, ax = plt.subplots(2,2, figsize=(10,6))
for idx, cell_type in enumerate(cells_pd['type'].unique()):

    c = idx%2
    r = idx//2
    sub_pd = cells_pd.loc[cells_pd['type']==cell_type]
    sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
        palette= palette_deepsup, ax= ax[r,c])

    ax[r,c].set_title(cell_type)
    ax[r,c].set_ylim([0, 1.8])
    ax[r,c].set_xlim([0,0.65])
plt.tight_layout()
plt.suptitle('Deep/Sup % of place cells')

plt.savefig(os.path.join(func_cells_dir,'func_cells_dist_scatter_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(func_cells_dir,'func_cells_dist_scatter_place_cells.png'), dpi = 400,bbox_inches="tight")

#__________________________________________________________________________
#|                                                                        |#
#|                                DUAL COLOR                              |#
#|________________________________________________________________________|#

base_dir = '/home/julio/Documents/DeepSup_project/DualColor/'
mice_list = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']

jRGECO_dir = os.path.join(base_dir,'Thy1jRGECO')
RCaMP_dir = os.path.join(base_dir,'ThyCalbRCaMP')
save_dir = os.path.join(base_dir,'figures')

model_dir = '/home/julio/Documents/SP_project/LT_DeepSup/model/exploration/010same_010xmirror_010remap_040remapnofield_030nofield_010noise'
model_dict = load_pickle(model_dir, 'statistic_model_results.pkl')
mode_params = (0.2, 0.1)
rot_angles = model_dict[mode_params]['rot_angles']
local_prob_list  = model_dict[mode_params]['local_prob_list']

perc_list = list()
type_list = list()
layer_list = list()
name_list = list()
angle_list = list()
dist_list = list()

for mouse in mice_list:
    if 'Thy1jRGECO' in mouse:
        rotation_dir = os.path.join(jRGECO_dir,'rotation', mouse)
        distance_dir = os.path.join(jRGECO_dir,'distance', mouse)
        functional_dir =os.path.join(jRGECO_dir,'functional_cells', mouse)

    else:
        rotation_dir = os.path.join(RCaMP_dir,'rotation', mouse)
        distance_dir = os.path.join(RCaMP_dir,'distance', mouse)
        functional_dir =os.path.join(RCaMP_dir,'functional_cells', mouse)


    rotation_dict = load_pickle(rotation_dir, mouse+'_rotation_dict.pkl')
    distance_dict = load_pickle(distance_dir, mouse+'_distance_dict.pkl')

    deep_rot = rotation_dict['deep']['deep_rotation_angle']
    sup_rot = rotation_dict['sup']['sup_rotation_angle']
    all_rot = rotation_dict['all']['all_rotation_angle']

    deep_dist = distance_dict['deep']['plane_deep_remap_dist']
    sup_dist = distance_dict['sup']['plane_sup_remap_dist']
    all_dist = distance_dict['all']['plane_all_remap_dist']

    #all cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_all_cellType.npy'))
    num_cells =cell_type.shape[0]
    static_cells = np.where(cell_type==0)[0].shape[0]
    perc_list.append(static_cells/num_cells)
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    perc_list.append(rot_cells/num_cells)
    remap_cells = np.where(cell_type==4)[0].shape[0]
    perc_list.append(remap_cells/num_cells)
    na_cells = np.where(cell_type==5)[0].shape[0]
    perc_list.append(na_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap', 'N/A']
    layer_list += ['all']*4
    name_list += [mouse]*4
    angle_list += [all_rot]*4
    dist_list += [all_dist]*4

    #deep cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_deep_cellType.npy'))
    num_cells =cell_type.shape[0]
    static_cells = np.where(cell_type==0)[0].shape[0]
    perc_list.append(static_cells/num_cells)
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    perc_list.append(rot_cells/num_cells)
    remap_cells = np.where(cell_type==4)[0].shape[0]
    perc_list.append(remap_cells/num_cells)
    na_cells = np.where(cell_type==5)[0].shape[0]
    perc_list.append(na_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap', 'N/A']
    layer_list += ['deep']*4
    name_list += [mouse]*4
    angle_list += [deep_rot]*4
    dist_list += [deep_dist]*4

    #sup cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_sup_cellType.npy'))
    num_cells =cell_type.shape[0]
    static_cells = np.where(cell_type==0)[0].shape[0]
    perc_list.append(static_cells/num_cells)
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    perc_list.append(rot_cells/num_cells)
    remap_cells = np.where(cell_type==4)[0].shape[0]
    perc_list.append(remap_cells/num_cells)
    na_cells = np.where(cell_type==5)[0].shape[0]
    perc_list.append(na_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap', 'N/A']
    layer_list += ['sup']*4
    name_list += [mouse]*4
    angle_list += [sup_rot]*4
    dist_list += [sup_dist]*4


cells_pd = pd.DataFrame(data={'type': type_list,
                            'layer': layer_list,
                            'mouse': name_list,
                            'percentage': perc_list,
                            'rotation': angle_list,
                            'distance': dist_list})


fig, ax = plt.subplots(2,2, figsize=(10,6))
for idx, cell_type in enumerate(cells_pd['type'].unique()):

    c = idx%2
    r = idx//2
    sub_pd = cells_pd.loc[cells_pd['type']==cell_type]
    sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
        palette= palette_dual, style= 'mouse', ax= ax[r,c])

    ax[r,c].plot(model_types[cell_type], mean_rot_model, color = 'k')
    ax[r,c].fill_between(model_types[cell_type], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
    ax[r,c].set_title(cell_type)
    ax[r,c].set_ylim([0, 180])
    ax[r,c].set_xlim([0,0.7])
plt.tight_layout()
plt.suptitle('Dual Color % of all cells')

plt.savefig(os.path.join(save_dir,'func_cells_rot_scatter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'func_cells_rot_scatter.png'), dpi = 400,bbox_inches="tight")

fig, ax = plt.subplots(2,2, figsize=(10,6))
for idx, cell_type in enumerate(cells_pd['type'].unique()):

    c = idx%2
    r = idx//2
    sub_pd = cells_pd.loc[cells_pd['type']==cell_type]
    sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
        palette= palette_dual, style= 'mouse', ax= ax[r,c])

    ax[r,c].set_title(cell_type)
    ax[r,c].set_ylim([0, 3])
    ax[r,c].set_xlim([0,0.65])
plt.tight_layout()
plt.suptitle('Dual Color % of all cells')


plt.savefig(os.path.join(save_dir,'func_cells_dist_scatter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'func_cells_dist_scatter.png'), dpi = 400,bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|                    DUAL COLOR ONLY PLACE CELLLS                        |#
#|________________________________________________________________________|#

base_dir = '/home/julio/Documents/DeepSup_project/DualColor/'
mice_list = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']

jRGECO_dir = os.path.join(base_dir,'Thy1jRGECO')
RCaMP_dir = os.path.join(base_dir,'ThyCalbRCaMP')
save_dir = os.path.join(base_dir,'figures')


model_dir = '/home/julio/Documents/SP_project/LT_DeepSup/model/exploration/010same_010xmirror_010remap_040remapnofield_030nofield_010noise'
model_dict = load_pickle(model_dir, 'statistic_model_results.pkl')
mode_params = (0.2, 0.1)
rot_angles = model_dict[mode_params]['rot_angles']
local_prob_list  = model_dict[mode_params]['local_prob_list']

perc_list = list()
type_list = list()
layer_list = list()
name_list = list()
angle_list = list()
dist_list = list()

for mouse in mice_list:
    if 'Thy1jRGECO' in mouse:
        rotation_dir = os.path.join(jRGECO_dir,'rotation', mouse)
        distance_dir = os.path.join(jRGECO_dir,'distance', mouse)
        functional_dir =os.path.join(jRGECO_dir,'functional_cells', mouse)

    else:
        rotation_dir = os.path.join(RCaMP_dir,'rotation', mouse)
        distance_dir = os.path.join(RCaMP_dir,'distance', mouse)
        functional_dir =os.path.join(RCaMP_dir,'functional_cells', mouse)


    rotation_dict = load_pickle(rotation_dir, mouse+'_rotation_dict.pkl')
    distance_dict = load_pickle(distance_dir, mouse+'_distance_dict.pkl')

    deep_rot = rotation_dict['deep']['deep_rotation_angle']
    sup_rot = rotation_dict['sup']['sup_rotation_angle']
    all_rot = rotation_dict['all']['all_rotation_angle']

    deep_dist = distance_dict['deep']['plane_deep_remap_dist']
    sup_dist = distance_dict['sup']['plane_sup_remap_dist']
    all_dist = distance_dict['all']['plane_all_remap_dist']

    #all cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_all_cellType.npy'))
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]

    num_cells = static_cells+rot_cells+remap_cells
    perc_list.append(static_cells/num_cells)
    perc_list.append(rot_cells/num_cells)
    perc_list.append(remap_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap']
    layer_list += ['all']*3
    name_list += [mouse]*3
    angle_list += [all_rot]*3
    dist_list += [all_dist]*3

    #deep cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_deep_cellType.npy'))
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]

    num_cells = static_cells+rot_cells+remap_cells
    perc_list.append(static_cells/num_cells)
    perc_list.append(rot_cells/num_cells)
    perc_list.append(remap_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap']
    layer_list += ['deep']*3
    name_list += [mouse]*3
    angle_list += [deep_rot]*3
    dist_list += [deep_dist]*3

    #sup cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_sup_cellType.npy'))
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]

    num_cells = static_cells+rot_cells+remap_cells
    perc_list.append(static_cells/num_cells)
    perc_list.append(rot_cells/num_cells)
    perc_list.append(remap_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap']
    layer_list += ['sup']*3
    name_list += [mouse]*3
    angle_list += [sup_rot]*3
    dist_list += [sup_dist]*3


cells_pd = pd.DataFrame(data={'type': type_list,
                            'layer': layer_list,
                            'mouse': name_list,
                            'percentage': perc_list,
                            'rotation': angle_list,
                            'distance': dist_list})

m = np.nanmean(rot_angles, axis=1)
sd = np.nanstd(rot_angles, axis=1)


fig, ax = plt.subplots(2,2, figsize=(10,6))
for idx, cell_type in enumerate(cells_pd['type'].unique()):

    c = idx%2
    r = idx//2
    sub_pd = cells_pd.loc[cells_pd['type']==cell_type]
    sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
        palette= palette_dual, style= 'mouse', ax= ax[r,c])

    ax[r,c].plot(model_types_place_cells[cell_type], mean_rot_model, color = 'k')
    ax[r,c].fill_between(model_types_place_cells[cell_type], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
    ax[r,c].set_title(cell_type)
    ax[r,c].set_ylim([0, 180])
    ax[r,c].set_xlim([0,0.7])
plt.tight_layout()
plt.suptitle('Dual Color % of place cells')

plt.savefig(os.path.join(save_dir,'func_cells_rot_scatter_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'func_cells_rot_scatter_place_cells.png'), dpi = 400,bbox_inches="tight")


fig, ax = plt.subplots(2,2, figsize=(10,6))
for idx, cell_type in enumerate(cells_pd['type'].unique()):

    c = idx%2
    r = idx//2
    sub_pd = cells_pd.loc[cells_pd['type']==cell_type]
    sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
        palette= palette_dual, style= 'mouse', ax= ax[r,c])

    ax[r,c].set_title(cell_type)
    ax[r,c].set_ylim([0, 3])
    ax[r,c].set_xlim([0,0.65])
plt.suptitle('Dual Color % of place cells')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'func_cells_dist_scatter_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'func_cells_dist_scatter_place_cells.png'), dpi = 400,bbox_inches="tight")





#__________________________________________________________________________
#|                                                                        |#
#|                            ONLY  MODEL                                 |#
#|________________________________________________________________________|#

save_dir = '/home/julio/Documents/DeepSup_project/model/figures/'

###################### ROTATION
model_dir = '/home/julio/Documents/SP_project/LT_DeepSup/model/exploration/010same_010xmirror_010remap_040remapnofield_030nofield_010noise'
model_dict = load_pickle(model_dir, 'statistic_model_results.pkl')
mode_params = (0.2, 0.1)
rot_angles = model_dict[mode_params]['rot_angles']
local_prob_list  = model_dict[mode_params]['local_prob_list']

no_field_prob = model_dict[mode_params]['field_type_probs']['no_field_prob']*(1-model_dict[mode_params]['cell_type_probs']['remap_prob'])
model_types = {
    'allocentric': local_prob_list[::-1]*(1-no_field_prob),
    'local-cue-anchored': local_prob_list*(1-no_field_prob),
    'remap': local_prob_list*0 + (model_dict[mode_params]['cell_type_probs']['remap_prob'] +model_dict[mode_params]['cell_type_probs']['remap_no_field_prob'])*(1-no_field_prob),
    'N/A': local_prob_list*0 + no_field_prob
}

model_types_place_cells = {
    'allocentric': local_prob_list[::-1],
    'local-cue-anchored': local_prob_list,
    'remap': local_prob_list*0 + (model_dict[mode_params]['cell_type_probs']['remap_prob'] +model_dict[mode_params]['cell_type_probs']['remap_no_field_prob']),
}


mean_rot_model = np.nanmean(rot_angles, axis=1)
std_rot_model = np.nanstd(rot_angles, axis=1)


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.plot(model_types['local-cue-anchored'], mean_rot_model, color = 'k')
ax.fill_between(model_types['local-cue-anchored'], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.7])
ax.set_ylabel('Rotation')
ax.set_xlabel('Percentage of local-cue-anchored')
plt.savefig(os.path.join(save_dir,'model_rotation_localcue.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_rotation_localcue.png'), dpi = 400,bbox_inches="tight")


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.plot(model_types_place_cells['local-cue-anchored'], mean_rot_model, color = 'k')
ax.fill_between(model_types_place_cells['local-cue-anchored'], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.7])
ax.set_ylabel('Rotation')
ax.set_xlabel('Percentage of local-cue-anchored place cells')
plt.savefig(os.path.join(save_dir,'model_rotation_localcue_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_rotation_localcue_place_cells.png'), dpi = 400,bbox_inches="tight")


###################### DISTANCE

model_dir_remap = '/home/julio/Documents/SP_project/LT_DeepSup/model/remapping/010same_010xmirror_010remap_040remapnofield_030nofield_010noise/'
model_dict_remap = load_pickle(model_dir_remap, 'statistic_model_results.pkl')

case = 'allo'
mode_params = (0.1, 0.1)
remap_list_place_cells = 1 - model_dict_remap[case][mode_params]['local_prob_list'] - mode_params[0]
no_field_prob = model_dict_remap[case][mode_params]['field_type_probs']['no_field_prob']*(1-model_dict_remap[case][mode_params]['cell_type_probs']['remap_prob'])
remap_list = (1 - model_dict_remap[case][mode_params]['local_prob_list'] - mode_params[0])*(1 -no_field_prob)


distances_no_ellipse = model_dict_remap[case][mode_params]['rot_distances_no_ellipse']
mean_dist_model = np.nanmean(distances_no_ellipse, axis=1)
std_dist_model = np.nanstd(distances_no_ellipse, axis=1)


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.plot(remap_list, mean_dist_model, color = 'k')
ax.fill_between(remap_list, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.set_ylabel('distance mean')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_dist_mean_remap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dist_mean_remap.png'), dpi = 400,bbox_inches="tight")
slope, intercept, r_value, p_value, std_err = stats.linregress(remap_list,mean_dist_model)
stats.pearsonr(remap_list,mean_dist_model)

mean_ellipse_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
std_ellipse_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances'], axis=1)

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.plot(remap_list, mean_ellipse_dist_model, color = 'k')
ax.fill_between(remap_list, mean_ellipse_dist_model-std_ellipse_dist_model, mean_ellipse_dist_model+std_ellipse_dist_model, color = 'k', alpha = .1)
ax.set_ylabel('distance ellipse')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_dist_ellipse_remap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dist_ellipse_remap.png'), dpi = 400,bbox_inches="tight")


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.plot(remap_list_place_cells, mean_dist_model, color = 'k')
ax.fill_between(remap_list_place_cells, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.set_ylabel('distance mean')
ax.set_xlabel('Percentage of remap place cells')
plt.savefig(os.path.join(save_dir,'model_dist_mean_remap_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dist_mean_remap_place_cells.png'), dpi = 400,bbox_inches="tight")



mean_ellipse_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
std_ellipse_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances'], axis=1)

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.plot(remap_list_place_cells, mean_ellipse_dist_model, color = 'k')
ax.fill_between(remap_list_place_cells, mean_ellipse_dist_model-std_ellipse_dist_model, mean_ellipse_dist_model+std_ellipse_dist_model, color = 'k', alpha = .1)
ax.set_ylabel('distance ellipse')
ax.set_xlabel('Percentage of remap place cells')
plt.savefig(os.path.join(save_dir,'model_dist_ellipse_remap_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dist_ellipse_remap_place_cells.png'), dpi = 400,bbox_inches="tight")


###### PLOT MODEL EXAMPLES

num_cells = 400
noise = 0.1

field_type_probs = {
    'same_prob': 0.1,
    'xmirror_prob': 0.1,
    'remap_prob': 0.1,
    'remap_no_field_prob': 0.4,
    'no_field_prob': 0.3
}


mouse = 'GC2'
data_dir = '/home/julio/Documents/SP_project/LT_DeepSup/processed_data/'
file_path = os.path.join(data_dir, mouse)
animal_dict = load_pickle(file_path,mouse+'_df_dict.pkl')

fnames = list(animal_dict.keys())
fname_pre = [fname for fname in fnames if 'lt' in fname][0]
fname_rot = [fname for fname in fnames if 'rot' in fname][0]

pos = copy.deepcopy(np.concatenate(animal_dict[fname_pre]['pos'].values, axis=0))
real_dir_mat = copy.deepcopy(np.concatenate(animal_dict[fname_pre]['dir_mat'].values, axis=0))[:,0]
direction_mat = np.zeros((pos.shape[0],))*np.nan
direction_mat[real_dir_mat==1] = 0
direction_mat[real_dir_mat==2] = 1

rot_pos = copy.deepcopy(np.concatenate(animal_dict[fname_rot]['pos'].values, axis=0))
rot_real_dir_mat = copy.deepcopy(np.concatenate(animal_dict[fname_rot]['dir_mat'].values, axis=0))[:,0]
rot_direction_mat = np.zeros((rot_pos.shape[0],))*np.nan
rot_direction_mat[rot_real_dir_mat==1] = 0
rot_direction_mat[rot_real_dir_mat==2] = 1


cell_type_probs = {
    'local_anchored_prob': 0.0,
    'allo_prob': 0.7,
    'remap_prob': 0.2,
    'remap_no_field_prob': 0.1
}

model_test = StatisticModel(pos, direction_mat, rot_pos,rot_direction_mat, num_cells = num_cells)
model_test.compute_fields(**field_type_probs)
model_test.compute_cell_types(**cell_type_probs)
model_test.compute_rotation_fields()

model_test.compute_traces(noise_sigma = noise)
model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
model_test.clean_traces()

model_test.compute_umap_both()
model_test.clean_umap()

model_test.compute_rotation()
model_test.compute_distance()


def process_axis(ax):
    ax.set_xlabel('Dim 1', labelpad = -8)
    ax.set_ylabel('Dim 2', labelpad = -8)
    ax.set_zlabel('Dim 3', labelpad = -8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


fig = plt.figure()
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*model_test.nout_umap_emb_pre[:,:3].T, c = model_test.nout_position[:,0], s= 10, cmap = 'magma')
ax.scatter(*model_test.nout_umap_emb_rot[:,:3].T, c = model_test.nout_rot_position[:,0], s= 10, cmap = 'magma')
process_axis(ax)
plt.savefig(os.path.join(save_dir,'model_rotation_180_example.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_rotation_180_example.png'), dpi = 400,bbox_inches="tight")


fig = plt.figure()
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*model_test.nout_umap_emb_pre[:,:3].T, c = model_test.nout_position[:,0], s= 10, cmap = 'magma')
ax.scatter(*model_test.rotated_aligned_nout_umap_emb_rot[:,:3].T, c = model_test.nout_rot_position[:,0], s= 10, cmap = 'magma')
process_axis(ax)
plt.savefig(os.path.join(save_dir,'model_rotation_0_example.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_rotation_0_example.png'), dpi = 400,bbox_inches="tight")




cell_type_probs = {
    'local_anchored_prob': 0.8,
    'allo_prob': 0.2,
    'remap_prob': 0,
    'remap_no_field_prob': 0.0
}

model_test = StatisticModel(pos, direction_mat, rot_pos,rot_direction_mat, num_cells = num_cells)
model_test.compute_fields(**field_type_probs)
model_test.compute_cell_types(**cell_type_probs)
model_test.compute_rotation_fields()

model_test.compute_traces(noise_sigma = noise)
model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
model_test.clean_traces()

model_test.compute_umap_both()
model_test.clean_umap()

model_test.compute_rotation()
model_test.compute_distance()

fig = plt.figure()
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*model_test.nout_umap_emb_pre[:,:3].T, c = model_test.nout_position[:,0], s= 10, cmap = 'magma')
ax.scatter(*model_test.nout_umap_emb_rot[:,:3].T, c = model_test.nout_rot_position[:,0], s= 10, cmap = 'magma')
process_axis(ax)
plt.savefig(os.path.join(save_dir,'model_distance_0_example.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_distance_0_example.png'), dpi = 400,bbox_inches="tight")


cell_type_probs = {
    'local_anchored_prob': 0,
    'allo_prob': 0.8,
    'remap_prob': 0.05,
    'remap_no_field_prob': 0.15
}

model_test = StatisticModel(pos, direction_mat, rot_pos,rot_direction_mat, num_cells = num_cells)
model_test.compute_fields(**field_type_probs)
model_test.compute_cell_types(**cell_type_probs)
model_test.compute_rotation_fields()

model_test.compute_traces(noise_sigma = noise)
model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
model_test.clean_traces()

model_test.compute_umap_both()
model_test.clean_umap()

model_test.compute_rotation()
model_test.compute_distance()

model_test.remap_dist

fig = plt.figure()
ax = plt.subplot(1,1,1, projection = '3d')
ax.scatter(*model_test.nout_umap_emb_pre[:,:3].T, c = model_test.nout_position[:,0], s= 10, cmap = 'magma')
ax.scatter(*model_test.nout_umap_emb_rot[:,:3].T, c = model_test.nout_rot_position[:,0], s= 10, cmap = 'magma')
process_axis(ax)
plt.savefig(os.path.join(save_dir,'model_distance_0_example.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_distance_0_example.png'), dpi = 400,bbox_inches="tight")



##################################### DEEP SUP
func_cells_dir = '/home/julio/Documents/DeepSup_project/DeepSup/functional_cells/'
rotation_dir = '/home/julio/Documents/DeepSup_project/DeepSup/rotation/'
distance_dir = '/home/julio/Documents/DeepSup_project/DeepSup/distance/'
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

perc_list = list()
type_list = list()
layer_list = list()
name_list = list()
angle_list = list()
dist_list = list()
ellipse_dist = list()
for mouse in mice_list:

    cell_type = np.load(os.path.join(func_cells_dir,mouse, mouse+'_cellType.npy'))
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]

    num_cells = static_cells+rot_cells+remap_cells
    num_cells = cell_type.shape[0]
    perc_list.append(static_cells/num_cells)
    perc_list.append(rot_cells/num_cells)
    perc_list.append(remap_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap']
    if mouse in deepMice:
        layer_list += ['deep']*3
    elif mouse in supMice:
        layer_list += ['sup']*3
    name_list += [mouse]*3

    rotation_dict = load_pickle(rotation_dir, f'{mouse}_rotation_dict.pkl')
    angle_rot = rotation_dict['umap']['rotation_angle']
    angle_list += [angle_rot]*3

    distance_dict = load_pickle(distance_dir, f'{mouse}_distance_dict.pkl')
    remap_dist = distance_dict['umap']['remap_dist']
    dist_list += [remap_dist]*3

    remap_dist = distance_dict['umap']['plane_remap_dist']
    ellipse_dist += [remap_dist]*3

deepsup_pd = pd.DataFrame(data={'type': type_list,
                            'layer': layer_list,
                            'mouse': name_list,
                            'percentage': perc_list,
                            'rotation': angle_list,
                            'distance': dist_list,
                            'plane_dist': ellipse_dist})


##################################### DUAL COLOR
base_dir = '/home/julio/Documents/DeepSup_project/DualColor/'
mice_list = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']

jRGECO_dir = os.path.join(base_dir,'Thy1jRGECO')
RCaMP_dir = os.path.join(base_dir,'ThyCalbRCaMP')

perc_list = list()
type_list = list()
layer_list = list()
name_list = list()
angle_list = list()
dist_list = list()
ellipse_dist = list()

for mouse in mice_list:
    if 'Thy1jRGECO' in mouse:
        rotation_dir = os.path.join(jRGECO_dir,'rotation', mouse)
        distance_dir = os.path.join(jRGECO_dir,'distance', mouse)
        functional_dir =os.path.join(jRGECO_dir,'functional_cells', mouse)

    else:
        rotation_dir = os.path.join(RCaMP_dir,'rotation', mouse)
        distance_dir = os.path.join(RCaMP_dir,'distance', mouse)
        functional_dir =os.path.join(RCaMP_dir,'functional_cells', mouse)


    rotation_dict = load_pickle(rotation_dir, mouse+'_rotation_dict.pkl')
    distance_dict = load_pickle(distance_dir, mouse+'_distance_dict.pkl')

    deep_rot = rotation_dict['deep']['deep_rotation_angle']
    sup_rot = rotation_dict['sup']['sup_rotation_angle']
    all_rot = rotation_dict['all']['all_rotation_angle']

    plane_deep_dist = distance_dict['deep']['plane_deep_remap_dist']
    plane_sup_dist = distance_dict['sup']['plane_sup_remap_dist']
    plane_all_dist = distance_dict['all']['plane_all_remap_dist']

    deep_dist = distance_dict['deep']['deep_remap_dist']
    sup_dist = distance_dict['sup']['sup_remap_dist']
    all_dist = distance_dict['all']['all_remap_dist']
    #all cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_all_cellType.npy'))
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]

    num_cells = static_cells+rot_cells+remap_cells
    perc_list.append(static_cells/num_cells)
    perc_list.append(rot_cells/num_cells)
    perc_list.append(remap_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap']
    layer_list += ['all']*3
    name_list += [mouse]*3
    angle_list += [all_rot]*3
    dist_list += [all_dist]*3
    ellipse_dist += [plane_all_dist]*3

    #deep cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_deep_cellType.npy'))
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]

    num_cells = static_cells+rot_cells+remap_cells
    perc_list.append(static_cells/num_cells)
    perc_list.append(rot_cells/num_cells)
    perc_list.append(remap_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap']
    layer_list += ['deep']*3
    name_list += [mouse]*3
    angle_list += [deep_rot]*3
    dist_list += [deep_dist]*3
    ellipse_dist += [plane_deep_dist]*3

    #sup cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_sup_cellType.npy'))
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]

    num_cells = cell_type.shape[0]#static_cells+rot_cells+remap_cells
    perc_list.append(static_cells/num_cells)
    perc_list.append(rot_cells/num_cells)
    perc_list.append(remap_cells/num_cells)

    type_list += ['allocentric', 'local-cue-anchored','remap']
    layer_list += ['sup']*3
    name_list += [mouse]*3
    angle_list += [sup_rot]*3
    dist_list += [sup_dist]*3
    ellipse_dist += [plane_sup_dist]*3

dual_pd = pd.DataFrame(data={'type': type_list,
                            'layer': layer_list,
                            'mouse': name_list,
                            'percentage': perc_list,
                            'rotation': angle_list,
                            'distance': dist_list,
                            'plane_dist': ellipse_dist})

##################################### DEEP/SUP + MODEL
mode_params = (0.2,0.1)

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)

sub_pd = deepsup_pd.loc[deepsup_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_deepsup, ax= ax)
mean_rot_model = np.nanmean(model_dict[mode_params]['rot_angles'], axis=1)
std_rot_model = np.nanstd(model_dict[mode_params]['rot_angles'], axis=1)
ax.plot(model_types['local-cue-anchored'], mean_rot_model, color = 'k')
ax.fill_between(model_types['local-cue-anchored'], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.7])
ax.set_ylabel('Rotation')
ax.set_xlabel('Percentage of local-cue-anchored')
plt.savefig(os.path.join(save_dir,'model_deepsup_rotation.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_deepsup_rotation.png'), dpi = 400,bbox_inches="tight")

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)

sub_pd = deepsup_pd.loc[deepsup_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_deepsup, ax= ax)
mean_rot_model = np.nanmean(model_dict[mode_params]['rot_angles'], axis=1)
std_rot_model = np.nanstd(model_dict[mode_params]['rot_angles'], axis=1)
ax.plot(model_types_place_cells['local-cue-anchored'], mean_rot_model, color = 'k')
ax.fill_between(model_types_place_cells['local-cue-anchored'], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.7])
ax.set_ylabel('Rotation')
ax.set_xlabel('Percentage of local-cue-anchored place cellls')
plt.savefig(os.path.join(save_dir,'model_deepsup_rotation_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_deepsup_rotation_place_cells.png'), dpi = 400,bbox_inches="tight")




mode_params = (0.1,0.1)

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = deepsup_pd.loc[deepsup_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='plane_dist', hue='layer', 
    palette= palette_deepsup, ax= ax)
mean_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
std_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
ax.fill_between(remap_list, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.plot(remap_list, mean_dist_model, color = 'k')
ax.set_ylabel('distance ellipse')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_deepsup_dist_ellipse_remap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_deepsup_dist_ellipse_remap.png'), dpi = 400,bbox_inches="tight")


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = deepsup_pd.loc[deepsup_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
    palette= palette_deepsup, ax= ax)
mean_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances_no_ellipse'], axis=1)
std_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances_no_ellipse'], axis=1)
ax.fill_between(remap_list, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.plot(remap_list, mean_dist_model, color = 'k')
ax.set_ylabel('distance mean')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_deepsup_dist_mean_remap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_deepsup_dist_mean_remap.png'), dpi = 400,bbox_inches="tight")



fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = deepsup_pd.loc[deepsup_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='plane_dist', hue='layer', 
    palette= palette_deepsup, ax= ax)
mean_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
std_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
ax.fill_between(remap_list_place_cells, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.plot(remap_list_place_cells, mean_dist_model, color = 'k')
ax.set_ylabel('distance ellipse')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_deepsup_dist_ellipse_remap_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_deepsup_dist_ellipse_remap_place_cells.png'), dpi = 400,bbox_inches="tight")


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = deepsup_pd.loc[deepsup_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
    palette= palette_deepsup, ax= ax)
mean_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances_no_ellipse'], axis=1)
std_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances_no_ellipse'], axis=1)
ax.fill_between(remap_list_place_cells, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.plot(remap_list_place_cells, mean_dist_model, color = 'k')
ax.set_ylabel('distance mean')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_deepsup_dist_mean_remap_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_deepsup_dist_mean_remap_place_cells.png'), dpi = 400,bbox_inches="tight")





##################################### DUAL + MODEL

mode_params = (0.2,0.1)
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_dual, ax= ax)
mean_rot_model = np.nanmean(model_dict[mode_params]['rot_angles'], axis=1)
std_rot_model = np.nanstd(model_dict[mode_params]['rot_angles'], axis=1)
ax.plot(model_types['local-cue-anchored'], mean_rot_model, color = 'k')
ax.fill_between(model_types['local-cue-anchored'], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.7])
ax.set_ylabel('Rotation')
ax.set_xlabel('Percentage of local-cue-anchored')
plt.savefig(os.path.join(save_dir,'model_dual_rotation.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dual_rotation.png'), dpi = 400,bbox_inches="tight")


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_dual, ax= ax)
mean_rot_model = np.nanmean(model_dict[mode_params]['rot_angles'], axis=1)
std_rot_model = np.nanstd(model_dict[mode_params]['rot_angles'], axis=1)
ax.plot(model_types_place_cells['local-cue-anchored'], mean_rot_model, color = 'k')
ax.fill_between(model_types_place_cells['local-cue-anchored'], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.7])
ax.set_ylabel('Rotation')
ax.set_xlabel('Percentage of local-cue-anchored place cellls')
plt.savefig(os.path.join(save_dir,'model_dual_rotation_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dual_rotation_place_cells.png'), dpi = 400,bbox_inches="tight")




mode_params = (0.1,0.1)
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='plane_dist', hue='layer', 
    palette= palette_dual, ax= ax)
mean_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
std_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
ax.fill_between(remap_list, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.plot(remap_list, mean_dist_model, color = 'k')
ax.set_ylabel('distance ellipse')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_dual_dist_ellipse_remap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dual_dist_ellipse_remap.png'), dpi = 400,bbox_inches="tight")


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
    palette= palette_dual, ax= ax)
mean_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances_no_ellipse'], axis=1)
std_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances_no_ellipse'], axis=1)
ax.fill_between(remap_list, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.plot(remap_list, mean_dist_model, color = 'k')
ax.set_ylabel('distance mean')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_dual_dist_mean_remap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dual_dist_mean_remap.png'), dpi = 400,bbox_inches="tight")



fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='plane_dist', hue='layer', 
    palette= palette_dual, ax= ax)
mean_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
std_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances'], axis=1)
ax.fill_between(remap_list_place_cells, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.plot(remap_list_place_cells, mean_dist_model, color = 'k')
ax.set_ylabel('distance ellipse')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_dual_dist_ellipse_remap_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dual_dist_ellipse_remap_place_cells.png'), dpi = 400,bbox_inches="tight")


fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
    palette= palette_dual, ax= ax)
mean_dist_model = np.nanmean(model_dict_remap[case][mode_params]['rot_distances_no_ellipse'], axis=1)
std_dist_model = np.nanstd(model_dict_remap[case][mode_params]['rot_distances_no_ellipse'], axis=1)
ax.fill_between(remap_list_place_cells, mean_dist_model-std_dist_model, mean_dist_model+std_dist_model, color = 'k', alpha = .1)
ax.plot(remap_list_place_cells, mean_dist_model, color = 'k')
ax.set_ylabel('distance mean')
ax.set_xlabel('Percentage of remap')
plt.savefig(os.path.join(save_dir,'model_dual_dist_mean_remap_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'model_dual_dist_mean_remap_place_cells.png'), dpi = 400,bbox_inches="tight")





##################################### DEEP/SUP + DUAL


mode_params = (0.2,0.1)
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_dual, style='mouse', markers = ['<','D','s'], ax= ax)
sub_pd = deepsup_pd.loc[deepsup_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_deepsup, ax= ax)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.6])
ax.set_ylabel('Rotation')
ax.set_xlabel('Percentage of local-cue-anchored')

mean_rot_model = np.nanmean(model_dict[mode_params]['rot_angles'], axis=1)
std_rot_model = np.nanstd(model_dict[mode_params]['rot_angles'], axis=1)
ax.plot(model_types['local-cue-anchored'], mean_rot_model, color = 'k')
ax.fill_between(model_types['local-cue-anchored'], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
plt.savefig(os.path.join(save_dir,'dual_deepsup_model_rotation.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dual_deepsup_model_rotation.png'), dpi = 400,bbox_inches="tight")

mode_params = (0.2,0.1)
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_dual, style='mouse', markers = ['<','D','s'], ax= ax)
sub_pd = deepsup_pd.loc[deepsup_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_deepsup, ax= ax)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.6])
ax.set_ylabel('Rotation')
ax.set_xlabel('Percentage of local-cue-anchored')

mean_rot_model = np.nanmean(model_dict[mode_params]['rot_angles'], axis=1)
std_rot_model = np.nanstd(model_dict[mode_params]['rot_angles'], axis=1)
ax.plot(model_types_place_cells['local-cue-anchored'], mean_rot_model, color = 'k')
ax.fill_between(model_types_place_cells['local-cue-anchored'], mean_rot_model-std_rot_model, mean_rot_model+std_rot_model, color = 'k', alpha = .1)
plt.savefig(os.path.join(save_dir,'dual_deepsup_model_rotation_place_cells.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dual_deepsup_model_rotation_place_cells.png'), dpi = 400,bbox_inches="tight")



from scipy.optimize import curve_fit, least_squares
def sigmoid(x, x0, a, b, c):
    return a/(b + np.exp(-c*(x-x0)))

def sigmoid(x, x0, c):
    return 1/(1 + np.exp(-c*(x-x0)))

ydata = deepsup_pd.loc[deepsup_pd['type']=='local-cue-anchored']['rotation'].to_list() + dual_pd.loc[dual_pd['type']=='local-cue-anchored']['rotation'].to_list()
xdata = deepsup_pd.loc[deepsup_pd['type']=='local-cue-anchored']['percentage'].to_list() + dual_pd.loc[dual_pd['type']=='local-cue-anchored']['percentage'].to_list()
p0 = [0.25, 40] # this is an mandatory initial guess


norm_ydata = (ydata - np.percentile(ydata,5))/np.percentile(ydata - np.percentile(ydata,5),95)
popt, pcov = curve_fit(sigmoid, xdata, norm_ydata)



popt = np.array([0.255, 41])

x = np.linspace(0, 0.55, 500)
y = sigmoid(x, *popt)*np.percentile(ydata - np.percentile(ydata,5),95) + np.percentile(ydata,5)


fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_dual, style='mouse', markers = ['<','D','s'], ax= ax)
sub_pd = deepsup_pd.loc[deepsup_pd['type']=='local-cue-anchored']
sns.scatterplot(data = sub_pd, x ='percentage', y='rotation', hue='layer', 
    palette= palette_deepsup, ax= ax, alpha=0.3)
ax.set_ylim([0, 180])
ax.set_xlim([0,0.55])
ax.set_ylabel('Rotation')
plt.plot(x,y, 'k--', label='fit', linewidth=3, alpha = .3)
ax.set_xlabel('Percentage of local-cue-anchored')
plt.title(popt)

plt.savefig(os.path.join(save_dir,'dual_deepsup_rotation.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dual_deepsup_rotation.png'), dpi = 400,bbox_inches="tight")




def line(x, a, b):
    return a*x + b

ydata = deepsup_pd.loc[deepsup_pd['type']=='remap']['distance'].to_list() + dual_pd.loc[dual_pd['type']=='remap']['distance'].to_list()
xdata = deepsup_pd.loc[deepsup_pd['type']=='remap']['percentage'].to_list() + dual_pd.loc[dual_pd['type']=='remap']['percentage'].to_list()

popt, pcov = curve_fit(line, xdata, ydata)
x = np.linspace(0.2, 0.8, 500)
y = line(x, *popt)

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111)
sub_pd = dual_pd.loc[dual_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
    palette= palette_dual, style='mouse', markers = ['<','D','s'], ax= ax)
sub_pd = deepsup_pd.loc[deepsup_pd['type']=='remap']
sns.scatterplot(data = sub_pd, x ='percentage', y='distance', hue='layer', 
    palette= palette_deepsup, ax= ax, alpha=0.3)

ax.set_ylabel('Distance')
plt.plot(x,y, 'k--', label='fit', linewidth=3, alpha = .3)
ax.set_xlabel('Percentage of remap')
plt.title(popt)

plt.savefig(os.path.join(save_dir,'dual_deepsup_distance.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dual_deepsup_distance.png'), dpi = 400,bbox_inches="tight")



from sklearn.metrics import r2_score
r2_score(ydata, line(np.array(xdata), *popt))

