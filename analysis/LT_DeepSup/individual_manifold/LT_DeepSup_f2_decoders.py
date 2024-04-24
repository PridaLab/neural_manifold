from sklearn.metrics import median_absolute_error
import pandas as pd 
from os import listdir
from os.path import isfile, join





supMice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']
# deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1','ChZ7', 'ChZ8', 'GC7']

##############################################
dec_dir = '/home/julio/Documents/SP_project/Fig2/decoders'
cues_dir = '/home/julio/Documents/SP_project/LT_DeepSup/data'

dec_R2s = load_pickle(dec_dir, 'dec_R2s_dict.pkl')
dec_pred = load_pickle(dec_dir, 'dec_pred_dict.pkl')
miceList = list(dec_R2s.keys())

label_name = 'posx'
label_idx = 0
dec_name = 'xgb'
dec_idx = 2

n_bins = 12

error_bin_sup = np.zeros((len(supMice),n_bins,2))
error_bin_deep = np.zeros((len(deepMice),n_bins,2))
count_bin_sup = np.zeros((len(supMice),n_bins))
count_bin_deep = np.zeros((len(deepMice),n_bins))
count_bin_pred_sup = np.zeros((len(supMice),n_bins,2))
count_bin_pred_deep = np.zeros((len(deepMice),n_bins,2))

cues_bin_sup= np.zeros((len(supMice),n_bins))
cues_bin_deep= np.zeros((len(deepMice),n_bins))


for idx, mouse in enumerate(miceList):

    #load cues position
    cues_file = [f for f in listdir(join(cues_dir, mouse)) if 'cues_info.csv' in f and 'lt' in f][0]
    cues_info = pd.read_csv(join(cues_dir, mouse, cues_file))
    #get position info to create grid
    pos = dec_pred[mouse][label_idx][dec_idx][0,:,1]
    min_pos = np.percentile(pos,1)
    max_pos = np.percentile(pos,99)
    st_cue = cues_info['x_start_cm'][0]
    end_cue = cues_info['x_end_cm'][0]

    #create grid
    grid_edges = list()
    steps = (max_pos - min_pos)/n_bins
    edges = np.linspace(min_pos, max_pos, n_bins+1).reshape(-1,1)
        #adapt to local cues
    st_cue_bin = np.argmin(np.abs(edges[:-1]-st_cue))
    en_cue_bin = st_cue_bin+3
    edges[st_cue_bin] = st_cue
    edges[en_cue_bin] = end_cue
    grid_edges = np.concatenate((edges[:-1], edges[1:]), axis = 1)

    grid = np.empty([grid_edges.shape[0]], object)

    #load predicitons
    test = dec_pred[mouse][label_idx][dec_idx][:,:,0].reshape(-1,1) == 1
    ground_truth = dec_pred[mouse][label_idx][dec_idx][:,:,1].reshape(-1,1)[test]
    og_pred = dec_pred[mouse][label_idx][dec_idx][:,:,2].reshape(-1,1)[test]
    umap_pred = dec_pred[mouse][label_idx][dec_idx][:,:,-1].reshape(-1,1)[test]

    #fill in grid
    grid = grid.ravel()
    grid_pred = copy.deepcopy(grid)
    grid_pred_umap = copy.deepcopy(grid)
    for idx in range(n_bins):
        logic = np.zeros(ground_truth.shape[0])
        min_edge = grid_edges[idx,0]
        max_edge = grid_edges[idx,1]
        logic = np.logical_and(ground_truth>=min_edge,ground_truth<=max_edge)
        grid[idx] = list(np.where(logic)[0])

        logic = np.logical_and(og_pred>=min_edge,og_pred<=max_edge)
        grid_pred[idx] =  list(np.where(logic)[0])

        logic = np.logical_and(umap_pred>=min_edge,umap_pred<=max_edge)
        grid_pred_umap[idx] =  list(np.where(logic)[0])

    #compute error in each bin 
    if mouse in deepMice:
        deep_idx = [x for x in range(len(deepMice)) if deepMice[x] == mouse][0]
        cues_bin_deep[deep_idx, st_cue_bin:en_cue_bin+1] = 1
        for b in range(len(grid)):
            error_bin_deep[deep_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_deep[deep_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_deep[deep_idx,b] = len(grid[b])
            count_bin_pred_deep[deep_idx,b,0] = len(grid_pred[b])
            count_bin_pred_deep[deep_idx,b,1] = len(grid_pred_umap[b])
    elif mouse in supMice:
        sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
        cues_bin_sup[sup_idx, st_cue_bin:en_cue_bin+1] = 1
        for b in range(len(grid)):
            error_bin_sup[sup_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_sup[sup_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_sup[sup_idx,b] = len(grid[b])
            count_bin_pred_sup[sup_idx,b,0] = len(grid_pred[b])
            count_bin_pred_sup[sup_idx,b,1] = len(grid_pred_umap[b])


for mouse in ['GC2', 'GC3', 'CZ3', 'CZ4']:
    if mouse in deepMice:
        deep_idx = [x for x in range(len(deepMice)) if deepMice[x] == mouse][0]
        error_bin_deep[deep_idx,:,0] = error_bin_deep[deep_idx,::-1,0]
        error_bin_deep[deep_idx,:,1] = error_bin_deep[deep_idx,::-1,1]
        count_bin_deep[deep_idx,:] = count_bin_deep[deep_idx,::-1]
        count_bin_pred_deep[deep_idx,:,0] = count_bin_pred_deep[deep_idx,::-1,0]
        count_bin_pred_deep[deep_idx,:,1] = count_bin_pred_deep[deep_idx,::-1,0]
        cues_bin_deep[deep_idx,:] = cues_bin_deep[deep_idx,::-1]
    elif mouse in supMice:
        sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
        error_bin_sup[sup_idx,:,0] = error_bin_sup[sup_idx,::-1,0]
        error_bin_sup[sup_idx,:,1] = error_bin_sup[sup_idx,::-1,1]
        count_bin_sup[sup_idx,:] = count_bin_sup[sup_idx,::-1]
        count_bin_pred_sup[sup_idx,:,0] = count_bin_pred_sup[sup_idx,::-1,0]
        count_bin_pred_sup[sup_idx,:,1] = count_bin_pred_sup[sup_idx,::-1,0]
        cues_bin_sup[sup_idx,:] = cues_bin_sup[sup_idx,::-1]


sup_color = '#9900ffff'
deep_color = '#cc9900ff'

bin_space = np.linspace(0,120,n_bins)

plt.figure()
ax = plt.subplot(2,1,1)
m = np.nanmean(error_bin_deep[:,:,0],axis=0)
sd = np.nanstd(error_bin_deep[:,:,0],axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
m = np.nanmean(error_bin_sup[:,:,0],axis=0)
sd = np.nanstd(error_bin_sup[:,:,0],axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Original space')
ax.set_ylabel('mean abs error posx')
ax.legend()

ax = plt.subplot(2,1,2)
m = np.nanmean(error_bin_deep[:,:,1],axis=0)
sd = np.nanstd(error_bin_deep[:,:,1],axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
m = np.nanmean(error_bin_sup[:,:,1],axis=0)
sd = np.nanstd(error_bin_sup[:,:,1],axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Umap space')
ax.set_ylabel('mean abs error posx')
ax.legend()

plt.savefig(os.path.join(data_dir,'DeepSup_error_by_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'DeepSup_error_by_pos.png'), dpi = 400,bbox_inches="tight")



plt.figure()
ax = plt.subplot(2,1,1)
vals = (count_bin_pred_deep[:,:,0] - count_bin_deep)/(count_bin_deep)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
vals = (count_bin_pred_sup[:,:,0] - count_bin_sup)/(count_bin_sup)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Original space')
ax.set_ylabel('norm change in prediction')
ax.legend()

ax = plt.subplot(2,1,2)
vals = (count_bin_pred_deep[:,:,1] - count_bin_deep)/(count_bin_deep)
vals = (count_bin_pred_deep[:,:,1] - count_bin_deep)/(count_bin_deep)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
vals = (count_bin_pred_sup[:,:,1] - count_bin_sup)/(count_bin_sup)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Umap space')
ax.set_ylabel('norm change in prediction')
ax.legend()
plt.savefig(os.path.join(data_dir,'DeepSup_pred_pos_by_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'DeepSup_pred_pos_by_pos.png'), dpi = 400,bbox_inches="tight")



###############################################

dec_dir = '/home/julio/Documents/SP_project/Fig2/decoders'
cues_dir = '/home/julio/Documents/SP_project/LT_DeepSup/data'


dec_R2s = load_pickle(dec_dir, 'dec_R2s_dict.pkl')
dec_pred = load_pickle(dec_dir, 'dec_pred_dict.pkl')

miceList = list(dec_R2s.keys())

label_name = 'posx'
label_idx = 0
dec_name = 'xgb'
dec_idx = 2


error_in_cue = np.zeros((len(miceList),2))*np.nan
error_out_cue =  np.zeros((len(miceList),2))*np.nan
num_pred = np.zeros((len(miceList),3))*np.nan
for idx, mouse in enumerate(miceList):

    test = dec_pred[mouse][label_idx][dec_idx][:,:,0].reshape(-1,1) == 1
    ground_truth = dec_pred[mouse][label_idx][dec_idx][:,:,1].reshape(-1,1)[test]
    og_pred = dec_pred[mouse][label_idx][dec_idx][:,:,2].reshape(-1,1)[test]
    umap_pred = dec_pred[mouse][label_idx][dec_idx][:,:,-1].reshape(-1,1)[test]

    #load cues position
    cues_file = [f for f in listdir(join(cues_dir, mouse)) if 'cues_info.csv' in f and 'lt' in f][0]
    cues_info = pd.read_csv(join(cues_dir, mouse, cues_file))

    in_cues_ground_truth = np.logical_and(ground_truth>=cues_info['x_start_cm'][0], ground_truth<=cues_info['x_end_cm'][0])
    in_cues_og_pred = np.logical_and(og_pred>=cues_info['x_start_cm'][0], og_pred<=cues_info['x_end_cm'][0])
    in_cues_umap_pred = np.logical_and(umap_pred>=cues_info['x_start_cm'][0], umap_pred<=cues_info['x_end_cm'][0])



    out_cues_ground_truth = np.logical_and(ground_truth>=out_cues_start, ground_truth<=out_cues_end)
    out_cues_og_pred = np.logical_and(og_pred>=out_cues_start, og_pred<=out_cues_end)
    out_cues_umap_pred = np.logical_and(umap_pred>=out_cues_start, umap_pred<=out_cues_end)


    error_in_cue[idx,0] = median_absolute_error(ground_truth[in_cues_ground_truth], og_pred[in_cues_ground_truth])
    error_out_cue[idx,0] = median_absolute_error(ground_truth[out_cues_ground_truth], og_pred[out_cues_ground_truth])

    error_in_cue[idx,1] = median_absolute_error(ground_truth[in_cues_ground_truth], umap_pred[in_cues_ground_truth])
    error_out_cue[idx,1] = median_absolute_error(ground_truth[out_cues_ground_truth], umap_pred[out_cues_ground_truth])

    num_pred[idx,:] = [np.sum(in_cues_ground_truth),np.sum(in_cues_og_pred),np.sum(in_cues_umap_pred)]


num_pred_list = []
error_pred_list = []
mouse_list = []
condition_list = []
layer_list = []
for idx, mouse in enumerate(miceList):
    num_pred_list.append((num_pred[idx,1]-num_pred[idx,0])/num_pred[idx,0])
    error_pred_list.append(error_in_cue[idx,0]/error_out_cue[idx,0])
    condition_list.append('og')
    num_pred_list.append((num_pred[idx,2]-num_pred[idx,0])/num_pred[idx,0])
    error_pred_list.append(error_in_cue[idx,1]/error_out_cue[idx,1])
    condition_list.append('emb')
    mouse_list += [mouse]*2
    if mouse in deepMice:
        layer_list += ['deep']*2
    elif mouse in supMice:
        layer_list += ['sup']*2


pd_decoders = pd.DataFrame(data={'mouse': mouse_list,
                     'num_pred': num_pred_list,
                     'error': error_pred_list,
                     'condition': condition_list,
                     'layer': layer_list})    

palette= ["#cc9900ff", "#9900ffff"]
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(1,2,1)
sns.boxplot(x='condition', y='error', data=pd_decoders, hue='layer',
            palette = palette, linewidth = 1, width= .5, ax = ax)

ax = plt.subplot(1,2,2)
sns.boxplot(x='condition', y='num_pred', data=pd_decoders, hue='layer',
            palette = palette, linewidth = 1, width= .5, ax = ax)

######

error_pred_list = []
mouse_list = []
condition_list = []
layer_list = []
location_list = []
for idx, mouse in enumerate(miceList):
    error_pred_list.append(error_in_cue[idx,0])
    condition_list.append('og')
    location_list.append('in-cue')

    error_pred_list.append(error_out_cue[idx,0])
    condition_list.append('og')
    location_list.append('out-cue')


    error_pred_list.append(error_in_cue[idx,1])
    condition_list.append('emb')
    location_list.append('in-cue')

    error_pred_list.append(error_out_cue[idx,1])
    condition_list.append('emb')
    location_list.append('out-cue')

    mouse_list += [mouse]*4
    if mouse in deepMice:
        layer_list += ['deep']*4
    elif mouse in supMice:
        layer_list += ['sup']*4


pd_decoders = pd.DataFrame(data={'mouse': mouse_list,
                     'location': location_list,
                     'error': error_pred_list,
                     'condition': condition_list,
                     'layer': layer_list})    

palette= ["#cc9900ff", "#9900ffff"]
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(1,2,1)
sns.boxplot(x='location', y='error', data=pd_decoders[pd_decoders['condition']=='og'], hue='layer',
            palette = palette, linewidth = 1, width= .5, ax = ax)
ax.set_title('og')
ax = plt.subplot(1,2,2)
sns.boxplot(x='location', y='error', data=pd_decoders[pd_decoders['condition']=='emb'], hue='layer',
            palette = palette, linewidth = 1, width= .5, ax = ax)
ax.set_title('emb')


deep_in_cue = pd_decoders.loc[(pd_decoders['condition']=='emb') & 
                                (pd_decoders['layer']=='deep') &
                                (pd_decoders['location']=='in-cue'),'error'].to_list()
sup_in_cue = pd_decoders.loc[(pd_decoders['condition']=='emb') & 
                                (pd_decoders['layer']=='sup') &
                                (pd_decoders['location']=='in-cue'),'error'].to_list()

deep_out_cue = pd_decoders.loc[(pd_decoders['condition']=='emb') & 
                                (pd_decoders['layer']=='deep') &
                                (pd_decoders['location']=='out-cue'),'error'].to_list()
sup_out_cue = pd_decoders.loc[(pd_decoders['condition']=='emb') & 
                                (pd_decoders['layer']=='sup') &
                                (pd_decoders['location']=='out-cue'),'error'].to_list()

deepShapiro = shapiro(deep_in_cue)
supShapiro = shapiro(sup_in_cue)

if deepShapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
    print('Rate:',stats.ks_2samp(deep_in_cue, sup_in_cue))
else:
    print('Rate:', stats.ttest_ind(deep_in_cue, sup_in_cue))


deep_cue = [deep_in_cue[idx]/deep_out_cue[idx] for idx in range(len(deep_in_cue))]
sup_cue = [sup_in_cue[idx]/sup_out_cue[idx] for idx in range(len(sup_in_cue))]

deepShapiro = shapiro(deep_cue)
supShapiro = shapiro(sup_cue)

if deepShapiro.pvalue<=0.05 or supShapiro.pvalue<=0.05:
    print('Decoder:',stats.ks_2samp(deep_cue, sup_cue))
else:
    print('Decoder:', stats.ttest_ind(deep_cue, sup_cue))


#__________________________________________________________________________
#|                                                                        |#
#|                                DECODERS                                |#
#|________________________________________________________________________|#

data_dir = '/home/julio/Documents/SP_project/Fig2/decoders'
dec_R2s = load_pickle(data_dir, 'dec_R2s_dict.pkl')
dec_R2s_shifted = load_pickle(data_dir, 'dec_R2s_dict_shifted.pkl')

palette= ['grey', '#5bb95bff', '#ff8a17ff', '#249aefff']
label_list = ['posx', 'posy','vel', 'index_mat', 'dir_mat']
signal_list = ['base_signal', 'pca', 'isomap', 'umap']
decoder_list = ['wf','wc','xgb','svr']
miceList = list(dec_R2s.keys())

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

        b = sns.barplot(x='signal', y='R2s', data=pd_R2s, hue='layer',
                palette = palette, linewidth = 1, width= .5, ax = ax[dec_idx])
        sns.swarmplot(x='signal', y='R2s', data=pd_R2s, hue='layer',
                    palette = 'dark:gray', edgecolor = 'gray', ax = ax[dec_idx])

        ax[dec_idx].set_ylabel(f'R2s {label_name}')
        ax[dec_idx].set_title(dec_name)

    fig.suptitle(label_name)
    plt.savefig(os.path.join(data_dir,f'dec_{label_name}_test.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(data_dir,f'dec_{label_name}_test.png'), dpi = 400,bbox_inches="tight")




from sklearn.metrics import median_absolute_error

data_dir = '/home/julio/Documents/SP_project/Fig2/decoders'
dec_R2s = load_pickle(data_dir, 'dec_R2s_dict.pkl')
dec_pred = load_pickle(data_dir, 'dec_pred_dict.pkl')
miceList = list(dec_R2s.keys())
izq = ['GC2', 'GC3', 'CZ3']
label_name = 'posx'
label_idx = 0
dec_name = 'xgb'
dec_idx = 2
miceList = list(dec_R2s.keys())

n_bins = 12

error_bin_sup = np.zeros((len(supMice),n_bins,2))
error_bin_deep = np.zeros((len(deepMice),n_bins,2))
count_bin_sup = np.zeros((len(supMice),n_bins))
count_bin_deep = np.zeros((len(deepMice),n_bins))
count_bin_pred_sup = np.zeros((len(supMice),n_bins,2))
count_bin_pred_deep = np.zeros((len(deepMice),n_bins,2))


for mouse in miceList:

    test = dec_pred[mouse][label_idx][dec_idx][:,:,0].reshape(-1,1) == 1
    ground_truth = dec_pred[mouse][label_idx][dec_idx][:,:,1].reshape(-1,1)[test]
    og_pred = dec_pred[mouse][label_idx][dec_idx][:,:,2].reshape(-1,1)[test]
    umap_pred = dec_pred[mouse][label_idx][dec_idx][:,:,-1].reshape(-1,1)[test]


    min_x = np.min(ground_truth)
    max_x = np.max(ground_truth)
    ndims = 1
    grid_edges = list()
    steps = (max_x - min_x)/n_bins
    edges = np.linspace(min_x, max_x, n_bins+1).reshape(-1,1)
    grid_edges = np.concatenate((edges[:-1], edges[1:]), axis = 1)

    #Generate the grid containing the indices of the points of the label and 
    #the coordinates as the mid point of edges

    grid = np.empty([grid_edges.shape[0]], object)
    mesh = meshgrid2(tuple([np.arange(s) for s in grid.shape]))
    meshIdx = np.vstack([col.ravel() for col in mesh]).T
    grid = grid.ravel()
    grid_pred = copy.deepcopy(grid)
    grid_pred_umap = copy.deepcopy(grid)
    for elem, idx in enumerate(meshIdx):
        logic = np.zeros(ground_truth.shape[0])
        min_edge = grid_edges[idx,0]
        max_edge = grid_edges[idx,1]
        logic = 1*np.logical_and(ground_truth>=min_edge,ground_truth<=max_edge)
        grid[elem] = list(np.where(logic == meshIdx.shape[1])[0])

        logic = 1*np.logical_and(og_pred>=min_edge,og_pred<=max_edge)
        grid_pred[elem] =  list(np.where(logic == meshIdx.shape[1])[0])

        logic = 1*np.logical_and(umap_pred>=min_edge,umap_pred<=max_edge)
        grid_pred_umap[elem] =  list(np.where(logic == meshIdx.shape[1])[0])

    for b in range(len(grid)):
        if mouse in deepMice:
            deep_idx = [x for x in range(len(deepMice)) if deepMice[x] == mouse][0]
            error_bin_deep[deep_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_deep[deep_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_deep[deep_idx,b] = len(grid[b])
            count_bin_pred_deep[deep_idx,b,0] = len(grid_pred[b])
            count_bin_pred_deep[deep_idx,b,1] = len(grid_pred_umap[b])
        else:
            sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
            error_bin_sup[sup_idx,b,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_sup[sup_idx,b,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_sup[sup_idx,b] = len(grid[b])
            count_bin_pred_sup[sup_idx,b,0] = len(grid_pred[b])
            count_bin_pred_sup[sup_idx,b,1] = len(grid_pred_umap[b])


for mouse in ['GC2', 'GC3', 'CZ3']:
    if mouse in deepMice:
        deep_idx = [x for x in range(len(deepMice)) if deepMice[x] == mouse][0]
        error_bin_deep[deep_idx,:,0] = error_bin_deep[deep_idx,::-1,0]
        error_bin_deep[deep_idx,:,1] = error_bin_deep[deep_idx,::-1,1]
        count_bin_deep[deep_idx,:] = count_bin_deep[deep_idx,::-1]
        count_bin_pred_deep[deep_idx,:,0] = count_bin_pred_deep[deep_idx,::-1,0]
        count_bin_pred_deep[deep_idx,:,1] = count_bin_pred_deep[deep_idx,::-1,0]



sup_color = '#9900ffff'
deep_color = '#cc9900ff'

bin_space = np.linspace(0,120,n_bins)

plt.figure()
ax = plt.subplot(2,1,1)
m = np.nanmean(error_bin_deep[:,:,0],axis=0)
sd = np.nanstd(error_bin_deep[:,:,0],axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
m = np.nanmean(error_bin_sup[:,:,0],axis=0)
sd = np.nanstd(error_bin_sup[:,:,0],axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Original space')
ax.set_ylabel('mean abs error posx')
ax.legend()

ax = plt.subplot(2,1,2)
m = np.nanmean(error_bin_deep[:,:,1],axis=0)
sd = np.nanstd(error_bin_deep[:,:,1],axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
m = np.nanmean(error_bin_sup[:,:,1],axis=0)
sd = np.nanstd(error_bin_sup[:,:,1],axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Umap space')
ax.set_ylabel('mean abs error posx')
ax.legend()

plt.savefig(os.path.join(data_dir,'DeepSup_error_by_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'DeepSup_error_by_pos.png'), dpi = 400,bbox_inches="tight")



plt.figure()
ax = plt.subplot(2,1,1)
vals = (count_bin_pred_deep[:,:,0] - count_bin_deep)/(count_bin_deep)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
vals = (count_bin_pred_sup[:,:,0] - count_bin_sup)/(count_bin_sup)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Original space')
ax.set_ylabel('norm change in prediction')
ax.legend()

ax = plt.subplot(2,1,2)
vals = (count_bin_pred_deep[:,:,1] - count_bin_deep)/(count_bin_deep)
vals = (count_bin_pred_deep[:,:,1] - count_bin_deep)/(count_bin_deep)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=deep_color, linewidth= 2,label='deep')
ax .fill_between(bin_space, m-sd, m+sd,color=deep_color,alpha = 0.3)
vals = (count_bin_pred_sup[:,:,1] - count_bin_sup)/(count_bin_sup)
m = np.nanmean(vals,axis=0)
sd = np.nanstd(vals,axis=0)
ax .plot(bin_space, m, color=sup_color, linewidth= 2,label='sup')
ax .fill_between(bin_space, m-sd, m+sd,color=sup_color,alpha = 0.3)
ax.set_title('Umap space')
ax.set_ylabel('norm change in prediction')
ax.legend()
plt.savefig(os.path.join(data_dir,'DeepSup_pred_pos_by_pos.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'DeepSup_pred_pos_by_pos.png'), dpi = 400,bbox_inches="tight")