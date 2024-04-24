mouse_list = list()
remap_dist_veh_list = list()
remap_dist_CNO_list = list()


calb_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb'
for mouse in ['CalbCharly2', 'CalbCharly11_concat']:#, 'CalbV23', 'DD2']:
    mouse_list.append(mouse)
    for case in ['veh','CNO']:
        data_dir = os.path.join(calb_dir,'distance', mouse+'_'+case)
        rot_error_dict = load_pickle(data_dir, f"{mouse}_{case}_distance_dict.pkl")
        if 'veh' in case:
            remap_dist_veh_list.append(rot_error_dict['umap']['remap_dist'])
        else:
            remap_dist_CNO_list.append(rot_error_dict['umap']['remap_dist'])

chrna7 = '/home/julio/Documents/DeepSup_project/DREADDs/ChRNA7'
for mouse in ['ChRNA7Charly1', 'ChRNA7Charly2']:
    mouse_list.append(mouse)
    for case in ['veh','CNO']:
        data_dir = os.path.join(chrna7,'distance', mouse+'_'+case)
        rot_error_dict = load_pickle(data_dir, f"{mouse}_{case}_distance_dict.pkl")
        if 'veh' in case:
            remap_dist_veh_list.append(rot_error_dict['umap']['remap_dist'])
        else:
            remap_dist_CNO_list.append(rot_error_dict['umap']['remap_dist'])


rotation_pd = pd.DataFrame(data={'dist_veh': remap_dist_veh_list,
                            'dist_CNO': remap_dist_CNO_list,
                            'mouse': mouse_list})


perc_sup = [0.94, 0.95, 0.02, 0.00]
perc_deep = [0.48, 0.22, 0.43, 0.67]

rotation_pd['perc_sup'] = perc_sup
rotation_pd['perc_deep'] = perc_deep
rotation_pd['dist_change'] = rotation_pd['dist_CNO']/rotation_pd['dist_veh']


fig, ax = plt.subplots(3, 2, figsize=(6,6))
b = sns.scatterplot(x='perc_sup', y='dist_veh',data=rotation_pd, ax = ax[0,0])
b = sns.scatterplot(x='perc_deep', y='dist_veh',data=rotation_pd, ax = ax[0,1])
b = sns.scatterplot(x='perc_sup', y='dist_CNO',data=rotation_pd, ax = ax[1,0])
b = sns.scatterplot(x='perc_deep', y='dist_CNO',data=rotation_pd, ax = ax[1,1])
b = sns.scatterplot(x='perc_sup', y='dist_change',data=rotation_pd, ax = ax[2,0])
b = sns.scatterplot(x='perc_deep', y='dist_change',data=rotation_pd, ax = ax[2,1])






#__________________________________________________________________________
#|                                                                        |#
#|                      DECODER MEDIAN ABSOLUTE ERRO                      |#
#|________________________________________________________________________|#



from sklearn.metrics import median_absolute_error, explained_variance_score

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

error_bin_sup = np.zeros((len(supMice),n_bins,10,2))
error_bin_deep = np.zeros((len(deepMice),n_bins,10,2))
count_bin_sup = np.zeros((len(supMice),n_bins,10))
count_bin_deep = np.zeros((len(deepMice),n_bins,10))
count_bin_pred_sup = np.zeros((len(supMice),n_bins,10,2))
count_bin_pred_deep = np.zeros((len(deepMice),n_bins,10,2))


for mouse in miceList:
    for it in range(10):
        test = dec_pred[mouse][label_idx][dec_idx][it,:,0] == 1
        ground_truth = dec_pred[mouse][label_idx][dec_idx][it,:,1][test]
        og_pred = dec_pred[mouse][label_idx][dec_idx][it,:,2][test]
        umap_pred = dec_pred[mouse][label_idx][dec_idx][it,:,-1][test]

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
                error_bin_deep[deep_idx,b,it, 0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
                error_bin_deep[deep_idx,b,it, 1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
                count_bin_deep[deep_idx,b, it] = len(grid[b])
                count_bin_pred_deep[deep_idx,b,it,0] = len(grid_pred[b])
                count_bin_pred_deep[deep_idx,b,it,1] = len(grid_pred_umap[b])
            else:
                sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
                error_bin_sup[sup_idx,b,it,0] = median_absolute_error(ground_truth[grid[b]], og_pred[grid[b]])
                error_bin_sup[sup_idx,b,it,1] = median_absolute_error(ground_truth[grid[b]], umap_pred[grid[b]])
                count_bin_sup[sup_idx,b,it] = len(grid[b])
                count_bin_pred_sup[sup_idx,b,it,0] = len(grid_pred[b])
                count_bin_pred_sup[sup_idx,b,it,1] = len(grid_pred_umap[b])


for mouse in ['GC2', 'GC3', 'CZ3']:
    if mouse in deepMice:
        deep_idx = [x for x in range(len(deepMice)) if deepMice[x] == mouse][0]
        for it in range(10):
            error_bin_deep[deep_idx,:,it,0] = error_bin_deep[deep_idx,::-1,it,0]
            error_bin_deep[deep_idx,:,it,1] = error_bin_deep[deep_idx,::-1,it,1]
            count_bin_deep[deep_idx,:,it] = count_bin_deep[deep_idx,::-1,it]
            count_bin_pred_deep[deep_idx,:,it,0] = count_bin_pred_deep[deep_idx,::-1,it,0]
            count_bin_pred_deep[deep_idx,:,it,1] = count_bin_pred_deep[deep_idx,::-1,it,0]


error_bin_deep = np.nanstd(error_bin_deep,axis=2)/np.nanmean(error_bin_deep,axis=2)
error_bin_sup = np.nanstd(error_bin_sup,axis=2)/np.nanmean(error_bin_sup,axis=2)


count_bin_deep = np.nanstd(count_bin_deep,axis=2)/np.nanmean(count_bin_deep,axis=2)
count_bin_sup = np.nanstd(count_bin_sup,axis=2)/np.nanmean(count_bin_sup,axis=2)
count_bin_pred_deep = np.nanstd(count_bin_pred_deep,axis=2)/np.nanmean(count_bin_pred_deep,axis=2)
count_bin_pred_sup = np.nanstd(count_bin_pred_sup,axis=2)/np.nanmean(count_bin_pred_sup,axis=2)



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



#__________________________________________________________________________
#|                                                                        |#
#|                       DECODER EXPLAINED VARIANCE                       |#
#|________________________________________________________________________|#

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

n_bins = 10

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
            error_bin_deep[deep_idx,b,0] = explained_variance_score(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_deep[deep_idx,b,1] = explained_variance_score(ground_truth[grid[b]], umap_pred[grid[b]])
            count_bin_deep[deep_idx,b] = len(grid[b])
            count_bin_pred_deep[deep_idx,b,0] = len(grid_pred[b])
            count_bin_pred_deep[deep_idx,b,1] = len(grid_pred_umap[b])
        else:
            sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
            error_bin_sup[sup_idx,b,0] = explained_variance_score(ground_truth[grid[b]], og_pred[grid[b]])
            error_bin_sup[sup_idx,b,1] = explained_variance_score(ground_truth[grid[b]], umap_pred[grid[b]])
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





error_bin_deep[::-1] = error_bin_deep[1,2,5]

sup_color = '#9900ffff'
deep_color = '#cc9900ff'

bin_space = np.linspace(0,70,n_bins)

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


deep_error = np.mean(error_bin_deep[:,2:10,1], axis=1)
sup_error = np.mean(error_bin_sup[:,2:10,1], axis=1)

deep_shapiro = shapiro(deep_error)
sup_shapiro = shapiro(sup_error)
if deep_shapiro.pvalue<=0.05 or sup_shapiro.pvalue<=0.05:
    print(f'Umap Kstest : {stats.ks_2samp(deep_error, sup_error)}')
else:
    print(f'Umap ttest : {stats.ttest_ind(deep_error, sup_error)}')


#__________________________________________________________________________
#|                                                                        |#
#|                    DECODER EXPLAINED VARIANCE ALL                      |#
#|________________________________________________________________________|#

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


error_bin_sup = np.zeros((len(supMice),10,2))
error_bin_deep = np.zeros((len(deepMice),10,2))



for mouse in miceList:
    
    test = dec_pred[mouse][label_idx][dec_idx][:,:,0].reshape(-1,1) == 1
    ground_truth = dec_pred[mouse][label_idx][dec_idx][:,:,1].reshape(-1,1)[test]
    og_pred = dec_pred[mouse][label_idx][dec_idx][:,:,2].reshape(-1,1)[test]
    umap_pred = dec_pred[mouse][label_idx][dec_idx][:,:,-1].reshape(-1,1)[test]
    min_x = np.min(ground_truth)
    max_x = np.max(ground_truth)

    upper_lim = max_x - 0.1*(max_x - min_x)
    lower_lim = min_x + 0.5*(max_x - min_x)

    if mouse in izq:
        upper_lim = max_x - 0.5*(max_x - min_x)
        lower_lim = min_x + 0.1*(max_x - min_x)


    for it in range(10):
        test = dec_pred[mouse][label_idx][dec_idx][it,:,0].reshape(-1,1) == 1
        ground_truth = dec_pred[mouse][label_idx][dec_idx][it,:,1].reshape(-1,1)[test]
        og_pred = dec_pred[mouse][label_idx][dec_idx][it,:,2].reshape(-1,1)[test]
        umap_pred = dec_pred[mouse][label_idx][dec_idx][it,:,-1].reshape(-1,1)[test]
        points_in = 1*np.logical_and(ground_truth>=lower_lim,ground_truth<=upper_lim)
        if mouse in deepMice:
            deep_idx = [x for x in range(len(deepMice)) if deepMice[x] == mouse][0]
            error_bin_deep[deep_idx,it,0] = explained_variance_score(ground_truth[points_in], og_pred[points_in])
            error_bin_deep[deep_idx,it,1] = explained_variance_score(ground_truth[points_in], umap_pred[points_in])
        else:
            sup_idx = [x for x in range(len(supMice)) if supMice[x] == mouse][0]
            error_bin_sup[sup_idx,it,0] = explained_variance_score(ground_truth[points_in], og_pred[points_in])
            error_bin_sup[sup_idx,it,1] = explained_variance_score(ground_truth[points_in], umap_pred[points_in])


deep_error = np.nanmean(error_bin_deep[:,:,1], axis=1)
sup_error =  np.nanmean(error_bin_sup[:,:,1], axis=1)

deep_shapiro = shapiro(deep_error)
sup_shapiro = shapiro(sup_error)
if deep_shapiro.pvalue<=0.05 or sup_shapiro.pvalue<=0.05:
    print(f'Umap Kstest : {stats.ks_2samp(deep_error, sup_error)}')
else:
    print(f'Umap ttest : {stats.ttest_ind(deep_error, sup_error)}')