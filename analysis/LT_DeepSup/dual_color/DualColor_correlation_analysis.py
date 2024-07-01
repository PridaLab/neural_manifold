import copy, os, pickle
import numpy as np
import matplotlib.pyplot as plt

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))

#__________________________________________________________________________
#|                                                                        |#
#|                              pre to rot                                |#
#|________________________________________________________________________|#


mice_list = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']

base_dir = '/home/julio/Documents/DeepSup_project/DualColor/'

jRGECO_dir = os.path.join(base_dir,'Thy1jRGECO')
RCaMP_dir = os.path.join(base_dir,'ThyCalbRCaMP')
save_dir = os.path.join(base_dir,'correlation')
for mouse in mice_list:

    print(f"Working on {mouse}:")
    if 'Thy1jRGECO' in mouse:
        load_dir = os.path.join(jRGECO_dir,'processed_data',mouse)
        data_file = f"{mouse}_data_dict.pkl"
    elif 'ThyCalbRCaMP' in mouse:
        load_dir = os.path.join(RCaMP_dir,'processed_data',mouse)
        data_file = f"{mouse}_df_dict.pkl"
    mouse_dict = load_pickle(load_dir, data_file)


    if 'Thy1jRGECO' in mouse:
        signal_deep_pre = mouse_dict['registered_clean_traces'][f"signal_deep_pre"].copy()
        signal_deep_rot = mouse_dict['registered_clean_traces'][f"signal_deep_rot"].copy()

        signal_sup_pre = mouse_dict['registered_clean_traces'][f"signal_sup_pre"].copy()
        signal_sup_rot = mouse_dict['registered_clean_traces'][f"signal_sup_rot"].copy()

    elif 'ThyCalbRCaMP' in mouse:
        fnames = list(mouse_dict.keys())
        fname_pre = [fname for fname in fnames if 'lt' in fname][0]
        fname_rot = [fname for fname in fnames if 'rot' in fname][0]

        signal_deep_pre = get_signal(mouse_dict[fname_pre], f"deep_clean_traces")
        signal_deep_rot = get_signal(mouse_dict[fname_rot], f"deep_clean_traces")

        signal_sup_pre = get_signal(mouse_dict[fname_pre], f"sup_clean_traces")
        signal_sup_rot = get_signal(mouse_dict[fname_rot], f"sup_clean_traces")

    signal_all_pre = np.concatenate((signal_sup_pre, signal_deep_pre), axis=1)
    signal_all_rot = np.concatenate((signal_sup_rot, signal_deep_rot), axis=1)

    #compute correlation matrix
    corr_mat_pre = np.corrcoef(signal_all_pre.T)
    corr_mat_pre = (corr_mat_pre + corr_mat_pre.T)/2 #symmetric
    corr_mat_pre[np.isnan(corr_mat_pre)] = 0


    corr_mat_rot = np.corrcoef(signal_all_rot.T)
    corr_mat_rot = (corr_mat_rot + corr_mat_rot.T)/2 #symmetric
    corr_mat_rot[np.isnan(corr_mat_rot)] = 0


    #cluster it
    pdist_uncondensed = 1.0 - corr_mat_pre
    pdist_condensed = np.concatenate([row[i+1:] for i, row in enumerate(pdist_uncondensed)])
    linkage = spc.linkage(pdist_condensed, method='complete')
    th = 0.9
    if 'ThyCalbRCaMP' in mouse: th = 0.95
    idx = spc.fcluster(linkage, th * pdist_condensed.max(), 'distance')
    new_order = np.argsort(idx)
    ordered_corr_mat_pre = corr_mat_pre[:, new_order]
    ordered_corr_mat_pre = ordered_corr_mat_pre[new_order,:]


    cluster_mat_pre = np.zeros(corr_mat_pre.shape)*np.nan
    old_cluster = 1
    start_cluster = 0
    for neu in range(cluster_mat_pre.shape[0]):
        cluster_id = idx[new_order[neu]]
        if cluster_id == old_cluster:
            continue;
        else:
            end_cluster = neu -1;
            cluster_mat_pre[start_cluster:end_cluster+1, start_cluster:end_cluster+1] = old_cluster-1;
            start_cluster = neu
            old_cluster = cluster_id
    cluster_mat_pre[start_cluster:, start_cluster:] = cluster_id-1;

    for neu in range(cluster_mat_pre.shape[0]):
        if new_order[neu] < signal_sup_pre.shape[1]:
            cluster_mat_pre[neu,neu] = 12
        else:
            cluster_mat_pre[neu,neu] = 8


    #apply order to rot
    ordered_corr_mat_rot = corr_mat_rot[:, new_order]
    ordered_corr_mat_rot = ordered_corr_mat_rot[new_order,:]

    fig = plt.figure(figsize= (14,8))
    ax = plt.subplot(2,2,1)
    b = ax.matshow(corr_mat_pre, vmin= 0, vmax = 0.8)
    ax.set_title(f'Original pre (<{signal_sup_pre.shape[1]} sup)')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)

    ax = plt.subplot(2,2,2)
    b = ax.matshow(ordered_corr_mat_pre, vmin= 0, vmax = 0.8)
    ax.set_title('Order pre')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)

    ax = plt.subplot(2,2,3)
    b = ax.matshow(cluster_mat_pre, cmap='tab20')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_title('Cluster pre')

    ax = plt.subplot(2,2,4)
    b = ax.matshow(ordered_corr_mat_rot, vmin= 0, vmax = 0.8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_title('Rot ordered by pre')

    plt.suptitle(mouse)

    plt.savefig(os.path.join(save_dir,f'{mouse}_correlation_pre_to_rot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_correlation_pre_to_rot.png'), dpi = 400,bbox_inches="tight")


    correlation_dict = {
        'mouse': mouse,
        'signal_sup_pre': signal_sup_pre,
        'signal_sup_rot': signal_sup_rot, 

        'signal_deep_pre': signal_deep_pre,
        'signal_deep_rot': signal_deep_rot, 

        'signal_all_pre': signal_all_pre,
        'signal_all_rot': signal_all_rot,

        'corr_mat_pre': corr_mat_pre, 
        'corr_mat_rot': corr_mat_rot,

        'pdist_uncondensed': pdist_uncondensed,
        'pdist_condensed': pdist_condensed,
        'linkage': linkage,
        'idx': idx,
        'new_order': new_order,
        'ordered_corr_mat_pre': ordered_corr_mat_pre,
        'cluster_mat_pre': cluster_mat_pre,
        'ordered_corr_mat_rot': ordered_corr_mat_rot
    }

    with open(os.path.join(save_dir, f"{mouse}_correlation_dict.pkl"), "wb") as file:
        pickle.dump(correlation_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                              rot to pre                                |#
#|________________________________________________________________________|#

mice_list = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']

base_dir = '/home/julio/Documents/DeepSup_project/DualColor/'

jRGECO_dir = os.path.join(base_dir,'Thy1jRGECO')
RCaMP_dir = os.path.join(base_dir,'ThyCalbRCaMP')
save_dir = os.path.join(base_dir,'correlation')
for mouse in mice_list:

    print(f"Working on {mouse}:")
    if 'Thy1jRGECO' in mouse:
        load_dir = os.path.join(jRGECO_dir,'processed_data',mouse)
        data_file = f"{mouse}_data_dict.pkl"
    elif 'ThyCalbRCaMP' in mouse:
        load_dir = os.path.join(RCaMP_dir,'processed_data',mouse)
        data_file = f"{mouse}_df_dict.pkl"
    mouse_dict = load_pickle(load_dir, data_file)


    if 'Thy1jRGECO' in mouse:
        signal_deep_pre = mouse_dict['registered_clean_traces'][f"signal_deep_pre"].copy()
        signal_deep_rot = mouse_dict['registered_clean_traces'][f"signal_deep_rot"].copy()

        signal_sup_pre = mouse_dict['registered_clean_traces'][f"signal_sup_pre"].copy()
        signal_sup_rot = mouse_dict['registered_clean_traces'][f"signal_sup_rot"].copy()

    elif 'ThyCalbRCaMP' in mouse:
        fnames = list(mouse_dict.keys())
        fname_pre = [fname for fname in fnames if 'lt' in fname][0]
        fname_rot = [fname for fname in fnames if 'rot' in fname][0]

        signal_deep_pre = get_signal(mouse_dict[fname_pre], f"deep_clean_traces")
        signal_deep_rot = get_signal(mouse_dict[fname_rot], f"deep_clean_traces")

        signal_sup_pre = get_signal(mouse_dict[fname_pre], f"sup_clean_traces")
        signal_sup_rot = get_signal(mouse_dict[fname_rot], f"sup_clean_traces")

    signal_all_pre = np.concatenate((signal_sup_pre, signal_deep_pre), axis=1)
    signal_all_rot = np.concatenate((signal_sup_rot, signal_deep_rot), axis=1)

    #compute correlation matrix
    corr_mat_pre = np.corrcoef(signal_all_pre.T)
    corr_mat_pre = (corr_mat_pre + corr_mat_pre.T)/2 #symmetric
    corr_mat_pre[np.isnan(corr_mat_pre)] = 0


    corr_mat_rot = np.corrcoef(signal_all_rot.T)
    corr_mat_rot = (corr_mat_rot + corr_mat_rot.T)/2 #symmetric
    corr_mat_rot[np.isnan(corr_mat_rot)] = 0


    #cluster it

    pdist_uncondensed = 1.0 - corr_mat_rot
    pdist_condensed = np.concatenate([row[i+1:] for i, row in enumerate(pdist_uncondensed)])
    linkage = spc.linkage(pdist_condensed, method='complete')
    th = 0.9
    if 'ThyCalbRCaMP' in mouse: th = 0.95
    idx = spc.fcluster(linkage, th * pdist_condensed.max(), 'distance')
    new_order = np.argsort(idx)
    ordered_corr_mat_rot = corr_mat_rot[:, new_order]
    ordered_corr_mat_rot = ordered_corr_mat_rot[new_order,:]


    cluster_mat_rot = np.zeros(corr_mat_rot.shape)*np.nan
    old_cluster = 1
    start_cluster = 0
    for neu in range(cluster_mat_rot.shape[0]):
        cluster_id = idx[new_order[neu]]
        if cluster_id == old_cluster:
            continue;
        else:
            end_cluster = neu -1;
            cluster_mat_rot[start_cluster:end_cluster+1, start_cluster:end_cluster+1] = old_cluster-1;
            start_cluster = neu
            old_cluster = cluster_id
    cluster_mat_rot[start_cluster:, start_cluster:] = cluster_id-1;

    for neu in range(cluster_mat_rot.shape[0]):
        if new_order[neu] < signal_sup_rot.shape[1]:
            cluster_mat_rot[neu,neu] = 12
        else:
            cluster_mat_rot[neu,neu] = 8


    #apply order to rot
    ordered_corr_mat_pre = corr_mat_pre[:, new_order]
    ordered_corr_mat_pre = ordered_corr_mat_pre[new_order,:]

    fig = plt.figure(figsize= (14,8))
    ax = plt.subplot(2,2,1)
    b = ax.matshow(corr_mat_rot, vmin= 0, vmax = 0.8)
    ax.set_title(f'Original rot (<{signal_sup_rot.shape[1]} sup)')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)

    ax = plt.subplot(2,2,2)
    b = ax.matshow(ordered_corr_mat_rot, vmin= 0, vmax = 0.8)
    ax.set_title('Order rot')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)

    ax = plt.subplot(2,2,3)
    b = ax.matshow(cluster_mat_rot, cmap='tab20')
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_title('Cluster rot')

    ax = plt.subplot(2,2,4)
    b = ax.matshow(ordered_corr_mat_pre, vmin= 0, vmax = 0.8)
    cbar = fig.colorbar(b, ax=ax, location='right',anchor=(0, 0.3), shrink=0.8)
    ax.set_title('Pre ordered by rot')

    plt.suptitle(mouse)

    plt.savefig(os.path.join(save_dir,f'{mouse}_correlation_rot_to_pre.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_correlation_rot_to_pre.png'), dpi = 400,bbox_inches="tight")


    correlation_dict = {
        'mouse': mouse,
        'signal_sup_pre': signal_sup_pre,
        'signal_sup_rot': signal_sup_rot, 

        'signal_deep_pre': signal_deep_pre,
        'signal_deep_rot': signal_deep_rot, 

        'signal_all_pre': signal_all_pre,
        'signal_all_rot': signal_all_rot,

        'corr_mat_pre': corr_mat_pre, 
        'corr_mat_rot': corr_mat_rot,

        'pdist_uncondensed': pdist_uncondensed,
        'pdist_condensed': pdist_condensed,
        'linkage': linkage,
        'idx': idx,
        'new_order': new_order,
        'ordered_corr_mat_rot': ordered_corr_mat_rot,
        'cluster_mat_rot': cluster_mat_rot,

        'ordered_corr_mat_pre': ordered_corr_mat_pre
    }

    with open(os.path.join(save_dir, f"{mouse}_correlation_inverse_dict.pkl"), "wb") as file:
        pickle.dump(correlation_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


#__________________________________________________________________________
#|                                                                        |#
#|                                  PLOT                                  |#
#|________________________________________________________________________|#

#PLOT
import pandas as pd
import seaborn as sns
save_dir = os.path.join(base_dir,'correlation')

#pre to rot
size = []
mean_corr_pre = []
mean_corr_rot = []
ratio = []
animal_list = []
for mouse in mice_list:
    correlation_dict = load_pickle(save_dir,f"{mouse}_correlation_dict.pkl")
    idx = correlation_dict['idx']
    signal_sup_pre = correlation_dict['signal_sup_pre']
    corr_mat_pre = correlation_dict['corr_mat_pre']
    corr_mat_rot = correlation_dict['corr_mat_rot']
    for cluster_id in np.unique(idx):

        size.append(np.sum(idx==cluster_id))
        animal_list.append(mouse)
        ratio.append(np.sum(idx[:signal_sup_pre.shape[1]]==cluster_id)/np.sum(idx[signal_sup_pre.shape[1]:]==cluster_id))
        mean_corr_pre.append(np.nanmean(corr_mat_pre[idx==cluster_id][:, idx==cluster_id]))
        mean_corr_rot.append(np.nanmean(corr_mat_rot[idx==cluster_id][:, idx==cluster_id]))



corr_pd =  pd.DataFrame(data={'idx': [x for x in range(len(mean_corr_pre))]*2,
                            'mouse': animal_list*2,
                            'size': size*2,
                            'corr': mean_corr_pre+mean_corr_rot,
                            'ratio sup/deep': ratio + ratio,
                            'cond': ['pre']*len(mean_corr_pre) + ['rot']*len(mean_corr_rot)})


#rot to pre
size = []
mean_corr_pre = []
mean_corr_rot = []
ratio = []
animal_list = []
for mouse in mice_list:
    correlation_dict = load_pickle(save_dir,f"{mouse}_correlation_inverse_dict.pkl")
    idx = correlation_dict['idx']
    signal_sup_pre = correlation_dict['signal_sup_pre']
    corr_mat_pre = correlation_dict['corr_mat_pre']
    corr_mat_rot = correlation_dict['corr_mat_rot']
    for cluster_id in np.unique(idx):
        size.append(np.sum(idx==cluster_id))
        animal_list.append(mouse)
        ratio.append(np.sum(idx[:signal_sup_pre.shape[1]]==cluster_id)/np.sum(idx[signal_sup_pre.shape[1]:]==cluster_id))
        mean_corr_pre.append(np.nanmean(corr_mat_pre[idx==cluster_id][:, idx==cluster_id]))
        mean_corr_rot.append(np.nanmean(corr_mat_rot[idx==cluster_id][:, idx==cluster_id]))



corr_inv_pd =  pd.DataFrame(data={'idx': [x for x in range(len(mean_corr_pre))]*2,
                            'mouse': animal_list*2,
                            'size': size*2,
                            'corr': mean_corr_pre+mean_corr_rot,
                            'ratio sup/deep': ratio + ratio,
                            'cond': ['pre']*len(mean_corr_pre) + ['rot']*len(mean_corr_rot)})



plt.figure()
ax = plt.subplot(1,2,1)
sns.barplot(x='cond', y='corr', data=corr_pd, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='cond', y='corr', data=corr_pd, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='cond', y= 'corr', data=corr_pd, units = 'idx', ax = ax, estimator = None, color = ".7", markers = True)
ax = plt.subplot(1,2,2)
sns.barplot(x='cond', y='corr', data=corr_inv_pd, linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='cond', y='corr', data=corr_inv_pd, palette = 'dark:gray', edgecolor = 'gray', ax = ax)
sns.lineplot(x='cond', y= 'corr', data=corr_inv_pd, units = 'idx', ax = ax, estimator = None, color = ".7", markers = True)
plt.savefig(os.path.join(save_dir,f'cluster_correlation_inverse_barplot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'cluster_correlation_inverse_barplot.png'), dpi = 400,bbox_inches="tight")


corr_pd_temp = corr_pd[corr_pd['cond']=='pre']
corr_inv_pd_temp = corr_inv_pd[corr_inv_pd['cond']=='rot']
corr_pd_both = pd.concat([corr_pd_temp, corr_inv_pd_temp])

plt.figure()
ax = plt.subplot(1,2,1)
sns.scatterplot(x='ratio sup/deep', y='corr', data=corr_pd_both, hue = 'cond', style = 'mouse', ax = ax)
ax = plt.subplot(1,2,2)
sns.scatterplot(x='ratio sup/deep', y='size', data=corr_pd_both, hue = 'cond', style = 'mouse', ax = ax)
plt.savefig(os.path.join(save_dir,f'cluster_correlation_scatter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'cluster_correlation_scatter.png'), dpi = 400,bbox_inches="tight")



correlation_dict = load_pickle(save_dir,f"{mouse}_correlation_dict.pkl")
idx = correlation_dict['idx']
signal_sup_pre = correlation_dict['signal_sup_pre']
corr_mat_pre = correlation_dict['corr_mat_pre']
corr_mat_rot = correlation_dict['corr_mat_rot']


sup_to_deep_corr_mat = corr_mat_pre[:signal_sup_pre.shape[1], signal_sup_pre.shape[1]:]
pdist_uncondensed = 1.0 - sup_to_deep_corr_mat
pdist_condensed = np.concatenate([row[i+1:] for i, row in enumerate(pdist_uncondensed)])
linkage = spc.linkage(pdist_condensed, method='complete')
th = 0.9
if 'ThyCalbRCaMP' in mouse: th = 0.95
idx = spc.fcluster(linkage, th * pdist_condensed.max(), 'distance')
new_order = np.argsort(idx)




#rot to pre
mean_corr_pre = []
mean_corr_rot = []
animal_list = []
case = []
for mouse in mice_list:
    correlation_dict = load_pickle(save_dir,f"{mouse}_correlation_inverse_dict.pkl")
    idx = correlation_dict['idx']
    signal_sup_pre = correlation_dict['signal_sup_pre']
    corr_mat_pre = correlation_dict['corr_mat_pre']
    corr_mat_rot = correlation_dict['corr_mat_rot']
    for neu_a in range(corr_mat_pre.shape[0]):
        for neu_b in range(neu_a+1, corr_mat_pre.shape[0]):
            animal_list.append(mouse)

            mean_corr_pre.append(corr_mat_pre[neu_a, neu_b])
            mean_corr_rot.append(corr_mat_rot[neu_a, neu_b])
            if neu_a < signal_sup_pre.shape[1]:
                pre_name = 'sup'
            else:
                pre_name = 'deep'
            if neu_b < signal_sup_pre.shape[1]:
                rot_name = 'sup'
            else:
                rot_name = 'deep'
            case.append(pre_name + '_to_' + rot_name)



corr_pre_rot_pd =  pd.DataFrame(data={
                            'mouse': animal_list,
                            'corr_pre': mean_corr_pre,
                            'corr_rot': mean_corr_rot,
                            'case': case})



plt.figure()
ax = plt.subplot(1,1,1)
sns.scatterplot(x='corr_pre', y='corr_rot', data=corr_pre_rot_pd, hue = 'case', style = 'mouse', ax = ax, alpha = 0.3,
    palette = ["#9900ffff", [0.5,0.5,0.5],"#cc9900ff"])
ax.set_aspect('equal')
plt.savefig(os.path.join(save_dir,f'deep_sup_correlation_scatter.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'deep_sup_correlation_scatter.png'), dpi = 400,bbox_inches="tight")



plt.figure()
ax = plt.subplot(1,2,1)
sns.kdeplot(corr_pre_rot_pd, x='corr_pre', fill=True, alpha=0.3, hue = 'case',
    common_norm=False, palette = ["#9900ffff", [0.5,0.5,0.5],"#cc9900ff"])
ax = plt.subplot(1,2,2)
sns.kdeplot(corr_pre_rot_pd, x='corr_rot', fill=True, alpha=0.3, hue = 'case',
    common_norm=False, palette = ["#9900ffff", [0.5,0.5,0.5],"#cc9900ff"])
plt.savefig(os.path.join(save_dir,f'deep_sup_correlation_kde.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'deep_sup_correlation_kde.png'), dpi = 400,bbox_inches="tight")