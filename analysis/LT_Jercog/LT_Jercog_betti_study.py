from ripser import ripser as tda
import os, copy
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx
#__________________________________________________________________________
#|                                                                        |#
#|                             BETTI NUMBERS                              |#
#|________________________________________________________________________|#
from ripser import ripser as tda
import os, copy
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import umap
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

def filter_noisy_outliers(data, D=None):
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,5), axis=1)
    noiseIdx = nn_dist < np.percentile(nn_dist, 20)
    sum(noiseIdx)
    return noiseIdx

def plot_betti_numbers(h0,h1):
    h0[~np.isfinite(h0)] = max_dist
    # Plot the 30 longest barcodes only
    to_plot = []
    for curr_h in [h0, h1]:
         bar_lens = curr_h[:,1] - curr_h[:,0]
         plot_h = curr_h[(-bar_lens).argsort()[:30]]
         to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
                lw=1.5)
        ax.set_ylabel('H' + str(curr_betti))
        ax.set_xlim([-0.5, np.max(np.vstack((h0,h1)))+0.5])
        # ax.set_xticks([0, xlim])
        ax.set_ylim([-1, len(curr_bar)])
    return fig
#__________________________________________________________________________
#|                                                                        |#
#|                             BETTI NUMBERS                              |#
#|________________________________________________________________________|#

col_list = ['r', 'g', 'm', 'c']
mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
signal_name= 'clean_traces'

# base_dir = '/lustre/home/ic/lcn/spatial_navigation/LT_Jercog/results'
# save_dir = os.path.join(base_dir, 'betti_numbers')
# data_dir = '/lustre/home/ic/lcn/spatial_navigation/LT_Jercog/processed_data'

base_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/'
save_dir = os.path.join(base_dir, 'betti_numbers')
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/processed_data/'

for mouse in mice_list:
    print('')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = load_pickle(file_path, file_name)
    fnames = list(mouse_dict.keys())
    betti_dict = dict()
    save_name = mouse + '_betti_nums.pkl'
    for f_idx, fname in enumerate(fnames):
        betti_dict[fname] = dict()
        print(f"Working on session: {fname} ({f_idx+1}/{len(fnames)})")
        pd_struct= copy.deepcopy(mouse_dict[fname])
        #signal
        signal = np.concatenate(pd_struct[signal_name].values, axis = 0)
        idx = np.random.permutation(np.arange(signal.shape[0]))[:n_points]
        print(f"UMAP:")
        for s in range(2,20):
            #UMAP
            model_umap = umap.UMAP(n_neighbors =120, n_components=s+1, min_dist=0.1)
            model_umap.fit(signal)
            emb_umap = model_umap.transform(signal)
            len_diff = list()
            betti_dict[fname]['umap'] = dict()
            semb_umap = emb_umap.copy()[idx,:]
            #clean outliers
            D = pairwise_distances(semb_umap)
            noiseIdx = filter_noisy_outliers(semb_umap,D=D)
            clean_emb = semb_umap[~noiseIdx,:]
            max_dist = np.round(np.nanmax(D))
            barcodes = tda(clean_emb, maxdim=1, coeff=2, thresh=max_dist)['dgms']
            results = {
                'h0': barcodes[0],
                'h1': barcodes[1],
                'signal': clean_emb,
                'noiseIdx': noiseIdx,
                'max_dist': max_dist,
                'signal_name': signal_name
            }
            betti_dict[fname]['umap'][s+1] = results
            bar_lens = barcodes[1][:,1] - barcodes[1][:,0]
            bar_lens =  bar_lens[(-bar_lens).argsort()[:30]]
            len_diff.append(bar_lens[0]/bar_lens[1])
            print(f"\tUsing {s+1} components - len: {len_diff[-1]:.2f}")
            betti_dict[fname]['umap']['len_diff'] = len_diff
            with open(os.path.join(save_dir, save_name), "wb") as file:
                pickle.dump(betti_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
            fig = plot_betti_numbers(barcodes[0],barcodes[1])
            plt.suptitle(f"{fname}: UMAP {s+1} dim")
            plt.savefig(os.path.join(save_dir,f'{fname}_{s+1}_umap_betti.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
            plt.savefig(os.path.join(save_dir,f'{fname}_{s+1}_umap_betti.svg'), dpi = 400,bbox_inches="tight",transparent=True)
            plt.close(fig)





col_list = ['r', 'g', 'm', 'c']

mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
signal_name= 'clean_traces'

base_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/'
save_dir = os.path.join(base_dir, 'betti_numbers')
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/processed_data/'

for mouse in mice_list:
    print('')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = load_pickle(file_path, file_name)
    fnames = list(mouse_dict.keys())
    betti_dict = dict()
    save_name = mouse + '_betti_nums.pkl'
    for f_idx, fname in enumerate(fnames):
        betti_dict[fname] = dict()
        print(f"Working on session: {fname} ({f_idx+1}/{len(fnames)})")
        pd_struct= copy.deepcopy(mouse_dict[fname])
        #signal
        signal = np.concatenate(pd_struct[signal_name].values, axis = 0)

        #
        model_pca = PCA(signal.shape[1])
        model_pca.fit(signal)
        emb = model_pca.transofrm(signal)[:,:10]
        #clean outliers
        D = pairwise_distances(emb)
        noiseIdx = filter_noisy_outliers(emb,D=D)
        clean_emb = emb[~noiseIdx,:]
        max_dist = np.round(np.nanmax(D))
        sub_perc = np.arange(10, 110, 10)
        stop_flag = False
        for s in sub_perc:
            if not stop_flag:
                n_points = int((s / 100) * clean_signal.shape[0])
                print(f"Using {s}% of points: {n_points}/{clean_signal.shape[0]}")
                idx = np.random.choice(clean_signal.shape[0], n_points)
                sclean_signal = clean_signal.copy()[idx,:]
                try:
                    barcodes = tda(clean_signal, maxdim=1, coeff=2, thresh=max_dist)['dgms']
                    results = {
                        'h0': barcodes[0],
                        'h1': barcodes[1],
                        'h2': barcodes[2],
                        'signal': sclean_signal,
                        'signal_idx': idx,
                        'noiseIdx': noiseIdx,
                        'max_dist': max_dist,
                        'signal_name': signal_name
                    }
                    betti_dict[fname][s] = results
                    with open(os.path.join(save_dir, save_name), "wb") as file:
                        pickle.dump(betti_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


                    h0, h1, h2 = results['h0'], results['h1'], results['h2']
                    # replace the infinity bar (-1) in H0 by a really large number
                    h0[~np.isfinite(h0)] = max_dist
                    # Plot the 30 longest barcodes only
                    to_plot = []
                    for curr_h in [h0, h1, h2]:
                         bar_lens = curr_h[:,1] - curr_h[:,0]
                         plot_h = curr_h[(-bar_lens).argsort()[:30]]
                         to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

                    fig = plt.figure(figsize=(10, 8))
                    gs = gridspec.GridSpec(3, 4)
                    for curr_betti, curr_bar in enumerate(to_plot):
                        ax = fig.add_subplot(gs[curr_betti, :])
                        for i, interval in enumerate(reversed(curr_bar)):
                            ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
                                lw=1.5)
                        ax.set_ylabel('H' + str(curr_betti))
                        ax.set_xlim([-0.5, np.max(np.vstack((h0,h1,h2)))+0.5])
                        # ax.set_xticks([0, xlim])
                        ax.set_ylim([-1, len(curr_bar)])
                    plt.suptitle(f"{fname}: {s}% ({n_points})")
                    plt.savefig(os.path.join(save_dir,f'{fname}_{s}_betti.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
                    plt.savefig(os.path.join(save_dir,f'{fname}_{s}_betti.svg'), dpi = 400,bbox_inches="tight",transparent=True)
                    plt.close(fig)

                except:
                    print(f"Error using {s}% of points: {n_points}/{clean_signal.shape[0]}")
                    stop_flag = True
                    continue



n_points = 3000
for mouse in mice_list:
    print('')
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    mouse_dict = load_pickle(file_path, file_name)
    fnames = list(mouse_dict.keys())
    betti_dict = dict()
    save_name = mouse + '_betti_nums.pkl'
    for f_idx, fname in enumerate(fnames):
        betti_dict[fname] = dict()
        print(f"Working on session: {fname} ({f_idx+1}/{len(fnames)})")
        pd_struct= copy.deepcopy(mouse_dict[fname])
        #signal
        signal = np.concatenate(pd_struct[signal_name].values, axis = 0)
        #clean outliers
        D = pairwise_distances(signal)
        noiseIdx = filter_noisy_outliers(signal,D=D)
        clean_signal = signal[~noiseIdx,:]
        max_dist = np.round(np.nanmax(D))
        for it in range(5):
            print(f"Iteration {it+1}/5")
            idx = np.random.permutation(np.arange(clean_signal.shape[0]))[:n_points]
            sclean_signal = clean_signal.copy()[idx,:]
            barcodes = tda(sclean_signal, maxdim=2, coeff=47, thresh=max_dist)['dgms']
            results = {
                'h0': barcodes[0],
                'h1': barcodes[1],
                'h2': barcodes[2],
                'signal': sclean_signal,
                'signal_idx': idx,
                'noiseIdx': noiseIdx,
                'max_dist': max_dist,
                'signal_name': signal_name
            }
            betti_dict[fname][it] = results
            with open(os.path.join(save_dir, save_name), "wb") as file:
                pickle.dump(betti_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
            h0, h1, h2 = results['h0'], results['h1'], results['h2']
            # replace the infinity bar (-1) in H0 by a really large number
            h0[~np.isfinite(h0)] = max_dist
            # Plot the 30 longest barcodes only
            to_plot = []
            for curr_h in [h0, h1, h2]:
                 bar_lens = curr_h[:,1] - curr_h[:,0]
                 plot_h = curr_h[(-bar_lens).argsort()[:30]]
                 to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(3, 4)
            for curr_betti, curr_bar in enumerate(to_plot):
                ax = fig.add_subplot(gs[curr_betti, :])
                for i, interval in enumerate(reversed(curr_bar)):
                    ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
                        lw=1.5)
                ax.set_ylabel('H' + str(curr_betti))
                ax.set_xlim([-0.5, np.max(np.vstack((h0,h1,h2)))+0.5])
                # ax.set_xticks([0, xlim])
                ax.set_ylim([-1, len(curr_bar)])
            plt.suptitle(f"{fname}: {it}% ({n_points})")
            plt.savefig(os.path.join(save_dir,f'{fname}_{it}it_betti.jpg'), dpi = 400,bbox_inches="tight",transparent=True)
            plt.savefig(os.path.join(save_dir,f'{fname}_{it}it_betti.svg'), dpi = 400,bbox_inches="tight",transparent=True)
            plt.close(fig)




def get_betti_results(betti_nums_dict, session):
    max_len = np.zeros((5,))
    mean_len = np.zeros((5,))

    last_idx = -1
    count_idx = 0
    fnames = list(betti_nums_dict.keys())
    for s_idx, s_name in enumerate(fnames):
        if s_idx==0:
            last_idx+=1
            count_idx = 1
        else:
            old_s_name = fnames[s_idx-1]
            old_s_name = old_s_name[:old_s_name.find('_',-5)]
            new_s_name = s_name[:s_name.find('_',-5)]
            if new_s_name == old_s_name:
                count_idx += 1
            else:
                max_len[session[last_idx]] = max_len[session[last_idx]]/count_idx
                mean_len[session[last_idx]] = mean_len[session[last_idx]]/count_idx
                last_idx +=1
                count_idx = 1
                
        pd_struct = betti_nums_dict[s_name]
        tmax_len = 0
        tmean_len = 0
        for it in range(5):
            h1 = pd_struct[it]['h1']
            bar_lens = np.diff(h1,axis=1)
            bar_lens = h1[:,1] - h1[:,0]
            h1_keep =  h1[(-bar_lens).argsort()[:30]]

            tmax_len += np.max(np.diff(h1_keep,axis=1))
            tmean_len += np.mean(np.diff(h1_keep,axis=1))

        max_len[session[last_idx]] += tmax_len/5
        mean_len[session[last_idx]] += tmean_len/5


    max_len[session[last_idx]] = max_len[session[last_idx]]/count_idx
    mean_len[session[last_idx]] = mean_len[session[last_idx]]/count_idx
    
    return max_len, mean_len

#%% LOAD DATA
if "M2019_betti_dict" not in locals():
    M2019_betti_dict = load_pickle(save_dir, 'M2019_betti_nums.pkl')
if "M2021_betti_dict" not in locals():
    M2021_betti_dict = load_pickle(save_dir, 'M2021_betti_nums.pkl')
if "M2023_betti_dict" not in locals():
    M2023_betti_dict = load_pickle(save_dir, 'M2023_betti_nums.pkl')
if "M2024_betti_dict" not in locals():
    M2024_betti_dict = load_pickle(save_dir, 'M2024_betti_nums.pkl')
if "M2025_betti_dict" not in locals():
    M2025_betti_dict = load_pickle(save_dir, 'M2025_betti_nums.pkl')
if "M2026_betti_dict" not in locals():
    M2026_betti_dict = load_pickle(save_dir, 'M2026_betti_nums.pkl')



max_len = np.zeros((5,6))
mean_len = np.zeros((5,6))
#M2019
max_len[:,0], mean_len[:,0] = get_betti_results(M2019_betti_dict, [0,1,2,4])
#M2021
max_len[:,1], mean_len[:,1] = get_betti_results(M2021_betti_dict, [0,1,2,4])
#M2023
max_len[:,2], mean_len[:,2] = get_betti_results(M2023_betti_dict, [0,1,2,4])
#M2024
max_len[:,3], mean_len[:,3] = get_betti_results(M2024_betti_dict, [0,1,2,3])
#M2025
max_len[:,4], mean_len[:,4] = get_betti_results(M2025_betti_dict, [0,1,2,3])
#M2026
max_len[:,5], mean_len[:,5] = get_betti_results(M2026_betti_dict, [0,1,2,3])



                 plot_h = curr_h[(-bar_lens).argsort()[:30]]
curr_h[(-bar_lens).argsort()[:30]]

# def sample_from_ring(r,sigma):
#     theta = random.uniform(0, 2*np.pi-0.05)

#     x = r*np.sin(theta)+ random.gauss(0, sigma)
#     y = r*np.cos(theta)+ random.gauss(0, sigma)
#     return (x,y), theta


# emb = np.empty((1500, 2))
# label = np.empty(emb.shape[0])

# for ii in range(emb.shape[0]):
#     emb[ii,:], label[ii] = sample_from_ring(1,0.01)

R = 2
r = 1
angle = np.linspace(0, 2*np.pi-0.05, 30)
x = np.zeros((angle.shape[0]**2,))
y = np.zeros((angle.shape[0]**2,))
z = np.zeros((angle.shape[0]**2,))

l_theta = np.zeros(x.shape)
count = 0
for theta in angle:
    for phi in angle:
        x[count] =  (R + r*np.cos(phi))*np.cos(theta) + np.random.rand()*0.05 - 0.025
        y[count] = (R + r*np.cos(phi))*np.sin(theta) + np.random.rand()*0.05 - 0.025
        z[count] = r * np.sin(phi)+ np.random.rand()*0.05 - 0.025
        l_theta[count] = theta
        count += 1
emb = np.array([x,y,z]).T


d= 10
sigma = 0.01
new_emb = np.zeros((emb.shape[0],d))
new_emb[:,:emb.shape[1]] = emb
if (d - emb.shape[1])>0:
    dim_sigma = np.sqrt((sigma**2)/(d-emb.shape[1]))
for ad in range(emb.shape[1],d):
    new_emb[:,ad] = np.random.normal(0,dim_sigma,emb.shape[0])
R,_ = np.linalg.qr(np.random.randn(d,d)) 
new_emb = np.matmul(R, new_emb.T).T

D = pairwise_distances(new_emb)
max_dist = np.round(np.nanmax(D))
noiseIdx = filter_noisy_outliers(new_emb,D=D)
barcodes = tda(new_emb, maxdim=1, coeff=2, thresh=max_dist)['dgms']
barcodes.append(barcodes[1])
h0, h1, h2 = barcodes[0], barcodes[1], barcodes[2]
col_list = ['r', 'g', 'm', 'c']
h0[~np.isfinite(h0)] = max_dist
# Plot the 30 longest barcodes only
to_plot = []
for curr_h in [h0, h1, h2]:
     bar_lens = curr_h[:,1] - curr_h[:,0]
     plot_h = curr_h[(-bar_lens).argsort()[:30]]
     to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 4)
for curr_betti, curr_bar in enumerate(to_plot):
    ax = fig.add_subplot(gs[curr_betti, :])
    for i, interval in enumerate(reversed(curr_bar)):
        ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
            lw=1.5)
    ax.set_ylabel('H' + str(curr_betti))
    ax.set_xlim([-0.5, np.max(np.vstack((h0,h1,h2)))+0.5])
    # ax.set_xticks([0, xlim])
    ax.set_ylim([-1, len(curr_bar)])


s_emb = np.zeros((new_emb.shape[0],new_emb.shape[1]))
for n in range(new_emb.shape[1]):
    lag = np.random.choice(new_emb.shape[0], 1)[0]
    s_emb[:lag,n] = new_emb[-lag:,n]
    s_emb[lag:,n] = new_emb[:-lag,n]

sbarcodes = list()
sbarcodes = tda(s_emb, maxdim=1, coeff=2, thresh=max_dist)['dgms']
sbarcodes.append(sbarcodes[1])
for t in range(9):
    print(t)
    s_emb = np.zeros((new_emb.shape[0],new_emb.shape[1]))
    for n in range(s_emb.shape[1]):
        lag = np.random.choice(new_emb.shape[0], 1)[0]
        s_emb[:lag,n] = new_emb[-lag:,n]
        s_emb[lag:,n] = new_emb[:-lag,n]

    tbarcodes = tda(s_emb, maxdim=1, coeff=2, thresh=max_dist)['dgms']
    sbarcodes[0] = np.concatenate((sbarcodes[0], tbarcodes[0]))
    sbarcodes[1] = np.concatenate((sbarcodes[1], tbarcodes[1]))
    sbarcodes[2] = np.concatenate((sbarcodes[2], tbarcodes[1]))


h0, h1, h2 = barcodes[0], barcodes[1], barcodes[2]
sh0, sh1, sh2 = sbarcodes[0], sbarcodes[1], sbarcodes[2]
h0[~np.isfinite(h0)] = max_dist
sh0[~np.isfinite(sh0)] = max_dist


slen = list()
slen.append(np.nanpercentile(np.diff(sh0, axis= 1), 99))
slen.append(np.nanpercentile(np.diff(sh1, axis= 1), 99))
slen.append(np.nanpercentile(np.diff(sh2, axis= 1), 99))

col_list = ['r', 'g', 'm', 'c']
h0[~np.isfinite(h0)] = max_dist
# Plot the 30 longest barcodes only
to_plot = []
for curr_h in [h0, h1, h2]:
     bar_lens = curr_h[:,1] - curr_h[:,0]
     plot_h = curr_h[(-bar_lens).argsort()[:30]]
     to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 4)
for curr_betti, curr_bar in enumerate(to_plot):
    ax = fig.add_subplot(gs[curr_betti, :])
    for i, interval in enumerate(reversed(curr_bar)):
        ax.plot([interval[0], interval[0]+slen[curr_betti]], [i, i], color=[.5,.5,.5],
            lw=3)
        ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
            lw=1.5)

    ax.set_ylabel('H' + str(curr_betti))
    ax.set_xlim([-0.5, np.max(np.vstack((h0,h1,h2)))+0.5])
    # ax.set_xticks([0, xlim])
    ax.set_ylim([-1, len(curr_bar)])