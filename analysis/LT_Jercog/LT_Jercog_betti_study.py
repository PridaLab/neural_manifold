from ripser import ripser as tda
import os, copy
from sklearn.metrics import pairwise_distances
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

col_list = ['r', 'g', 'm', 'c']

mice_list = ['M2019', 'M2021', 'M2023', 'M2024', 'M2025', 'M2026']
signal_name= 'clean_traces'

base_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving'
save_dir = os.path.join(base_dir, signal_name, 'betti_numbers')
data_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results/moving/same_len_data/'

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
        sub_perc = np.arange(10, 110, 10)
        stop_flag = False
        for s in sub_perc:
            if not stop_flag:
                n_points = int((s / 100) * clean_signal.shape[0])
                print(f"Using {s}% of points: {n_points}/{clean_signal.shape[0]}")
                idx = np.random.choice(clean_signal.shape[0], n_points)
                sclean_signal = clean_signal.copy()[idx,:]
                try:
                    barcodes = tda(sclean_signal, maxdim=2, coeff=2, thresh=max_dist)['dgms']
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

