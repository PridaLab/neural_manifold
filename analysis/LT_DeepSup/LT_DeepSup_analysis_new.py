import umap, math
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import neural_manifold.decoders as dec 
import neural_manifold.general_utils as gu
import copy
import matplotlib.pyplot as plt

def add_dir_mat_field(pd_struct):
    out_pd = copy.deepcopy(pd_struct)
    if 'dir_mat' not in out_pd.columns:
        out_pd["dir_mat"] = [np.zeros((out_pd["pos"][idx].shape[0],1)).astype(int)+
                            ('L' == out_pd["dir"][idx])+ 2*('R' == out_pd["dir"][idx])
                            for idx in out_pd.index]
    return out_pd


def preprocess_traces(pd_struct_p, pd_struct_r, signal_field, sigma = 5, MAD_th = 5):

    out_pd_p = copy.deepcopy(pd_struct_p)
    out_pd_r = copy.deepcopy(pd_struct_r)

    out_pd_p["index_mat"] = [np.zeros((out_pd_p[signal_field][idx].shape[0],1))+out_pd_p["trial_id"][idx] 
                                  for idx in range(out_pd_p.shape[0])]                     
    index_mat_p = np.concatenate(out_pd_p["index_mat"].values, axis=0)

    out_pd_r["index_mat"] = [np.zeros((out_pd_r[signal_field][idx].shape[0],1))+out_pd_r["trial_id"][idx] 
                                  for idx in range(out_pd_r.shape[0])]
    index_mat_r = np.concatenate(out_pd_r["index_mat"].values, axis=0)

    signal_p_og = copy.deepcopy(np.concatenate(pd_struct_p[signal_field].values, axis=0))
    lowpass_p = uniform_filter1d(signal_p_og, size = 4000, axis = 0)
    signal_p = gaussian_filter1d(signal_p_og, sigma = sigma, axis = 0)

    signal_r_og = copy.deepcopy(np.concatenate(pd_struct_r[signal_field].values, axis=0))
    lowpass_r = uniform_filter1d(signal_r_og, size = 4000, axis = 0)
    signal_r = gaussian_filter1d(signal_r_og, sigma = sigma, axis = 0)

    for nn in range(signal_p.shape[1]):
        base_p = np.histogram(signal_p_og[:,nn], 100)
        base_p = base_p[1][np.argmax(base_p[0])]
        base_p = base_p + lowpass_p[:,nn] - np.min(lowpass_p[:,nn]) 

        base_r = np.histogram(signal_r_og[:,nn], 100)
        base_r = base_r[1][np.argmax(base_r[0])]   
        base_r = base_r + lowpass_r[:,nn] - np.min(lowpass_r[:,nn])   

        concat_signal = np.concatenate((signal_p[:,nn]-base_p, signal_r[:,nn]-base_r))

        concat_signal = concat_signal/np.max(concat_signal,axis = 0)
        concat_signal[concat_signal<0] = 0
        MAD = np.median(abs(concat_signal - np.median(concat_signal)), axis=0)

        concat_signal[concat_signal<MAD_th*MAD] = 0

        signal_p[:,nn] = concat_signal[:signal_p.shape[0]]
        signal_r[:,nn] = concat_signal[signal_p.shape[0]:]

    out_pd_p['clean_traces'] = [signal_p[index_mat_p[:,0]==out_pd_p["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_p.shape[0])]
    out_pd_r['clean_traces'] = [signal_r[index_mat_r[:,0]==out_pd_r["trial_id"][idx] ,:] 
                                                                for idx in range(out_pd_r.shape[0])]
    return out_pd_p, out_pd_r



signal_field = 'raw_traces'
vel_th = 8
nn_val = 60
dim = 3
mouse = 'TGrin1'
load_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/' + mouse
sigma = 5
MAD_th = 5

animal = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")
fnames = list(animal.keys())

animal_p= copy.deepcopy(animal[fnames[0]])
animal_r= copy.deepcopy(animal[fnames[1]])

animal_p = add_dir_mat_field(animal_p)
animal_r = add_dir_mat_field(animal_r)

animal_r,_ = gu.keep_only_moving(animal_r, vel_th)
animal_p,_ = gu.keep_only_moving(animal_p, vel_th)

animal_p, animal_r = preprocess_traces(animal_p, animal_r, signal_field, sigma=sigma, MAD_th=MAD_th)









signal_p = copy.deepcopy(np.concatenate(animal_p['clean_traces'].values, axis=0))
pos_p = copy.deepcopy(np.concatenate(animal_p['pos'].values, axis=0))
dir_mat_p = copy.deepcopy(np.concatenate(animal_p['dir_mat'].values, axis=0))
vel_p = copy.deepcopy(np.concatenate(animal_p['vel'].values, axis=0))

signal_r = copy.deepcopy(np.concatenate(animal_r['clean_traces'].values, axis=0))
pos_r = copy.deepcopy(np.concatenate(animal_r['pos'].values, axis=0))
dir_mat_r = copy.deepcopy(np.concatenate(animal_r['dir_mat'].values, axis=0))
vel_r = copy.deepcopy(np.concatenate(animal_r['vel'].values, axis=0))



#%%all data
index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
concat_signal = np.vstack((signal_p, signal_r))
# concat_signal = np.vstack((signal_p[:, cells_to_keep], signal_r[:, cells_to_keep]))

model = umap.UMAP(n_neighbors = nn_val, n_components =3, min_dist=0.1)
model.fit(concat_signal)
concat_emb = model.transform(concat_signal)
emb_p = concat_emb[index[:,0]==0,:]
emb_r = concat_emb[index[:,0]==1,:]


TAB, RAB = dec.align_manifolds_1D(emb_p, emb_r, pos_p[:,0], pos_r[:,0], dir_mat_p, dir_mat_r, ndims = 3, nCentroids =20)   
tr = (np.trace(RAB)-1)/2
if abs(tr)>1:
    tr = round(tr,2)
    if abs(tr)>1:
        tr = np.nan
angle_10 = math.acos(tr)*180/np.pi
print(f"Umap all 10: {angle_10:.2f}º ")

#%%
plt.figure()
ax = plt.subplot(1,2,1, projection = '3d')
ax.scatter(*emb_p[:,:3].T, color ='r', s= 30, cmap = 'magma')
ax.scatter(*emb_r[:,:3].T, color = 'b', s= 30, cmap = 'magma')

ax = plt.subplot(1,2,2, projection = '3d')
ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')
ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')


#%%
plt.figure()
ax = plt.subplot(2,3,1, projection = '3d')
ax.scatter(*emb_p[:,:3].T, c = pos_p[:,0], s= 30, cmap = 'magma')

ax = plt.subplot(2,3,4, projection = '3d')
ax.scatter(*emb_r[:,:3].T, c = pos_r[:,0], s= 30, cmap = 'magma')

ax = plt.subplot(2,3,2, projection = '3d')
ax.scatter(*emb_p[:,:3].T, c = dir_mat_p, s= 30, cmap = 'magma')
ax = plt.subplot(2,3,5, projection = '3d')
ax.scatter(*emb_r[:,:3].T, c = dir_mat_r, s= 30, cmap = 'magma')

ax = plt.subplot(2,3,3, projection = '3d')
ax.scatter(*emb_p[:,:3].T, c = np.arange(pos_p.shape[0]), s= 30, cmap = 'magma')
ax = plt.subplot(2,3,6, projection = '3d')
ax.scatter(*emb_r[:,:3].T, c = np.arange(pos_r.shape[0]), s= 30, cmap = 'magma')









from scipy.signal import correlate
temp = np.zeros((3,3))
for ii in range(3):
    for jj in range(ii+1,3):
        temp[ii,jj] = np.corrcoef(emb[:,ii], emb[:,jj])[0,1]
        temp[jj,ii] = temp[ii,jj]


cells_to_keep = np.zeros((signal_p_og.shape[1],), dtype=bool)*False

for n in range(signal_p_og.shape[1]):

    ylims = [np.min([np.min(signal_p_og[:,n]), np.min(signal_r_og[:,n])]),
            1.1*np.max([np.max(signal_p_og[:,n]), np.max(signal_r_og[:,n])]) ]

    f = plt.figure()
    ax = plt.subplot(2,2,1)
    ax.plot(signal_p_og[:,n])
    base = np.histogram(signal_p_og[:,n], 100)
    base_p = base[1][np.argmax(base[0])]
    base_p = base_p + lowpass_p[:,n] - np.min(lowpass_p[:,n])   
    ax.plot(base_p, color = 'r')
    ax.set_ylim(ylims)

    ax = plt.subplot(2,2,2)
    ax.plot(signal_r_og[:,n])
    base = np.histogram(signal_r_og[:,n], 100)
    base_r = base[1][np.argmax(base[0])]
    base_r = base_r + lowpass_r[:,n] - np.min(lowpass_r[:,n])   
    ax.plot(base_r, color = 'r')
    ax.set_ylim(ylims)

    ax = plt.subplot(2,2,3)
    ax.plot(signal_p[:,n])
    ax.set_ylim([-0.05, 1.5])

    ax = plt.subplot(2,2,4)
    ax.plot(signal_r[:,n])
    ax.set_ylim([-0.05, 1.5])
    plt.suptitle(n)
    a = input()
    cells_to_keep[n] = not any(a)
    plt.close(f)



xlim = [3,13]
ylim = [7,15]
zlim=  [3, 12]


x_idx = (emb_r[:,0]>xlim[0])*(emb_r[:,0]<xlim[1])
y_idx = (emb_r[:,1]>ylim[0])*(emb_r[:,1]<ylim[1])
z_idx = (emb_r[:,2]>zlim[0])*(emb_r[:,2]<zlim[1])
in_idx = np.where(x_idx*y_idx*z_idx)[0]


signal_r = signal_r[in_idx]
pos_r = pos_r[in_idx]
vel_r = vel_r[in_idx]
dir_mat_r = dir_mat_r[in_idx]

index = np.vstack((np.zeros((signal_p.shape[0],1)),np.zeros((signal_r.shape[0],1))+1))
concat_signal = np.vstack((signal_p, signal_r))

model = umap.UMAP(n_neighbors = nn_val, n_components =dim, min_dist=0.1)
model.fit(concat_signal)
concat_emb = model.transform(concat_signal)
emb_p = concat_emb[index[:,0]==0,:]
emb_r = concat_emb[index[:,0]==1,:]


    # ylims = [np.min([np.min(signal_p_og[:,nn]), np.min(signal_r_og[:,nn])]),1.1*np.max([np.max(signal_p_og[:,nn]), np.max(signal_r_og[:,nn])]) ]
    # f = plt.figure()
    # ax = plt.subplot(2,2,1)
    # ax.plot(signal_p_og[:,nn])
    # ax.plot(base_p, color = 'r')
    # ax.plot(base_p-2*std_p, color = 'r', linestyle= '--')
    # ax.plot(base_p+2*std_p, color = 'r', linestyle= '--')
    # ax.set_ylim(ylims)
    # ax = plt.subplot(2,2,2)
    # ax.plot(signal_r_og[:,nn])
    # ax.plot(base_r, color = 'r')
    # ax.plot(base_r-2*std_r, color = 'r', linestyle= '--')
    # ax.plot(base_r+2*std_r, color = 'r', linestyle= '--')
    # ax.set_ylim(ylims)

    # ax = plt.subplot(2,1,2)
    # ax.plot(np.arange(signal_p.shape[0]), signal_p[:,nn])
    # ax.plot(np.arange(signal_r.shape[0])+ signal_p.shape[0]+100, signal_r[:,nn])
    
    # plt.suptitle(nn)
    # a = input()
    # # cells_to_keep[n] = not any(a)
    # plt.close(f)





    signal_p[:,nn] = (signal_p[:,nn] - base)/(np.percentile(signal_p[:,nn], 99.9, axis = 0)-base)
signal_p[signal_p<0] = 0
MAD_p = np.median(abs(signal_p - np.tile(np.median(signal_p, axis=0), (signal_p.shape[0],1))), axis=0)
signal_p[signal_p<5*np.tile(MAD_p,(signal_p.shape[0],1))] = 0
# signal_p[signal_p_og<1] = 0

signal_r_og = copy.deepcopy(np.concatenate(animal_r[signal_field].values, axis=0))
lowpass_r = uniform_filter1d(signal_r_og, size = 4000, axis = 0)
signal_r = gaussian_filter1d(signal_r_og, sigma = 5, axis = 0)
for nn in range(signal_r.shape[1]):
    base = np.histogram(signal_r[:,nn], 100)
    base = base[1][np.argmax(base[0])]   
    base = base + lowpass_r[:,nn] - np.min(lowpass_r[:,nn])   
    signal_r[:,nn] = (signal_r[:,nn] - base)/(np.percentile(signal_r[:,nn], 99.9, axis = 0)-base)
signal_r[signal_r<0] = 0
MAD_r = np.median(abs(signal_r - np.tile(np.median(signal_r, axis=0), (signal_r.shape[0],1))), axis=0)
signal_r[signal_r<5*np.tile(MAD_r,(signal_r.shape[0],1))] = 0

