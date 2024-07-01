from sklearn.decomposition import PCA
import time
from sklearn.metrics import pairwise_distances
from os import listdir
from os.path import isfile, join
from scipy.ndimage import gaussian_filter1d


mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
cues_dir = '/home/julio/Documents/SP_project/LT_DeepSup/data'






def get_signal(pd_struct, signal):
    return copy.deepcopy(np.concatenate(pd_mouse[signal].values, axis=0))


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def get_dir_color(dir_mat):
    dir_color = np.zeros((dir_mat.shape[0],3))
    for point in range(dir_mat.shape[0]):
        if dir_mat[point]==0:
            dir_color[point] = [14/255,14/255,143/255]
        elif dir_mat[point]==-1:
            dir_color[point] = [12/255,136/255,249/255]
        elif dir_mat[point]==1:
            dir_color[point] = [17/255,219/255,224/255]
    return dir_color


def get_centroids(cloud, label, dire = None, num_centroids = 20):
    dims = cloud.shape[1]
    if label.ndim>1:
        label = label[:,0]

    #compute label max and min to divide into centroids
    label_lims = np.array([(np.percentile(label,5), np.percentile(label,95))]).T[:,0] 
    #find centroid size
    cent_size = (label_lims[1] - label_lims[0]) / (num_centroids)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    cent_edges = np.column_stack((np.linspace(label_lims[0],label_lims[0]+cent_size*(num_centroids),num_centroids),
                                np.linspace(label_lims[0],label_lims[0]+cent_size*(num_centroids),num_centroids)+cent_size))

    if isinstance(dire, type(None)):
        cent = np.zeros((num_centroids,dims))
        cent_label = np.mean(cent_edges,axis=1).reshape(-1,1)
        num_centroids = np.zeros((num_centroids,))
        for c in range(num_centroids):
            points = cloud[np.logical_and(label >= cent_edges[c,0], label<cent_edges[c,1]),:]
            cent[c,:] = np.median(points, axis=0)
            num_centroids[c] = points.shape[0]

    else:
        cloud_left = copy.deepcopy(cloud[dire==-1,:])
        label_left = copy.deepcopy(label[dire==-1])
        cloud_right = copy.deepcopy(cloud[dire==1,:])
        label_right = copy.deepcopy(label[dire==1])
        

        
        cent = np.zeros((2*num_centroids,dims))
        num_points = np.zeros((2*num_centroids,))
        
        cent_dir = np.zeros((2*num_centroids, ))
        cent_label = np.tile(np.mean(cent_edges,axis=1),(2,1)).T.reshape(-1,1)
        for c in range(num_centroids):
            points_left = cloud_left[np.logical_and(label_left >= cent_edges[c,0], label_left<cent_edges[c,1]),:]
            cent[2*c,:] = np.median(points_left, axis=0)
            num_points[2*c] = points_left.shape[0]
            points_right = cloud_right[np.logical_and(label_right >= cent_edges[c,0], label_right<cent_edges[c,1]),:]
            cent[2*c+1,:] = np.median(points_right, axis=0)
            num_points[2*c+1] = points_right.shape[0]

            cent_dir[2*c] = -1
            cent_dir[2*c+1] = 1

    del_cent_nan = np.all(np.isnan(cent), axis= 1)
    del_cent_num = num_points<20
    del_cent = del_cent_nan + del_cent_num
    
    cent = np.delete(cent, del_cent, 0)
    cent_label = np.delete(cent_label, del_cent, 0)
    if isinstance(dire, type(None)):
        return cent, cent_label
    else:
        cent_dir = np.delete(cent_dir, del_cent, 0)
        return cent, cent_label, cent_dir

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=3,
                    shrinkA=0, shrinkB=0, color= 'r')
    ax.annotate('', v1, v0, arrowprops=arrowprops)

#__________________________________________________________________________
#|                                                                        |#
#|                           COMPUTE DYNAMICS                             |#
#|________________________________________________________________________|#

import matplotlib
import math

save_dir = '/home/julio/Documents/SP_project/Fig2/dynamics/'

for mouse in mice_list:
    tic()
    print(f"Working on mouse {mouse}: ")

    #main data
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)
        #signal
    traces = np.concatenate(pd_mouse['clean_traces'].values, axis = 0)
    umap = np.concatenate(pd_mouse['umap'].values, axis = 0)
    pos = get_signal(pd_mouse, 'pos')
    dir_mat = get_signal(pd_mouse, 'dir_mat')[:,0]
    dir_mat[dir_mat==1] = -1
    dir_mat[dir_mat==2] = 1


    dir_mov = np.diff(pos,axis=0)*20
    dir_mov = np.concatenate([dir_mov[0,:].reshape(-1,2), dir_mov], axis= 0)[:,0]
    dir_mov = gaussian_filter1d(dir_mov, sigma = 5, axis = 0)
    dir_mov[dir_mov<0] = -1
    dir_mov[dir_mov>=0] = 1
    dir_mat[dir_mat==0] = dir_mov[dir_mat==0]



    plt.figure(); 
    plt.plot(pos[:,0]/np.max(pos[:,0]))
    plt.plot(dir_mat)

    vel = get_signal(pd_mouse, 'vel')
    trial = get_signal(pd_mouse, 'index_mat')
    dir_color = get_dir_color(dir_mat)

    #cues
        #load cues position
    cues_file = [f for f in listdir(join(cues_dir, mouse)) if 'cues_info.csv' in f and 'lt' in f][0]
    cues_info = pd.read_csv(join(cues_dir, mouse, cues_file))
    st_cue = cues_info['x_start_cm'][0]
    end_cue = cues_info['x_end_cm'][0]
    in_cue_points = np.logical_and(pos[:,0]>=st_cue, pos[:,0]<end_cue)
    out_cue_points = np.invert(in_cue_points)

    umap_2D = PCA(2).fit_transform(umap)
    cent, cent_pos, cent_dir = get_centroids(umap_2D, pos[:,0], dir_mat, num_centroids=20) 
    dist_to_cents = np.min(pairwise_distances(umap_2D, cent),axis=1)
    off_ring_points = dist_to_cents>np.percentile(dist_to_cents, 75)
    in_ring_points = np.invert(off_ring_points)

    i = 0
    while i<len(cent)-1:
        if cent_pos[i] == cent_pos[i+1]:
            i = i+2
        else:
            cent = np.delete(cent, i, axis=0)
            cent_pos = np.delete(cent_pos, i, axis=0)
            cent_dir = np.delete(cent_dir, i)
    if len(cent_pos)%2>0:
        cent = np.delete(cent, -1, axis=0)
        cent_pos = np.delete(cent_pos, -1, axis=0)
        cent_dir = np.delete(cent_dir, -1)

    cent_1D = np.mean(cent.reshape(-1,2,2), axis=1)
    gradient_pca = PCA(2).fit(cent_1D)
    current_axis = gradient_pca.components_[:,0]

    aligned_umap_2D = gradient_pca.transform(umap_2D)
    aligned_cent = gradient_pca.transform(cent)


    #check low position left, high position right
    low_pos = aligned_umap_2D[pos[:,0]<np.percentile(pos[:,0],20)]
    high_pos = aligned_umap_2D[pos[:,0]>np.percentile(pos[:,0],80)]
    if np.mean(low_pos[:,0])>np.mean(high_pos[:,0]):
        aligned_umap_2D[:,0] *= -1
        aligned_cent[:,0] *= -1
        current_axis[0] += -1

    #check left top and right bottom
    left_dir = aligned_umap_2D[dir_mat==-1]
    right_dir = aligned_umap_2D[dir_mat==+1]
    if np.mean(left_dir[:,1])<np.mean(right_dir[:,1]):
        aligned_umap_2D[:,1] *= -1
        aligned_cent[:,1] *= -1
    


    #get flow
    diff = np.diff(aligned_umap_2D,n=2, axis=0)
    flow_angle = np.arctan2(diff[:,1], diff[:,0])*180/np.pi
    pos_angle = np.arctan2(aligned_umap_2D[:-2,1], aligned_umap_2D[:-2,0])*180/np.pi

    #get radial and tangential
    radial_angle = 180 - pos_angle + flow_angle
    radial_dist = np.linalg.norm(aligned_umap_2D[:-2,:], axis=1)
    flow_modulus = np.linalg.norm(diff, axis=1)
    radial_comp = np.cos(radial_angle*np.pi/180)*flow_modulus
    tan_comp = np.sin(radial_angle*np.pi/180)*flow_modulus

    col = 4
    row = 3

    fig = plt.figure(figsize=(15,8))
    ax = plt.subplot(row,col,1, projection = '3d')
    b = ax.scatter(*umap[:,:3].T, color = dir_color,s = 10)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(row,col,col+1, projection = '3d')
    b = ax.scatter(*umap[:,:3].T, c = pos[:,0],s = 10, cmap = 'inferno', vmin= 0, vmax = 70)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(row,col,2*col+1, projection = '3d')
    b = ax.scatter(*umap[:,:3].T, c = trial[:],s = 10, cmap = 'YlGn_r', vmax = np.percentile(trial, 95))
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(row,col,2)
    b = ax.scatter(*aligned_umap_2D.T, color = dir_color, s = 10)
    b = ax.scatter(*aligned_umap_2D[in_cue_points,:].T,c = None, edgecolors = 'r', linewidths = 0.5, s = 10,alpha =0.7, label = 'incue')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    ax = plt.subplot(row,col,col+2)
    b = ax.scatter(*aligned_umap_2D.T, c = pos[:,0], s = 10,cmap = 'inferno', vmin=0, vmax=70)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(row,col,2*col+2)
    b = ax.scatter(*aligned_umap_2D[in_ring_points,:].T, color = 'gray', s = 10,alpha = 0.05)
    b = ax.scatter(*aligned_umap_2D[off_ring_points,:].T, color = 'red', s = 10,alpha = 0.05)
    b = ax.scatter(*aligned_cent[:,:].T, c = cent_pos[:,0], cmap = 'inferno', vmin=0, vmax=70, s = 50)
    ax.set_aspect('equal', adjustable='box')

    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    ax = plt.subplot(row,col,3)
    ax.scatter(*aligned_umap_2D[in_cue_points,:].T, color = 'gray', s = 10,alpha = 0.05)
    for idx in range(0,aligned_umap_2D.shape[0]-2,10):
        st = aligned_umap_2D[idx,:]
        en = aligned_umap_2D[idx+2,:]
        angle = math.atan2(en[1]-st[1], en[0]-st[0])
        color = matplotlib.cm.hsv(norm(angle), bytes=True)[:-1]
        color = [x/255 for x in color]
        ax.arrow(st[0], st[1], en[0]-st[0], en[1]-st[1], color = color)
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(row,col,col+3)
    ax.scatter(pos_angle, flow_angle, s = 5, alpha = 0.1, c = 'k') #c = '#cc9900ff')
    ax.set_xlabel('manifold angle (º)')
    ax.set_ylabel('flow angle (º)')
    ax.set_aspect('equal', adjustable='box')

    ax = plt.subplot(row,col,2*col+3)
    ax.scatter(tan_comp, radial_comp, s= 5, alpha = .3)
    ax.set_xlabel('tang component')
    ax.set_ylabel('radian component')
    ax.set_aspect('equal', adjustable='box')
    
    ax = plt.subplot(row,col, 4)
    ax.scatter(flow_modulus, np.abs(tan_comp), s= 5, alpha = .5, label = 'tangent comp')
    ax.scatter(flow_modulus, np.abs(radial_comp), s= 5, alpha = .5, label = 'radial comp')

    ax.set_xlabel('flow modulus')
    ax.set_ylabel('radial or tangent component')
    ax.legend()

    ax = plt.subplot(row,col,col+4)
    ax.scatter(tan_comp[out_cue_points[:-2]], pos_angle[out_cue_points[:-2]], s= 5, alpha = .3)
    ax.scatter(tan_comp[in_cue_points[:-2]], pos_angle[in_cue_points[:-2]], s= 5, alpha = .3)
    ax.set_xlabel('tan_comp modulus')
    ax.set_ylabel('angular position')

    ax = plt.subplot(row,col,2*col+4)
    ax.scatter(flow_modulus, vel[:-2], s= 5, alpha = .5)
    ax.set_xlabel('flow modulus')
    ax.set_ylabel('velocity')

    plt.savefig(os.path.join(save_dir,f'{mouse}_dynamics_plot.svg'), dpi = 400,bbox_inches="tight")
    plt.savefig(os.path.join(save_dir,f'{mouse}_dynamics_plot.png'), dpi = 400,bbox_inches="tight")
    plt.close(fig)

    print('|flow| vs |radial|', np.corrcoef(flow_modulus, np.abs(radial_comp))[0,1])
    print('|flow| vs |tan|   ', np.corrcoef(flow_modulus, np.abs(tan_comp))[0,1])
    print('|flow| vs |radial in| ', np.corrcoef(flow_modulus[in_ring_points[:-2]], np.abs(radial_comp[in_ring_points[:-2]]))[0,1])
    print('|flow| vs |radial out|', np.corrcoef(flow_modulus[off_ring_points[:-2]], np.abs(radial_comp[off_ring_points[:-2]]))[0,1])
    print('|flow| vs |tan in| ', np.corrcoef(flow_modulus[in_ring_points[:-2]], np.abs(tan_comp[in_ring_points[:-2]]))[0,1])
    print('|flow| vs |tan out|', np.corrcoef(flow_modulus[off_ring_points[:-2]], np.abs(tan_comp[off_ring_points[:-2]]))[0,1])
    print('|flow| vs |vel|   ', np.corrcoef(flow_modulus, vel[:-2])[0,1])
    print(f"|radial|/|tan| in  = {np.mean(np.abs(radial_comp[in_ring_points[:-2]])/np.abs(tan_comp[in_ring_points[:-2]])):.4f}")
    print(f"|radial|/|tan| out = {np.mean(np.abs(radial_comp[off_ring_points[:-2]])/np.abs(tan_comp[off_ring_points[:-2]])):.4f}\n")

    dynamic_dict = {
        'umap': umap,
        'pos': pos,
        'dir_mat': dir_mat, 
        'vel': vel,
        'trial': trial,
        'dir_color': dir_color,

        'umap_2D': umap_2D,
        'cent': cent,
        'cent_pos': cent_pos,
        'cent_dir': cent_dir,

        'off_ring_points': off_ring_points,
        'in_ring_points': in_ring_points,


        'cues_info': cues_info,
        'st_cue': st_cue,
        'end_cue': end_cue,
        'in_cue_points': in_cue_points,
        'out_cue_points': out_cue_points,


        'cent_1D': cent_1D,
        'current_axis': current_axis,
        'aligned_umap_2D': aligned_umap_2D,
        'aligned_cent': aligned_cent,


        'flow': diff,
        'flow_modulus': flow_modulus,
        'flow_angle': flow_angle,

        'pos_angle': pos_angle,
        'radial_angle': radial_angle,
        'radial_comp': radial_comp,
        'tan_comp': tan_comp,
        'radial_dist': radial_dist

    }

    with open(os.path.join(save_dir, mouse+"_dynamic_dict.pkl"), "wb") as file:
        pickle.dump(dynamic_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    # if mouse in deep_mice:
    #     deep_flow_angle.append(flow_angle)
    #     deep_pos_angle.append(pos_angle)
    # elif mouse in sup_mice:
    #     sup_flow_angle.append(flow_angle)
    #     sup_pos_angle.append(pos_angle)


#__________________________________________________________________________
#|                                                                        |#
#|                        PLOT RADIAL/TAN IN VS OUT                       |#
#|________________________________________________________________________|#

mouse_list = []
layer_list = []
in_off_list = []
val_list = []

for mouse in mice_list:

    dynamic_dict = load_pickle(save_dir, mouse+"_dynamic_dict.pkl" )

    flow_modulus = dynamic_dict['flow_modulus']
    radial_comp = dynamic_dict['radial_comp']
    tan_comp = dynamic_dict['tan_comp']
    # in_ring_points = dynamic_dict['in_ring_points']
    # off_ring_points = dynamic_dict['off_ring_points']


    umap_2D = dynamic_dict['umap_2D']
    cent = dynamic_dict['cent']
    dist_to_cents = np.min(pairwise_distances(umap_2D, cent),axis=1)
    off_ring_points = dist_to_cents>np.percentile(dist_to_cents, 80)
    if mouse in ['GC3', 'ChZ8']:
        off_ring_points = dist_to_cents>np.percentile(dist_to_cents, 90)
    in_ring_points = np.invert(off_ring_points)


    vel = dynamic_dict['vel']
    # print(f"\n{mouse}: ")
    # print('|flow| vs |radial|', np.corrcoef(flow_modulus, np.abs(radial_comp))[0,1])
    # print('|flow| vs |tan|   ', np.corrcoef(flow_modulus, np.abs(tan_comp))[0,1])
    # print('|flow| vs |radial in| ', np.corrcoef(flow_modulus[in_ring_points[:-2]], np.abs(radial_comp[in_ring_points[:-2]]))[0,1])
    # print('|flow| vs |radial out|', np.corrcoef(flow_modulus[off_ring_points[:-2]], np.abs(radial_comp[off_ring_points[:-2]]))[0,1])
    # print('|flow| vs |tan in| ', np.corrcoef(flow_modulus[in_ring_points[:-2]], np.abs(tan_comp[in_ring_points[:-2]]))[0,1])
    # print('|flow| vs |tan out|', np.corrcoef(flow_modulus[off_ring_points[:-2]], np.abs(tan_comp[off_ring_points[:-2]]))[0,1])
    # print('|flow| vs |vel|   ', np.corrcoef(flow_modulus, vel[:-2])[0,1])
    # print(f"|radial|/|tan| in  = {np.median(np.abs(radial_comp[in_ring_points[:-2]])/np.abs(tan_comp[in_ring_points[:-2]])):.4f}")
    # print(f"|radial|/|tan| out = {np.median(np.abs(radial_comp[off_ring_points[:-2]])/np.abs(tan_comp[off_ring_points[:-2]])):.4f}")

    mouse_list += [mouse]*2
    if mouse in sup_mice:
        layer_list += ['sup']*2
    elif mouse in deep_mice:
        layer_list += ['deep']*2

    val_list.append(np.median(np.abs(radial_comp[in_ring_points[:-2]])/np.abs(tan_comp[in_ring_points[:-2]])))
    in_off_list.append('in')
    val_list.append(np.median(np.abs(radial_comp[off_ring_points[:-2]])/np.abs(tan_comp[off_ring_points[:-2]])))
    in_off_list.append('out')


pd_data = pd.DataFrame(data={'mouse': mouse_list,
                             'layer': layer_list,
                             'position': in_off_list,
                             'rad/tan': val_list})

fig = plt.figure()
ax = plt.subplot(111)

b = sns.barplot(x='layer', y='rad/tan', data=pd_data, hue='position',
        linewidth = 1, width= .5, ax = ax, errorbar= 'ci')
for mouse in pd_data['mouse'].unique():

    mouse_pd = pd_data[pd_data['mouse']==mouse]
    case = mouse_pd['layer'].unique()[0]

    if case=='deep':
        x = [-0.13,0.13]
    else:
        x = [0.87, 1.13]


    in_val = mouse_pd[mouse_pd['position']=='in']['rad/tan'].item()
    out_val = mouse_pd[mouse_pd['position']=='out']['rad/tan'].item()
    ax.plot(x, [in_val, out_val], 'gray')
    ax.scatter(x[0], in_val, color = [.5,.5,.5])
    ax.scatter(x[1], out_val,color = [0,0,0])

plt.savefig(os.path.join(save_dir,f'rad_tan_attractor_plot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'rad_tan_attractor_plot.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)

in_vals = pd_data[pd_data['position']=='in']['rad/tan'].to_list()
out_vals = pd_data[pd_data['position']=='out']['rad/tan'].to_list()
if stats.shapiro(in_vals).pvalue<=0.05 or stats.shapiro(out_vals).pvalue<=0.05:
    print('in vs out:',stats.ks_2samp(in_vals, out_vals))
else:
    print('in vs out:',stats.ttest_rel(in_vals, out_vals))
deep_pd = pd_data[pd_data['layer']=='deep']
in_deep_vals = deep_pd[deep_pd['position']=='in']['rad/tan'].to_list()
out_deep_vals = deep_pd[deep_pd['position']=='out']['rad/tan'].to_list()

sup_pd = pd_data[pd_data['layer']=='sup']
in_sup_vals = sup_pd[sup_pd['position']=='in']['rad/tan'].to_list()
out_sup_vals = sup_pd[sup_pd['position']=='out']['rad/tan'].to_list()
if stats.shapiro(in_deep_vals).pvalue<=0.05 or stats.shapiro(out_deep_vals).pvalue<=0.05:
    print('(deep in) vs (deep out):',stats.ks_2samp(in_deep_vals, out_deep_vals))
else:
    print('(deep in) vs (deep out):',stats.ttest_rel(in_deep_vals, out_deep_vals))

if stats.shapiro(in_sup_vals).pvalue<=0.05 or stats.shapiro(out_sup_vals).pvalue<=0.05:
    print('(sup in) vs (sup out):',stats.ks_2samp(in_sup_vals, out_sup_vals))
else:
    print('(sup in) vs (sup out):',stats.ttest_rel(in_sup_vals, out_sup_vals))

if stats.shapiro(in_sup_vals).pvalue<=0.05 or stats.shapiro(in_deep_vals).pvalue<=0.05:
    print('(sup in) vs (deep in):',stats.ks_2samp(in_sup_vals, in_deep_vals))
else:
    print('(sup in) vs (deep in):',stats.ttest_ind(in_sup_vals, in_deep_vals))

if stats.shapiro(out_sup_vals).pvalue<=0.05 or stats.shapiro(out_deep_vals).pvalue<=0.05:
    print('(sup out) vs (deep out):',stats.ks_2samp(out_sup_vals, out_deep_vals))
else:
    print('(sup out) vs (deep out):',stats.ttest_ind(out_sup_vals, out_deep_vals))



#__________________________________________________________________________
#|                                                                        |#
#|                          GET FLUX/POS RELATION                         |#
#|________________________________________________________________________|#

mouse_list = []
layer_list = []
in_off_list = []
val_list = []

for mouse in mice_list:

    dynamic_dict = load_pickle(save_dir, mouse+"_dynamic_dict.pkl" )

    flow_modulus = dynamic_dict['flow_modulus']
    radial_comp = dynamic_dict['radial_comp']
    tan_comp = dynamic_dict['tan_comp']
    in_ring_points = dynamic_dict['in_ring_points']
    off_ring_points = dynamic_dict['off_ring_points']
    vel = dynamic_dict['vel']
    print(f"\n{mouse}: ")
    print('|flow| vs |radial|', np.corrcoef(flow_modulus, np.abs(radial_comp))[0,1])
    print('|flow| vs |tan|   ', np.corrcoef(flow_modulus, np.abs(tan_comp))[0,1])
    print('|flow| vs |radial in| ', np.corrcoef(flow_modulus[in_ring_points[:-2]], np.abs(radial_comp[in_ring_points[:-2]]))[0,1])
    print('|flow| vs |radial out|', np.corrcoef(flow_modulus[off_ring_points[:-2]], np.abs(radial_comp[off_ring_points[:-2]]))[0,1])
    print('|flow| vs |tan in| ', np.corrcoef(flow_modulus[in_ring_points[:-2]], np.abs(tan_comp[in_ring_points[:-2]]))[0,1])
    print('|flow| vs |tan out|', np.corrcoef(flow_modulus[off_ring_points[:-2]], np.abs(tan_comp[off_ring_points[:-2]]))[0,1])
    print('|flow| vs |vel|   ', np.corrcoef(flow_modulus, vel[:-2])[0,1])
    print(f"|radial|/|tan| in  = {np.median(np.abs(radial_comp[in_ring_points[:-2]])/np.abs(tan_comp[in_ring_points[:-2]])):.4f}")
    print(f"|radial|/|tan| out = {np.median(np.abs(radial_comp[off_ring_points[:-2]])/np.abs(tan_comp[off_ring_points[:-2]])):.4f}")

    mouse_list += [mouse]*2
    if mouse in sup_mice:
        layer_list += ['sup']*2
    elif mouse in deep_mice:
        layer_list += ['deep']*2

    val_list.append(np.median(np.abs(radial_comp[in_ring_points[:-2]])/np.abs(tan_comp[in_ring_points[:-2]])))
    in_off_list.append('in')
    val_list.append(np.median(np.abs(radial_comp[off_ring_points[:-2]])/np.abs(tan_comp[off_ring_points[:-2]])))
    in_off_list.append('out')


pd_data = pd.DataFrame(data={'mouse': mouse_list,
                             'layer': layer_list,
                             'position': in_off_list,
                             'rad/tan': val_list})

plt.figure()
ax = plt.subplot(111)

b = sns.barplot(x='layer', y='rad/tan', data=pd_data, hue='position',
        linewidth = 1, width= .5, ax = ax, errorbar= 'sd')
sns.swarmplot(x='layer', y='rad/tan', data=pd_data, hue= 'position',
            palette = 'dark:gray', ax = ax)
sns.lineplot(x = 'layer', y= 'rad/tan', data=pd_data, hue = 'position', units = 'mouse',
            ax = ax, estimator = None, color = ".7", markers = True)


in_vals = pd_data[pd_data['position']=='in']['rad/tan'].to_list()
out_vals = pd_data[pd_data['position']=='out']['rad/tan'].to_list()


if stats.shapiro(in_vals).pvalue<=0.05 or stats.shapiro(out_vals).pvalue<=0.05:
    print('in vs out:',stats.ks_2samp(in_vals, out_vals))
else:
    print('in vs out:',stats.ttest_rel(in_vals, out_vals))

deep_pd = pd_data[pd_data['layer']=='deep']
in_deep_vals = deep_pd[deep_pd['position']=='in']['rad/tan'].to_list()
out_deep_vals = deep_pd[deep_pd['position']=='out']['rad/tan'].to_list()

sup_pd = pd_data[pd_data['layer']=='sup']
in_sup_vals = sup_pd[sup_pd['position']=='in']['rad/tan'].to_list()
out_sup_vals = sup_pd[sup_pd['position']=='out']['rad/tan'].to_list()

if stats.shapiro(in_deep_vals).pvalue<=0.05 or stats.shapiro(out_deep_vals).pvalue<=0.05:
    print('(deep in) vs (deep out):',stats.ks_2samp(in_deep_vals, out_deep_vals))
else:
    print('(deep in) vs (deep out):',stats.ttest_rel(in_deep_vals, out_deep_vals))

if stats.shapiro(in_sup_vals).pvalue<=0.05 or stats.shapiro(out_sup_vals).pvalue<=0.05:
    print('(sup in) vs (sup out):',stats.ks_2samp(in_sup_vals, out_sup_vals))
else:
    print('(sup in) vs (sup out):',stats.ttest_rel(in_sup_vals, out_sup_vals))

if stats.shapiro(in_sup_vals).pvalue<=0.05 or stats.shapiro(in_deep_vals).pvalue<=0.05:
    print('(sup in) vs (deep in):',stats.ks_2samp(in_sup_vals, in_deep_vals))
else:
    print('(sup in) vs (deep in):',stats.ttest_ind(in_sup_vals, in_deep_vals))

if stats.shapiro(out_sup_vals).pvalue<=0.05 or stats.shapiro(out_deep_vals).pvalue<=0.05:
    print('(sup out) vs (deep out):',stats.ks_2samp(out_sup_vals, out_deep_vals))
else:
    print('(sup out) vs (deep out):',stats.ttest_ind(out_sup_vals, out_deep_vals))
















def draw_vector(v0, v1, txt = '',  ax=None, color = 'r'):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=1,
                    shrinkA=0, shrinkB=0, color= color)
    ax.annotate(txt, v1, v0, arrowprops=arrowprops)

plt.figure()
ax = plt.subplot(111)
ax.scatter(0,0, s = 10, marker ='x', color = 'k')
b = ax.scatter(*aligned_umap_2D.T, color = 'gray', s = 10,alpha = 0.05)
for ii in range(10):
    idx = np.random.choice(range(aligned_umap_2D.shape[0]),size=1)[0]
    color = list(np.random.choice(range(256), size=3))
    color = [x/255 for x in color]
    draw_vector(aligned_umap_2D[idx,:],aligned_umap_2D[idx+2,:], str(idx), ax=ax, color = color)
    ax.scatter(*aligned_umap_2D[idx,:].T, s = 10, marker ='x', color =color)
    ax.plot([0,aligned_umap_2D[idx,0]], [0, aligned_umap_2D[idx,1]], color = color, linestyle = '--')


deep_flow_angle = np.concatenate(deep_flow_angle, axis=0)
deep_pos_angle = np.concatenate(deep_pos_angle, axis=0)
sup_flow_angle = np.concatenate(sup_flow_angle, axis=0)
sup_pos_angle = np.concatenate(sup_pos_angle, axis=0)

plt.figure()
ax = plt.subplot(1,2,1)
ax.scatter(sup_pos_angle, sup_flow_angle, s = 5, alpha = 0.1, c = 'k') #c = '#9900ffff')
ax.set_xlabel('manifold angle (º)')
ax.set_ylabel('flow angle (º)')
ax.set_aspect('equal', adjustable='box')
ax.set_title('sup')

ax = plt.subplot(1,2,2)
ax.scatter(deep_pos_angle, deep_flow_angle, s = 5, alpha = 0.1, c = 'k') #c = '#cc9900ff')
ax.set_xlabel('manifold angle (º)')
ax.set_ylabel('flow angle (º)')
ax.set_aspect('equal', adjustable='box')
ax.set_title('deep')

diff1 = np.abs(sup_flow_angle- sup_pos_angle).reshape(-1,1)
diff2 = np.abs(sup_flow_angle + 360 - sup_pos_angle).reshape(-1,1)
diff3 = np.abs(sup_flow_angle - 360 - sup_pos_angle).reshape(-1,1)
sup_diff = np.min(np.concatenate((diff1, diff2, diff3),axis=1), axis=1)

diff1 = np.abs(deep_flow_angle- deep_pos_angle).reshape(-1,1)
diff2 = np.abs(deep_flow_angle + 360 - deep_pos_angle).reshape(-1,1)
diff3 = np.abs(deep_flow_angle - 360 - deep_pos_angle).reshape(-1,1)
deep_diff = np.min(np.concatenate((diff1, diff2, diff3),axis=1), axis=1)

plt.figure()
ax = plt.subplot(1,2,1)
ax.scatter(sup_pos_angle, np.abs(sup_diff-90), s = 5, alpha = 0.1, c = 'k') #c = '#9900ffff')
ax.set_xlabel('manifold angle (º)')
ax.set_ylabel('flow-pos angle diff (º)')
# ax.set_aspect('equal', adjustable='box')
ax.set_title('sup')

ax = plt.subplot(1,2,2)
ax.scatter(deep_pos_angle, np.abs(deep_diff-90), s = 5, alpha = 0.1, c = 'k') #c = '#cc9900ff')
ax.set_xlabel('manifold angle (º)')
ax.set_ylabel('flow-pos angle diff (º)')
# ax.set_aspect('equal', adjustable='box')
ax.set_title('deep')

print('sup', np.median(np.abs(sup_diff-90)))
print('deep', np.median(np.abs(deep_diff-90)))
#__________________________________________________________________________
#|                                                                        |#
#|                                  START                                 |#
#|________________________________________________________________________|#

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
data_dir =  '/home/julio/Documents/SP_project/Fig2/processed_data/'
save_dir = '/home/julio/Documents/SP_project/Fig2/npoints/'


deep_corr_left = []
deep_corr_right = []

sup_corr_right = []
sup_corr_left = []
for mouse in mice_list:
    tic()
    print(f"Working on mouse {mouse}: ")
    file_name =  mouse+'_df_dict.pkl'
    file_path = os.path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,file_name)

    #signal
    signal = get_signal(pd_mouse,'clean_traces')
    dir_mat = get_signal(pd_mouse,'dir_mat')[:,0]


    left = signal[dir_mat==1, :]
    right = signal[dir_mat==2,:]

    corr_left = np.corrcoef(left.T).flatten().reshape(-1,1)
    corr_right = np.corrcoef(right.T).flatten().reshape(-1,1)

    if mouse in sup_mice:
        sup_corr_left.append(corr_left)
        sup_corr_right.append(corr_right)
    elif mouse in deep_mice:
        deep_corr_left.append(corr_left)
        deep_corr_right.append(corr_right)

deep_corr_left = np.concatenate(deep_corr_left,axis=0)
deep_corr_right = np.concatenate(deep_corr_right,axis=0)
sup_corr_left = np.concatenate(sup_corr_left,axis=0)
sup_corr_right = np.concatenate(sup_corr_right,axis=0)

plt.figure()
ax = plt.subplot(121)

deep_rand = np.arange(deep_corr_left.shape[0])
np.random.shuffle(deep_rand)

sup_rand = np.arange(sup_corr_left.shape[0])
np.random.shuffle(sup_rand)
for idx in range(350000):
    ax.scatter(deep_corr_left[deep_rand[idx]], deep_corr_right[deep_rand[idx]], s = 1, alpha = 0.05, color = 'b') 
    ax.scatter(sup_corr_left[sup_rand[idx]], sup_corr_right[sup_rand[idx]], s = 1, alpha = 0.05, color = 'g') 
ax.set_aspect('equal')

ax = plt.subplot(122)
ax.