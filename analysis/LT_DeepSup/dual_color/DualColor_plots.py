import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy, os, pickle
import seaborn as sns
from scipy import stats
from bioinfokit.analys import stat

from statsmodels.formula.api import ols
import statsmodels.api as sm

mice_list = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']
base_dir = '/home/julio/Documents/DeepSup_project/DualColor/'
jRGECO_dir = os.path.join(base_dir,'Thy1jRGECO')
RCaMP_dir = os.path.join(base_dir,'ThyCalbRCaMP')
save_dir = os.path.join(base_dir,'figures')

def load_pickle(path,name):
    with open(os.path.join(path, name), 'rb') as sf:
        data = pickle.load(sf)
    return data

supMice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepMice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4']

palette_deepsup = ["#cc9900ff", "#9900ffff"]
palette_dual = ["gray"]+palette_deepsup
#__________________________________________________________________________
#|                                                                        |#
#|                          PLOT ROTATION ANGLES                          |#
#|________________________________________________________________________|#

rot_angle = list()
mouse_list = list()
channel_list = list()
for mouse in mice_list:
    if 'Thy1jRGECO' in mouse:
        rotation_dir = os.path.join(jRGECO_dir,'rotation', mouse)
    else:
        rotation_dir = os.path.join(RCaMP_dir,'rotation', mouse)

    rotation_dict = load_pickle(rotation_dir, mouse+'_rotation_dict.pkl')
    rot_angle.append(rotation_dict['deep']['deep_rotation_angle'])
    channel_list.append('deep')
    mouse_list.append(mouse)
    rot_angle.append(rotation_dict['sup']['sup_rotation_angle'])
    channel_list.append('sup')
    mouse_list.append(mouse)
    rot_angle.append(rotation_dict['all']['all_rotation_angle'])
    channel_list.append('all')
    mouse_list.append(mouse)


pdAngle = pd.DataFrame(data={'mouse': mouse_list,
                     'channel': channel_list,
                     'angle': rot_angle})    

fig, ax = plt.subplots(1, 1, figsize=(6,6))

b = sns.boxplot(x='channel', y='angle', data=pdAngle,
            linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='channel', y='angle', data=pdAngle,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
ax.set_ylim([-2.5, 180.5])
ax.set_yticks([0,45,90,135,180]);
plt.savefig(os.path.join(save_dir,f'dual_rot_angle.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dual_rot_angle.png'), dpi = 400,bbox_inches="tight")


deepAngle = pdAngle.loc[pdAngle['channel']=='deep']['angle']
calbAngle = pdAngle.loc[pdAngle['channel']=='sup']['angle']
allAngle = pdAngle.loc[pdAngle['channel']=='all']['angle']

deepAngle_norm = stats.shapiro(deepAngle)
calbAngle_norm = stats.shapiro(calbAngle)
allAngle_norm = stats.shapiro(allAngle)

if deepAngle_norm.pvalue<=0.05 or calbAngle_norm.pvalue<=0.05:
    print('deepAngle vs calbAngle:',stats.ks_2samp(deepAngle, calbAngle))
else:
    print('deepAngle vs calbAngle:', stats.ttest_rel(deepAngle, calbAngle))

if deepAngle_norm.pvalue<=0.05 or allAngle_norm.pvalue<=0.05:
    print('deepAngle vs allAngle:',stats.ks_2samp(deepAngle, allAngle))
else:
    print('deepAngle vs allAngle:',stats.ttest_rel(deepAngle, allAngle))

if calbAngle_norm.pvalue<=0.05 or allAngle_norm.pvalue<=0.05:
    print('calbAngle vs allAngle:',stats.ks_2samp(calbAngle, allAngle))
else:
    print('calbAngle vs allAngle:', stats.ttest_rel(calbAngle, allAngle))

res = stat()
res.anova_stat(df=pdAngle, res_var='angle', anova_model='angle~C(channel)+C(mouse)+C(channel):C(mouse)')
res.anova_summary

#perform two-way ANOVA
model = ols('angle ~ C(channel) + C(mouse) + C(channel):C(mouse)', data=pdAngle).fit()
sm.stats.anova_lm(model, typ=2)


model = ols('angle ~ C(channel)', data=pdAngle).fit()
sm.stats.anova_lm(model, typ=1)
#__________________________________________________________________________
#|                                                                        |#
#|                            PLOT DISTANCE                               |#
#|________________________________________________________________________|#


########################### perimeter
emb_distance_list = list()
mouse_list = list()
channel_list = list()
emb_distance_plane_list = list()
emb_distance_pairwise_list = list()

for mouse in mice_list:
    if 'Thy1jRGECO' in mouse:
        distance_dir = os.path.join(jRGECO_dir,'distance', mouse)
    else:
        distance_dir = os.path.join(RCaMP_dir,'distance', mouse)

    distance_dict = load_pickle(distance_dir, mouse+'_distance_dict.pkl')

    mouse_list += [mouse]*3
    emb_distance_list.append(distance_dict['deep']['deep_remap_dist'])
    emb_distance_plane_list.append(distance_dict['deep']['plane_deep_remap_dist'])
    channel_list.append('deep')

    emb_distance_list.append(distance_dict['sup']['sup_remap_dist'])
    emb_distance_plane_list.append(distance_dict['sup']['plane_sup_remap_dist'])
    channel_list.append('sup')

    emb_distance_list.append(distance_dict['all']['all_remap_dist'])
    emb_distance_plane_list.append(distance_dict['all']['plane_all_remap_dist'])
    channel_list.append('all')


pd_distance = pd.DataFrame(data={'mouse': mouse_list,
                     'channel': channel_list,
                     'distance': emb_distance_list,
                     'plane_distance': emb_distance_plane_list})    

fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='channel', y='distance', data=pd_distance,
            linewidth = 1, width= .5, ax = ax, order = ['sup', 'deep', 'all'])
sns.swarmplot(x='channel', y='distance', data=pd_distance,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax, order = ['sup', 'deep', 'all'])
plt.savefig(os.path.join(save_dir,f'dual_distance.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dual_distance.png'), dpi = 400,bbox_inches="tight")

fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='channel', y='plane_distance', data=pd_distance,
            linewidth = 1, width= .5, ax = ax, order = ['sup', 'deep', 'all'])
sns.swarmplot(x='channel', y='plane_distance', data=pd_distance,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax, order = ['sup', 'deep', 'all'])
plt.savefig(os.path.join(save_dir,f'dual_perimeter_distance.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dual_perimeter_distance.png'), dpi = 400,bbox_inches="tight")



model = ols('distance ~ C(channel)', data=pd_distance).fit()
sm.stats.anova_lm(model, typ=1)

deep_dist = pd_distance.loc[pd_distance['channel']=='deep']['distance']
sup_dist = pd_distance.loc[pd_distance['channel']=='sup']['distance']
all_dist = pd_distance.loc[pd_distance['channel']=='all']['distance']

deep_dist_norm = stats.shapiro(deep_dist)
sup_dist_norm = stats.shapiro(sup_dist)
all_dist_norm = stats.shapiro(all_dist)

if deep_dist_norm.pvalue<=0.05 or sup_dist_norm.pvalue<=0.05:
    print('deep_dist vs sup_dist:',stats.ks_2samp(deep_dist, sup_dist))
else:
    print('deep_dist vs sup_dist:', stats.ttest_rel(deep_dist, sup_dist))

if deep_dist_norm.pvalue<=0.05 or all_dist_norm.pvalue<=0.05:
    print('deep_dist vs all_dist:',stats.ks_2samp(deep_dist, all_dist))
else:
    print('deep_dist vs all_dist:',stats.ttest_rel(deep_dist, all_dist))

if sup_dist_norm.pvalue<=0.05 or all_dist_norm.pvalue<=0.05:
    print('sup_dist vs all_dist:',stats.ks_2samp(sup_dist, all_dist))
else:
    print('sup_dist vs all_dist:', stats.ttest_rel(sup_dist, all_dist))


#perform two-way ANOVA
model = ols('distance ~ C(channel) + C(mouse) + C(channel):C(mouse)', data=pd_distance).fit()
sm.stats.anova_lm(model, typ=2)

########################### mean

emb_distance_list = list()
mouse_list = list()
channel_list = list()

for mouse in mice_list:
    if 'Thy1jRGECO' in mouse:
        distance_dir = os.path.join(jRGECO_dir,'distance', mouse)
    else:
        distance_dir = os.path.join(RCaMP_dir,'distance', mouse)

    distance_dict = load_pickle(distance_dir, mouse+'_distance_dict.pkl')

    mouse_list += [mouse]*3
    emb_distance_list.append(distance_dict['deep']['deep_remap_dist'])
    channel_list.append('deep')
    emb_distance_list.append(distance_dict['sup']['sup_remap_dist'])
    channel_list.append('sup')
    emb_distance_list.append(distance_dict['all']['all_remap_dist'])
    channel_list.append('all')


pd_distance = pd.DataFrame(data={'mouse': mouse_list,
                     'channel': channel_list,
                     'distance': emb_distance_list})    

fig, ax = plt.subplots(1, 1, figsize=(6,6))
b = sns.barplot(x='channel', y='distance', data=pd_distance,
            linewidth = 1, width= .5, ax = ax)
sns.swarmplot(x='channel', y='distance', data=pd_distance,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax)
plt.savefig(os.path.join(save_dir,f'dual_mean_distance.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,f'dual_mean_distance.png'), dpi = 400,bbox_inches="tight")


deep_dist = pd_distance.loc[pd_distance['channel']=='deep']['distance']
sup_dist = pd_distance.loc[pd_distance['channel']=='sup']['distance']
all_dist = pd_distance.loc[pd_distance['channel']=='all']['distance']

deep_dist_norm = stats.shapiro(deep_dist)
sup_dist_norm = stats.shapiro(sup_dist)
all_dist_norm = stats.shapiro(all_dist)

if deep_dist_norm.pvalue<=0.05 or sup_dist_norm.pvalue<=0.05:
    print('deep_dist vs sup_dist:',stats.ks_2samp(deep_dist, sup_dist))
else:
    print('deep_dist vs sup_dist:', stats.ttest_rel(deep_dist, sup_dist))

if deep_dist_norm.pvalue<=0.05 or all_dist_norm.pvalue<=0.05:
    print('deep_dist vs all_dist:',stats.ks_2samp(deep_dist, all_dist))
else:
    print('deep_dist vs all_dist:',stats.ttest_rel(deep_dist, all_dist))

if sup_dist_norm.pvalue<=0.05 or all_dist_norm.pvalue<=0.05:
    print('sup_dist vs all_dist:',stats.ks_2samp(sup_dist, all_dist))
else:
    print('sup_dist vs all_dist:', stats.ttest_rel(sup_dist, all_dist))


#perform two-way ANOVA
model = ols('distance ~ C(channel) + C(mouse) + C(channel):C(mouse)', data=pd_distance).fit()
sm.stats.anova_lm(model, typ=2)




#__________________________________________________________________________
#|                                                                        |#
#|                       PLOT DEEP/SUP CORRELATION                        |#
#|________________________________________________________________________|#


save_dir = '/home/julio/Documents/DeepSup_project/DualColor/figures/'
base_dir = '/home/julio/Documents/DeepSup_project/DualColor/'
mice_list = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']

jRGECO_dir = os.path.join(base_dir,'Thy1jRGECO')
RCaMP_dir = os.path.join(base_dir,'ThyCalbRCaMP')

layer_list = list()
name_list = list()
angle_list = list()
dist_list = list()
ellipse_dist = list()
deep_perc = list()
sup_perc = list()
allo_list = list()
local_list = list()
remap_list = list()
animal_type_list = list()
num_cells_list = list()
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
    num_cells_list.append(cell_type.shape[0])
    allo_list.append(static_cells/num_cells)
    local_list.append(rot_cells/num_cells)
    remap_list.append(remap_cells/num_cells)
    layer_list.append('all')
    name_list.append(mouse)
    angle_list.append(all_rot)
    dist_list.append(all_dist)
    ellipse_dist.append(plane_all_dist)

    #deep cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_deep_cellType.npy'))
    num_deep_cells = cell_type.shape[0]
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]
    num_cells = static_cells+rot_cells+remap_cells
    num_cells_list.append(cell_type.shape[0])

    allo_list.append(static_cells/num_cells)
    local_list.append(rot_cells/num_cells)
    remap_list.append(remap_cells/num_cells)
    layer_list.append('deep')
    name_list.append(mouse)
    angle_list.append(deep_rot)
    dist_list.append(deep_dist)
    ellipse_dist.append(plane_deep_dist)

    #sup cells
    cell_type = np.load(os.path.join(functional_dir,  mouse+'_sup_cellType.npy'))
    num_sup_cells = cell_type.shape[0]

    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]
    num_cells = static_cells+rot_cells+remap_cells
    num_cells_list.append(cell_type.shape[0])

    allo_list.append(static_cells/num_cells)
    local_list.append(rot_cells/num_cells)
    remap_list.append(remap_cells/num_cells)
    layer_list.append('sup')
    name_list.append(mouse)
    angle_list.append(sup_rot)
    dist_list.append(sup_dist)
    ellipse_dist.append(plane_sup_dist)


    deep_perc.append(num_deep_cells/(num_deep_cells+num_sup_cells))    
    sup_perc.append(num_sup_cells/(num_deep_cells+num_sup_cells))    

    deep_perc.append(1)    
    sup_perc.append(0)    

    deep_perc.append(0)    
    sup_perc.append(1)

    animal_type_list += ['dual']*3

func_cells_dir = '/home/julio/Documents/DeepSup_project/DeepSup/functional_cells/'
rotation_dir = '/home/julio/Documents/DeepSup_project/DeepSup/rotation/'
distance_dir = '/home/julio/Documents/DeepSup_project/DeepSup/distance/'
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']


for mouse in mice_list:

    name_list.append(mouse)

    cell_type = np.load(os.path.join(func_cells_dir,mouse, mouse+'_cellType.npy'))
    static_cells = np.where(cell_type==0)[0].shape[0]
    rot_cells = np.where(np.logical_and(cell_type<4,cell_type>0))[0].shape[0]
    remap_cells = np.where(cell_type==4)[0].shape[0]
    num_cells = cell_type.shape[0]
    num_cells_list.append(num_cells)

    allo_list.append(static_cells/num_cells)
    local_list.append(rot_cells/num_cells)
    remap_list.append(remap_cells/num_cells)

    rotation_dict = load_pickle(rotation_dir, f'{mouse}_rotation_dict.pkl')
    angle_rot = rotation_dict['umap']['rotation_angle']
    angle_list.append(angle_rot)

    distance_dict = load_pickle(distance_dir, f'{mouse}_distance_dict.pkl')
    remap_dist = distance_dict['umap']['remap_dist']
    plane_dist = distance_dict['umap']['plane_remap_dist']
    dist_list.append(remap_dist)
    ellipse_dist.append(plane_dist)


    if mouse in deepMice:
        layer_list.append('deep')
        deep_perc.append(1)    
        sup_perc.append(0)    
    elif mouse in supMice:
        layer_list.append('sup')
        deep_perc.append(0)    
        sup_perc.append(1)
    animal_type_list.append('single')

mouse_pd = pd.DataFrame(data={'layer': layer_list,
                            'mouse': name_list,
                            'rotation': angle_list,
                            'distance': dist_list,
                            'plane_dist': ellipse_dist,
                            'allocentric': allo_list,
                            'local-cue-anchored': local_list,
                            'remapping': remap_list,
                            'deep_perc': deep_perc,
                            'sup_perc': sup_perc,
                            'animal_type': animal_type_list,
                            'num_cells': num_cells_list})
 

 ##################################### ADD LENSE POSITION ##################################
#Thy1jRGECO22-Thy1jRGECO23-ThyCalbRCaMP2
lense_x = [1.75]*3+[1.75]*3+[2]*3
lense_y = [-2.4]*3+[-2.5]*3+[-2.75]*3

#GC2-GC3-GC5-TGrin1-ChZ4                 -                   CZ3-CZ6-CZ8-CZ9-CGrin1
lense_x += [ 1.25,  1.25,  1.50,  2.00,  1.25,               1.30,  1.25,  1.50,  1.50,  2.00]
lense_y += [-1.45, -1.75, -2.30, -2.50, -2.20,              -2.40, -2.25, -2.65, -2.40, -2.30]

mouse_pd['lense_x'] = lense_x
mouse_pd['lense_y'] = lense_y


 ##################################### ADD HISTOLOGY  CELLS ##################################


#Thy1jRGECO22-Thy1jRGECO23-ThyCalbRCaMP2
# perc_sup_hist =  [0.65, 0.55, 0.83] + [0.55, 0.40, 1.00] + [0.34, 0.00, 0.93]
# perc_deep_hist = [0.35, 0.45, 0.17] + [0.45, 0.60, 0.00] + [0.66, 1.00, 0.07]
# perc_sup_hist =  [0.84, 0.38, 0.83] + [0.56, 0.32, 1.00] + [0.34, 0.00, 0.93]
# perc_deep_hist = [0.16, 0.62, 0.17] + [0.44, 0.68, 0.00] + [0.66, 1.00, 0.07]

perc_sup_hist =  [0.68, 0.33, 0.67] + [0.50, 0.27, 0.76] + [0.47, 0.27, 0.80]
perc_deep_hist = [0.32, 0.67, 0.33] + [0.50, 0.73, 0.24] + [0.53, 0.73, 0.20]

                  #GC2   GC3   GC5   TGrin1  ChZ4         CZ3   CZ6   CZ8   CZ9   CGrin1
perc_sup_hist +=  [0.00, 0.00, 0.31, 0.43,   0.33,        1.00, 0.83, 0.85, 0.81, 0.78]
perc_deep_hist += [1.00, 1.00, 0.69, 0.57,   0.67,        0.00, 0.17, 0.15, 0.19, 0.22]

mouse_pd['sup_perc_hist'] = perc_sup_hist
mouse_pd['deep_perc_hist'] = perc_deep_hist


plt.figure()
ax = plt.subplot(2,2,1)
sns.scatterplot(data=mouse_pd, y='rotation', x='deep_perc_hist', 
    style='animal_type', ax= ax)

ax = plt.subplot(2,2,2)
sns.scatterplot(data=mouse_pd, y='rotation', x='sup_perc_hist',
    style='animal_type', ax= ax)

ax = plt.subplot(2,2,3)
sns.scatterplot(data=mouse_pd, y='rotation', x='num_cells',
    style='animal_type', ax= ax)

ax = plt.subplot(2,2,4)
sns.scatterplot(data=mouse_pd, y='rotation', x='distance',
    style='animal_type', ax= ax)

plt.savefig(os.path.join(save_dir,'dual_deepsup_scatters.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dual_deepsup_scatters.png'), dpi = 400,bbox_inches="tight")



plt.figure()
ax = plt.subplot(2,2,1)
sns.scatterplot(data=mouse_pd, x='lense_x', y='lense_y', hue='rotation',
    style='animal_type', ax= ax, palette='viridis')


ax = plt.subplot(2,2,2)
sns.scatterplot(data=mouse_pd, x='lense_x', y='lense_y', hue='distance',
    style='animal_type', ax= ax, palette='viridis')

ax = plt.subplot(2,2,3)
sns.scatterplot(data=mouse_pd, x='lense_x', y='lense_y', hue='num_cells',
    style='animal_type', ax= ax, palette='viridis')




from scipy import stats
temp_pd = mouse_pd[mouse_pd["mouse"]!="GC2"]
temp_pd = temp_pd[temp_pd["mouse"]!="GC3"]
temp_pd = temp_pd[temp_pd["mouse"]!="CZ3"]


#ROTATION
slope, intercept, r_value, p_value, std_err = stats.linregress(temp_pd["deep_perc_hist"].to_list(),temp_pd["rotation"].to_list())
stats.pearsonr(temp_pd["deep_perc_hist"].to_list(),temp_pd["rotation"].to_list())

plt.figure()
ax = plt.subplot(2,2,1)
sns.scatterplot(data=temp_pd, y='rotation', x='deep_perc', hue='layer',
    palette = palette_dual, style='animal_type', ax= ax)

ax = plt.subplot(2,2,2)
sns.scatterplot(data=temp_pd, y='rotation', x='deep_perc_hist', hue='layer',
    palette = palette_dual, style='animal_type', ax= ax)
ax.plot([0,1], [intercept, slope+intercept], 'k--')
ax = plt.subplot(2,2,3)
sns.scatterplot(data=temp_pd, y='rotation', x='sup_perc', hue='layer',
    palette = palette_dual, style='animal_type', ax= ax)

ax = plt.subplot(2,2,4)
sns.scatterplot(data=temp_pd, y='rotation', x='sup_perc_hist', hue='layer',
    palette = palette_dual, style='animal_type', ax= ax)


#DISTANCE
slope, intercept, r_value, p_value, std_err = stats.linregress(temp_pd["deep_perc_hist"].to_list(),temp_pd["distance"].to_list())
stats.pearsonr(temp_pd["deep_perc_hist"].to_list(),temp_pd["distance"].to_list())


#DISTANCE
slope, intercept, r_value, p_value, std_err = stats.linregress(mouse_pd["remapping"].to_list(),mouse_pd["distance"].to_list())
stats.pearsonr(mouse_pd["remapping"].to_list(),mouse_pd["distance"].to_list())


plt.figure()
ax = plt.subplot(2,2,1)
sns.scatterplot(data=temp_pd, y='rotation', x='deep_perc', hue='layer',
    palette = palette_dual, style='animal_type', ax= ax)

ax = plt.subplot(2,2,2)
sns.scatterplot(data=temp_pd, y='rotation', x='deep_perc_hist', hue='layer',
    palette = palette_dual, style='animal_type', ax= ax)
ax.plot([0,1], [intercept, slope+intercept], 'k--')
ax = plt.subplot(2,2,3)
sns.scatterplot(data=temp_pd, y='rotation', x='sup_perc', hue='layer',
    palette = palette_dual, style='animal_type', ax= ax)

ax = plt.subplot(2,2,4)
sns.scatterplot(data=temp_pd, y='rotation', x='sup_perc_hist', hue='layer',
    palette = palette_dual, style='animal_type', ax= ax)




plt.savefig(os.path.join(save_dir,'dual_deepsup_scatters_hist.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dual_deepsup_scatters_hist.png'), dpi = 400,bbox_inches="tight")


