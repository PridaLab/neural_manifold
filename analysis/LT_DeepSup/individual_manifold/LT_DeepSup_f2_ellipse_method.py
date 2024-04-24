from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

sup_mice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deep_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']


data_dir = '/home/julio/Documents/SP_project/Fig2/eccentricity/'
ellipse_dict = load_pickle(data_dir, 'ellipse_fit_dict.pkl')


explained_variance_list = list()
ellipse_error_list = list()
name_list = list()
layer_list = list()
strain_list = list()


for mouse in list(ellipse_dict.keys()):


    name_list.append(mouse)
    if mouse in deep_mice:
        layer_list.append('deep')
    elif mouse in sup_mice:
        layer_list.append('sup')

    if ('GC' in mouse) or ('TG' in mouse):
        strain_list.append('Thy1')
    elif ('CZ' in mouse) or ('CG' in mouse):
        strain_list.append('Calb')
    elif 'Ch' in mouse:
        strain_list.append('ChRNA7')

    cent = ellipse_dict[mouse]['cent']
    model_pca = PCA(3)
    model_pca.fit(cent)
    explained_variance_list.append(sum(model_pca.explained_variance_ratio_[:2]))
    
    cent_2D = ellipse_dict[mouse]['cent2D']
    ellipse_points = ellipse_dict[mouse]['ellipsePoints']
    distances = pairwise_distances(cent_2D, ellipse_points)
    mean_error = np.sum(np.min(distances,axis=1))
    ellipse_error_list.append(mean_error)



pd_ellipse = pd.DataFrame(data={'mouse': name_list,
                     'layer': layer_list,
                     'strain': strain_list,
                     'pca_variance': explained_variance_list,
                     'ellipse_error':ellipse_error_list})    



fig, ax = plt.subplots(1, 2, figsize=(10,6))

b = sns.barplot(x='layer', y='pca_variance', data=pd_ellipse,
            linewidth = 1, width= .5, ax = ax[0])
sns.swarmplot(x='layer', y='pca_variance', data=pd_ellipse,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax[0])
b = sns.barplot(x='layer', y='ellipse_error', data=pd_ellipse,
            linewidth = 1, width= .5, ax = ax[1])
sns.swarmplot(x='layer', y='ellipse_error', data=pd_ellipse,
        palette = 'dark:gray', edgecolor = 'gray', ax = ax[1])
plt.tight_layout()


plt.savefig(os.path.join(data_dir,'ellipse_method_quantification.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'ellipse_method_quantification.png'), dpi = 400,bbox_inches="tight")

pd_ellipse.to_csv(os.path.join(data_dir, 'ellipse_method_quantification.csv'))



#############################################################################################

palette_deepsup = ["#cc9900ff", "#9900ffff"]

sup_mice = ['CZ3', 'CZ4','CZ6', 'CZ8', 'CZ9', 'CGrin1']
deep_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7']

mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']

data_dir = '/home/julio/Documents/DeepSup_project/DeepSup/rotation/'


align_angle_list = list()
name_list = list()
layer_list = list()
strain_list = list()
emb_list = list()

for mouse in mice_list:
    for emb in ['pca','isomap','umap']:
        name_list.append(mouse)
        if mouse in deep_mice:
            layer_list.append('deep')
        elif mouse in sup_mice:
            layer_list.append('sup')

        if ('GC' in mouse) or ('TG' in mouse):
            strain_list.append('Thy1')
        elif ('CZ' in mouse) or ('CG' in mouse):
            strain_list.append('Calb')
        elif 'Ch' in mouse:
            strain_list.append('ChRNA7')

        rotation_dict = load_pickle(data_dir, mouse+'_rotation_dict.pkl')
        align_angle_list.append(rotation_dict[emb]['align_angle'])
        emb_list.append(emb)

pd_align = pd.DataFrame(data={'mouse': name_list,
                         'layer': layer_list,
                         'strain': strain_list,
                         'align_angle': align_angle_list,
                         'emb': emb_list})




fig, ax = plt.subplots(1, 3, figsize=(10,6))
for idx, emb in enumerate(['pca', 'isomap', 'umap']):
    emb_pd = pd_align[pd_align['emb']==emb]
    b = sns.barplot(x='layer', y='align_angle', data=emb_pd,
                palette=palette_deepsup, linewidth = 1, width= .5, ax = ax[idx])
    sns.swarmplot(x='layer', y='align_angle', data=emb_pd,
            palette = 'dark:gray', edgecolor = 'gray', ax = ax[idx])
    ax[idx].set_title(emb)
plt.tight_layout()


plt.savefig(os.path.join(data_dir,'align_method_quantification.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir,'align_method_quantification.png'), dpi = 400,bbox_inches="tight")

pd_align.to_csv(os.path.join(data_dir, 'align_method_quantification.csv'))

