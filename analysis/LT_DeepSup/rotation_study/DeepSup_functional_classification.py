palette_deepsup = ["#cc9900ff", "#9900ffff"]
mice_list = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
sup_mice = ['CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deep_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4', 'ChZ12']
base_dir = '/home/julio/Documents/DeepSup_project/DeepSup/'


from os import path
from sklearn.feature_selection import mutual_info_regression


def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))


#__________________________________________________________________________
#|                                                                        |#
#|                          COMPUTE INFORMATION                           |#
#|________________________________________________________________________|#


data_dir = path.join(base_dir, 'processed_data')
functional_dir = path.join(base_dir, 'functional_cells')
save_dir = path.join(base_dir, 'mutual_info')

for mouse in mice_list:
    print(f"\nWorking on mouse {mouse}: ")
    file_path = path.join(data_dir, mouse)
    pd_mouse = load_pickle(file_path,mouse+'_df_dict.pkl')

    mi_scores = {
        'label_order': ['global_pos','local_pos']
    }

    fnames = list(pd_mouse.keys())
    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]
    pd_pre = copy.deepcopy(pd_mouse[fname_pre])
    pd_rot = copy.deepcopy(pd_mouse[fname_rot])

    #signal
    traces_pre = get_signal(pd_pre, 'clean_traces')
    pos_pre = get_signal(pd_pre, 'pos')[:,0]
    pos_pre = (pos_pre-np.min(pos_pre))/(np.max(pos_pre)-np.min(pos_pre))

    traces_rot = get_signal(pd_rot, 'clean_traces')
    pos_rot = get_signal(pd_rot, 'pos')[:,0]
    pos_rot = (pos_rot-np.min(pos_rot))/(np.max(pos_rot)-np.min(pos_rot))

    traces_both = np.concatenate((traces_pre, traces_rot), axis=0)
    global_pos = np.concatenate((pos_pre, pos_rot), axis=0)
    local_pos = np.concatenate((pos_pre, np.abs(pos_rot-1)), axis=0)

    x = np.concatenate((global_pos.reshape(-1,1),local_pos.reshape(-1,1)),axis=1)
    mi_regression = np.zeros((x.shape[1], traces_both.shape[1]))*np.nan
    for cell in range(traces_both.shape[1]):
        mi_regression[:, cell] = mutual_info_regression(x, traces_both[:,cell], n_neighbors = 50, random_state = 16)
    mi_scores['mi_scores'] = copy.deepcopy(mi_regression)

    save_file = open(path.join(save_dir, f'{mouse}_mi_scores_dict.pkl'), "wb")
    pickle.dump(mi_scores, save_file)
    save_file.close()


#__________________________________________________________________________
#|                                                                        |#
#|                          COMPUTE INFORMATION                           |#
#|________________________________________________________________________|#



mutual_info_dir = path.join(base_dir, 'mutual_info')
functional_dir = path.join(base_dir, 'functional_cells')
save_dir = path.join(base_dir, 'mutual_info')


cell_type_list = list()
global_mi_list = list()
local_mi_list = list()
mouse_list = list()
layer_list = list()

global_corr_list = list()
rot_corr_list = list()
xmirror_corr_list = list()
ymirror_corr_list = list()


for mouse in mice_list:
    cell_type = np.load(path.join(functional_dir, mouse, f'{mouse}_cellType.npy'))
    cell_type[np.logical_and(cell_type>0,cell_type<4)] == 1
    cell_type[cell_type==4] = 2
    cell_type[cell_type==5] = 3
    name_types = ['Global', 'Local', 'Remap', 'N/A']
    cell_type_list.append([name_types[int(x)] for x in cell_type])


    cell_corr = np.load(path.join(functional_dir, mouse, f'{mouse}_cellTypeCorr.npy')).T
    cell_corr[np.isnan(cell_corr)]=0
    global_corr_list.append(list(cell_corr[0,:]))
    rot_corr_list.append(list(cell_corr[1,:]))
    xmirror_corr_list.append(list(cell_corr[2,:]))
    ymirror_corr_list.append(list(cell_corr[3,:]))




    mi_scores = load_pickle(mutual_info_dir, f'{mouse}_mi_scores_dict.pkl')['mi_scores']
    global_mi_list.append(list(mi_scores[0,:]))
    local_mi_list.append(list(mi_scores[1,:]))

    mouse_list.append([mouse]*len(cell_type))
    if mouse in deep_mice:
        layer_list.append(['deep']*len(cell_type))
    elif mouse in sup_mice:
        layer_list.append(['sup']*len(cell_type))


mouse_list = sum(mouse_list, []) #[cell for day in mouse_list cell in day]
layer_list = sum(layer_list, []) #[cell for day in mouse_list cell in day]

global_mi_list = sum(global_mi_list, [])
local_mi_list = sum(local_mi_list, [])
cell_type_list = sum(cell_type_list, [])

global_corr_list = sum(global_corr_list, [])
rot_corr_list = sum(rot_corr_list, [])
xmirror_corr_list = sum(xmirror_corr_list, [])
ymirror_corr_list = sum(ymirror_corr_list, [])

pd_cell_mi = pd.DataFrame(data={'mouse': mouse_list,
                             'layer': layer_list,
                             'global_mi': global_mi_list, 
                             'local_mi': local_mi_list,
                             'cell_type': cell_type_list,
                             'global_corr': global_corr_list,
                             'rot_corr': rot_corr_list,
                             'xmirror_corr': xmirror_corr_list,
                             'ymirror_corr': ymirror_corr_list})


fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.violinplot(data = pd_cell_mi, x = 'cell_type', y = 'global_mi', ax=ax[0])
sns.violinplot(data = pd_cell_mi, x = 'cell_type', y = 'local_mi', ax=ax[1])

from sklearn.svm import SVC #For support vector classification (SVM)

x_data = np.concatenate((np.array(pd_cell_mi["global_mi"].to_list()).reshape(-1,1), 
    np.array(pd_cell_mi["local_mi"].to_list()).reshape(-1,1),
    np.array(pd_cell_mi["global_corr"].to_list()).reshape(-1,1),
    np.array(pd_cell_mi["rot_corr"].to_list()).reshape(-1,1),
    np.array(pd_cell_mi["xmirror_corr"].to_list()).reshape(-1,1),
    np.array(pd_cell_mi["ymirror_corr"].to_list()).reshape(-1,1)),axis=1)


y_data = np.array([name_types.index(x) for x in pd_cell_mi["cell_type"].to_list()])

idx = np.arange(len(y_data))
np.random.shuffle(idx)
x_data = x_data[idx,:]
y_data = y_data[idx]




model = SVC(C=3, max_iter = -1)
model.fit(x_data[:1200], y_data[:1200])
y_predict = model.predict(x_data[1200:])

median_absolute_error(y_data[1200:], y_predict)


from sklearn import metrics

print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_data[1200:], y_predict)}\n"
)


# Classification report for classifier SVC(C=3):
#               precision    recall  f1-score   support

#            0       0.79      0.43      0.56       222
#            1       0.67      0.28      0.39        94
#            2       0.53      0.69      0.60       242
#            3       0.36      0.64      0.46       110

#     accuracy                           0.54       668
#    macro avg       0.59      0.51      0.50       668
# weighted avg       0.61      0.54      0.53       668


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_data[1200:], y_predict)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()