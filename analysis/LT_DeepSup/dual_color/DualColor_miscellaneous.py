activity = {}
for case in ['veh', 'CNO']:
    print(f"\tcondition: {case}")
    case_dir = os.path.join(mouse_dir, mouse+'_'+case)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                               LOAD DATA                                |#
    #|________________________________________________________________________|#
    animal = gu.load_files(case_dir, '*_PyalData_struct.mat', verbose=True, struct_type = "PyalData")
    fnames = list(animal.keys())

    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
    fname_rot = [fname for fname in fnames if 'rot' in fname][0]

    animal_p = copy.deepcopy(animal[fname_pre])
    animal_r = copy.deepcopy(animal[fname_rot])

    #__________________________________________________________________________
    #|                                                                        |#
    #|               CHANGE COLUMN NAMES AND ADD NEW ONES                     |#
    #|________________________________________________________________________|#

    for column in columns_to_drop:
        if column in animal_p.columns: animal_p.drop(columns=[column], inplace=True)
        if column in animal_r.columns: animal_r.drop(columns=[column], inplace=True)

    for old, new in columns_to_rename.items():
        if old in animal_p.columns: animal_p.rename(columns={old:new}, inplace=True)
        if old in animal_r.columns: animal_r.rename(columns={old:new}, inplace=True)

    gu.add_trial_id_mat_field(animal_p)
    gu.add_trial_id_mat_field(animal_r)

    gu.add_mov_direction_mat_field(animal_p)
    gu.add_mov_direction_mat_field(animal_r)

    gu.add_trial_type_mat_field(animal_p)
    gu.add_trial_type_mat_field(animal_r)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                          KEEP ONLY MOVING                              |#
    #|________________________________________________________________________|#

    if vel_th>0:
        animal_p, animal_p_still = gu.keep_only_moving(animal_p, vel_th)
        animal_r, animal_r_still = gu.keep_only_moving(animal_r, vel_th)


    #__________________________________________________________________________
    #|                                                                        |#
    #|                          PREPROCESS TRACES                             |#
    #|________________________________________________________________________|#

    animal_p, animal_r = preprocess_traces(animal_p, animal_r, signal_field, sigma=sigma, sig_up = sig_up, sig_down = sig_down)
    animal_p['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}
    animal_r['clean_traces_params'] = {'sigma': sigma, 'sig_up': sig_up, 'sig_down': sig_down}




    signal_p = get_signal(animal_p,'clean_traces')
    signal_r = get_signal(animal_r,'clean_traces')

    activity[case] = np.zeros((signal_p.shape[1], 2))*np.nan

    for nn in range(signal_p.shape[1]):
        peaks_p, _ =find_peaks(signal_p[:,nn],height=0.01)
        activity[case][nn,0] = len(peaks_p)    

        peaks_r, _ =find_peaks(signal_r[:,nn],height=0.01)
        activity[case][nn,1] = len(peaks_r)



act_change =  np.concatenate((np.diff(activity['veh'], axis=1)/activity['veh'][:,0].reshape(-1,1), np.diff(activity['CNO'], axis=1)/activity['CNO'][:,0].reshape(-1,1)),axis=0)
act_change[act_change==np.inf] = -10
act_change[np.isnan(act_change)] = 0
condition = ['veh']*activity['veh'].shape[0] + ['CNO']*activity['CNO'].shape[0]
pd_act = pd.DataFrame(data={'act_change': act_change[:,0],
                     'condition': condition})    


plt.figure();
ax= plt.subplot(2,2,1)
ax.plot(activity['veh'])
ax.set_ylabel('event rate')
ax.set_xlabel('neuron number')
ax.set_title('veh')

ax= plt.subplot(2,2,3)
ax.plot(activity['CNO'])
ax.set_ylabel('event rate')
ax.set_xlabel('neuron number')
ax.set_title('CNO')


ax= plt.subplot(1,2,2)
sns.violinplot(data=pd_act, x='condition', y='act_change', ax=ax, palette = ['#666666ff', '#aa0007ff'])
ax.set_ylim([-5,5])
plt.suptitle(mouse)
plt.savefig(os.path.join('/home/julio/Pictures/',f'{mouse}_activity_change.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join('/home/julio/Pictures/',f'{mouse}_activity_change.png'), dpi = 400,bbox_inches="tight")






base_dir = '/home/julio/Documents/DeepSup_project/DualColor/Thy1jRGECO/'

for mouse in ['Thy1jRGECO22', 'Thy1jRGECO23']:
    save_dir = os.path.join(base_dir, 'processed_data', mouse)
    mouse_dict = load_pickle(save_dir, mouse+'_data_dict.pkl')

    og_red_signal = mouse_dict['original_signals']['signal_red_pre']

    signal_length = og_red_signal.shape[0]
    color_dir = os.path.join(base_dir, f'data/{mouse}/Inscopix_data/color_registration/')

    matched_signal = pd.read_csv(os.path.join(color_dir,mouse+'_matched_raw.csv')).to_numpy()[1:,1:].astype(np.float64)
    unmatched_signal = pd.read_csv(os.path.join(color_dir,mouse+'_unmatched_raw.csv')).to_numpy()[1:,1:].astype(np.float64)
    uncertain_signal = pd.read_csv(os.path.join(color_dir,mouse+'_uncertain_raw.csv')).to_numpy()[1:,1:].astype(np.float64)

    matched_indexes = np.zeros((matched_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_red_signal.shape[1]).astype(int)
    for cell_matched in range(matched_signal.shape[1]):
        for cell_red in cells_to_check:
            corr_coeff = np.corrcoef(matched_signal[:signal_length,cell_matched], og_red_signal[:,cell_red])[0,1]
            if corr_coeff > 0.999:
                matched_indexes[cell_matched] = cell_red;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_red)[0])
                break;
    matched_indexes = matched_indexes.astype(int)

    unmatched_indexes = np.zeros((unmatched_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_red_signal.shape[1]).astype(int)
    for cell_unmatched in range(unmatched_signal.shape[1]):
        for cell_red in cells_to_check:
            corr_coeff = np.corrcoef(unmatched_signal[:signal_length,cell_unmatched], og_red_signal[:,cell_red])[0,1]
            if corr_coeff > 0.999:
                unmatched_indexes[cell_unmatched] = cell_red;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_red)[0])
                break;
    unmatched_indexes = unmatched_indexes.astype(int)

    uncertain_indexes = np.zeros((uncertain_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_red_signal.shape[1]).astype(int)
    for cell_uncertain in range(uncertain_signal.shape[1]):
        for cell_red in cells_to_check:
            corr_coeff = np.corrcoef(uncertain_signal[:signal_length,cell_uncertain], og_red_signal[:,cell_red])[0,1]
            if corr_coeff > 0.999:
                uncertain_indexes[cell_uncertain] = cell_red;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_red)[0])
                break;
    uncertain_indexes = uncertain_indexes.astype(int)

    registered_red_cells = mouse_dict['registered_clean_traces']['pre_red_cells']
    registered_matched_indexes = [];
    for matched_index in matched_indexes:
        if matched_index in registered_red_cells:
            new_index = np.where(registered_red_cells==matched_index)[0][0]
            registered_matched_indexes.append(new_index)

    registered_unmatched_indexes = [];
    for unmatched_index in unmatched_indexes:
        if unmatched_index in registered_red_cells:
            new_index = np.where(registered_red_cells==unmatched_index)[0][0]
            registered_unmatched_indexes.append(new_index)

    registered_uncertain_indexes = [];
    for uncertain_index in uncertain_indexes:
        if uncertain_index in registered_red_cells:
            new_index = np.where(registered_red_cells==uncertain_index)[0][0]
            registered_uncertain_indexes.append(new_index)




    og_green_signal = mouse_dict['original_signals']['signal_green_pre']

    matched_indexes = np.zeros((matched_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_green_signal.shape[1]).astype(int)
    for cell_matched in range(matched_signal.shape[1]):
        for cell_green in cells_to_check:
            corr_coeff = np.corrcoef(matched_signal[:signal_length,cell_matched], og_green_signal[:,cell_green])[0,1]
            if corr_coeff > 0.6:
                matched_indexes[cell_matched] = cell_green;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_green)[0])
                break;
    matched_indexes = matched_indexes.astype(int)

    unmatched_indexes = np.zeros((unmatched_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_green_signal.shape[1]).astype(int)
    for cell_unmatched in range(unmatched_signal.shape[1]):
        for cell_green in cells_to_check:
            corr_coeff = np.corrcoef(unmatched_signal[:signal_length,cell_unmatched], og_green_signal[:,cell_green])[0,1]
            if corr_coeff > 0.6:
                unmatched_indexes[cell_unmatched] = cell_green;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_green)[0])
                break;
    unmatched_indexes = unmatched_indexes.astype(int)

    uncertain_indexes = np.zeros((uncertain_signal.shape[1],))*np.nan
    cells_to_check = np.arange(og_green_signal.shape[1]).astype(int)
    for cell_uncertain in range(uncertain_signal.shape[1]):
        for cell_green in cells_to_check:
            corr_coeff = np.corrcoef(uncertain_signal[:signal_length,cell_uncertain], og_green_signal[:,cell_green])[0,1]
            if corr_coeff > 0.6:
                uncertain_indexes[cell_uncertain] = cell_green;
                cells_to_check = np.delete(cells_to_check, np.where(cells_to_check==cell_green)[0])
                break;
    uncertain_indexes = uncertain_indexes.astype(int)

    registered_green_cells = mouse_dict['registered_clean_traces']['pre_green_cells']
    registered_matched_indexes = [];
    for matched_index in matched_indexes:
        if matched_index in registered_green_cells:
            new_index = np.where(registered_green_cells==matched_index)[0][0]
            registered_matched_indexes.append(new_index)

    registered_unmatched_indexes = [];
    for unmatched_index in unmatched_indexes:
        if unmatched_index in registered_green_cells:
            new_index = np.where(registered_green_cells==unmatched_index)[0][0]
            registered_unmatched_indexes.append(new_index)

    registered_uncertain_indexes = [];
    for uncertain_index in uncertain_indexes:
        if uncertain_index in registered_green_cells:
            new_index = np.where(registered_green_cells==uncertain_index)[0][0]
            registered_uncertain_indexes.append(new_index)
