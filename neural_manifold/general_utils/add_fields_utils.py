import numpy as np
from neural_manifold.general_utils import data_preprocessing

def add_mov_direction_mat_field(pd_struct):
    def compute_movement_direction(position, speed=None, sf = 20, speed_th= 2):
        if isinstance(speed, type(None)):
            speed = np.sqrt(np.sum((np.diff(position, axis=0)*sf)**2,axis=1))
            speed = data_preprocessing.smooth_data(np.hstack((speed[0], speed)),bin_size=1/sf, std=0.5)

        mov_direction = np.zeros((speed.shape[0],))*np.nan
        mov_direction[speed<speed_th] = 0
        x_speed = np.diff(position[:,0])/(1/sf)
        x_speed = data_preprocessing.smooth_data(np.hstack((x_speed[0], x_speed)),bin_size=1/sf, std=0.5)
        right_moving = np.logical_and(speed>speed_th, x_speed>0)
        mov_direction[right_moving] = 1
        left_moving = np.logical_and(speed>speed_th, x_speed<0)
        mov_direction[left_moving] = -1
        mov_direction = np.round(data_preprocessing.smooth_data(mov_direction,bin_size=1/sf, std=0.5),0).astype(int).copy()

        mov_direction_dict = {0: 'non-moving', 1: 'moving to the right', -1: 'moving to the left'}
        return mov_direction, mov_direction_dict
    
    columns_name = [col for col in pd_struct.columns.values]
    lower_columns_name = [col.lower() for col in pd_struct.columns.values]

    if 'bin_size' in lower_columns_name:
        sf = 1/pd_struct.iloc[0][columns_name[lower_columns_name.index("bin_size")]]
    elif 'fs' in lower_columns_name:
        sf = pd_struct.iloc[0][columns_name[lower_columns_name.index("fs")]]
    elif 'sf' in lower_columns_name:
        sf = pd_struct.iloc[0][columns_name[lower_columns_name.index("sf")]]

    position = np.concatenate(pd_struct["position"].values, axis=0)
    if "speed" in pd_struct.columns:
        speed = np.concatenate(pd_struct["speed"].values, axis=0)
    else: 
        speed = None

    mov_direction, mov_direction_dict = compute_movement_direction(position, speed, sf)
    if "trial_id_mat" not in lower_columns_name:
        add_trial_id_mat_field(pd_struct)

    trial_id_mat = np.concatenate(pd_struct["trial_id_mat"].values, axis=0).reshape(-1,)

    pd_struct["mov_direction"] = [mov_direction[trial_id_mat==pd_struct["trial_id"][idx]] 
                                   for idx in pd_struct.index]

    pd_struct["mov_direction_dict"] = mov_direction_dict         

def add_trial_id_mat_field(pd_struct):
    pd_struct["trial_id_mat"] = [np.zeros((pd_struct["position"][idx].shape[0],1))+pd_struct["trial_id"][idx] 
                              for idx in pd_struct.index]

def add_trial_type_mat_field(pd_struct):

    pd_struct["trial_type_mat"] = [np.zeros((pd_struct["position"][idx].shape[0],1)).astype(int)+
                        ('L' == pd_struct["dir"][idx])+ 2*('R' == pd_struct["dir"][idx])+
                        4*('F' in pd_struct["dir"][idx]) for idx in pd_struct.index]
