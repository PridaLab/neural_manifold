folder = uigetdir(cd, 'Select folder with mouse data');
files = dir(folder);
files_LT = files(contains({files(:).name},'LT_PyalData_struct', 'IgnoreCase',true));
for idx=1:size(files_LT,1)
    name = files_LT(idx,1).name;
    load(fullfile(folder, name));
    pos = trial_data(1,1).pos;
    for trial=2:size(trial_data,2)
        pos = [pos; trial_data(1,trial).pos];
    end
    LT_length = nanmax(pos(:,1))-nanmin(pos(:,1));
    fprintf('\nFile %s: \n\tCurrent length %.2f', name, LT_length)
    if LT_length>600
        scale_factor = 120/705;
    else
        scale_factor = 120/135;
    end
    scaled_pos = scale_factor*pos;
    fprintf('\n\tNew length %.2f', scale_factor*LT_length)
    scaled_vel = abs(diff(vecnorm(scaled_pos,2,2)))*20;
    scaled_vel = [scaled_vel(1); scaled_vel];
    for trial=1:size(trial_data,2)
        idx_start = trial_data(1,trial).idx_trial_start;
        idx_end = trial_data(1,trial).idx_trial_end;
        
        trial_pos = trial_data(1,trial).pos;
        scaled_trial_pos = scaled_pos(idx_start:idx_end,:);
        scaled_trial_vel = scaled_vel(idx_start:idx_end);
        assert(all(size(trial_pos)==size(scaled_trial_pos)))
        trial_data(1,trial).pos = scaled_trial_pos;
        trial_data(1,trial).vel = scaled_trial_vel;
        trial_data(1,trial).idx_peak_speed = nanmax(scaled_trial_vel);
    end
    save(fullfile(folder, name), 'trial_data')
    clear trial_data
end

%% 
clear all
folder = uigetdir(cd, 'Select folder with mouse data');
files = dir(folder);
files_LT = files(contains({files(:).name},'LT_PyalData_struct', 'IgnoreCase',true));
for idx=1:size(files_LT,1)
    name = files_LT(idx,1).name;
    fprintf('File %s\n', name)
    load(fullfile(folder, name));
    pos = trial_data(1,1).pos;
    for trial=2:size(trial_data,2)
        pos = [pos; trial_data(1,trial).pos];
    end
    scaled_vel = abs(diff(vecnorm(pos,2,2)))*20;
    scaled_vel = [scaled_vel(1); scaled_vel];
    for trial=1:size(trial_data,2)
        idx_start = trial_data(1,trial).idx_trial_start;
        idx_end = trial_data(1,trial).idx_trial_end;
        
        trial_vel = trial_data(1,trial).vel;
        scaled_trial_vel = scaled_vel(idx_start:idx_end);
        assert(all(size(trial_vel)==size(scaled_trial_vel)))
        trial_data(1,trial).vel = scaled_trial_vel;
        trial_data(1,trial).idx_peak_speed = nanmax(scaled_trial_vel);
    end
    save(fullfile(folder, name), 'trial_data')
    clear trial_data
end

    
    

