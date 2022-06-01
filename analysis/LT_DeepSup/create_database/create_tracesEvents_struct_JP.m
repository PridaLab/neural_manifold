function create_tracesEvents_struct_JP(mouse_name, varargin)
    w = warning;
    warning('off','all')
    
    je = inputParser;
    addParameter(je,'folder_data','None',@ischar)
    addParameter(je,'save_data','None',@ischar)
    addParameter(je,'session_names',{'lt', 'rot'},@iscell)
    addParameter(je,'session_number',[5,7],@isvector)
    parse(je,varargin{:})

    folder_data = je.Results.folder_data; %folder where the data is
    save_data = je.Results.save_data; %folder in which to save the results
    session_names = je.Results.session_names; %name of the sessions
    session_number = je.Results.session_number; %number of the sessions

    if strcmp('None', folder_data)
        folder_data = uigetdir(cd, 'Select folder with mouse data');
    end
    if strcmp('None', save_data)
        save_data = uigetdir(cd, 'Select folder to save data');
    end
    fileID = fopen(fullfile(save_data,'logFile.txt'),'w');
    files = dir(folder_data); 
    double_fprintf(fileID, 'logFile: %s\n\nfolder_data: %s\nsave_data: %s\n', ...
        datetime,folder_data, save_data);
    %infer which protocol to use based on the name of the files inside the
    %folder
    if sum(contains(session_names, 'lt','IgnoreCase',true))*...
            sum(contains(session_names, 'rot','IgnoreCase',true)) == 1

        session_type = 'LT_ROT';
        double_fprintf(fileID,'LT to Rotation pipeline selected.\n')
    elseif sum(contains(session_names, 'LT','IgnoreCase',true))*...
            sum(contains(session_names, 'ALO','IgnoreCase',true))*...
            sum(contains(session_names, 'RL_EAST','IgnoreCase',true))*...
            sum(contains(session_names, 'RL_WEST','IgnoreCase',true))== 1

        session_type = 'Full_Castle';
        double_fprintf(fileID,'Full Castle pipeline selected.\n')
    else
        session_type = 'None';
    end
    if length(session_names)>1
        %pipeline of analysis in case of LT_ROT (registration)
        if contains(session_type, 'LT_ROT', 'IgnoreCase',true)
            %get guide for first and second session files
            files_LT = files(contains({files(:).name}, session_names{1}, 'IgnoreCase',true));
            files_ROT = files(contains({files(:).name}, session_names{2}, 'IgnoreCase',true));
            %Load correspondance table
            correspondence_file_index = contains({files(:).name}, 'correspondence', 'IgnoreCase',true);
            if ~any(correspondence_file_index)
                error("Couldn't find correspondance file. Please check " + ...
                "that it is on the folder with the word 'correspondence' in the name.")            
            elseif sum(correspondence_file_index)>1
                error("More than one correspondence file found in the folder.")
            end

            correspondence_file_index = find(correspondence_file_index);
            correspondence_table = readtable([files(correspondence_file_index).folder, ...
                                '/',files(correspondence_file_index).name]);
            %Get local_to_global_guide for LT and ROT
            accepted_to_global_guide_LT = correspondence_table{correspondence_table{:,3}==0,1:2};
            accepted_to_global_guide_ROT = correspondence_table{correspondence_table{:,3}==1,1:2};

            %Keep only cells that appear in both sessions
            Lia  = ismember(accepted_to_global_guide_LT(:,1), accepted_to_global_guide_ROT(:,1));
            Lib = ismember(accepted_to_global_guide_ROT(:,1), accepted_to_global_guide_LT(:,1));
            accepted_to_global_guide_LT = accepted_to_global_guide_LT(Lia,:);
            accepted_to_global_guide_ROT = accepted_to_global_guide_ROT(Lib,:);
            double_fprintf(fileID,"Global cells: %i\n%s accepted cells: %i\n%s accepted cells: %i\n",sum(Lia), ...
                                            session_names{1},size(Lia,1),session_names{2}, size(Lib,1))
            %LT
            double_fprintf(fileID,'\n-%s:\n',session_names{1})
            tracesEvents = get_tracesEvents(files_LT,accepted_to_global_guide_LT, session_names{1}, fileID);
            tracesEvents.mouse = mouse_name;
            tracesEvents.session = session_number(1);
            save([save_data, '/',mouse_name,'_', session_names{1}, '_events_s', int2str(session_number(1)), '.mat'], "tracesEvents")
      
            %ROT
            double_fprintf(fileID,'\n-%s:\n',session_names{2})

            tracesEvents = get_tracesEvents(files_ROT,accepted_to_global_guide_ROT, session_names{2}, fileID);
            tracesEvents.mouse = mouse_name;
            tracesEvents.session = session_number(2);
            save([save_data, '/',mouse_name,'_', session_names{2}, '_events_s',int2str(session_number(2)), '.mat'], "tracesEvents")
        end
    else 
        tracesEvents = get_tracesEvents(files,[], session_names{1}, fileID);
        tracesEvents.mouse = mouse_name;
        tracesEvents.session = session_number(1);
        save([save_data, '/',mouse_name,'_', session_names{1}, '_events_s', int2str(session_number(1)), '.mat'], "tracesEvents")
    end
    warning(w)
end


function [tracesEvents] = get_tracesEvents(files,accepted_to_global_guide, condition, fileID)

    tracesEvents = struct();
    tracesEvents.test = condition;
    %Load raw traces props to get mapping from "accepted" to "local" cells
    raw_file_index = contains({files(:).name}, 'raw', 'IgnoreCase',true) ...
                            .*contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(raw_file_index)<1
        error("Couldn't find a raw file for condition %s. Please check that it " + ...
            "is on the folder with the word 'raw' & ""props"" in the name.", condition)
    elseif sum(raw_file_index)>1
        error("More than one raw files found for the condition %s in the folder.", condition)
    end
    raw_file_index = find(raw_file_index);
    raw_props_table = readtable([files(raw_file_index).folder, '/',files(raw_file_index).name]);
    raw_props_mat = (0:size(raw_props_table,1)-1)';

    accepted_cells = cellfun(@(c) (contains(c,'accepted') || contains(c, 'undecided')), raw_props_table{:,2});
    double_fprintf(fileID,"\tDetected cells: %i\n\tAccepted cells: %i\n", size(raw_props_mat,1), sum(accepted_cells))

    accepted_to_local_guide = [(0:sum(accepted_cells)-1)',raw_props_mat(accepted_cells)]; 
    if isempty(accepted_to_global_guide)
        accepted_to_global_guide = [accepted_to_local_guide(:,1),accepted_to_local_guide(:,1)];
    end
    gla_guide = array2table(accepted_to_global_guide(:,1));
    gla_guide.Properties.VariableNames{1} = 'global';
    gla_guide(:,2) = array2table(accepted_to_global_guide(:,2));
    gla_guide.Properties.VariableNames{2} = 'local_accepted';
    gla_guide(:,3) = array2table(accepted_to_global_guide(:,2)*0);
    gla_guide.Properties.VariableNames{3} = 'local_all';
    %add local_all guide 
    for cell= 1:size(gla_guide)
        accepted_num = gla_guide{cell,2};
        local_idx = accepted_to_local_guide(:,1)==accepted_num;
        gla_guide(cell,3) = array2table(accepted_to_local_guide(local_idx,2));
    end
    tracesEvents.gla_guide = gla_guide;

    %Load raw traces (local guide)
    double_fprintf(fileID,'\tLoading raw traces...')
    raw_file_index = contains({files(:).name}, 'raw', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(raw_file_index)<1
        error("Couldn't find a raw file for condition %s. Please check that it " + ...
            "is on the folder with the word 'raw' in the name.", condition)
    elseif sum(raw_file_index)>1
        error("More than one raw files found for the condition %s in the folder.", condition)
    end
    raw_file_index = find(raw_file_index);
    double_fprintf(fileID,'\b\b\b: %s - ',files(raw_file_index).name)
    raw_table = readtable([files(raw_file_index).folder, '/',files(raw_file_index).name]);
    tracesEvents.raw_traces = raw_table{:, tracesEvents.gla_guide{:,3}+2}; %+2: +1 because first column is time, +1 because cells in guide start in 0
    double_fprintf(fileID,' Done\n')

    %Load denoised traces (accepted guide)
    double_fprintf(fileID,'\tLoading denoised traces... ')
    traces_file_index = contains({files(:).name}, 'traces', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(traces_file_index)<1
        traces_file_index = contains({files(:).name}, 'denoised', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
        if sum(traces_file_index)<1
            error("Couldn't find a trace file for condition %s. Please check that it " + ...
                "is on the folder with the word 'traces' or 'denoised' in the name.", condition)
        end
    elseif sum(traces_file_index)>1
        error("More than one traces files found for the condition %s in the folder.", condition)
    end
    traces_file_index = find(traces_file_index);
    double_fprintf(fileID,'\b\b\b: %s - ',files(traces_file_index).name)

    traces_table = readtable([files(traces_file_index).folder, '/',files(traces_file_index).name]);
    tracesEvents.denoised_traces = traces_table{:,tracesEvents.gla_guide{:,2}+2}; %+2: +1 because first column is time, +1 because cells in guide start in 0
    double_fprintf(fileID,' Done\n')
    %Get sampling frequency
    times = traces_table{:,1};
    tracesEvents.sF = 1/median(diff(times));
    %Load spikes (local guide)
    double_fprintf(fileID,'\tLoading spikes...')
    spikes_file_index = contains({files(:).name}, 'spikes', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    spikes_file_index = find(spikes_file_index);
    double_fprintf(fileID,'\b\b\b: %i files found:\n',length(spikes_file_index))
    for file_num= 1:length(spikes_file_index)
        spikes_idx = spikes_file_index(file_num);
        double_fprintf(fileID,'\t\t%i/%i: %s: ', file_num, length(spikes_file_index), files(spikes_idx).name);
        spikes_table = readtable([files(spikes_idx).folder, '/',files(spikes_idx).name]);
        spikes_bi_array = zeros(size(traces_table,1), size(raw_props_table,1));
        spikes_amp_array = zeros(size(traces_table,1), size(raw_props_table,1));
        spikes_index = zeros(size(spikes_table,1),1);
        %check og # of neurons on raw_props
        div_edges = (1:10000:size(spikes_table,1)+1);
        if div_edges(end)<=size(spikes_table,1)
            div_edges = [div_edges, size(spikes_table,1)+1]; %#ok<AGROW> 
        end
        double_fprintf(fileID,'batch X/X')
        del_s = '\b\b\b';
        for batch = 1:length(div_edges)-1
            eval(strcat("double_fprintf(fileID,'", del_s, "%i/%i',batch, length(div_edges)-1)"))
            del_s = repmat({'\b'},1,length(int2str(batch))+length(int2str(length(div_edges)-1))+1);
            del_s = strcat(del_s{:});
            [~,batch_idx] = min(abs(spikes_table{div_edges(batch):div_edges(batch+1)-1,1}-times'),[],2);
            spikes_index(div_edges(batch):div_edges(batch+1)-1,1) = batch_idx;
        end
        spikes_neuron = cell2mat(arrayfun(@(spike) str2double(spike{1}(2:end)), spikes_table{:,2}, 'UniformOutput', false));
        spikes_amplitude = spikes_table{:,3};
        linearidx = sub2ind(size(spikes_bi_array), spikes_index(:,1), spikes_neuron(:,1)+1);
        spikes_bi_array(linearidx) = 1; %#ok<NASGU> 
        spikes_amp_array(linearidx) = spikes_amplitude; %#ok<NASGU> 
        name_file = files(spikes_idx).name;
        name_field = strcat('spikes_',name_file(strfind(lower(name_file), 'spikes_')+7:end-4));
        eval(strcat('tracesEvents.', name_field, ' = spikes_bi_array(:,tracesEvents.gla_guide{:,3}+1);'))
        eval(strcat('tracesEvents.', name_field, '_amp = spikes_amp_array(:,tracesEvents.gla_guide{:,3}+1);'))
        double_fprintf(fileID,' - Done\n')
    end
    %Load events (local guide)
    double_fprintf(fileID,'\tLoading events...')
    events_file_index = contains({files(:).name}, 'events', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    events_file_index = find(events_file_index);
    double_fprintf(fileID,'\b\b\b: %i files found:\n',length(events_file_index))
    for file_num= 1:length(events_file_index)
        events_idx = events_file_index(file_num);
        double_fprintf(fileID,'\t\t%i/%i: %s: ', file_num, length(events_file_index), files(events_idx).name);
        events_table = readtable([files(events_idx).folder, '/',files(events_idx).name]);
        num_cells = events_table{end,2}{1};
        num_cells = str2double(num_cells(2:end));
        events_bi_array = zeros(size(traces_table,1), num_cells+1);
        events_amp_array = zeros(size(traces_table,1), num_cells+1);
        events_index = zeros(size(events_table,1),1);
        %check og # of neurons on raw_props
        div_edges = (1:10000:size(events_table,1)+1);
        if div_edges(end)<=size(events_table,1)
            div_edges = [div_edges, size(events_table,1)+1]; %#ok<AGROW> 
        end
        double_fprintf(fileID,'batch X/X')
        del_s = '\b\b\b';
        for batch = 1:length(div_edges)-1
            eval(strcat("double_fprintf(fileID,'", del_s, "%i/%i',batch, length(div_edges)-1)"))
            del_s = repmat({'\b'},1,length(int2str(batch))+length(int2str(length(div_edges)-1))+1);
            del_s = strcat(del_s{:});
            [~,batch_idx] = min(abs(events_table{div_edges(batch):div_edges(batch+1)-1,1}-times'),[],2);
            events_index(div_edges(batch):div_edges(batch+1)-1,1) = batch_idx;
        end
        events_neuron = cell2mat(arrayfun(@(event) str2double(event{1}(2:end)), events_table{:,2}, 'UniformOutput', false));
        events_amplitude = events_table{:,3};
        linearidx = sub2ind(size(events_bi_array), events_index(:,1), events_neuron(:,1)+1);
        events_bi_array(linearidx) = 1; %#ok<NASGU> 
        events_amp_array(linearidx) = events_amplitude; %#ok<NASGU> 
        name_file = files(events_idx).name;
        name_field = strcat('events_',name_file(strfind(lower(name_file), 'events_')+7:end-4));
        eval(strcat('tracesEvents.', name_field, ' = events_bi_array(:,tracesEvents.gla_guide{:,2}+1);'))
        eval(strcat('tracesEvents.', name_field, '_amp = events_amp_array(:,tracesEvents.gla_guide{:,2}+1);'))
        double_fprintf(fileID,' - Done\n')
    end
    %Load position
    double_fprintf(fileID,'\tLoading position... ')
    position_file_index = contains({files(:).name}, 'position', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(position_file_index)<1
        error("Couldn't find a position file for condition %s. Please check that it " + ...
            "is on the folder with the word 'position' in the name.", condition)
    elseif sum(position_file_index)>1
        error("More than one position files found for the condition %s in the folder.", condition)
    end
    position_file_index = find(position_file_index);
    double_fprintf(fileID,'\b\b\b: %s - ',files(position_file_index).name)

    if contains(files(position_file_index).name(end-3:end), '.csv')
        position_table = readtable([files(position_file_index).folder, '/',files(position_file_index).name]);
        tracesEvents.position = position_table{:,:};
    else
        temp = load([files(position_file_index).folder, '/',files(position_file_index).name]);
        field_temp = fieldnames(temp);
        for field = 1:size(field_temp,1)
            if contains(field_temp{field,1}, 'position', 'IgnoreCase', true)
                tracesEvents.position = temp.(field_temp{field,1});
            elseif contains(field_temp{field,1}, 'droppedFrames', 'IgnoreCase', true)
                tracesEvents.droppedFrames = temp.(field_temp{field,1})+1;
            elseif contains(field_temp{field,1}, 'frame', 'IgnoreCase', true)
                tracesEvents.frame = temp.(field_temp{field,1});
            elseif contains(field_temp{field,1}, 'scale', 'IgnoreCase', true)
                tracesEvents.pixel_scale = temp.(field_temp{field,1});
            end
        end
    end
    if  sum(contains(fieldnames(tracesEvents), 'scale', 'IgnoreCase',true))>0
        tracesEvents.position = tracesEvents.position*tracesEvents.pixel_scale;
    end
    tracesEvents.velocity = sqrt(diff(tracesEvents.position(:,1)).^2+ diff(tracesEvents.position(:,2)).^2)/(1/20); 
    tracesEvents.velocity = [tracesEvents.velocity(1,1); tracesEvents.velocity];
    double_fprintf(fileID,'Done\n')
    double_fprintf(fileID,'\t\tOriginal shapes: Inscopix %i - Position %i\n', size(tracesEvents.raw_traces,1),size(tracesEvents.position,1))
    %Delete dropped frames
    tracesEvents.time = times;
    if isfield(tracesEvents,'droppedFrames')
        if ~isempty(tracesEvents.droppedFrames)
            if (size(tracesEvents.position,1)+length(tracesEvents.droppedFrames)) == size(tracesEvents.denoised_traces,1)
                double_fprintf(fileID,'\tInterpolating dropped frames on position (%i)...', length(tracesEvents.droppedFrames))

                og_frames = (1:size(tracesEvents.denoised_traces,1))';
                pos_frames = og_frames;
                pos_frames(tracesEvents.droppedFrames) = [];
                pos_query = interp1(pos_frames, tracesEvents.position, tracesEvents.droppedFrames);
                new_pos = zeros(size(tracesEvents.denoised_traces,1),2);
                new_pos(pos_frames,:) = tracesEvents.position;
                new_pos(tracesEvents.droppedFrames,:) = pos_query;
                tracesEvents.position = new_pos;
                tracesEvents.velocity = sqrt(diff(tracesEvents.position(:,1)).^2+ diff(tracesEvents.position(:,2)).^2)/(1/20); 
                tracesEvents.velocity = [tracesEvents.velocity(1,1); tracesEvents.velocity];
                double_fprintf(fileID,' Done\n')
            else
                double_fprintf(fileID,'\tRemoving dropped frames (%i)...', length(tracesEvents.droppedFrames))
                tracesEvents.time(tracesEvents.droppedFrames,:) = [];
                tracesEvents.raw_traces(tracesEvents.droppedFrames,:) = [];
                tracesEvents.denoised_traces(tracesEvents.droppedFrames,:) = [];
                fields = fieldnames(tracesEvents);
                spike_fields = fields(contains(fields,'spikes_'));
                for idx = 1:length(spike_fields)
                    eval(strcat('tracesEvents.',spike_fields{idx},'(tracesEvents.droppedFrames,:) = [];'))
                end
                event_fields = fields(contains(fields,'events_'));
                for idx = 1:length(event_fields)
                    eval(strcat('tracesEvents.',event_fields{idx},'(tracesEvents.droppedFrames,:) = [];'))
                end
                double_fprintf(fileID,' Done\n')
            end
            double_fprintf(fileID,'\t\tResulting shapes: Inscopix %i - Position %i\n', size(tracesEvents.raw_traces,1),size(tracesEvents.position,1))
        end
    end
    % Check position initial zeros
    if all(tracesEvents.position(1,:) == 0)
        keyboard;
        double_fprintf(fileID,'\tRemoving zeros at the beggining of position...')
        initial_zeros = find(all(tracesEvents.position == 0, 2));
        double_fprintf(fileID,'\b\b\b: %i entries compromised',length(initial_zeros))
        tracesEvents.time(initial_zeros,:) = [];
        tracesEvents.position(initial_zeros,:) = [];
        tracesEvents.velocity(initial_zeros,:) = [];
        tracesEvents.raw_traces(initial_zeros,:) = [];
        tracesEvents.denoised_traces(initial_zeros,:) = [];
        fields = fieldnames(tracesEvents);
        spike_fields = fields(contains(fields,'spikes_'));
        for idx = 1:length(spike_fields)
            eval(strcat('tracesEvents.',spike_fields{idx},'(initial_zeros,:) = [];'))
        end
        event_fields = fields(contains(fields,'events_'));
        for idx = 1:length(event_fields)
            eval(strcat('tracesEvents.',event_fields{idx},'(initial_zeros,:) = [];'))
        end
        double_fprintf(fileID,' - Done\n')
        double_fprintf(fileID,'\t\tResulting shapes: Inscopix %i - Position %i\n', size(tracesEvents.raw_traces,1),size(tracesEvents.position,1))
    end
    % Check position size with inscopix signals
    if size(tracesEvents.position,1)<size(tracesEvents.denoised_traces,1)
        end_idx = size(tracesEvents.position,1)+1;
        double_fprintf(fileID,'\tPosition still %i frames sorter than Inscopix signals. Dropping final edges of Inscopix...', ...
                                                size(tracesEvents.denoised_traces,1)-size(tracesEvents.position,1));
        tracesEvents.time(end_idx:end,:) = [];
        tracesEvents.raw_traces(end_idx:end,:) = [];
        tracesEvents.denoised_traces(end_idx:end,:) = [];
        fields = fieldnames(tracesEvents);
        spike_fields = fields(contains(fields,'spikes_'));
        for idx = 1:length(spike_fields)
            eval(strcat('tracesEvents.',spike_fields{idx},'(end_idx:end,:) = [];'))
        end
        event_fields = fields(contains(fields,'events_'));
        for idx = 1:length(event_fields)
            eval(strcat('tracesEvents.',event_fields{idx},'(end_idx:end,:) = [];'))
        end
        double_fprintf(fileID,' - Done\n')
        double_fprintf(fileID,'\t\tResulting shapes: Inscopix %i - Position %i\n', size(tracesEvents.raw_traces,1),size(tracesEvents.position,1))
    end

    if size(tracesEvents.position,1)>size(tracesEvents.denoised_traces,1)
        double_fprintf(fileID,'\tPosition still %i frames longer than Inscopix signals. Dropping final edges of Inscopix...', ...
                                                size(tracesEvents.position,1)-size(tracesEvents.denoised_traces,1));
        if size(tracesEvents.position,1)-size(tracesEvents.denoised_traces,1) > 10
            warning('More than 10 frames more on position than on traces')
        end
        end_idx = size(tracesEvents.denoised_traces,1)+1;
        tracesEvents.position(end_idx:end,:) = [];
        tracesEvents.velocity(end_idx:end,:) = [];
        double_fprintf(fileID,' Done\n')

    end
    %CHECK NANS IN TRACES
    if sum(isnan(tracesEvents.denoised_traces(:,1)))>0
        double_fprintf(fileID,'\tRemoving nans found on traces on all signals...')
        nan_idx = isnan(tracesEvents.denoised_traces(:,1));
        nan_num = sum(nan_idx);
        double_fprintf(fileID,'\b\b\b: %i entries compromised',sum(nan_num))
        tracesEvents.time(nan_idx,:) = [];
        tracesEvents.raw_traces(nan_idx,:) = [];
        tracesEvents.denoised_traces(nan_idx,:) = [];
        tracesEvents.position(nan_idx,:) = [];
        tracesEvents.velocity(nan_idx,:) = [];
        fields = fieldnames(tracesEvents);
        spike_fields = fields(contains(fields,'spikes_'));
        for idx = 1:length(spike_fields)
            eval(strcat('tracesEvents.',spike_fields{idx},'(nan_idx,:) = [];'))
        end
        event_fields = fields(contains(fields,'events_'));
        for idx = 1:length(event_fields)
            eval(strcat('tracesEvents.',event_fields{idx},'(nan_idx,:) = [];'))
        end
        double_fprintf(fileID,' - Done\n')
        double_fprintf(fileID,'\t\tResulting shapes: Inscopix %i - Position %i\n', size(tracesEvents.raw_traces,1),size(tracesEvents.position,1))
    end
    %load cells images
    image_file_index = contains({files(:).name}, '.tif', 'IgnoreCase',true) ...
                           .*contains({files(:).name}, 'contours', 'IgnoreCase',true);
    if sum(image_file_index)>0
        image_file_index = find(image_file_index);
        image = Tiff([files(image_file_index).folder, '/',files(image_file_index).name]);
        imageData = read(image);
        tracesEvents.cell_image = imageData;
        close(image);
    end
    %load behavioral frame
    image_file_index = contains({files(:).name}, '.tif', 'IgnoreCase',true) ...
                           .*contains({files(:).name}, 'frame', 'IgnoreCase',true);
    if sum(image_file_index)>0
        image_file_index = find(image_file_index);
        image = Tiff([files(image_file_index).folder, '/',files(image_file_index).name]);
        imageData = read(image);
        tracesEvents.frame_image = imageData;
        close(image);
    end
end


