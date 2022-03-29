function create_tracesEvents_struct_JP(mouse_name, varargin)
    w = warning;
    warning('off','all')

    je = inputParser;
    addParameter(je,'folder_data','None',@ischar)
    addParameter(je,'save_data','None',@ischar)
    addParameter(je,'session_names',{'LT5', 'ROT1'},@iscell)
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

    files = dir(folder_data); 
    %infer which protocol to use based on the name of the files inside the
    %folder
    if sum(contains(session_names, 'LT','IgnoreCase',true))*...
            sum(contains(session_names, 'Rot','IgnoreCase',true)) == 1

        session_type = 'LT_ROT';
        fprintf('LT to Rotation pipeline selected.')

    elseif sum(contains(session_names, 'LT','IgnoreCase',true))*...
            sum(contains(session_names, 'ALO','IgnoreCase',true))*...
            sum(contains(session_names, 'RL_EAST','IgnoreCase',true))*...
            sum(contains(session_names, 'RL_WEST','IgnoreCase',true))== 1

        session_type = 'Full_Castle';
        fprintf('Full Castle pipeline selected.')

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
            local_to_global_guide_LT = correspondence_table{correspondence_table{:,3}==0,1:2};
            local_to_global_guide_ROT = correspondence_table{correspondence_table{:,3}==1,1:2};

            %Keep only cells that appear in both sessions
            Lia  = ismember(local_to_global_guide_LT(:,1), local_to_global_guide_ROT(:,1));
            Lib = ismember(local_to_global_guide_ROT(:,1), local_to_global_guide_LT(:,1));
            local_to_global_guide_LT = local_to_global_guide_LT(Lia,:);
            local_to_global_guide_ROT = local_to_global_guide_ROT(Lib,:);
            
            %LT
            tracesEvents = get_tracesEvents(files_LT,local_to_global_guide_LT, 'LT');
            tracesEvents.mouse = mouse_name;
            tracesEvents.session = session_number(1);
            save([save_data, '/',mouse_name,'_LTm_events_s', int2str(session_number(1)), '.mat'], "tracesEvents")
            
            %ROT1
            tracesEvents = get_tracesEvents(files_ROT,local_to_global_guide_ROT, 'ROT');
            tracesEvents.mouse = mouse_name;
            tracesEvents.session = session_number(2);
            save([save_data, '/',mouse_name,'_Rot_events_s', int2str(session_number(2)), '.mat'], "tracesEvents")
        end
    else 
        tracesEvents = get_tracesEvents(files,[], session_names{1});
        tracesEvents.mouse = mouse_name;
        tracesEvents.session = session_number(1);
        save([save_data, '/',mouse_name,'_', session_names{1}, '_events_s', int2str(session_number(1)), '.mat'], "tracesEvents")
    end
    warning(w)
end


function [tracesEvents] = get_tracesEvents(files,local_to_global_guide, condition)
    tracesEvents = struct();
    tracesEvents.test = condition;
    
    %Load raw traces props to get mapping from "local_guide" to "accepted_local_guide"
    raw_file_index = contains({files(:).name}, 'raw', 'IgnoreCase',true) ...
                            .*contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(raw_file_index)<1
        error("Couldn't find a raw file for condition %s. Please check that it " + ...
            "is on the folder with the word 'raw' % ""props"" in the name.", condition)
    elseif sum(raw_file_index)>1
        error("More than one raw files found for the condition %s in the folder.", condition)
    end
    raw_file_index = find(raw_file_index);
    raw_props_table = readtable([files(raw_file_index).folder, '/',files(raw_file_index).name]);
    raw_props_mat = (0:size(raw_props_table,1)-1)';
    accepted_cells = cellfun(@(c) contains(c,'accepted'), raw_props_table{:,2});
    accepted_to_local_guide = [(0:sum(accepted_cells)-1)',raw_props_mat(accepted_cells)];
    tracesEvents.accepted_to_local_guide = accepted_to_local_guide;
    if isempty(local_to_global_guide)
        local_to_global_guide = [accepted_to_local_guide(:,2),accepted_to_local_guide(:,2)];
    end
    tracesEvents.local_to_global_guide = local_to_global_guide;

    %Create acceptd to global guide
    accepted_to_global_guide = local_to_global_guide;
    for cell= 1:size(local_to_global_guide)
        local_num = local_to_global_guide(cell,2);
        accepted_to_global_guide(cell,2) = find(accepted_to_local_guide(:,2)==local_num)-1;
    end
    tracesEvents.accepted_to_global_guide = accepted_to_local_guide;

    %Load traces
    traces_file_index = contains({files(:).name}, 'traces', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(traces_file_index)<1
        traces_file_index = contains({files(:).name}, 'denoised', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
        if sum(traces_file_index)<1
            error("Couldn't find a trace file for condition %s. Please check that it " + ...
                "is on the folder with the word 'traces' in the name.", condition)
        end
    elseif sum(traces_file_index)>1
        error("More than one traces files found for the condition %s in the folder.", condition)
    end
    traces_file_index = find(traces_file_index);
    traces_table = readtable([files(traces_file_index).folder, '/',files(traces_file_index).name]);
    tracesEvents.traces = traces_table{:,accepted_to_global_guide(:,2)+2};

    %Get sampling frequency
    times = traces_table{:,1};
    tracesEvents.sF = 1/median(diff(times));

    %Load spikes
    spikes_file_index = contains({files(:).name}, 'spikes', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    
    spikes_file_index = find(spikes_file_index);
    for file_num= 1:length(spikes_file_index)
        spikes_idx = spikes_file_index(file_num);
        spikes_table = readtable([files(spikes_idx).folder, '/',files(spikes_idx).name]);
        spikes_bi_array = zeros(size(traces_table,1), size(raw_props_table,1));
        spikes_amp_array = zeros(size(traces_table,1), size(raw_props_table,1));
        [~, spikes_index]  = min(abs(spikes_table{:,1}-times'),[],2);
        spikes_neuron = cell2mat(arrayfun(@(spike) str2double(spike{1}(2:end)), spikes_table{:,2}, 'UniformOutput', false));
        spikes_amplitude = spikes_table{:,3};
        linearidx = sub2ind(size(spikes_bi_array), spikes_index(:,1), spikes_neuron(:,1)+1);
        spikes_bi_array(linearidx) = 1;
        spikes_amp_array(linearidx) = spikes_amplitude;
    
        if contains(files(spikes_idx).name, 'SNR3')
            tracesEvents.spikes_SNR3 = spikes_bi_array(:,local_to_global_guide(:,2)+1);
            tracesEvents.spikes_SNR3_amp = spikes_amp_array(:,local_to_global_guide(:,2)+1);
    
        elseif contains(files(spikes_idx).name, 'SNR2')
            tracesEvents.spikes_SNR2 = spikes_bi_array(:,local_to_global_guide(:,2)+1);
            tracesEvents.spikes_SNR2_amp = spikes_amp_array(:,local_to_global_guide(:,2)+1);
           
        elseif contains(files(spikes_idx).name, 'SNR1_5')
            tracesEvents.spikes_SNR1_5 = spikes_bi_array(:,local_to_global_guide(:,2)+1);
            tracesEvents.spikes_SNR1_5_amp = spikes_amp_array(:,local_to_global_guide(:,2)+1);
        else
            tracesEvents.spikes_SNR1 = spikes_bi_array(:,local_to_global_guide(:,2)+1);
            tracesEvents.spikes_SNR1_amp = spikes_amp_array(:,local_to_global_guide(:,2)+1);
        end
    end
    %{
    if sum(spikes_file_index)<1
        error("Couldn't find a spikes file for condition %s. Please check that it " + ...
            "is on the folder with the word 'spikes' in the name.", condition)
    elseif sum(spikes_file_index)>1
        error("More than one spikes files found for the condition %s in the folder.", condition)
    end
    spikes_file_index = find(spikes_file_index);
    spikes_table = readtable([files(spikes_file_index).folder, '/',files(spikes_file_index).name]);
    spikes_bi_array = zeros(size(traces_table,1), size(raw_props_table,1));
    spikes_amp_array = zeros(size(traces_table,1), size(raw_props_table,1));
    [~, spikes_index]  = min(abs(spikes_table{:,1}-times'),[],2);
    spikes_neuron = cell2mat(arrayfun(@(spike) str2double(spike{1}(2:end)), spikes_table{:,2}, 'UniformOutput', false));
    spikes_amplitude = spikes_table{:,3};
    linearidx = sub2ind(size(spikes_bi_array), spikes_index(:,1), spikes_neuron(:,1)+1);
    spikes_bi_array(linearidx) = 1;
    spikes_amp_array(linearidx) = spikes_amplitude;
    tracesEvents.spikes = spikes_bi_array(:,local_to_global_guide(:,2)+1);
    tracesEvents.spikes_amp = spikes_amp_array(:,local_to_global_guide(:,2)+1);
    %}

    %Load events
    events_file_index = contains({files(:).name}, 'events', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(events_file_index)<1
        error("Couldn't find a events file for condition %s. Please check that it " + ...
            "is on the folder with the word 'events' in the name.", condition)
    elseif sum(events_file_index)>1
        error("More than one events files found for the condition %s in the folder.", condition)
    end
    events_file_index = find(events_file_index);
    events_table = readtable([files(events_file_index).folder, '/',files(events_file_index).name]);
    events_array = zeros(size(traces_table,1), size(traces_table,2)-1);

    [~, events_index]  = min(abs(events_table{:,1}-times'),[],2);
    events_neuron = cell2mat(arrayfun(@(event) str2double(event{1}(2:end)), events_table{:,2}, 'UniformOutput', false));
    events_amplitude = events_table{:,3};

    linearidx = sub2ind(size(events_array), events_index(:,1), events_neuron(:,1)+1);
    events_array(linearidx) = events_amplitude;
    tracesEvents.events = events_array(:,accepted_to_global_guide(:,2)+1);

    %Load position
    position_file_index = contains({files(:).name}, 'position', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(position_file_index)<1
        error("Couldn't find a position file for condition %s. Please check that it " + ...
            "is on the folder with the word 'position' in the name.", condition)
    elseif sum(position_file_index)>1
        error("More than one position files found for the condition %s in the folder.", condition)
    end
    position_file_index = find(position_file_index);
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

    %Delete dropped frames
    tracesEvents.time = times;
    if isfield(tracesEvents,'droppedFrames')
        if ~isempty(tracesEvents.droppedFrames)
            tracesEvents.time(tracesEvents.droppedFrames,:) = [];

            tracesEvents.spikes_SNR3(tracesEvents.droppedFrames,:) = [];
            tracesEvents.spikes_SNR3_amp(tracesEvents.droppedFrames,:) = [];
            tracesEvents.spikes_SNR2(tracesEvents.droppedFrames,:) = [];
            tracesEvents.spikes_SNR2_amp(tracesEvents.droppedFrames,:) = [];
            tracesEvents.spikes_SNR1_5(tracesEvents.droppedFrames,:) = [];
            tracesEvents.spikes_SNR1_5_amp(tracesEvents.droppedFrames,:) = [];
            tracesEvents.spikes_SNR1(tracesEvents.droppedFrames,:) = [];
            tracesEvents.spikes_SNR1_amp(tracesEvents.droppedFrames,:) = [];

            tracesEvents.events(tracesEvents.droppedFrames,:) = [];
            tracesEvents.traces(tracesEvents.droppedFrames,:) = [];
            if length(tracesEvents.position)>length(tracesEvents.time)
                tracesEvents.position(tracesEvents.droppedFrames,:) = [];
                tracesEvents.velocity(tracesEvents.droppedFrames,:) = [];
            end
        end

    end
    if sum(isnan(tracesEvents.traces(:,1)))>0
        tracesEvents.time(isnan(tracesEvents.traces(:,1)),:) = [];
        tracesEvents.spikes(isnan(tracesEvents.traces(:,1)),:) = [];
        tracesEvents.spikes_amp(isnan(tracesEvents.traces(:,1)),:) = [];
        tracesEvents.events(isnan(tracesEvents.traces(:,1)),:) = [];
        if length(tracesEvents.position)>length(tracesEvents.time)
            tracesEvents.position(isnan(tracesEvents.traces(:,1)),:) = [];
            tracesEvents.velocity(isnan(tracesEvents.traces(:,1)),:) = [];
            if length(tracesEvents.position)>length(tracesEvents.time)
                tracesEvents.position = tracesEvents.position(1:length(tracesEvents.time),:);
                tracesEvents.velocity = tracesEvents.velocity(1:length(tracesEvents.time),:);
            end
    
        end
        tracesEvents.traces(isnan(tracesEvents.traces(:,1)),:) = [];
    end
    %Check for size consistency across all signals
    if length(unique([size(tracesEvents.position,1),size(tracesEvents.traces,1), ...
            size(tracesEvents.spikes_SNR3,1),size(tracesEvents.events,1)]))>1
        warning("Signals for condition %s do not have the same time duration!", condition)
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


