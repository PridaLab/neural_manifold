function create_tracesEvents_dual_struct_JP(mouse_name, varargin)
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
        if 1==1
            %get guide for first and second session files
            files_LT = files(contains({files(:).name}, session_names{1}, 'IgnoreCase',true));
            files_ROT = files(contains({files(:).name}, session_names{2}, 'IgnoreCase',true));
            %Load correspondance table GREEN
            correspondence_file_index_green = contains({files(:).name}, 'correspondence', 'IgnoreCase',true)...
                            .*contains({files(:).name}, 'green', 'IgnoreCase',true);
            if ~any(correspondence_file_index_green)
                error("Couldn't find correspondance file. Please check " + ...
                "that it is on the folder with the word 'correspondence' in the name.")            
            elseif sum(correspondence_file_index_green)>1
                error("More than one correspondence file found in the folder.")
            end

            correspondence_file_index_green = find(correspondence_file_index_green);
            correspondence_table_green = readtable([files(correspondence_file_index_green).folder, ...
                                '/',files(correspondence_file_index_green).name]);
            %Get local_to_global_guide for LT and ROT
            accepted_to_global_guide_LT_green = correspondence_table_green{correspondence_table_green{:,3}==0,1:2};
            accepted_to_global_guide_ROT_green = correspondence_table_green{correspondence_table_green{:,3}==1,1:2};

            %Keep only cells that appear in both sessions
            Lia  = ismember(accepted_to_global_guide_LT_green(:,1), accepted_to_global_guide_ROT_green(:,1));
            Lib = ismember(accepted_to_global_guide_ROT_green(:,1), accepted_to_global_guide_LT_green(:,1));
            accepted_to_global_guide_LT_green = accepted_to_global_guide_LT_green(Lia,:);
            accepted_to_global_guide_ROT_green = accepted_to_global_guide_ROT_green(Lib,:);
            double_fprintf(fileID,"Green: Global cells: %i\n%s accepted cells: %i\n%s accepted cells: %i\n",sum(Lia), ...
                                            session_names{1},size(Lia,1),session_names{2}, size(Lib,1))

            %Load correspondance table red
            correspondence_file_index_red = contains({files(:).name}, 'correspondence', 'IgnoreCase',true)...
                            .*contains({files(:).name}, 'red', 'IgnoreCase',true);
            if ~any(correspondence_file_index_red)
                error("Couldn't find correspondance file. Please check " + ...
                "that it is on the folder with the word 'correspondence' in the name.")            
            elseif sum(correspondence_file_index_red)>1
                error("More than one correspondence file found in the folder.")
            end
            correspondence_file_index_red = find(correspondence_file_index_red);
            correspondence_table_red = readtable([files(correspondence_file_index_red).folder, ...
                                '/',files(correspondence_file_index_red).name]);
            %Get local_to_global_guide for LT and ROT
            accepted_to_global_guide_LT_red = correspondence_table_red{correspondence_table_red{:,3}==0,1:2};
            accepted_to_global_guide_ROT_red = correspondence_table_red{correspondence_table_red{:,3}==1,1:2};

            %Keep only cells that appear in both sessions
            Lia  = ismember(accepted_to_global_guide_LT_red(:,1), accepted_to_global_guide_ROT_red(:,1));
            Lib = ismember(accepted_to_global_guide_ROT_red(:,1), accepted_to_global_guide_LT_red(:,1));
            accepted_to_global_guide_LT_red = accepted_to_global_guide_LT_red(Lia,:);
            accepted_to_global_guide_ROT_red = accepted_to_global_guide_ROT_red(Lib,:);
            double_fprintf(fileID,"Red: Global cells: %i\n%s accepted cells: %i\n%s accepted cells: %i\n",sum(Lia), ...
                                            session_names{1},size(Lia,1),session_names{2}, size(Lib,1))

            %LT
            double_fprintf(fileID,'\n-%s:\n',session_names{1})
            tracesEvents = get_tracesEvents(files_LT,accepted_to_global_guide_LT_green, ...
                accepted_to_global_guide_LT_red, session_names{1}, fileID);
            tracesEvents.mouse = mouse_name;
            tracesEvents.session = session_number(1);
            save([save_data, '/',mouse_name,'_', session_names{1}, '_events_s', int2str(session_number(1)), '.mat'], "tracesEvents")
      
            %ROT
            double_fprintf(fileID,'\n-%s:\n',session_names{2})
            tracesEvents = get_tracesEvents(files_ROT,accepted_to_global_guide_ROT_green, ...
                accepted_to_global_guide_ROT_red, session_names{2}, fileID);
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


function [tracesEvents] = get_tracesEvents(files,accepted_to_global_guide_green, accepted_to_global_guide_red, condition, fileID)
    tracesEvents = struct();
    tracesEvents.test = condition;
    %GREEN: Load raw traces props to get mapping from "accepted" to "local" cells
    raw_file_index = contains({files(:).name}, 'green_raw', 'IgnoreCase',true) ...
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
    double_fprintf(fileID,"\tGreen: Detected cells: %i\n\tAccepted cells: %i\n", size(raw_props_mat,1), sum(accepted_cells))

    accepted_to_local_guide_green = [(0:sum(accepted_cells)-1)',raw_props_mat(accepted_cells)]; 
    if isempty(accepted_to_global_guide_green)
        accepted_to_global_guide_green = [accepted_to_local_guide_green(:,1),accepted_to_local_guide_green(:,1)];
    end
    gla_guide_green = array2table(accepted_to_global_guide_green(:,1));
    gla_guide_green.Properties.VariableNames{1} = 'global';
    gla_guide_green(:,2) = array2table(accepted_to_global_guide_green(:,2));
    gla_guide_green.Properties.VariableNames{2} = 'local_accepted';
    gla_guide_green(:,3) = array2table(accepted_to_global_guide_green(:,2)*0);
    gla_guide_green.Properties.VariableNames{3} = 'local_all';
    
    %add local_all guide 
    for cell= 1:size(gla_guide_green)
        accepted_num = gla_guide_green{cell,2};
        local_idx = accepted_to_local_guide_green(:,1)==accepted_num;
        gla_guide_green(cell,3) = array2table(accepted_to_local_guide_green(local_idx,2));
    end
    tracesEvents.gla_guide_green = gla_guide_green;

    %RED: Load raw traces props to get mapping from "accepted" to "local" cells
    raw_file_index = contains({files(:).name}, 'red_raw', 'IgnoreCase',true) ...
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
    double_fprintf(fileID,"\tRed: Detected cells: %i\n\tAccepted cells: %i\n", size(raw_props_mat,1), sum(accepted_cells))

    accepted_to_local_guide_red = [(0:sum(accepted_cells)-1)',raw_props_mat(accepted_cells)]; 
    if isempty(accepted_to_global_guide_red)
        accepted_to_global_guide_red = [accepted_to_local_guide_red(:,1),accepted_to_local_guide_red(:,1)];
    end
    gla_guide_red = array2table(accepted_to_global_guide_red(:,1));
    gla_guide_red.Properties.VariableNames{1} = 'global';
    gla_guide_red(:,2) = array2table(accepted_to_global_guide_red(:,2));
    gla_guide_red.Properties.VariableNames{2} = 'local_accepted';
    gla_guide_red(:,3) = array2table(accepted_to_global_guide_red(:,2)*0);
    gla_guide_red.Properties.VariableNames{3} = 'local_all';
    
    %add local_all guide 
    for cell= 1:size(gla_guide_red)
        accepted_num = gla_guide_red{cell,2};
        local_idx = accepted_to_local_guide_red(:,1)==accepted_num;
        gla_guide_red(cell,3) = array2table(accepted_to_local_guide_red(local_idx,2));
    end
    tracesEvents.gla_guide_red = gla_guide_red;

      %if correspondesdonce table has been done with CNMFE then use this
      %section
%     gla_guide_green = array2table(accepted_to_global_guide(:,1));
%     gla_guide_green.Properties.VariableNames{1} = 'global';
%     gla_guide_green(:,2) = array2table(accepted_to_global_guide(:,2)*0);
%     gla_guide_green.Properties.VariableNames{2} = 'local_accepted';
%     gla_guide_green(:,3) = array2table(accepted_to_global_guide(:,2));
%     gla_guide_green.Properties.VariableNames{3} = 'local_all';
%     %add local_all guide 
%     for cell= 1:size(gla_guide_green)
%         accepted_num = gla_guide_green{cell,3};
%         local_idx = accepted_to_local_guide(:,2)==accepted_num;
%         gla_guide_green(cell,2) = array2table(accepted_to_local_guide(local_idx,1));
%     end
%     tracesEvents.gla_guide_green = gla_guide_green;

    %GREEN: Load raw traces (local guide)
    double_fprintf(fileID,'\tLoading green raw traces...')
    raw_file_index = contains({files(:).name}, 'green_raw', 'IgnoreCase',true) ...
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
    green_raw_table = raw_table{:, tracesEvents.gla_guide_green{:,3}+2}; %+2: +1 because first column is time, +1 because cells in guide start in 0

    double_fprintf(fileID,' Done\n')
    %RED: Load raw traces (local guide)
    double_fprintf(fileID,'\tLoading red raw traces...')
    raw_file_index = contains({files(:).name}, 'red_raw', 'IgnoreCase',true) ...
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
    red_raw_table = raw_table{:, tracesEvents.gla_guide_red{:,3}+2}; %+2: +1 because first column is time, +1 because cells in guide start in 0
    double_fprintf(fileID,' Done\n')
    if size(green_raw_table,1)>size(red_raw_table,1)
        green_raw_table = green_raw_table(1:size(red_raw_table,1),:);
    elseif size(red_raw_table,1)>size(green_raw_table,1)
        red_raw_table = red_raw_table(1:size(green_raw_table,1),:);
    end
    tracesEvents.green_raw_traces = green_raw_table;
    tracesEvents.red_raw_traces = red_raw_table;
    % double_fprintf(fileID,'\tInterpolating raw traces...')
    % xq = 1:max([size(green_raw_table,1),size(red_raw_table,1)])*2;
    % %interpolate green:
    % x = 1:2:size(green_raw_table,1)*2;
    % sq_green = interp1(x,green_raw_table,xq, 'linear');
    % tracesEvents.green_raw_traces = sq_green;
    % %interpolate red:
    % x = 1:2:size(red_raw_table,1)*2;
    % sq_red = interp1(x,red_raw_table,xq, 'linear');
    % tracesEvents.red_raw_traces = sq_red;
    % double_fprintf(fileID,'\b\b\b: Done\n')

    %GREEN: Load denoised traces (accepted guide)
    double_fprintf(fileID,'\tLoading green denoised traces... ')
    traces_file_index = contains({files(:).name}, 'green_traces', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(traces_file_index)<1
        traces_file_index = contains({files(:).name}, 'green_denoised', 'IgnoreCase',true) ...
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
    green_denoised_traces = traces_table{:,tracesEvents.gla_guide_green{:,2}+2}; %+2: +1 because first column is time, +1 because cells in guide start in 0
    double_fprintf(fileID,' Done\n')
    %GREEN: Load denoised traces (accepted guide)
    double_fprintf(fileID,'\tLoading red denoised traces... ')
    traces_file_index = contains({files(:).name}, 'red_traces', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
    if sum(traces_file_index)<1
        traces_file_index = contains({files(:).name}, 'red_denoised', 'IgnoreCase',true) ...
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
    red_denoised_traces = traces_table{:,tracesEvents.gla_guide_red{:,2}+2}; %+2: +1 because first column is time, +1 because cells in guide start in 0
    double_fprintf(fileID,' Done\n')

    if size(green_denoised_traces,1)>size(red_denoised_traces,1)
        green_denoised_traces = green_denoised_traces(1:size(red_denoised_traces,1),:);
    elseif size(red_denoised_traces,1)>size(green_denoised_traces,1)
        red_denoised_traces = red_denoised_traces(1:size(green_denoised_traces,1),:);
    end
    tracesEvents.green_denoised_traces = green_denoised_traces;
    tracesEvents.red_denoised_traces = red_denoised_traces;
    % 
    % double_fprintf(fileID,'\tInterpolating denoised traces...')
    % xq = 1:max([size(green_denoised_traces,1),size(red_denoised_traces,1)])*2;
    % %interpolate green:
    % x = 1:2:size(green_denoised_traces,1)*2;
    % sq_green = interp1(x,green_denoised_traces,xq, 'linear');
    % tracesEvents.green_denoised_traces = sq_green;
    % %interpolate red:
    % x = 1:2:size(red_denoised_traces,1)*2;
    % sq_red = interp1(x,red_denoised_traces,xq, 'linear');
    % tracesEvents.red_denoised_traces = sq_red;
    % double_fprintf(fileID,'\b\b\b: Done\n')
    %Get sampling frequency
    tracesEvents.sF = 10;

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
        tracesEvents.position = tracesEvents.position(1:2:end,:);
    else
        temp = load([files(position_file_index).folder, '/',files(position_file_index).name]);
        field_temp = fieldnames(temp);
        for field = 1:size(field_temp,1)
            if contains(field_temp{field,1}, 'position', 'IgnoreCase', true)
                tracesEvents.position = temp.(field_temp{field,1});
                tracesEvents.position = tracesEvents.position(1:2:end,:);
            elseif contains(field_temp{field,1}, 'droppedFrames', 'IgnoreCase', true)
                tracesEvents.droppedFrames = temp.(field_temp{field,1})+1;
                tracesEvents.droppedFrames = round(tracesEvents.droppedFrames/2);
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
    tracesEvents.velocity = sqrt(diff(tracesEvents.position(:,1)).^2+ diff(tracesEvents.position(:,2)).^2)/(1/tracesEvents.sF); 
    tracesEvents.velocity = [tracesEvents.velocity(1,1); tracesEvents.velocity];
    double_fprintf(fileID,'Done\n')
    double_fprintf(fileID,'\t\tOriginal shapes: Inscopix %i - Position %i\n', size(tracesEvents.green_raw_traces,1),size(tracesEvents.position,1))
    %Delete dropped frames
    times = (0:size(tracesEvents.green_raw_traces,1)-1)'/tracesEvents.sF;
    tracesEvents.time = times;
    if isfield(tracesEvents,'droppedFrames')
        if ~isempty(tracesEvents.droppedFrames)
            if (size(tracesEvents.position,1)+length(tracesEvents.droppedFrames)) == size(tracesEvents.green_denoised_traces,1)
                double_fprintf(fileID,'\tInterpolating dropped frames on position (%i)...', length(tracesEvents.green_droppedFrames))

                og_frames = (1:size(tracesEvents.denoised_traces,1))';
                pos_frames = og_frames;
                pos_frames(tracesEvents.green_droppedFrames) = [];
                pos_query = interp1(pos_frames, tracesEvents.position, tracesEvents.green_droppedFrames);
                new_pos = zeros(size(tracesEvents.green_denoised_traces,1),2);
                new_pos(pos_frames,:) = tracesEvents.position;
                new_pos(tracesEvents.droppedFrames,:) = pos_query;
                tracesEvents.position = new_pos;
                tracesEvents.velocity = sqrt(diff(tracesEvents.position(:,1)).^2+ diff(tracesEvents.position(:,2)).^2)/(1/tracesEvents.sF); 
                tracesEvents.velocity = [tracesEvents.velocity(1,1); tracesEvents.velocity];
                double_fprintf(fileID,' Done\n')
            else
                double_fprintf(fileID,'\tRemoving dropped frames (%i)...', length(tracesEvents.green_droppedFrames))
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
    if size(tracesEvents.position,1)<size(tracesEvents.green_denoised_traces,1)
        end_idx = size(tracesEvents.position,1)+1;
        double_fprintf(fileID,'\tPosition still %i frames sorter than Inscopix signals. Dropping final edges of Inscopix...', ...
                                                size(tracesEvents.green_denoised_traces,1)-size(tracesEvents.position,1));
        tracesEvents.time(end_idx:end,:) = [];
        tracesEvents.green_raw_traces(end_idx:end,:) = [];
        tracesEvents.red_raw_traces(end_idx:end,:) = [];


        tracesEvents.green_denoised_traces(end_idx:end,:) = [];
        tracesEvents.red_denoised_traces(end_idx:end,:) = [];

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
        double_fprintf(fileID,'\t\tResulting shapes: Inscopix %i - Position %i\n', size(tracesEvents.green_raw_traces,1),size(tracesEvents.position,1))
    end

    if size(tracesEvents.position,1)>size(tracesEvents.green_raw_traces,1)
        double_fprintf(fileID,'\tPosition still %i frames longer than Inscopix signals. Dropping final edges of Inscopix...', ...
                                                size(tracesEvents.position,1)-size(tracesEvents.green_raw_traces,1));
        if size(tracesEvents.position,1)-size(tracesEvents.green_raw_traces,1) > 10
            warning('More than 10 frames more on position than on traces')
        end
        end_idx = size(tracesEvents.green_raw_traces,1)+1;
        tracesEvents.position(end_idx:end,:) = [];
        tracesEvents.velocity(end_idx:end,:) = [];
        double_fprintf(fileID,' Done\n')

    end
    %CHECK NANS IN TRACES
    if sum(isnan(tracesEvents.green_raw_traces(:,1)+tracesEvents.red_denoised_traces(:,1)))>0
        double_fprintf(fileID,'\tRemoving nans found on traces on all signals...')
        nan_idx = isnan(tracesEvents.green_denoised_traces(:,1)+tracesEvents.red_denoised_traces(:,1));
        nan_num = sum(nan_idx);
        double_fprintf(fileID,'\b\b\b: %i entries compromised',sum(nan_num))
        tracesEvents.time(nan_idx,:) = [];
        tracesEvents.green_raw_traces(nan_idx,:) = [];
        tracesEvents.red_raw_traces(nan_idx,:) = [];

        tracesEvents.green_denoised_traces(nan_idx,:) = [];
        tracesEvents.red_denoised_traces(nan_idx,:) = [];

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
        double_fprintf(fileID,'\t\tResulting shapes: Inscopix %i - Position %i\n', size(tracesEvents.red_denoised_traces,1),size(tracesEvents.position,1))
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


