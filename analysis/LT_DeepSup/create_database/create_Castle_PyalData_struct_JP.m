%% RESTART
close all; clear; clc;
%% FILE
data_file =  'GC2_alo_events_s3.mat';

% Load file
load(data_file);
%% DEFINE MAZE SECTIONS
fh1 = figure('units','normalized','outerposition',[0 0 1 1]);
imshow(tracesEvents.frame)
hold on;
frame_position = tracesEvents.position/tracesEvents.pixel_scale;
plot(frame_position(:,1), frame_position(:,2),'Color', [.5,.5,.5], 'LineWidth', 2);

%Draw house polygons and center
H0_default_position = [623.3657 7.4446; 623.3657 157.0213; 725.6 157.0213;
  725.6 186.0819; 836.2 186.0819; 836.2 156.1666; 955.8533 156.1666;
  955.8533 7.4446];

H0 = drawpolygon(gca, 'Position', int32(H0_default_position), 'Label', 'H0', ...
    'LabelVisible', 'hover');

H3_default_position = [1.1658 0.4914; 1.1658 0.5967; 1.2047 0.5967; 1.2047 0.7039; 
        1.3716 0.7039; 1.3716 0.3794; 1.2047 0.3794; 1.2047 0.4914]*1e3;
H3 = drawpolygon(gca, 'Position', int32(H3_default_position), 'Label', 'H3', ...
    'LabelVisible', 'hover');

H6_default_position = [0.8362 0.9196; 0.7256 0.9196; 0.7256 0.9572; 0.6275 0.9572;
    0.6275 1.0543; 0.9092 1.0543; 0.9092 0.9572; 0.8362 0.9572]*1e3;
H6 = drawpolygon(gca, 'Position', int32(H6_default_position), 'Label', 'H6', ...
    'LabelVisible', 'hover');

H9_default_position = [421.5728 491.4; 376.6155 491.4; 376.6155 382.2340; 
  215.9933 382.2340; 215.9933 702.6041; 376.6155 702.6041; 376.6155 596.6697;
  421.5728 596.7];
H9 = drawpolygon(gca, 'Position', int32(H9_default_position), 'Label', 'H9', ...
    'LabelVisible', 'hover');

[~,H0_idx] = sort(H0.Position(:,2)); %bottom 2 corners
H0_idx = H0_idx(end-1:end);
[~,order] = sort(H0.Position(H0_idx,1)); %order them from left to right
H0_idx = H0_idx(order); %invert

[~,H3_idx] = sort(H3.Position(:,1)); %left 2 corners
H3_idx = H3_idx(1:2);
[~,order] = sort(H3.Position(H3_idx,2), 'descend'); %order them from bot to top
H3_idx = H3_idx(order); 

[~,H6_idx] = sort(H6.Position(:,2)); %top 2 corners
H6_idx = H6_idx(1:2);
[~,order] = sort(H6.Position(H6_idx,1)); %order them from left to right
H6_idx = H6_idx(order);

[~,H9_idx] = sort(H9.Position(:,1), 'descend'); %right 2 corners
H9_idx = H9_idx(1:2);
[~,order] = sort(H9.Position(H9_idx,2)); %order them from top to bot
H9_idx = H9_idx(order); 

CB_default_position = [H0.Position(H0_idx(1),1), H9.Position(H9_idx(1),2);
    H6.Position(H6_idx(1),1), H9.Position(H9_idx(2),2);
    H6.Position(H6_idx(2),1), H3.Position(H3_idx(1),2);
    H0.Position(H0_idx(2),1), H3.Position(H3_idx(2),2)];

CB = drawpolygon(gca, 'Position', int32(CB_default_position), 'Label', 'CB', ...
    'LabelVisible', 'hover');

%TODO: maybe adapt polygon default position to center of mass when animal
%is still (probably drinking on the home sections.
title("Press enter to continue")
pause;

%Infer A0 arm
[~,CB_idx] = sort(CB.Position(:,2)); %top 2 corners
CB_idx = CB_idx(1:2);
[~,order] = sort(CB.Position(CB_idx,1)); %order them from left to right
CB_idx = CB_idx(order);

[~,H0_idx] = sort(H0.Position(:,2)); %bottom 2 corners
H0_idx = H0_idx(end-1:end);
[~,order] = sort(H0.Position(H0_idx,1), 'descend'); %order them from right to left
H0_idx = H0_idx(order); %invert 

A0_position = [CB.Position(CB_idx,:); H0.Position(H0_idx,:)];
A0 = drawpolygon(gca, 'Position', int32(A0_position), 'Label', 'A0', ...
    'LabelVisible', 'hover');

%Infer A3 arm
[~,CB_idx] = sort(CB.Position(:,1)); %right 2 corners
CB_idx = CB_idx(end-1:end);
[~,order] = sort(CB.Position(CB_idx,2)); %order them from top to bot
CB_idx = CB_idx(order);

[~,H3_idx] = sort(H3.Position(:,1)); %left 2 corners
H3_idx = H3_idx(1:2);
[~,order] = sort(H3.Position(H3_idx,2), 'descend'); %order them from bot to top
H3_idx = H3_idx(order); 

A3_position = [CB.Position(CB_idx,:); H3.Position(H3_idx,:)];
A3 = drawpolygon(gca, 'Position', int32(A3_position), 'Label', 'A3', ...
    'LabelVisible', 'hover');

%Infer A6 arm
[~,CB_idx] = sort(CB.Position(:,2)); %bottom 2 corners
CB_idx = CB_idx(end-1:end);
[~,order] = sort(CB.Position(CB_idx,1), 'descend'); %order them from right to left
CB_idx = CB_idx(order);

[~,H6_idx] = sort(H6.Position(:,2)); %top 2 corners
H6_idx = H6_idx(1:2);
[~,order] = sort(H6.Position(H6_idx,1)); %order them from left to right
H6_idx = H6_idx(order);

A6_position = [CB.Position(CB_idx,:); H6.Position(H6_idx,:)];
A6 = drawpolygon(gca, 'Position', int32(A6_position), 'Label', 'A6', ...
    'LabelVisible', 'hover');

%Infer A9 arm
[~,CB_idx] = sort(CB.Position(:,1)); %left 2 corners
CB_idx = CB_idx(1:2);
[~,order] = sort(CB.Position(CB_idx,2), 'descend'); %order them from bot to top
CB_idx = CB_idx(order);

[~,H9_idx] = sort(H9.Position(:,1), 'descend'); %right 2 corners
H9_idx = H9_idx(1:2);
[~,order] = sort(H9.Position(H9_idx,2)); %order them from top to bot
H9_idx = H9_idx(order); 

A9_position = [H9.Position(H9_idx,:);CB.Position(CB_idx,:)];
A9 = drawpolygon(gca, 'Position', int32(A9_position), 'Label', 'A9', ...
    'LabelVisible', 'hover');

% Infer R9_0 arm
[~,H0_idx] = sort(H0.Position(:,1)); %left points
H0_idx = H0_idx(1:2);
[~,order] = sort(H0.Position(H0_idx,2)); %order them from top to bot
H0_idx = H0_idx(order);

[~,H9_idx] = sort(H9.Position(:,2)); %top points
H9_idx = H9_idx(1:2);
[~,order] = sort(H9.Position(H9_idx,1)); %order them left to right
H9_idx = H9_idx(order);

R9_0_position = [H9.Position(H9_idx(1),1), H0.Position(H0_idx(1),2);
            H0.Position(H0_idx,:);
            H0.Position(H0_idx(2),1), H9.Position(H9_idx(2),2);
            H9.Position(H9_idx(end:-1:1),:)];
                  
R9_0 = drawpolygon(gca, 'Position', int32(R9_0_position), 'Label', 'R9_0', ...
    'LabelVisible', 'hover');

% Infer R0_3 arm
[~,H0_idx] = sort(H0.Position(:,1), 'descend'); %right points
H0_idx = H0_idx(1:2);
[~,order] = sort(H0.Position(H0_idx,2)); %order them from top to bot
H0_idx = H0_idx(order);

[~,H3_idx] = sort(H3.Position(:,2)); %top points
H3_idx = H3_idx(1:2);
[~,order] = sort(H3.Position(H3_idx,1)); %order them left to right
H3_idx = H3_idx(order);

R0_3_position = [H0.Position(H0_idx,:);
            H0.Position(H0_idx(2),1), H3.Position(H3_idx(1),2);
            H3.Position(H3_idx,:);
            H3.Position(H3_idx(2),1), H0.Position(H0_idx(1),2)];
                  
R0_3 = drawpolygon(gca, 'Position', int32(R0_3_position), 'Label', 'R0_3', ...
    'LabelVisible', 'hover');

% Infer R3_6 arm
[~,H3_idx] = sort(H3.Position(:,2), 'descend'); %bot points
H3_idx = H3_idx(1:2);
[~,order] = sort(H3.Position(H3_idx,1), 'descend'); %order them right to left
H3_idx = H3_idx(order);

[~,H6_idx] = sort(H6.Position(:,1), 'descend'); %right points
H6_idx = H6_idx(1:2);
[~,order] = sort(H6.Position(H6_idx,2)); %order them from top to bot
H6_idx = H6_idx(order);

R3_6_position = [H3.Position(H3_idx,:);
            H6.Position(H6_idx(1),1), H3.Position(H3_idx(2),2);
            H6.Position(H6_idx,:);
            H3.Position(H3_idx(1),1), H6.Position(H6_idx(2),2)];
                  
R3_6 = drawpolygon(gca, 'Position', int32(R3_6_position), 'Label', 'R3_6', ...
    'LabelVisible', 'hover');

% Infer R6_9 arm
[~,H9_idx] = sort(H9.Position(:,2), 'descend'); %bot points
H9_idx = H9_idx(1:2);
[~,order] = sort(H9.Position(H9_idx,1), 'descend'); %order them right to left
H9_idx = H9_idx(order);

[~,H6_idx] = sort(H6.Position(:,1)); %left points
H6_idx = H6_idx(1:2);
[~,order] = sort(H6.Position(H6_idx,2), 'descend'); %order them from bot to top
H6_idx = H6_idx(order);

R6_9_position = [H9.Position(H9_idx,:);
            H9.Position(H9_idx(2),1), H6.Position(H6_idx(1),2);
            H6.Position(H6_idx,:);
            H6.Position(H6_idx(2),1), H9.Position(H9_idx(1),2);];
                  
R6_9 = drawpolygon(gca, 'Position', int32(R6_9_position), 'Label', 'R6_9', ...
    'LabelVisible', 'hover');

%% SEE IN WHICH SECTION IS THE ANIMAL AT EACH TIMESTAMP 
%order H0,H3,H6,H9,CB,A0,A3,A6,A9,R0_3,R3_6,R6_9,R9_0
section_legend = {'H0','H3','H6', 'H9','CB','A0','A3','A6','A9','R0_3', ...
                                                'R3_6', 'R6_9', 'R9_0'};
sections.H0 = H0.Position;
sections.H3 = H3.Position;
sections.H6 = H6.Position;
sections.H9 = H9.Position;
sections.CB = CB.Position;  

sections.A0 = A0.Position;
sections.A3 = A3.Position;
sections.A6 = A6.Position;
sections.A9 = A9.Position;

sections.R0_3 = R0_3.Position;
sections.R3_6 = R3_6.Position;
sections.R6_9 = R6_9.Position;
sections.R9_0 = R9_0.Position;

pos_sec = zeros(size(frame_position,1), 13);
section_names = fieldnames(sections);
for sec_idx= 1:size(section_names,1)
    sec_name = section_names{sec_idx,1};
    pos_sec(:,sec_idx) = inpolygon(frame_position(:,1), frame_position(:,2), ...
                        sections.(sec_name)(:,1),sections.(sec_name)(:,2));
end

%check points that are not in any of the sections by asigning them the
%closest one
non_labeled = sum(pos_sec,2)==0;
non_labeled_idx = find(non_labeled==true);
if sum(non_labeled)>0

    dist_sec =  zeros(sum(non_labeled), 13);
    non_pos = frame_position(non_labeled,:);
    figure(fh1)
    scatter(non_pos(:,1), non_pos(:,2), 'r', 'filled')
    for sec_idx= 1:size(section_names,1)
        sec_name = section_names{sec_idx,1};
        dist_sec(:,sec_idx) = p_poly_dist(non_pos(:,1)', non_pos(:,2)', ...
                            sections.(sec_name)(:,1),sections.(sec_name)(:,2));
    end
    [~,sec_idx] = min(dist_sec, [], 2);
    for point= 1:size(non_pos,1)
        pos_sec(non_labeled_idx(point),sec_idx(point)) = 1;
    end
end
%Reformat way of saving area from integer to the actual name of the section
[~,pos_sec] = max(pos_sec, [], 2);
pos_sec_name = cell(size(pos_sec,1),1);
for point = 1:size(pos_sec,1)
    pos_sec_name{point,1} = section_legend{1,pos_sec(point)};
end
%% DIVIDE INTO TRIALS
trial_type = cell(size(pos_sec,1),1); 
trial_phase = cell(size(pos_sec,1),1); 

%find return phases
return_arms = find(contains(pos_sec_name, 'R'));
end_return = find(diff(return_arms)>40); %at least 2 seconds between
return_intervals = [[return_arms(1); return_arms(end_return+1)], ...
    [return_arms(end_return); return_arms(end)]];

%start dividing into trials
    %first see the direction of trial-0
trial_0 = find(contains(pos_sec_name(1:return_intervals(1,1)-1), 'H'));
if isempty(trial_0)
    trial_0 = find(contains(pos_sec_name(1:return_intervals(2,1)-1), 'H'));
end
if contains(pos_sec_name{trial_0(1),1}, 'H3')
    last_dir = 'E';
elseif contains(pos_sec_name{trial_0(1),1}, 'H9')
    last_dir = 'W';
end
for point = 1:return_intervals(1,1)-1
    trial_type{point,1} = 'trial_0';
    trial_phase{point,1} = 'trial_0';
end
for return_idx = 1:size(return_intervals,1)

    for point = return_intervals(return_idx,1):return_intervals(return_idx,2)
        trial_type{point,1} = 'R';
        trial_phase{point,1} = 'R';
    end

    st_trial = return_intervals(return_idx,2)+1;
    if return_idx<size(return_intervals,1)
        en_trial = return_intervals(return_idx+1,1)-1;
    else
        en_trial = size(pos_sec_name,1);
    end
    trial = find(~contains(pos_sec_name(st_trial:en_trial), 'H')); 
    if isempty(trial)
        trial = en_trial+1;
    end
    en_house = pos_sec_name{en_trial,1};
    if contains(last_dir, 'E')
        if contains(en_house, 'H9')
            result = 'W';
            last_dir = 'W';
        else
            result = 'FW';
            last_dir = 'E';
        end
    elseif contains(last_dir, 'W')
        if contains(en_house, 'H3')
            result = 'E';
            last_dir = 'E';
        else
            result = 'FE';
            last_dir = 'Wen';
        end
    end

    for point = st_trial:en_trial
        trial_type{point,1} = result;
        if point-st_trial+1<trial(1)
            trial_phase{point,1} = 'pre_trial';
        elseif point-st_trial+1<trial(end)
            trial_phase{point,1} = 'trial';
        else
            trial_phase{point,1} = 'post_trial';
        end
    end        
end

%% CREATE TRIALDATA STRUCTURE
trial_data(1).mouse = tracesEvents.mouse;
trial_data(1).date = date;
trial_data(1).task = 'Castle-Allocentric';
trial_data(1).trial_id = 1;

trial_data(1).trial_type = 'trial_0';
trial_data(1).trial_type_mat = trial_type(1:return_intervals(1,1)-1,1);
trial_data(1).trial_phase = trial_phase(1:return_intervals(1,1)-1,1);
trial_data(1).pos_sec = pos_sec_name(1:return_intervals(1,1)-1,1);

trial_data(1).Fs =tracesEvents.sF;
trial_data(1).bin_size = 1/trial_data(1).Fs;
trial_data(1).idx_trial_start = 1;
trial_data(1).idx_trial_end = return_intervals(1,2)-1;
trial_data(1).pos = tracesEvents.position(1:return_intervals(1,1)-1,:);
trial_data(1).vel = tracesEvents.velocity(1:return_intervals(1,1)-1,:);
trial_data(1).raw_traces = tracesEvents.raw_traces(1:return_intervals(1,1)-1,:);
trial_data(1).denoised_traces = tracesEvents.denoised_traces(1:return_intervals(1,1)-1,:);

fields = fieldnames(tracesEvents);
spike_fields = fields(contains(fields,'spikes_'));
for idx = 1:length(spike_fields)
    eval(strcat('trial_data(1).',spike_fields{idx},' = tracesEvents.', spike_fields{idx}, '(1:return_intervals(1,1)-1,:);'))
end
events_fields = fields(contains(fields,'events_'));
for idx = 1:length(events_fields)
    eval(strcat('trial_data(1).',events_fields{idx},' = tracesEvents.', events_fields{idx}, '(1:return_intervals(1,1)-1,:);'))
end
trial_data(1).cell_idx = 1:size(tracesEvents.denoised_traces,2);


ii = 1;
for return_idx = 1:size(return_intervals,1)
    ii = ii +1;
    st_trial = return_intervals(return_idx,1);
    en_trial = return_intervals(return_idx,2);

    trial_data(ii).mouse = tracesEvents.mouse;
    trial_data(ii).date = date;
    trial_data(ii).task = 'Castle-Allocentric';
    trial_data(ii).trial_id = ii;
    
    trial_data(ii).trial_type = trial_type{st_trial,1};

    trial_data(ii).trial_type_mat = trial_type(st_trial:en_trial,1);
    trial_data(ii).trial_phase = trial_phase(st_trial:en_trial,1);
    trial_data(ii).pos_sec = pos_sec_name(st_trial:en_trial,1);
    
    trial_data(ii).Fs =tracesEvents.sF;
    trial_data(ii).bin_size = 1/trial_data(1).Fs;
    trial_data(ii).idx_trial_start = st_trial;
    trial_data(ii).idx_trial_end = en_trial;
    trial_data(ii).pos = tracesEvents.position(st_trial:en_trial,:);
    trial_data(ii).vel = tracesEvents.velocity(st_trial:en_trial,:);
    trial_data(ii).raw_traces = tracesEvents.raw_traces(st_trial:en_trial,:);
    trial_data(ii).denoised_traces = tracesEvents.denoised_traces(st_trial:en_trial,:);

    fields = fieldnames(tracesEvents);
    spike_fields = fields(contains(fields,'spikes_'));
    for idx = 1:length(spike_fields)
        eval(strcat('trial_data(ii).',spike_fields{idx},' = tracesEvents.', spike_fields{idx}, '(st_trial:en_trial,:);'))
    end
    events_fields = fields(contains(fields,'events_'));
    for idx = 1:length(events_fields)
        eval(strcat('trial_data(ii).',events_fields{idx},' = tracesEvents.', events_fields{idx}, '(st_trial:en_trial,:);'))
    end
    trial_data(ii).cell_idx = 1:size(tracesEvents.denoised_traces,2);

    ii = ii +1;
    st_trial = return_intervals(return_idx,2)+1;
    if return_idx<size(return_intervals,1)
        en_trial = return_intervals(return_idx+1,1)-1;
    else
        en_trial = size(pos_sec_name,1);
    end

    trial_data(ii).mouse = tracesEvents.mouse;
    trial_data(ii).date = date;
    trial_data(ii).task = 'Castle-Allocentric';
    trial_data(ii).trial_id = ii;
    
    trial_data(ii).trial_type = trial_type{st_trial,1};

    trial_data(ii).trial_type_mat = trial_type(st_trial:en_trial,1);
    trial_data(ii).trial_phase = trial_phase(st_trial:en_trial,1);
    trial_data(ii).pos_sec = pos_sec_name(st_trial:en_trial,1);
    
    trial_data(ii).Fs =tracesEvents.sF;
    trial_data(ii).bin_size = 1/trial_data(1).Fs;
    trial_data(ii).idx_trial_start = st_trial;
    trial_data(ii).idx_trial_end = en_trial;
    trial_data(ii).pos = tracesEvents.position(st_trial:en_trial,:);
    trial_data(ii).vel = tracesEvents.velocity(st_trial:en_trial,:);
    trial_data(ii).raw_traces = tracesEvents.raw_traces(st_trial:en_trial,:);
    trial_data(ii).denoised_traces = tracesEvents.denoised_traces(st_trial:en_trial,:);

    fields = fieldnames(tracesEvents);
    spike_fields = fields(contains(fields,'spikes_'));
    for idx = 1:length(spike_fields)
        eval(strcat('trial_data(ii).',spike_fields{idx},' = tracesEvents.', spike_fields{idx}, '(st_trial:en_trial,:);'))
    end
    events_fields = fields(contains(fields,'events_'));
    for idx = 1:length(events_fields)
        eval(strcat('trial_data(ii).',events_fields{idx},' = tracesEvents.', events_fields{idx}, '(st_trial:en_trial,:);'))
    end
    trial_data(ii).cell_idx = 1:size(tracesEvents.denoised_traces,2);
end

save([tracesEvents.mouse, '_', tracesEvents.test, '_PyalData_struct.mat'], 'trial_data');

