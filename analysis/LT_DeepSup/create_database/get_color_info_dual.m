folder_data = uigetdir(cd, 'Select folder with mouse data');
files = dir(folder_data); 
mouse = regexp(folder_data,filesep,'split');
mouse = mouse{1,end-1};

%% Load red correspondance table
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
accepted_to_global_guide_LT_red = accepted_to_global_guide_LT_red(Lia,:);

%Load raw traces props to get mapping from "accepted" to "local" cells
raw_file_index = contains({files(:).name}, 'lt_red_raw', 'IgnoreCase',true) ...
                        .*contains({files(:).name}, 'props', 'IgnoreCase',true);
if sum(raw_file_index)<1
    error("Couldn't find a raw file. Please check that it " + ...
        "is on the folder with the word 'raw' & ""props"" in the name.")
elseif sum(raw_file_index)>1
    error("More than one raw files found in the folder.")
end
raw_file_index = find(raw_file_index);
raw_props_table = readtable([files(raw_file_index).folder, '/',files(raw_file_index).name]);
raw_props_mat = (0:size(raw_props_table,1)-1)';

accepted_cells = cellfun(@(c) (contains(c,'accepted') || contains(c, 'undecided')), raw_props_table{:,2});
fprintf("\tRed: Detected cells: %i\n\tAccepted cells: %i\n", size(raw_props_mat,1), sum(accepted_cells))

accepted_to_local_guide_red = [(0:sum(accepted_cells)-1)',raw_props_mat(accepted_cells)]; 
if isempty(accepted_to_global_guide_LT_red)
    accepted_to_global_guide_LT_red = [accepted_to_local_guide_red(:,1),accepted_to_local_guide_red(:,1)];
end
gla_guide_red = array2table(accepted_to_global_guide_LT_red(:,1));
gla_guide_red.Properties.VariableNames{1} = 'global';
gla_guide_red(:,2) = array2table(accepted_to_global_guide_LT_red(:,2));
gla_guide_red.Properties.VariableNames{2} = 'local_accepted';
gla_guide_red(:,3) = array2table(accepted_to_global_guide_LT_red(:,2)*0);
gla_guide_red.Properties.VariableNames{3} = 'local_all';

%add local_all guide 
for red_cell= 1:size(gla_guide_red)
    accepted_num = gla_guide_red{red_cell,2};
    local_idx = accepted_to_local_guide_red(:,1)==accepted_num;
    gla_guide_red(red_cell,3) = array2table(accepted_to_local_guide_red(local_idx,2));
end
%% load pre red raw traces
raw_file_index = contains({files(:).name}, 'lt_red_raw', 'IgnoreCase',true) ...
                            .*~contains({files(:).name}, 'props', 'IgnoreCase',true);
if sum(raw_file_index)<1
    error("Couldn't find a raw file. Please check that it " + ...
        "is on the folder with the word 'raw' in the name.")
elseif sum(raw_file_index)>1
    error("More than one raw files found in the folder.")
end
raw_file_index = find(raw_file_index);
fprintf('\b\b\b: %s - ',files(raw_file_index).name)
raw_table = readtable([files(raw_file_index).folder, '/',files(raw_file_index).name]);
registered_raw_table = raw_table(:, gla_guide_red{:,3}+2); %+2: +1 because first column is time, +1 because cells in guide start in 0


%% load color raw traces
color_files = dir(fullfile(folder_data, 'color_registration')); 
%matched
matched_file_index = contains({color_files(:).name}, '_matched_raw', 'IgnoreCase',true) ...
                            .*~contains({color_files(:).name}, 'props', 'IgnoreCase',true);
matched_file_index = find(matched_file_index);
matched_table = readtable([color_files(matched_file_index).folder, '/',color_files(matched_file_index).name]);
matched_table = matched_table(:,2:end);

%unmatched
unmatched_file_index = contains({color_files(:).name}, '_unmatched_raw', 'IgnoreCase',true) ...
                            .*~contains({color_files(:).name}, 'props', 'IgnoreCase',true);
unmatched_file_index = find(unmatched_file_index);
unmatched_table = readtable([color_files(unmatched_file_index).folder, '/',color_files(unmatched_file_index).name]);
unmatched_table = unmatched_table(:,2:end);

%uncertain
uncertain_file_index = contains({color_files(:).name}, '_uncertain_raw', 'IgnoreCase',true) ...
                            .*~contains({color_files(:).name}, 'props', 'IgnoreCase',true);
uncertain_file_index = find(uncertain_file_index);
uncertain_table = readtable([color_files(uncertain_file_index).folder, '/',color_files(uncertain_file_index).name]);
uncertain_table = uncertain_table(:,2:end);

%% get guide between color traces and raw traces

%matched
matched_indexes = nan(1,size(matched_table,2)-1);
cells_to_check = (1:size(registered_raw_table,2));
for cell_matched = 1:size(matched_table,2)
    for cell_red = 1:length(cells_to_check)
        cell_red_idx = cells_to_check(cell_red);
        corr_coeff = corrcoef(matched_table{:, cell_matched}, registered_raw_table{:, cell_red_idx});
        if corr_coeff(1,2)>0.99
            matched_indexes(1, cell_matched) = cell_red_idx;
            cells_to_check(cell_red) = [];
            break;
        end
    end
end
matched_indexes(isnan(matched_indexes)) = [];

%unmatched
unmatched_indexes = nan(1,size(unmatched_table,2)-1);
cells_to_check = (1:size(registered_raw_table,2));
for cell_unmatched = 1:size(unmatched_table,2)
    for cell_red = 1:length(cells_to_check)
        cell_red_idx = cells_to_check(cell_red);
        corr_coeff = corrcoef(unmatched_table{:, cell_unmatched}, registered_raw_table{:, cell_red_idx});
        if corr_coeff(1,2)>0.99
            unmatched_indexes(1, cell_unmatched) = cell_red_idx;
            cells_to_check(cell_red) = [];
            break;
        end
    end
end
unmatched_indexes(isnan(unmatched_indexes)) = [];

%uncertain
uncertain_indexes = nan(1,size(uncertain_table,2)-1);
cells_to_check = (1:size(registered_raw_table,2));
for cell_uncertain = 1:size(uncertain_table,2)
    for cell_red = 1:length(cells_to_check)
        cell_red_idx = cells_to_check(cell_red);
        corr_coeff = corrcoef(uncertain_table{:, cell_uncertain}, registered_raw_table{:, cell_red_idx});
        if corr_coeff(1,2)>0.99
            uncertain_indexes(1, cell_uncertain) = cell_red_idx;
            cells_to_check(cell_red) = [];
            break;
        end
    end
end
uncertain_indexes(isnan(uncertain_indexes)) = [];

%% get guide between color traces and raw traces in terms of global cells
color_registration = cell(size(gla_guide_red,1),1);
for red_cell=1:size(gla_guide_red,1)
    if any(matched_indexes==red_cell)
        color_registration{red_cell} = 'matched';
    elseif any(unmatched_indexes==red_cell)
        color_registration{red_cell} = 'unmatched';
    elseif any(uncertain_indexes==red_cell)
        color_registration{red_cell} = 'uncertain';
    else
        color_registration{red_cell} = 'unknown';
    end
end
gla_guide_red.color_registration = categorical(color_registration);

%% save data
writetable(gla_guide_red, fullfile(folder_data, 'color_registration', strcat(mouse,'_gla_red_guide.csv')));
writematrix(matched_indexes-1, fullfile(folder_data, 'color_registration',strcat(mouse, '_matched_indexes.txt')))
writematrix(unmatched_indexes-1, fullfile(folder_data, 'color_registration',strcat(mouse, '_unmatched_indexes.txt')))
writematrix(uncertain_indexes-1, fullfile(folder_data, 'color_registration',strcat(mouse, '_uncertain_indexes.txt')))



