folder = '/media/julio/A6CB-53EE/';

%%%%% VEH PRE TO CNO
name_pre = 'DD2_veh_pre_CNO_correspondence_table_denoised_06.csv';
correspondence_table_pre = readtable(fullfile(folder, name_pre));
%detect which cellset index correspond to veh and which to CNO
[counts, cellnames] = groupcounts(correspondence_table_pre{:,1});
cellnames = cellnames(counts==2,:);
rows = find(correspondence_table_pre{:,1}==cellnames(1));

if correspondence_table_pre{rows(1),"local_cellset_index"}==1 && ... 
        correspondence_table_pre{rows(2),"local_cellset_index"}==2     %when veh is 0 and CNO is 1&2
    fprintf("veh comes first (cellset index 0) and CNO after (cellset index 1&2)")
    del_2 = correspondence_table_pre.local_cellset_index~=2;
    correspondence_table_pre = correspondence_table_pre(del_2,:);
elseif  correspondence_table_pre{rows(1),"local_cellset_index"}==0 && ...
        correspondence_table_pre{rows(2),"local_cellset_index"}==1   %when CNO is 0&1 and veh is 2
    fprintf("CNO comes first (cellset index 0&1) and veh after (cellset index 2)")
    del_2 = correspondence_table_pre.local_cellset_index~=0;
    correspondence_table_pre = correspondence_table_pre(del_2,:);
    correspondence_table_pre{correspondence_table_pre{:,3}==2,3} = 0;
else
    fprintf("Unable to detect which cellsets correspond to veh and which to CNO")
end

%keep only cells that are in both sets
[counts, cellnames] = groupcounts(correspondence_table_pre{:,1});
cellnames = cellnames(counts==2,:);

local_veh = zeros(length(cellnames),1);
local_cno = zeros(length(cellnames),1);
global_all = cellnames;

for idx=1:length(cellnames)
    cell_idx = cellnames(idx);
    rows = find(correspondence_table_pre{:,1}==cell_idx);

    if correspondence_table_pre{rows(1),"local_cellset_index"}==0
        local_veh(idx) = correspondence_table_pre{rows(1),"local_cell_index"};
        local_cno(idx) = correspondence_table_pre{rows(2),"local_cell_index"};
    else
        local_veh(idx) = correspondence_table_pre{rows(2),"local_cell_index"};
        local_cno(idx) = correspondence_table_pre{rows(1),"local_cell_index"};
    end
end


cellmap_pre = table(global_all, local_veh, local_cno, 'VariableNames', ["global_idx", "veh_idx", "CNO_idx"]);
writetable(cellmap_pre, fullfile(folder, 'DD2_veh_pre_CNO_cellmap.csv'))



%VEH ROT TO CNO
name_rot = 'DD2_veh_rot_CNO_correspondence_table_denoised_06.csv';
correspondence_table_rot = readtable(fullfile(folder, name_rot));

%detect which cellset index correspond to veh and which to CNO
[counts, cellnames] = groupcounts(correspondence_table_rot{:,1});
cellnames = cellnames(counts==2,:);
rows = find(correspondence_table_rot{:,1}==cellnames(1));
if correspondence_table_rot{rows(1),"local_cellset_index"}==1 && ... 
        correspondence_table_rot{rows(2),"local_cellset_index"}==2     %when veh is 0 and CNO is 1&2
    fprintf("veh comes first (cellset index 0) and CNO after (cellset index 1&2)")
    del_2 = correspondence_table_rot.local_cellset_index~=2;
    correspondence_table_rot = correspondence_table_rot(del_2,:);
elseif  correspondence_table_rot{rows(1),"local_cellset_index"}==0 && ...
        correspondence_table_rot{rows(2),"local_cellset_index"}==1   %when CNO is 0&1 and veh is 2
    fprintf("CNO comes first (cellset index 0&1) and veh after (cellset index 2)")
    del_2 = correspondence_table_rot.local_cellset_index~=0;
    correspondence_table_rot = correspondence_table_rot(del_2,:);
    correspondence_table_rot{correspondence_table_rot{:,3}==2,3} = 0;
else
    fprintf("Unable to detect which cellsets correspond to veh and which to CNO")
end

%keep only cells that are in both sets
[counts, cellnames] = groupcounts(correspondence_table_rot{:,1});
cellnames = cellnames(counts==2,:);

local_veh = zeros(length(cellnames),1);
local_cno = zeros(length(cellnames),1);
global_all = cellnames;

for idx=1:length(cellnames)
    cell_idx = cellnames(idx);
    rows = find(correspondence_table_rot{:,1}==cell_idx);

    if correspondence_table_rot{rows(1),"local_cellset_index"}==0
        local_veh(idx) = correspondence_table_rot{rows(1),"local_cell_index"};
        local_cno(idx) = correspondence_table_rot{rows(2),"local_cell_index"};
    else
        local_veh(idx) = correspondence_table_rot{rows(2),"local_cell_index"};
        local_cno(idx) = correspondence_table_rot{rows(1),"local_cell_index"};
    end
end

cellmap_rot = table(global_all, local_veh, local_cno, 'VariableNames', ["global_idx", "veh_idx", "CNO_idx"]);
writetable(cellmap_rot, fullfile(folder, 'DD2_veh_rot_CNO_cellmap.csv'))


%find cells that are in both cellmaps 
only_pre = [];
only_rot = [];
both = [];
for idx=1:size(cellmap_pre,1)
    global_idx = cellmap_pre{idx,1};
    veh_idx_pre = cellmap_pre{idx,2};
    cno_idx_pre = cellmap_pre{idx,3};

    rot_idx = find(cellmap_rot{:,1}==global_idx);
    if isempty(rot_idx)
        only_pre = [only_pre; global_idx, veh_idx_pre, cno_idx_pre];
    else
        veh_idx_rot = cellmap_rot{rot_idx, 2};
        cno_idx_rot = cellmap_rot{rot_idx, 3};
        both = [both; global_idx, veh_idx_pre, veh_idx_rot, cno_idx_pre, cno_idx_rot];
    end
end
for idx=1:size(cellmap_rot,1)
    global_idx = cellmap_rot{idx,1};
    veh_idx_rot = cellmap_rot{idx,2};
    cno_idx_rot = cellmap_rot{idx,3};

    veh_idx = find(cellmap_pre{:,1}==global_idx);
    if isempty(veh_idx)
        only_rot = [only_rot; global_idx, veh_idx_rot, cno_idx_rot];
    end
end
cellmap_both = table(both(:,1), both(:,2), both(:,3), zeros(size(both,1), 1)*nan, both(:,4), 'VariableNames', ["global_idx", "veh_idx_pre", ...
    "veh_idx_rot", "veh_idx", "CNO_idx"]);


if isempty(only_pre) && isempty(only_rot)
    fprintf("Same cells in veh pre and rot. Assuming concatenated version (same cellmap).")
    fpritnf("Saving common cellmap without adjusting for local to global index")
    writetable(cellmap_rot, fullfile(folder, 'DD2_veh_CNO_cellmap.csv'))
else
    fprintf("Different cells in veh pre and rot. Assuming non-concatenated version (different cellmap).")
    fprintf("\nLoad tracesEvents to find local to global index.")

    [file_pre, location_pre] = uigetfile('*.mat', 'Select PRE tracesEvents file');
    [file_rot, location_rot] = uigetfile(fullfile(location_pre, '*.mat'), 'Select ROT tracesEvents file');

    events_pre = load(fullfile(location_pre, file_pre));
    events_rot = load(fullfile(location_rot, file_rot));
    
    gla_pre = events_pre.tracesEvents.gla_guide;
    gla_rot = events_rot.tracesEvents.gla_guide;
    
    for idx=1:size(cellmap_both,1)
        fprintf("\nWorking on cell %i\n", idx)
        veh_idx_pre = cellmap_both{idx, 'veh_idx_pre'};
        veh_idx_rot = cellmap_both{idx, 'veh_idx_rot'};

        gla_pre_idx = find(gla_pre{:,'local_accepted'}==veh_idx_pre);
        gla_rot_idx = find(gla_rot{:,'local_accepted'}==veh_idx_rot);

        if ~isempty(gla_pre_idx)
            if isempty(gla_rot_idx) 
                fprintf("Different cells in veh pre and rot for the same CNO. Rot no in all. Keeping pre.")
                veh_idx = gla_pre_idx;
            elseif gla_pre{gla_pre_idx, 'global'} ~= gla_rot{gla_rot_idx, 'global'}
                fprintf("Different cells in veh pre and rot for the same CNO. Keeping only pre.")
                veh_idx = gla_pre_idx;
            else
                fprintf("Correct match in veh pre and rot for the same CNO.")
                veh_idx = gla_pre_idx;
            end
            
        elseif ~isempty(gla_rot_idx)
            fprintf("Different cells in veh pre and rot for the same CNO. Pre no in all. Keeping rot.")
            veh_idx = gla_rot_idx;
        end
        cellmap_both{idx, "veh_idx"} = veh_idx-1;

    end
    writetable(cellmap_both, fullfile(folder, 'DD2_veh_both_CNO_cellmap.csv'))

    %cellmap_rot
    cellmap_pre.Properties.VariableNames{'veh_idx'} = 'veh_idx_pre';
    cellmap_pre{:, 'veh_idx'} = nan;
    cellmap_pre{:, 'veh_idx_rot'} = nan;
    cellmap_pre = cellmap_pre(:, {'global_idx', 'veh_idx_pre', 'veh_idx_rot', 'veh_idx', 'CNO_idx'});
    for idx=1:size(cellmap_pre,1)
        fprintf("\nWorking on cell %i\n", idx)
        veh_idx_pre = cellmap_pre{idx, 'veh_idx_pre'};

        gla_pre_idx = find(gla_pre{:,'local_accepted'}==veh_idx_pre);
        
        if ~isempty(gla_pre_idx)
            veh_idx = gla_pre_idx;
            %find cell in veh rot
            veh_idx_rot = gla_rot{veh_idx,'local_accepted'};
            cellmap_pre{idx, 'veh_idx_rot'} = veh_idx_rot;
            cellmap_pre{idx, 'veh_idx'} = veh_idx-1;
        end

    end
    writetable(cellmap_pre, fullfile(folder, 'DD2_veh_pre_CNO_cellmap.csv'))

    %cellmap_rot
    cellmap_rot.Properties.VariableNames{'veh_idx'} = 'veh_idx_rot';
    cellmap_rot{:, 'veh_idx'} = nan;
    cellmap_rot{:, 'veh_idx_pre'} = nan;
    cellmap_rot = cellmap_rot(:, {'global_idx', 'veh_idx_pre', 'veh_idx_rot', 'veh_idx', 'CNO_idx'});
    for idx=1:size(cellmap_rot,1)
        fprintf("\nWorking on cell %i\n", idx)
        veh_idx_rot= cellmap_rot{idx, 'veh_idx_rot'};

        gla_rot_idx = find(gla_rot{:,'local_accepted'}==veh_idx_rot);
        
        if ~isempty(gla_rot_idx)
            veh_idx = gla_rot_idx;
            %find cell in veh rot
            veh_idx_pre = gla_pre{veh_idx,'local_accepted'};
            cellmap_rot{idx, 'veh_idx_pre'} = veh_idx_pre;
            cellmap_rot{idx, 'veh_idx'} = veh_idx-1;
        end

    end
    writetable(cellmap_rot, fullfile(folder, 'DD2_veh_rot_CNO_cellmap.csv'))

    %combine all
    cellmap_veh_CNO = [cellmap_both; cellmap_pre(~isnan(cellmap_pre{:, 'veh_idx'}),:); cellmap_rot(~isnan(cellmap_rot{:, 'veh_idx'}),:)];
    cellmap_veh_CNO = sortrows(cellmap_veh_CNO);
    %delete repeated pairs of veh_idx/CNO_idx
    [~,i] = unique(findgroups(cellmap_veh_CNO{:, 'veh_idx'}, cellmap_veh_CNO{:, 'CNO_idx'}));
    li = false(size(cellmap_veh_CNO, 1),1);
    li(i) = true;
    cellmap_veh_CNO = cellmap_veh_CNO(li, :);

    writetable(cellmap_veh_CNO, fullfile(folder, 'DD2_veh_CNO_cellmap.csv'))

end
