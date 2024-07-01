folder = '/media/julio/A6CB-53EE/';

%%%%% VEH PRE TO CNO
name_pre = 'CalbCharly2_veh_pre_CNO_correspondence_table_denoised_06.csv';
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
writetable(cellmap_pre, fullfile(folder, 'CalbCharly2_veh_pre_CNO_cellmap.csv'))


%VEH ROT TO CNO
name_rot = 'CalbCharly2_veh_rot_CNO_correspondence_table_denoised_06.csv';
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
writetable(cellmap_rot, fullfile(folder, 'CalbCharly2_veh_rot_CNO_cellmap.csv'))


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
    writetable(cellmap_rot, fullfile(folder, 'CalbCharly2_veh_CNO_cellmap.csv'))

end