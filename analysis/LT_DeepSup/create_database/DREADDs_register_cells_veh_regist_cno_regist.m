folder = '/home/julio/Documents/DeepSup_project/DREADDs/Calb/data/DREADDs_veh_cno_correspondence/CalbV23';

%%%%% VEH PRE TO CNO
mouse = 'CalbV23';
CNO_set = 0;
veh_set = 1;

%CREATE LOCAL CELLMAPS
for veh_case = ["pre", "rot"]
    for cno_case = ["pre", "rot"]
        file_name = strcat(mouse,"_veh_",veh_case,"_CNO_",cno_case,"_correspondence_table_denoised_06.csv");
        correspondence_table = readtable(fullfile(folder, file_name));

        %keep only cells that are in both sets
        [counts, cellnames] = groupcounts(correspondence_table{:,1});
        cellnames = cellnames(counts==2,:);
        local_veh = zeros(length(cellnames),1);
        local_cno = zeros(length(cellnames),1);
        global_all = cellnames;

        for idx=1:length(cellnames)
            cell_idx = cellnames(idx);
            rows = find(correspondence_table{:,1}==cell_idx);
            
            for row=rows(:)'
                if correspondence_table{row,"local_cellset_index"} == CNO_set
                    local_cno(idx) = correspondence_table{row,"local_cell_index"};
                elseif correspondence_table{row,"local_cellset_index"} == veh_set
                    local_veh(idx) = correspondence_table{row,"local_cell_index"};
                end

            end
        end

        cellmap = table(global_all, local_veh, local_cno, 'VariableNames', ...
                                ["global_idx", "veh_"+veh_case+"_idx", "CNO_" + cno_case+"_idx"]);

        save_name = strcat(mouse, "_veh_", veh_case, "_CNO_", cno_case, "_cellmap.csv");
        writetable(cellmap, fullfile(folder, save_name))

    end
end

%LINK TO GLOBAL CELLMAPS

%load gla_guide from tracesEvents
fprintf("\nLoad tracesEvents to find local to global index.")
mouse_folder = uigetdir(folder, 'Select mouse folder');
gla_guide_dict = dictionary();
for condition = ["veh", "CNO"]
    case_path = fullfile(mouse_folder, mouse+"_"+condition);
    files = dir(case_path);
    for cond_case = ["pre", "rot"]
    
        if cond_case == "pre"
            name_pattern = mouse + "_" + condition + "_lt_events";
        elseif cond_case == "rot"
            name_pattern = mouse + "_" + condition + "_rot_events";
        end
        
        case_file = files(contains({files(:).name},name_pattern, 'IgnoreCase',true));
        gla_guide = load(fullfile(case_file.folder, case_file.name), "tracesEvents").tracesEvents.gla_guide;
        dict_key = condition+"_"+cond_case;
        gla_guide_dict(dict_key) = {gla_guide};
        fprintf("\n%s | %s : %s (%i cells)", condition, cond_case, case_file.name, size(gla_guide,1))
    end
end

% find global indexes
for veh_case = ["pre", "rot"]
    for cno_case = ["pre", "rot"]

        cellmap_name = strcat(mouse, "_veh_", veh_case, "_CNO_", cno_case, "_cellmap.csv");
        cellmap = readtable(fullfile(folder,cellmap_name));
        if veh_case == "pre"
            veh_oposite_case = "rot";
        else
            veh_oposite_case = "pre";
        end

        if cno_case == "pre"
            cno_oposite_case = "rot";
        else
            cno_oposite_case = "pre";
        end
        cellmap{:,"veh_"+veh_oposite_case+"_idx"} = nan;
        cellmap{:,"CNO_"+cno_oposite_case+"_idx"} = nan;

        cellmap{:,"veh_idx"} = nan; 
        cellmap{:,"CNO_idx"} = nan;

        cellmap = cellmap(:, {'global_idx', 'veh_pre_idx', 'veh_rot_idx', ...
                        'veh_idx', 'CNO_pre_idx', 'CNO_rot_idx', 'CNO_idx'});

        for idx=1:size(cellmap,1)
            veh_local_idx =  cellmap{idx, "veh_"+veh_case+"_idx"};
            cno_local_idx =  cellmap{idx, "CNO_"+cno_case+"_idx"};


            gla_veh_idx = find(gla_guide_dict{"veh_"+veh_case}{:,'local_accepted'}==veh_local_idx);
            gla_cno_idx = find(gla_guide_dict{"CNO_"+cno_case}{:,'local_accepted'}==cno_local_idx);

            if ~isempty(gla_veh_idx) 
                cellmap{idx, "veh_idx"} = gla_veh_idx-1;
                veh_local_opposite_idx = gla_guide_dict{"veh_"+veh_oposite_case}{gla_veh_idx,'local_accepted'};
                cellmap{idx, "veh_"+veh_oposite_case+"_idx"} = veh_local_opposite_idx;

            end
            if ~isempty(gla_cno_idx)             
                cellmap{idx, "CNO_idx"} = gla_cno_idx-1;

                cno_local_opposite_idx = gla_guide_dict{"CNO_"+cno_oposite_case}{gla_cno_idx,'local_accepted'};
                cellmap{idx, "CNO_"+cno_oposite_case+"_idx"} = cno_local_opposite_idx;
            end
        end
        fprintf("\nveh %s | cno %s: %i/%i cells recovered", veh_case, cno_case, ...
            sum(~isnan(cellmap{:, "veh_idx"}.*cellmap{:, "CNO_idx"})), size(cellmap,1))
        writetable(cellmap, fullfile(folder, cellmap_name))

    end
end

%concat all cellmaps
concat_cellmap = nan;
for veh_case = ["pre", "rot"]
    for cno_case = ["pre", "rot"]
        cellmap_name = strcat(mouse, "_veh_", veh_case, "_CNO_", cno_case, "_cellmap.csv");
        cellmap = readtable(fullfile(folder,cellmap_name));

        if class(concat_cellmap)=="double"
            concat_cellmap = cellmap;
        else
            concat_cellmap = [concat_cellmap; cellmap];
        end
    end
end

%CONCAT ALL CELLMAPS
%delete NANs
keep_cells = ~isnan(concat_cellmap{:, "veh_idx"}.*concat_cellmap{:, "CNO_idx"});
concat_cellmap = concat_cellmap(keep_cells,:);
%delete repeated pairs of veh_idx/CNO_idx
[~,i] = unique(findgroups(concat_cellmap{:, 'veh_idx'}, concat_cellmap{:, 'CNO_idx'}));
li = false(size(concat_cellmap, 1),1);
li(i) = true;
concat_cellmap = concat_cellmap(li, :);
concat_cellmap = sortrows(concat_cellmap, [4,7]);
writetable(concat_cellmap, fullfile(folder, mouse+"_veh_CNO_cellmap.csv"))
