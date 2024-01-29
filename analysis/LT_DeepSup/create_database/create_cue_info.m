root_dir = uigetdir(cd, 'Select root folder');
mouse_folders = dir(root_dir);
delete_folders = [];
for idx = 1:size(mouse_folders,1)
    if (mouse_folders(idx).isdir==0 || mouse_folders(idx).name=="." || mouse_folders(idx).name =="..")
        delete_folders = [delete_folders, idx]; %#ok<AGROW>
    end
end
mouse_folders(delete_folders)= [];
og_cd = pwd();
for mouse_idx = 1:size(mouse_folders,1)
    fprintf("\nWorking on mouse: %s", mouse_folders(mouse_idx).name)
    mouse_path = fullfile(mouse_folders(mouse_idx).folder, mouse_folders(mouse_idx).name);
    files = dir(mouse_path); 
    cd(mouse_path)
    tracesEvents_files = files(contains({files(:).name}, '_events_', 'IgnoreCase',true));
    pyalData_files = files(contains({files(:).name}, 'PyalData_struct', 'IgnoreCase',true));
    for idx = 1:size(tracesEvents_files,1)
        fprintf("\n\ttracesEvents: %s | PyalData: %s",tracesEvents_files(idx).name, pyalData_files(idx).name)
        load(fullfile(tracesEvents_files(idx).folder,tracesEvents_files(idx).name));
        frame = tracesEvents.frame;
        tracesEvents_position = tracesEvents.position;
        pixel_scale = tracesEvents.pixel_scale;
        cero_xy = tracesEvents.ceroXY;
        side_lim = tracesEvents.sideLim;
    
    
        load(fullfile(pyalData_files(idx).folder,pyalData_files(idx).name));
        pyalData_position = vertcat(trial_data(:).pos);
    
        fh = figure();
        subplot(4,1,[1,2,3])
        imshow(frame);
        hold on;
        plot(tracesEvents_position(:,1)/pixel_scale + cero_xy(1)/pixel_scale, tracesEvents_position(:,2)/pixel_scale + cero_xy(2)/pixel_scale);
      
        cues_default_position = [405 475; 680 475; 680 600; 405 600];
        cues_polygon = drawpolygon(gca, 'Position', int32(cues_default_position), 'Label', 'cues', ...
            'LabelVisible', 'hover', color='green');
        title("Press enter to continue")
        pause;
    
        x_start = min(cues_polygon.Position(:,1));
        x_end = max(cues_polygon.Position(:,1));
        y_start = min(cues_polygon.Position(:,2));
        y_end = max(cues_polygon.Position(:,2));
        cues_position_pixel = [x_start, y_start; x_end, y_end];
        cues_position = cues_position_pixel.*pixel_scale - repmat(cero_xy,[2,1]);
    
        figure(fh)
        ax2 = subplot(4,1,4);
        plot(pyalData_position(:,1), pyalData_position(:,2));
        hold on;
        plot([side_lim(1), side_lim(1)], [min(pyalData_position(:,2)), max(pyalData_position(:,2))], 'm');
        plot([side_lim(2), side_lim(2)], [min(pyalData_position(:,2)), max(pyalData_position(:,2))], 'm');
        pgon = polyshape([cues_position(1,1), cues_position(2,1), cues_position(2,1), cues_position(1,1)], ...
            [cues_position(1,2), cues_position(1,2), cues_position(2,2), cues_position(2,2)]);
        plot(pgon,FaceColor='green')
        pbaspect(ax2,[5 1 1])
    
        title([tracesEvents.mouse, '_', tracesEvents.test, '_s', int2str(tracesEvents.session)])
        savefig([tracesEvents.mouse, '_', tracesEvents.test, '_s', int2str(tracesEvents.session), '_cues.fig'])
        saveas(fh, [tracesEvents.mouse, '_', tracesEvents.test, '_s', int2str(tracesEvents.session), '_cues.png'])
        close(fh);
        
        output.x_start_pixel = x_start;
        output.x_end_pixel = x_end;
        output.y_start_pixel = y_start;
        output.y_end_pixel = y_end;
    
        output.x_start_cm = cues_position(1,1);
        output.x_end_cm = cues_position(2,1);
        output.y_start_cm = cues_position(1,2);
        output.y_end_cm = cues_position(2,2);
    
        output.pixel_scale = pixel_scale;
        output.cero_x = cero_xy(1);
        output.cero_y = cero_xy(2);
        output.left_side_lim = side_lim(1);
        output.right_side_lim = side_lim(2);
    
        writetable(struct2table(output), [tracesEvents.mouse, '_', tracesEvents.test, '_s', ...
                                                int2str(tracesEvents.session), '_cues_info.csv'])
        imwrite(frame, [tracesEvents.mouse, '_', tracesEvents.test, '_s', ...
                                                int2str(tracesEvents.session), '_maze_frame.png']);
    end
end

cd(og_cd);