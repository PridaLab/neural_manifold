mouse = 'GC3';
ltm = load([mouse,'_LTm_events_s5.mat']);
rot = load([mouse, '_Rot_events_s7.mat']);

figure
select = nan(1,size(ltm.tracesEvents.traces,2));
for ii = 1:size(ltm.tracesEvents.traces,2)
    nr = max([ltm.tracesEvents.traces(:,ii);rot.tracesEvents.traces(:,ii)]);
    plot([ltm.tracesEvents.traces(:,ii);rot.tracesEvents.traces(:,ii)]/nr)
    hold on;
    xline(size(ltm.tracesEvents.traces(:,ii),1), 'r--', 'LineWidth', 3)
    yline(std(ltm.tracesEvents.traces(:,ii))/nr)
    yline(std(rot.tracesEvents.traces(:,ii))/nr)
    title(int2str(ii))
    select(ii) = waitforbuttonpress;
    hold off;
end
cells = 1:size(ltm.tracesEvents.traces,2);
cells_selected = cells(select==1);
save('cells_select.mat', 'cells_selected')

load([mouse, '_LT_s5_PyalData_struct.mat'])
for count=1:size(trial_data,2)
    trial_data(count).Inscopix_traces = trial_data(count).Inscopix_traces(:,cells_selected);
    fields = fieldnames(trial_data);
    spike_fields = fields(contains(fields,'spikes_'));
    for idx = 1:length(spike_fields)
        eval(strcat('trial_data(count).',spike_fields{idx},' = trial_data(count).', spike_fields{idx}, '(:,cells_selected);'))
    end
end
save([mouse, '_LT_s5_PyalData_struct.mat'], "trial_data")

load([mouse, '_ROT_s7_PyalData_struct.mat'])
for count=1:size(trial_data,2)
    trial_data(count).Inscopix_traces = trial_data(count).Inscopix_traces(:,cells_selected);
    for idx = 1:length(spike_fields)
        eval(strcat('trial_data(count).',spike_fields{idx},' = trial_data(count).', spike_fields{idx}, '(:,cells_selected);'))
    end
end
save([mouse, '_ROT_s7_PyalData_struct.mat'], "trial_data")