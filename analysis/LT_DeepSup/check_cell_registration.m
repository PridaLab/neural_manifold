mouse = 'GC1';
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
    trial_data(count).Inscopix_spikes = trial_data(count).Inscopix_spikes(:,cells_selected);
    trial_data(count).Inscopix_amp_spikes = trial_data(count).Inscopix_amp_spikes(:,cells_selected);
    trial_data(count).Inscopix_events_spikes = trial_data(count).Inscopix_events_spikes(:,cells_selected);
    trial_data(count).Inscopix_amp_events_spikes = trial_data(count).Inscopix_amp_events_spikes(:,cells_selected);
end
save([mouse, '_LT_s5_PyalData_struct.mat'], "trial_data")

load([mouse, '_ROT_s7_PyalData_struct.mat'])
for count=1:size(trial_data,2)
    trial_data(count).Inscopix_traces = trial_data(count).Inscopix_traces(:,cells_selected);
    trial_data(count).Inscopix_spikes = trial_data(count).Inscopix_spikes(:,cells_selected);
    trial_data(count).Inscopix_amp_spikes = trial_data(count).Inscopix_amp_spikes(:,cells_selected);
    trial_data(count).Inscopix_events_spikes = trial_data(count).Inscopix_events_spikes(:,cells_selected);
    trial_data(count).Inscopix_amp_events_spikes = trial_data(count).Inscopix_amp_events_spikes(:,cells_selected);
end
save([mouse, '_ROT_s7_PyalData_struct.mat'], "trial_data")