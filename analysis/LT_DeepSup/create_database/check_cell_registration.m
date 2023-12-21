mouse = 'GC2';
ltm = load([mouse,'_ltpre_events_s3.mat']);
rot = load([mouse, '_ltpost_events_s5.mat']);

figure
select = nan(1,size(ltm.tracesEvents.denoised_traces,2));
for ii = 1:size(ltm.tracesEvents.denoised_traces,2)
    nr = max([ltm.tracesEvents.denoised_traces(:,ii);rot.tracesEvents.denoised_traces(:,ii)]);
    plot([ltm.tracesEvents.denoised_traces(:,ii);rot.tracesEvents.denoised_traces(:,ii)]/nr)
    hold on;
    xline(size(ltm.tracesEvents.denoised_traces(:,ii),1), 'r--', 'LineWidth', 3)
    yline(std(ltm.tracesEvents.denoised_traces(:,ii))/nr)
    yline(std(rot.tracesEvents.denoised_traces(:,ii))/nr)
    title(int2str(ii))
    select(ii) = waitforbuttonpress;
    hold off;
end
cells = 1:size(ltm.tracesEvents.denoised_traces,2);
cells_selected = cells(select==1);
save('cells_select.mat', 'cells_selected')

load([mouse, '_ltpre_s3_PyalData_struct.mat'])
for count=1:size(trial_data,2)
    trial_data(count).raw_traces = trial_data(count).raw_traces(:,cells_selected);
    trial_data(count).denoised_traces = trial_data(count).denoised_traces(:,cells_selected);
    fields = fieldnames(trial_data);
    spike_fields = fields(contains(fields,'spikes_'));
    for idx = 1:length(spike_fields)
        eval(strcat('trial_data(count).',spike_fields{idx},' = trial_data(count).', spike_fields{idx}, '(:,cells_selected);'))
    end
    event_fields = fields(contains(fields,'events_'));
    for idx = 1:length(event_fields)
        eval(strcat('trial_data(count).',event_fields{idx},' = trial_data(count).', event_fields{idx}, '(:,cells_selected);'))
    end
end
save([mouse, '_ltpre_s3_PyalData_struct.mat'], "trial_data")

load([mouse, '_ltpost_s5_PyalData_struct.mat'])
for count=1:size(trial_data,2)
    trial_data(count).raw_traces = trial_data(count).raw_traces(:,cells_selected);
    trial_data(count).denoised_traces = trial_data(count).denoised_traces(:,cells_selected);
    fields = fieldnames(trial_data);
    spike_fields = fields(contains(fields,'spikes_'));
    for idx = 1:length(spike_fields)
        eval(strcat('trial_data(count).',spike_fields{idx},' = trial_data(count).', spike_fields{idx}, '(:,cells_selected);'))
    end
    event_fields = fields(contains(fields,'events_'));
    for idx = 1:length(event_fields)
        eval(strcat('trial_data(count).',event_fields{idx},' = trial_data(count).', event_fields{idx}, '(:,cells_selected);'))
    end
end
save([mouse, '_ltpost_s5_PyalData_struct.mat'], "trial_data")