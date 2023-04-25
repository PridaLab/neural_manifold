%% Plot
folder = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/data/M2019';
name = 'M2019_20150301_104001_LT_PyalData_struct.mat';
time_lim = [465.4, 565.4];
load(fullfile(folder, name));

pos = trial_data(1,1).pos;
vel = trial_data(1,1).vel;
for trial=2:size(trial_data,2)
    pos = [pos; trial_data(1,trial).pos];
    vel = [vel; trial_data(1,trial).vel];
end

t = (0:size(pos,1)-1)'./trial_data(1,1).Fs;

figure
ax1 = subplot(2,1,1);
hold on;
for ii = 1:size(trial_data,2)
    if strcmp(trial_data(1,ii).dir, 'R')
        col = [251,192,134]/255;
    elseif strcmp(trial_data(1,ii).dir, 'L')
        col = [190, 174, 212]/255;
    elseif strcmp(trial_data(1,ii).dir, 'N')
        col = [127,201,127]/255;  
    elseif strcmp(trial_data(1,ii).dir, 'FL') || strcmp(trial_data(1,ii).dir, 'FR')
        col = '#CD071E';
    else
        col = '#F76DBF';
    end
    idx_start = trial_data(1,ii).idx_trial_start;
    idx_end = trial_data(1,ii).idx_trial_end;
    area([idx_start/20, idx_end/20], [120, 120], 'FaceAlpha', 0.75, 'FaceColor', col)
end
plot(t, pos(:,1), 'k', 'LineWidth', 2)
ylim([0, 130])
xlim(time_lim)
xlabel('Time (s)')
ylabel('X position (mm)')
set(gca,'FontSize',20)
set(gca,'TickDir','out')

ax2 = subplot(2,1,2);
hold on;
for ii = 1:size(trial_data,2)
    if strcmp(trial_data(1,ii).dir, 'R')
        col = [251,192,134]/255;
    elseif strcmp(trial_data(1,ii).dir, 'L')
        col = [190, 174, 212]/255;
    elseif strcmp(trial_data(1,ii).dir, 'N')
        col = [127,201,127]/255;  
    elseif strcmp(trial_data(1,ii).dir, 'FL') || strcmp(trial_data(1,ii).dir, 'FR')
        col = '#CD071E';
    else
        col = '#F76DBF';
    end
    idx_start = trial_data(1,ii).idx_trial_start;
    idx_end = trial_data(1,ii).idx_trial_end;
    area([idx_start/20, idx_end/20], [60, 60], 'FaceAlpha', 0.75, 'FaceColor', col)
end
plot(t, abs(vel), 'k', 'LineWidth', 2)
xlim(time_lim)
ylim([0, 60])
xlabel('Time (s)')
ylabel('Velocity (cm/s)')
set(gca,'FontSize',20)
set(gca,'TickDir','out')
linkaxes([ax1,ax2],'x');

print('-vector, '-depsc', [name(1:5),'_behaviour'])