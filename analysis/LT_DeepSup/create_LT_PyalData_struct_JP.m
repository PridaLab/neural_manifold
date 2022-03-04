close all; clear all; clc;
%%
data_file =  'GC2_castle_alo_events_s2.mat';
%% Load struct
load(data_file);
%% Get start and end points of reward boxes
fh1 = figure('units','normalized','outerposition',[0 0 1 1]);
subplot(4,1,[1,2,3])
imshow(tracesEvents.frame)

title("Draw left line")
leftLim =  drawline(gca);
title("Draw right line")
rightLim = drawline(gca);
title("Press enter to continue")
pause;
figure(fh1)
if sum(contains(fieldnames(tracesEvents), 'ceroxy', 'IgnoreCase',true))==0
    title("Click on bottom-left corner of the track w/ right-click")
    [x,y] = getpts(fh1);
    tracesEvents.ceroXY = [x, y];
    if prctile(tracesEvents.position(:,1),3)>15
        tracesEvents.position(:,1) = tracesEvents.position(:,1) -tracesEvents.ceroXY(1)*tracesEvents.pixel_scale;
        tracesEvents.position(:,2) = tracesEvents.position(:,2) -tracesEvents.ceroXY(2)*tracesEvents.pixel_scale;
    end
end
art_idx = (1:length(tracesEvents.velocity));
art_idx = art_idx(tracesEvents.velocity>100);
og_length = length(tracesEvents.velocity);
if ~isempty(art_idx)
    fields = fieldnames(tracesEvents);
    for field = 1:size(fields,1)
         if length(tracesEvents.(fields{field,1}))== og_length
            tracesEvents.(fields{field,1})(art_idx,:) = [];
         end
    end
end

art_idx = (1:length(tracesEvents.velocity));
art_idx = art_idx(any(tracesEvents.position(:,1)<0, 2));
og_length = length(tracesEvents.velocity);
if ~isempty(art_idx)
    fields = fieldnames(tracesEvents);
    for field = 1:size(fields,1)
         if length(tracesEvents.(fields{field,1}))== og_length
            tracesEvents.(fields{field,1})(art_idx,:) = [];
         end
    end
end

leftLim = mean(leftLim.Position(:,1))-tracesEvents.ceroXY(1);
rightLim = mean(rightLim.Position(:,1))-tracesEvents.ceroXY(1);
if sum(contains(fieldnames(tracesEvents), 'scale', 'IgnoreCase',true))==1
    leftLim = leftLim*tracesEvents.pixel_scale;
    rightLim = rightLim*tracesEvents.pixel_scale;
end
ax2 = subplot(4,1,4);
plot(tracesEvents.position(:,1), tracesEvents.position(:,2));
hold on;
plot([leftLim, leftLim], [min(tracesEvents.position(:,2)), max(tracesEvents.position(:,2))], 'm');
plot([rightLim, rightLim], [min(tracesEvents.position(:,2)), max(tracesEvents.position(:,2))], 'm');
pbaspect(ax2,[5 1 1])
tracesEvents.position(:,2) = tracesEvents.position(:,2) - min(tracesEvents.position(:,2));
%% separate trials 
leftDep=[];
for entry=1:length(tracesEvents.position(:,1))-1
    if tracesEvents.position(entry,1)<leftLim && tracesEvents.position(entry+1,1)>leftLim
        leftDep= [leftDep;entry];
    end
end

leftArrival=[];
for entry=1:length(tracesEvents.position(:,1))-1
    if tracesEvents.position(entry,1)>leftLim && tracesEvents.position(entry+1,1)<leftLim
        leftArrival= [leftArrival;entry];
    end
end
rightDep=[];
for entry=1:length(tracesEvents.position(:,1))-1
    if tracesEvents.position(entry,1)>rightLim && tracesEvents.position(entry+1,1)<rightLim
        rightDep= [rightDep;entry];
    end
end
rightArrival=[];
for entry=1:length(tracesEvents.position(:,1))-1
    if tracesEvents.position(entry,1)<rightLim && tracesEvents.position(entry+1,1)>rightLim
        rightArrival=[rightArrival;entry];
    end
end

%% Visualize arrivals and departures
%{
figure
imshow(tracesEvents.frame)
hold on;
plot(tracesEvents.position(:,1)+ tracesEvents.ceroXY(1), tracesEvents.position(:,2)+tracesEvents.ceroXY(2),'Color', [.5,.5,.5], 'LineWidth', 2);
plot([leftLim+tracesEvents.ceroXY(1), leftLim+tracesEvents.ceroXY(1)], ...
    [min(tracesEvents.position(:,2))+tracesEvents.ceroXY(2), max(tracesEvents.position(:,2))+tracesEvents.ceroXY(2)], 'm');
plot([rightLim+tracesEvents.ceroXY(1), rightLim+tracesEvents.ceroXY(1)],...
    [min(tracesEvents.position(:,2))+tracesEvents.ceroXY(2), max(tracesEvents.position(:,2))+tracesEvents.ceroXY(2)], 'm');
xlim manual

for ii = 1:1:length(tracesEvents.position)
    if any(abs([leftArrival; rightArrival] -ii) <2)
        color = 'm';
        tstring = [int2str(ii), ' - arriving'];
        tpause = 0.5;
    elseif any(abs([leftDep; rightDep] -ii) <2)
        color ='g';
        tstring = [int2str(ii), ' - departing'];
        tpause = 0.5;
    elseif any(abs([leftDep; rightDep;leftArrival; rightArrival] - ii)<tracesEvents.sF)
        frames_diff = abs([leftDep; rightDep;leftArrival; rightArrival] - ii);
        frames_diff(frames_diff<-5) = [];
        tpause = 0.15*(1-min(abs(frames_diff))/tracesEvents.sF) + 0.15;
        color = 'b';
        tstring = [int2str(ii), ' - closing'];
    else
        color = 'r';
        tstring = int2str(ii);
        tpause = 0.05;
    end
    hplot = scatter(tracesEvents.position(ii,1)+tracesEvents.ceroXY(1), tracesEvents.position(ii,2)+tracesEvents.ceroXY(2),color, 'filled');
    title(tstring)
    pause(tpause)
    delete(hplot)
end
%}

%% Get left2right and right2left structures
rightDep(rightDep> max(leftArrival(end), rightArrival(end)))=[];
leftDep(leftDep> max(leftArrival(end), rightArrival(end)))=[];

% check left to left
L2L = [];
visit = 1;
stp = 0;
while stp == 0
    closerArrivalR = min(rightArrival(rightArrival>leftDep(visit)));
    if isempty(closerArrivalR); closerArrivalR = inf; end
    closerArrivalL = min(leftArrival(leftArrival>leftDep(visit)));
    if isempty(closerArrivalL); closerArrivalL = inf; end
    if closerArrivalL < closerArrivalR
        L2L(end+1,1) = leftDep(visit);
        L2L(end,2) = closerArrivalL;
        leftDep(visit) = [];
        leftArrival(leftArrival==closerArrivalL) = [];
    else
        visit = visit+1;
    end
    if visit>length(leftDep); stp = 1; end
end
L2L(diff(L2L,1,2)<floor(tracesEvents.sF/2),:) = []; %at least 0.5 ms
%check right to right     
R2R = [];
visit = 1;
stp = 0;
while stp == 0
    closerArrivalR = min(rightArrival(rightArrival>rightDep(visit)));
    if isempty(closerArrivalR); closerArrivalR = inf; end
    closerArrivalL = min(leftArrival(leftArrival>rightDep(visit)));
    if isempty(closerArrivalL); closerArrivalL = inf; end
    if closerArrivalR < closerArrivalL
        R2R(end+1,1) = rightDep(visit);
        R2R(end,2) = closerArrivalR;
        rightDep(visit) = [];
        rightArrival(rightArrival==closerArrivalR) = [];
    else
        visit = visit+1;
    end
    if visit>length(rightDep); stp = 1; end
end
R2R(diff(R2R,1,2)<floor(tracesEvents.sF/2),:) = []; %at least 0.5 ms
% left to right
rightArrival(rightArrival<leftDep(1))=[];
for visit=1:length(rightArrival)
    L2R(visit,1) = max(leftDep(leftDep<rightArrival(visit)));
    index_speed = max(1,L2R(visit,1)-int16(tracesEvents.sF/2)):min(L2R(visit,1)+int16(tracesEvents.sF/2),length(tracesEvents.position));
    index_speed = index_speed(tracesEvents.velocity(index_speed)>3);
    if ~isempty(index_speed)
        L2R(visit,1) = index_speed(1);
    end
    index_speed = rightArrival(visit):min(rightArrival(visit)+int16(tracesEvents.sF),length(tracesEvents.position));
    index_speed = index_speed(tracesEvents.velocity(index_speed)<3);
    if ~isempty(index_speed)
        rightArrival(visit) = index_speed(1);
    end
    L2R(visit,2)= rightArrival(visit);
end
visit=1;
stp=0;
while stp==0 
    if L2R(visit,1)== L2R(visit+1,1)
        L2R(visit,:)=[];
    else
        visit=visit+1;
    end
    if visit==length(L2R(:,1)); stp=1; end
end
%for visit = 1:size(L2R,1)
%    L2R(visit,3) = min(tracesEvents.reward(tracesEvents.reward>=L2R(visit,2)-3));
%end
% right to left
leftArrival(leftArrival<rightDep(1))=[];
for visit=1:length(leftArrival)
    R2L(visit,1)=max(rightDep(rightDep<leftArrival(visit)));
    index_speed = max(1,R2L(visit,1)-int16(tracesEvents.sF/2)):min(R2L(visit,1)+int16(tracesEvents.sF/2),length(tracesEvents.position));
    index_speed = index_speed(tracesEvents.velocity(index_speed)>3);
    if ~isempty(index_speed)
        R2L(visit,1) = index_speed(1);
    end

    index_speed = leftArrival(visit):min(leftArrival(visit)+int16(tracesEvents.sF),length(tracesEvents.position));
    index_speed = index_speed(tracesEvents.velocity(index_speed)<3);
    if ~isempty(index_speed)
        leftArrival(visit) = index_speed(1);
    end
    R2L(visit,2)=leftArrival(visit);
end
visit=1;
stp=0;
while stp==0 
    if R2L(visit,1)== R2L(visit+1,1)
        R2L(visit,:)=[];
    else
        visit=visit+1;
    end
   
    if visit==length(R2L(:,1)); stp=1; end
end
%for visit = 1:size(R2L,1)
%    R2L(visit,3) = min(tracesEvents.reward(tracesEvents.reward>=R2L(visit,2)-3));
%end
fprintf('\nL2L: %i\nR2R: %i\nL2R: %i\nR2L: %i\n', size(L2L,1), size(R2R,1), size(L2R,1), size(R2L,1))
%{
dur_L2R = (L2R(:,2)-L2R(:,1))/20;
for trial = 1:length(dur_L2R)
    if dur_L2R(trial)>7
        low_lim = max(1, L2R(trial,1)-10*int16(tracesEvents.sF));
        upper_lim = min(L2R(trial,2)+10*int16(tracesEvents.sF), length(tracesEvents.position));
        t = (double(low_lim):double(upper_lim))'/tracesEvents.sF;

        fH1 =figure;
        plot(t, tracesEvents.position(low_lim:upper_lim,1), 'k', 'LineWidth', 2)
        hold on;
        area([L2R(trial,1)/tracesEvents.sF, L2R(trial,2)/tracesEvents.sF], [100, 100], 'FaceAlpha', 0.75, 'FaceColor', '#3C93C2')
        [L2R(trial,1), ~] = getpts(gca);
        L2R(trial,1) = L2R(trial,1)*20;
        close(fH1)
    end
end

dur_R2L = (R2L(:,2)-R2L(:,1))/20;
for trial = 1:length(dur_R2L)
    if dur_R2L(trial)>7
        low_lim = max(1, R2L(trial,1)-10*int16(tracesEvents.sF));
        upper_lim = min(R2L(trial,2)+10*int16(tracesEvents.sF), length(tracesEvents.position));
        t = (double(low_lim):double(upper_lim))'/tracesEvents.sF;

        fH1 = figure;
        plot(t, tracesEvents.position(low_lim:upper_lim,1), 'k', 'LineWidth', 2)
        hold on;
        area([R2L(trial,1)/tracesEvents.sF, R2L(trial,2)/tracesEvents.sF], [100, 100], 'FaceAlpha', 0.75, 'FaceColor', '#3C93C2')
        [R2L(trial,1), ~] = getpts(gca);
        R2L(trial,1) = R2L(trial,1)*20;
        close(fH1)
    end
end
%}
%% Create structure
%code: 0-static, 1-L2R, 2-R2L, 3-L2L, 4-R2R, 5-RewardL2R, 6-RewardR2L
L2R(:,4) = 1;
R2L(:,4) = 2;
if ~isempty(L2L)
    L2L(:,3) = 0;
    L2L(:,4) = 3;
end
if ~isempty(R2R)    
    R2R(:,3) = 0;
    R2R(:,4) = 4;
end
cState = [R2R;L2L;L2R;R2L];
[~,ord] = sort(cState(:,1),1);
cState = cState(ord,:);
%add inbetween static periods and rewards
cState_exp = [];
entry = 1;
for trial = 1:size(cState,1)
    cState_exp(entry,:) = cState(trial, :);
    if trial<size(cState,1)
        entry = entry+1; %static
        cState_exp(entry,:) = [cState(trial,2)-1+1, cState(trial+1,1)-1,nan, 0];
    end
    entry = entry+1;
end

if cState_exp(1,1)>1
    cState_exp = [1, cState_exp(1,1)-1, NaN, 0; cState_exp];
end
if cState_exp(end,2)<size(tracesEvents.position,1)
    cState_exp = [cState_exp; cState_exp(end,2)+1, size(tracesEvents.position,1), NaN, 0];
end

%% Plot
t = (0:size(tracesEvents.position,1)-1)'./tracesEvents.sF;
figure
ax1 = subplot(1,2,1);
hold on;
for ii = 1:size(cState_exp,1)-1
    if cState_exp(ii,4) == 1
        col = '#3C93C2';
    elseif cState_exp(ii,4) == 2
        col = '#FEB24C';
    elseif cState_exp(ii,4) == 3 || cState_exp(ii,4) == 4
        col = '#CD071E';
    elseif cState_exp(ii,4) == 5 || cState_exp(ii,4) == 6
        col = '#F76DBF';
    else
        col = [0.7,0.7,0.7];
    end
    area([cState_exp(ii,1)/tracesEvents.sF, cState_exp(ii,2)/tracesEvents.sF], [1150, 1150], 'FaceAlpha', 0.75, 'FaceColor', col)
end
plot(t, tracesEvents.position(:,1), 'k', 'LineWidth', 2)
xlim([0, 80])
ylim([0, 100])
xlabel('Time (s)')
ylabel('X position (mm)')
set(gca,'FontSize',20)
set(gca,'TickDir','out')
ax2 = subplot(1,2,2);
hold on;
for ii = 1:size(cState_exp,1)-1
    if cState_exp(ii,4) == 1
        col = '#3C93C2';
    elseif cState_exp(ii,4) == 2
        col = '#FEB24C';
    elseif cState_exp(ii,4) == 3 || cState_exp(ii,4) == 4
        col = '#CD071E';
    elseif cState_exp(ii,4) == 5 || cState_exp(ii,4) == 6
        col = '#F76DBF';
    else
        col = [0.7,0.7,0.7];
    end
    area([cState_exp(ii,1)/tracesEvents.sF, cState_exp(ii,2)/tracesEvents.sF], [800, 800], 'FaceAlpha', 0.75, 'FaceColor', col)
end
plot(t, tracesEvents.velocity, 'k', 'LineWidth', 2)
xlim([0, 80])
ylim([0, 100])
xlabel('Time (s)')
ylabel('Velocity (cm/s)')
set(gca,'FontSize',20)
set(gca,'TickDir','out')
linkaxes([ax1,ax2],'x');


%% create trial structure
for ii = 1:size(cState_exp,1)-1
    trial_data(ii).mouse = tracesEvents.mouse;
    trial_data(ii).date = '2021';
    trial_data(ii).task = 'Linear-Track';
    trial_data(ii).session =  tracesEvents.session;

    trial_data(ii).trial_id = ii;
    trial_data(ii).mov = double(cState_exp(ii,4)>0);
    trial_data(ii).cross_middle =  double(cState_exp(ii,4)>0);

    if cState_exp(ii,4) == 1
        trial_data(ii).dir = 'L';
    elseif cState_exp(ii,4) == 2
        trial_data(ii).dir = 'R';
    elseif cState_exp(ii,4) == 3
        trial_data(ii).dir = 'L2L';
    elseif cState_exp(ii,4) == 4
        trial_data(ii).dir = 'R2R';
    else
        trial_data(ii).dir = 'N';
    end
   
    trial_data(ii).Fs =tracesEvents.sF;
    trial_data(ii).bin_size = 1/trial_data(ii).Fs;
    trial_data(ii).idx_trial_start = cState_exp(ii,1);
    [~ , trial_data(ii).idx_peak_speed] = max(tracesEvents.velocity(cState_exp(ii,1):cState_exp(ii,2)));
    trial_data(ii).idx_trial_end = cState_exp(ii,2);
    %trial_data(ii).reward = cState_exp(ii,3);
    trial_data(ii).pos = tracesEvents.position(cState_exp(ii,1):cState_exp(ii,2),:);
    trial_data(ii).vel = tracesEvents.velocity(cState_exp(ii,1):cState_exp(ii,2),:);

    trial_data(ii).Inscopix_traces = tracesEvents.traces(cState_exp(ii,1):cState_exp(ii,2),:);
    trial_data(ii).Inscopix_spikes = tracesEvents.spikes(cState_exp(ii,1):cState_exp(ii,2),:);
    trial_data(ii).Inscopix_amp_spikes = tracesEvents.spikes_amp(cState_exp(ii,1):cState_exp(ii,2),:);
    trial_data(ii).Inscopix_events_spikes = double(tracesEvents.events(cState_exp(ii,1):cState_exp(ii,2),:)>0);
    trial_data(ii).Inscopix_amp_events_spikes = double(tracesEvents.events(cState_exp(ii,1):cState_exp(ii,2),:));

    %trial_data(ii).cellAnaLoc = ones(size(tracesEvents.tracesMINI,2),2);
end
count = 1;
while count<=size(trial_data,2)
    if size(trial_data(count).pos,1)<2
        trial_data(count) = [];
    else
        count = count+1;
    end
end
save([tracesEvents.mouse, '_', tracesEvents.test, '_s', int2str(tracesEvents.session), '_PyalData_struct.mat'], 'trial_data');
       

    
