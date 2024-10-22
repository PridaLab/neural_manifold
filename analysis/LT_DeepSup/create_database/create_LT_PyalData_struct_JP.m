close all; clear ; clc;
%%
data_file =  'ChRNA7Charly3_veh_rot_events_s4.mat';
%% Load struct
load(data_file);
%% Get start and end points of reward boxes
fh1 = figure('units','normalized','outerposition',[0 0 1 1]);
subplot(4,1,[1,2,3])
imshow(tracesEvents.frame)
%title("Draw left line")
%leftLim =  drawline(gca);

%title("Draw right line")
%rightLim = drawline(gca);
%title("Press enter to continue")
%pause;

art_idx = (1:length(tracesEvents.velocity));
art_idx = art_idx(tracesEvents.velocity>100);
og_length = length(tracesEvents.velocity);
if ~isempty(art_idx)
    fields = fieldnames(tracesEvents);
    for field = 1:size(fields,1)
         if size(tracesEvents.(fields{field,1}),1) == og_length
            tracesEvents.(fields{field,1})(art_idx,:) = [];
         end
    end
end
art_idx = (1:length(tracesEvents.velocity));
art_idx = art_idx(any(tracesEvents.position(:,1)<0, 2));
og_length = size(tracesEvents.velocity,1);
if ~isempty(art_idx)
    fields = fieldnames(tracesEvents);
    for field = 1:size(fields,1)
         if size(tracesEvents.(fields{field,1}),1)== og_length
            tracesEvents.(fields{field,1})(art_idx,:) = [];
         end
    end
end
figure(fh1)
if sum(contains(fieldnames(tracesEvents), 'ceroxy', 'IgnoreCase',true))==0
    tracesEvents.ceroXY = min(tracesEvents.position,[],1);
    tracesEvents.position(:,1) = tracesEvents.position(:,1) -tracesEvents.ceroXY(1);
    tracesEvents.position(:,2) = tracesEvents.position(:,2) -tracesEvents.ceroXY(2);
end
leftLim = 8;
rightLim = 61;
%{
leftLim = mean(leftLim.Position(:,1));
rightLim = mean(rightLim.Position(:,1));
if sum(contains(fieldnames(tracesEvents), 'scale', 'IgnoreCase',true))==1
    leftLim = leftLim*tracesEvents.pixel_scale;
    rightLim = rightLim*tracesEvents.pixel_scale;
end
leftLim = leftLim-tracesEvents.ceroXY(1);
rightLim = rightLim-tracesEvents.ceroXY(1);
%}
ax2 = subplot(4,1,4);
plot(tracesEvents.position(:,1), tracesEvents.position(:,2));
hold on;
plot([leftLim, leftLim], [min(tracesEvents.position(:,2)), max(tracesEvents.position(:,2))], 'm');
plot([rightLim, rightLim], [min(tracesEvents.position(:,2)), max(tracesEvents.position(:,2))], 'm');
pbaspect(ax2,[5 1 1])
tracesEvents.sideLim = [leftLim, rightLim];
%%
save(data_file, "tracesEvents");

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
frame_position = (tracesEvents.position+tracesEvents.ceroXY)./tracesEvents.pixel_scale;
imshow(tracesEvents.frame)
hold on;
plot(frame_position(:,1), frame_position(:,2),'Color', [.5,.5,.5], 'LineWidth', 2);

plot(([leftLim, leftLim]+ tracesEvents.ceroXY(1))./tracesEvents.pixel_scale ,...
            [min(frame_position(:,2)), max(frame_position(:,2))], 'm');
plot(([rightLim, rightLim]+ tracesEvents.ceroXY(1))./tracesEvents.pixel_scale ,...
            [min(frame_position(:,2)), max(frame_position(:,2))], 'm');
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
    hplot = scatter(frame_position(ii,1), frame_position(ii,2),color, 'filled');
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
%% Adjust start and end trials manually
%smooth_vel = movmean(tracesEvents.velocity, round(tracesEvents.sF*1));
%vel_min = prctile(smooth_vel, 5);
%el_max = prctile(smooth_vel, 95);
%pos_min = prctile(tracesEvents.position(:,1), 5);
%pos_max = prctile(tracesEvents.position(:,1), 95);
%fh2 = figure;
for trial = 1:length(L2R)
    pos_5cm = find(tracesEvents.position(L2R(trial,1):L2R(trial,2))>leftLim+10);
    pos_5cm = pos_5cm(1);
    trial_st = max([0, pos_5cm- round(round(tracesEvents.sF))]);
    L2R(trial,1) = L2R(trial,1)+trial_st;
    %{
    hold off;
    plot(t(L2R(trial,1):L2R(trial,2)), tracesEvents.position(L2R(trial,1):L2R(trial,2)), 'LineWidth', 2)
    hold on
    plot(t(L2R(trial,1):L2R(trial,2)), smooth_vel(L2R(trial,1):L2R(trial,2)), 'LineWidth', 2)
    plot(t(L2R(trial,1):L2R(trial,1)+trial_st),tracesEvents.position(L2R(trial,1):L2R(trial,1)+trial_st), 'r', 'LineWidth', 2)
    ylim([pos_min, pos_max]);
    title(int2str(trial))
    [x,y] = getpts(fh2);
    if ~isempty(x)
        if length(x) == 2
            x_left = min(x);
            x_right = max(x);
            L2R(trial,1) = max([L2R(trial,1),round(x_left*tracesEvents.sF)]);
            L2R(trial,2) = min([L2R(trial,2),round(x_right*tracesEvents.sF)]);
        elseif length(x) == 1
            dist_to_edges = abs(L2R(trial,:)-x*tracesEvents.sF);
            if diff(dist_to_edges)>20
                [~, ind] = min(dist_to_edges);
            else %consider also y-dim
                dist_to_edges_y = abs(tracesEvents.position(L2R(trial,:),1)'-y);
                [~, ind] = min(sqrt(dist_to_edges.^2 + dist_to_edges_y.^2));
            end
            if ind == 1 %x_left
                L2R(trial,1) = max([L2R(trial,1),round(x*tracesEvents.sF)]);
            else
                L2R(trial,2) = min([L2R(trial,2),round(x*tracesEvents.sF)]);
            end
        end    
    end
    %}
end
for trial = 1:length(R2L)
    pos_5cm = find(tracesEvents.position(R2L(trial,1):R2L(trial,2))>rightLim-10);
    pos_5cm = pos_5cm(end);
    trial_st = max([0, pos_5cm-round(round(tracesEvents.sF))]);
    R2L(trial,1) = R2L(trial,1)+trial_st;
    %{
    hold off;
    plot(t(R2L(trial,1):R2L(trial,2)), tracesEvents.position(R2L(trial,1):R2L(trial,2)), 'LineWidth', 2)
    hold on
    plot(t(R2L(trial,1):R2L(trial,2)), smooth_vel(R2L(trial,1):R2L(trial,2)), 'LineWidth', 2)
    plot(t(R2L(trial,1):R2L(trial,1)+trial_st),tracesEvents.position(R2L(trial,1):R2L(trial,1)+trial_st), 'r', 'LineWidth', 2)
    ylim([pos_min, pos_max]);
    title(int2str(trial))
    [x,y] = getpts(fh2);
    if ~isempty(x)
        if length(x) == 2
            x_left = min(x);
            x_right = max(x);
            R2L(trial,1) = max([R2L(trial,1),round(x_left*tracesEvents.sF)]);
            R2L(trial,2) = min([R2L(trial,2),round(x_right*tracesEvents.sF)]);
        elseif length(x) == 1
            dist_to_edges = abs(R2L(trial,:)-x*tracesEvents.sF);
            if diff(dist_to_edges)>20
                [~, ind] = min(dist_to_edges);
            else %consider also y-dim
                dist_to_edges_y = abs(tracesEvents.position(R2L(trial,:),1)'-y);
                [~, ind] = min(sqrt(dist_to_edges.^2 + dist_to_edges_y.^2));
            end
            if ind == 1 %x_left
                R2L(trial,1) = max([R2L(trial,1),round(x*tracesEvents.sF)]);
            else
                R2L(trial,2) = min([R2L(trial,2),round(x*tracesEvents.sF)]);
            end
        end    
    end
    %}
end
%% Check vicarious trials
L2Rv = [];
% trial = 1;
% while trial<=length(L2R)
%     pos = tracesEvents.position(L2R(trial,1):L2R(trial,2),1);
%     vel = movmean(tracesEvents.velocity(L2R(trial,1):L2R(trial,2)), round(tracesEvents.sF*1));
%     vel_middle_st = find(pos<prctile(pos,40));
%     vel_middle_st_idx = find(diff(vel_middle_st)>1);
%     if ~isempty(vel_middle_st_idx)
%         vel_middle_st = vel_middle_st(vel_middle_st_idx(1));
%     else
%         vel_middle_st = vel_middle_st(end);
%     end
%     vel_middle_en = find(pos>prctile(pos,60));
%     vel_middle_en_idx = find(diff(vel_middle_en)>1);
%     if ~isempty(vel_middle_en_idx)
%         vel_middle_en = vel_middle_en(vel_middle_en_idx(end)+1);
%     else
%         vel_middle_en = vel_middle_en(1);
%     end
%     middle_vel = mean(vel(vel_middle_st:vel_middle_en));
%     edges_vel = mean([vel(1:vel_middle_st); vel(vel_middle_en:end)]);
%     if middle_vel<=0.7*edges_vel
%         L2Rv(end+1,:) = L2R(trial,:);
%         L2R(trial,:) = [];
%     else
%         trial = trial+1;
% 
%     end
% end

R2Lv = [];
% trial = 1;
% middle_vel_hist = zeros(size(R2L,1),2);
% while trial<=length(R2L)
%     pos = tracesEvents.position(R2L(trial,1):R2L(trial,2),1);
%     vel = movmean(tracesEvents.velocity(R2L(trial,1):R2L(trial,2)), round(tracesEvents.sF*1));
%     vel_middle_st = find(pos>prctile(pos,60));
%     vel_middle_st_idx = find(diff(vel_middle_st)>1);
%     if ~isempty(vel_middle_st_idx)
%         vel_middle_st = vel_middle_st(vel_middle_st_idx(1));
%     else
%         vel_middle_st = vel_middle_st(end);
%     end
%     vel_middle_en = find(pos<prctile(pos,35));
%     vel_middle_en_idx = find(diff(vel_middle_en)>1);
%     if ~isempty(vel_middle_en_idx)
%         vel_middle_en = vel_middle_en(vel_middle_en_idx(end)+1);
%     else
%         vel_middle_en = vel_middle_en(1);
%     end
%     
%     middle_vel = mean(vel(vel_middle_st:vel_middle_en));
%     edges_vel = mean([vel(1:vel_middle_st); vel(vel_middle_en:end)]);
%     if middle_vel<=0.7*edges_vel
%         R2Lv(end+1,:) = R2L(trial,:);
%         R2L(trial,:) = [];
%     else
%         trial = trial+1;
%     end
% end
fprintf('\nL2L: %i\nR2R: %i\nL2Rv: %i\nR2Lv: %i\nL2R: %i\nR2L: %i\n', size(L2L,1), ...
                            size(R2R,1), size(L2Rv,1), size(R2Lv,1),size(L2R,1), size(R2L,1))

%% 
%code: 0-static, 1-L2R, 2-R2L, 3-L2L, 4-R2R, 5-L2Rv, 6-R2Lv
L2R(:,3) = 1;
R2L(:,3) = 2;
if ~isempty(L2L)
    L2L(:,3) = 3;
end
if ~isempty(R2R)    
    R2R(:,3) = 4;
end
if ~isempty(L2Rv)
    L2Rv(:,3) =5;
end
if ~isempty(R2Lv)
    R2Lv(:,3) =6;
end
cState = [R2R;L2L;L2R;R2L;L2Rv;R2Lv];
[~,ord] = sort(cState(:,1),1);
cState = cState(ord,:);

%add inbetween static periods and rewards
cState_exp = [];
entry = 1;
for trial = 1:size(cState,1)
    cState_exp(entry,:) = cState(trial, :);
    if trial<size(cState,1)
        entry = entry+1; %static
        cState_exp(entry,:) = [cState(trial,2)-1+1, cState(trial+1,1)-1,0];
    end
    entry = entry+1;
end

if cState_exp(1,1)>1
    cState_exp = [1, cState_exp(1,1)-1, 0; cState_exp];
end
if cState_exp(end,2)<size(tracesEvents.position,1)
    cState_exp = [cState_exp; cState_exp(end,2)+1, size(tracesEvents.position,1), 0];
end
%% Plot

tracesEvents.velocity = movmean(tracesEvents.velocity, 20);
still_vel = [];
for ii = 1:size(cState_exp,1)-1
    if cState_exp(ii,3) == 0
        still_vel = [still_vel; tracesEvents.velocity(cState_exp(ii,1):cState_exp(ii,2))];
    end
end
min_vel = prctile(still_vel, 20);
tracesEvents.velocity = tracesEvents.velocity - min_vel;
tracesEvents.velocity(tracesEvents.velocity<0) = 0;

t = (0:size(tracesEvents.position,1)-1)'./tracesEvents.sF;
figure
ax1 = subplot(1,2,1);
hold on;
for ii = 1:size(cState_exp,1)-1
    if cState_exp(ii,3) == 1
        col = '#3C93C2';
    elseif cState_exp(ii,3) == 2
        col = '#FEB24C';
    elseif cState_exp(ii,3) == 3 || cState_exp(ii,3) == 4
        col = '#CD071E';
    elseif cState_exp(ii,3) == 5 || cState_exp(ii,3) == 6
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
    if cState_exp(ii,3) == 1
        col = '#3C93C2';
    elseif cState_exp(ii,3) == 2
        col = '#FEB24C';
    elseif cState_exp(ii,3) == 3 || cState_exp(ii,3) == 4
        col = '#CD071E';
    elseif cState_exp(ii,3) == 5 || cState_exp(ii,3) == 6
        col = '#F76DBF';
    else
        col = [0.7,0.7,0.7];
    end
    area([cState_exp(ii,1)/tracesEvents.sF, cState_exp(ii,2)/tracesEvents.sF], [800, 800], 'FaceAlpha', 0.75, 'FaceColor', col)
end
plot(t, tracesEvents.velocity, 'k', 'LineWidth', 1)
xlim([0, 80])
ylim([0, 100])
xlabel('Time (s)')
ylabel('Velocity (cm/s)')
set(gca,'FontSize',20)
set(gca,'TickDir','out')
linkaxes([ax1,ax2],'x');
savefig([tracesEvents.mouse, '_', tracesEvents.test, '_s', int2str(tracesEvents.session), '_beh.fig'])
%% create trial structure
for ii = 1:size(cState_exp,1)-1
    trial_data(ii).mouse = tracesEvents.mouse;
    trial_data(ii).date = datetime;
    trial_data(ii).task = 'Linear-Track';
    trial_data(ii).session =  tracesEvents.session;
    trial_data(ii).trial_id = ii;
    trial_data(ii).mov = double(cState_exp(ii,2)>0);
    trial_data(ii).cross_middle =  double(cState_exp(ii,3)>0);
    if cState_exp(ii,3) == 1
        trial_data(ii).dir = 'L';
    elseif cState_exp(ii,3) == 2
        trial_data(ii).dir = 'R';
    elseif cState_exp(ii,3) == 3
        trial_data(ii).dir = 'L2L';
    elseif cState_exp(ii,3) == 4
        trial_data(ii).dir = 'R2R';
    elseif cState_exp(ii,3) == 5
        trial_data(ii).dir = 'L2Rv';
    elseif cState_exp(ii,3) == 6
        trial_data(ii).dir = 'R2Lv';    
    else
        trial_data(ii).dir = 'N';
    end
    trial_data(ii).Fs =tracesEvents.sF;
    trial_data(ii).bin_size = 1/trial_data(ii).Fs;
    trial_data(ii).idx_trial_start = cState_exp(ii,1);
    [~ , trial_data(ii).idx_peak_speed] = max(tracesEvents.velocity(cState_exp(ii,1):cState_exp(ii,2)));
    trial_data(ii).idx_trial_end = cState_exp(ii,2);
    trial_data(ii).pos = tracesEvents.position(cState_exp(ii,1):cState_exp(ii,2),:);
    trial_data(ii).vel = tracesEvents.velocity(cState_exp(ii,1):cState_exp(ii,2),:);
    trial_data(ii).raw_traces = tracesEvents.raw_traces(cState_exp(ii,1):cState_exp(ii,2),:);
    trial_data(ii).denoised_traces = tracesEvents.denoised_traces(cState_exp(ii,1):cState_exp(ii,2),:);
%     trial_data(ii).conv_traces = tracesEvents.conv_traces(cState_exp(ii,1):cState_exp(ii,2),:);
    fields = fieldnames(tracesEvents);
    spike_fields = fields(contains(fields,'spikes_'));
    for idx = 1:length(spike_fields)
        eval(strcat('trial_data(ii).',spike_fields{idx},' = tracesEvents.', spike_fields{idx}, '(cState_exp(ii,1):cState_exp(ii,2),:);'))
    end
    event_fields = fields(contains(fields,'events_'));
    for idx = 1:length(event_fields)
        eval(strcat('trial_data(ii).',event_fields{idx},' = tracesEvents.', event_fields{idx}, '(cState_exp(ii,1):cState_exp(ii,2),:);'))
    end
    trial_data(ii).cell_idx = 1:size(tracesEvents.denoised_traces,2);
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
       

    
