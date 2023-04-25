load Mouse-2021-20150301_151117-linear-track-TracesAndEvents.mat
%See if frame detection artifacts (acceleration>2 cm/s2)  %CHANGE: artifacts must have a positive followed by a negative acceleration change take into account this shape for detection
Fs = 20;
tracesEvents.sF = Fs;
%1:106 for 2019-20150238-130909
%1:30 for 2019-20150301_104001
%18300:end for 2021-20150228_180537
%1:178 & 15868:end for 2022-20150228_142050 
%1:43 for 2020-20150301_154837
%1:212 & 17255:end for 2023-20150228_144258
%{
field_names = fields(tracesEvents);
for field_num= 1:numel(field_names)-1
    tracesEvents.(field_names{field_num})(18300:end,:) = [];
    %tracesEvents.(field_names{field_num})(1:212,:) = [];
end
%}
t = (0:size(tracesEvents.position,1)-1)'./Fs;
tracesEvents.velocity_check = tracesEvents.velocity;
tracesEvents.position_check = tracesEvents.position;
acc = [0; diff(tracesEvents.velocity)];
jerk = [0; diff(acc)];
art = (1:size(tracesEvents.velocity,1))';
art = art(abs(acc)>2);
%% compute number of artifacts
if ~isempty(art)
    loc_art = (1:size(art,1))';
    loc_art = [loc_art(diff(art)>30);loc_art(end)];  %CHANGE: take into acccount position intead of time diff
    %fix each artifact
    for ii = 1:size(loc_art,1)
        %locate artifact limits
        if ii == 1
            st = art(1);
        else
            st = art(loc_art(ii-1)+1);
        end
        en = art(loc_art(ii));
        %adjust edges to +-1 s according to jerk (m/s3)
        reg = (st-floor(Fs):en+floor(Fs))';
        reg(reg<=0) = [];
        reg(reg>length(t)) = [];
        mins =  reg(islocalmin(jerk(reg)));
        st = mins(1);
        en = mins(end);
        %interpolate
        query_points = st:en;
        points = 1:size(tracesEvents.velocity_check,1);
        points(st:en) = [];
        vel = tracesEvents.velocity_check;
        vel(st:en) = [];
        pos = tracesEvents.position_check;
        pos(st:en,:) = [];
        
        tracesEvents.velocity_check(st:en) = interp1(points,vel,query_points,'makima');
        tracesEvents.position_check(st:en,1) = interp1(points,pos(:,1),query_points,'makima');
        tracesEvents.position_check(st:en,2) = interp1(points,pos(:,2),query_points,'makima');
    end
end

%%Compute state (moving/steady) according to threshold (2 cm/s)
velocity_threshold = 3;
pMoving = movmean(tracesEvents.velocity_check,Fs)>velocity_threshold;
cState = (1:size(tracesEvents.velocity_check,1))';
cState = cState(diff(pMoving)~=0);
%Check that each section last a minimum duration (1 s)
durState = [cState(1);diff(cState)];
cState(durState<floor(Fs/2)) = [];
%Re-adjust the start and end of each state
    %Count points bellow 0.1
sig_min_th = (tracesEvents.velocity_check<0.5);
loc_min_th =  (1:size(tracesEvents.velocity_check,1))';
loc_min_th(sig_min_th==0) = [];
    %Count local minima
sig_min_dev = islocalmin(tracesEvents.velocity_check);
sig_min_dev(tracesEvents.velocity_check>1) = 0;
loc_min_dev = (1:size(tracesEvents.velocity_check,1))';
loc_min_dev(sig_min_dev==0) = [];
clear sig_min_dev sig_min_th
cState_exp = cState;
cState_exp = [1;cState_exp;size(tracesEvents.velocity_check,1)];
%asing state
cState_exp_states = nan(size(cState_exp));
for ii = 1:size(cState_exp,1)-1
    st = cState_exp(ii);
    en = cState_exp(ii+1);
    if median(tracesEvents.velocity_check(st:en))>2
        cState_exp_states(ii) = 1;
    else
        cState_exp_states(ii) = 0;
    end
end
%expand events and save in _exp variables
for ii = 2:size(cState_exp,1)-1
    if cState_exp_states(ii-1) == 0 %expand back
        pre_ev = cState_exp(ii-1);
        pre_min_th = find(loc_min_th<=cState_exp(ii));
        pre_min_dev = find(loc_min_dev<=cState_exp(ii));
        
        max_val = (tracesEvents.velocity_check>1.5);
        above_val = (1:size(tracesEvents.velocity_check,1))';
        above_val(max_val==0) = [];
        ind = find(above_val<cState_exp(ii)-Fs);
        if ~isempty(ind)
            above_val = above_val(ind(end));
        else
            above_val = NaN;
        end
        max_range = cState_exp(ii) - 1.5*Fs;
        pre_lim = max([pre_ev;above_val; max_range]);
        clear max_range above_val pre_ev
        if ~isempty(pre_min_dev)
            pre_min = loc_min_dev(pre_min_dev(end));
            if pre_min<pre_lim
                pre_min = NaN;
            end
        else
            pre_min = NaN;   
        end
        if isnan(pre_min)&&~isempty(pre_min_th)
            pre_min_th = loc_min_th(pre_min_th(end));
            if pre_min<pre_lim
                pre_min = NaN;
            end
        end
        if ~isnan(pre_min)
            cState_exp(ii) = pre_min;
        end   
    else
        pos_ev = cState_exp(ii+1);
        pos_min_th = find(loc_min_th>=cState_exp(ii));
        pos_min_dev = find(loc_min_dev>=cState_exp(ii));
        
        max_val = (tracesEvents.velocity_check>1.5);
        above_val = (1:size(tracesEvents.velocity_check,1))';
        above_val(max_val==0) = [];
        ind = find(above_val>cState_exp(ii)+Fs);
        if ~isempty(ind)
            above_val = above_val(ind(1));
        else
            above_val = NaN;
        end
        max_range = cState_exp(ii) + 1*Fs;
        pos_lim = min([pos_ev;above_val; max_range]);
        clear max_range above_val pos_ev
        if ~isempty(pos_min_dev)
            pos_min = loc_min_dev(pos_min_dev(1));
            if pos_min>pos_lim
                pos_min = NaN;
            end
        else
            pos_min = NaN;   
        end
        if isnan(pos_min)&&~isempty(pos_min_th)
            pos_min_th = loc_min_th(pos_min_th(1));
            if pos_min<pos_lim
                pos_min = NaN;
            end
        end
        if ~isnan(pos_min)
            cState_exp(ii) = pos_min;
        end   
    end
end
%asign movement/steady to each interval
cState_exp = [cState_exp, nan(size(cState_exp))];
for ii = 1:size(cState_exp,1)-1
    st = cState_exp(ii);
    en = cState_exp(ii+1);
    if prctile(tracesEvents.velocity_check(st:en),50)>=1.5
        cState_exp(ii,2) = 1;
    else
        cState_exp(ii,2) = 0;
    end
end

%merge continous states
merge = nan(size(cState_exp,1),1);
for ii = 2:size(cState_exp,1)-1
    if cState_exp(ii,2) == cState_exp(ii-1,2)
        merge(ii) = 1;
    elseif cState_exp(ii,2) ==cState_exp(ii+1,2)
        merge(ii+1) = 1;
    end
end
merge = merge==1;
cState_exp(merge,:) = [];
%{
%delete events that last less than 0.5sec and merge again
durState = [diff(cState_exp(:,1));1000];
cState_exp(durState<floor(Fs/2),:) = [];
merge = nan(size(cState_exp,1),1);
for trial_idx = 2:size(cState_exp,1)-1
    if cState_exp(trial_idx,2) == cState_exp(trial_idx-1,2)
        merge(trial_idx) = 1;
    elseif cState_exp(trial_idx,2) ==cState_exp(trial_idx+1,2)
        merge(trial_idx+1) = 1;
    end
end
merge = merge==1;
cState_exp(merge,:) = [];
%}
%check if animal crosses from one side to the other
min_x = prctile(tracesEvents.position_check(:,1),5);
max_x = prctile(tracesEvents.position_check(:,1),95);
cState_exp = [cState_exp, nan(size(cState_exp,1),1)];


%%
trial_idx =1;
%while trial_idx <=size(cState_exp,1)-1
while trial_idx <=size(cState_exp,1)-1

    st = cState_exp(trial_idx);
    en = cState_exp(trial_idx+1);
    dur = en-st;
    st_pos = tracesEvents.position_check(st);
    en_pos = tracesEvents.position_check(en);
    if st_pos>en_pos
        st_pos = prctile(tracesEvents.position_check(st:st+floor(0.1*dur),1),95);
        en_pos = prctile(tracesEvents.position_check(en-floor(0.1*dur):en,1),5);
    else  
        st_pos = prctile(tracesEvents.position_check(st:st+floor(0.1*dur),1),5);
        en_pos = prctile(tracesEvents.position_check(en-floor(0.1*dur):en,1),95);
    end
    if abs(en_pos - st_pos)>= 0.7*abs(max_x - min_x) %if crosses midle check whether moving or not
        if cState_exp(trial_idx,2) == 1 %if moving perfect trial
            cState_exp(trial_idx,3) = 1;
        else %if not moving inconsistency
            cState_exp(trial_idx,3) = NaN;
            fprintf('Inconsistency in trial %i, not moving but crossing middle\n', trial_idx);
        end
    else %if does not cross the middle
        if cState_exp(trial_idx,2) == 0 %if not moving static trial
            cState_exp(trial_idx,3) = 0;
        else %if moving but not crossing middle check whether fail trial
            count = 0;
            en_pos_fut = en_pos;
            if cState_exp(trial_idx-1,3) == -1
                th = 0.5;
                st_pos = cState_exp(trial_idx-1);
            elseif trial_idx>2
                if cState_exp(trial_idx-2,3) == -1 && cState_exp(trial_idx-1,3) == 0
                    th = 0.4;
                    st_pos = cState_exp(trial_idx-2);
                else
                    th = 0.6;
                end
            else
                th = 0.6;
            end
            while (abs(en_pos_fut - st_pos)> 0.1*abs(max_x + min_x)/2) && (abs(en_pos_fut - st_pos)<th*abs(max_x - min_x)) %check if still close to the middle
                count = count+1;
                st_fut = cState_exp(trial_idx+count);
                en_fut = cState_exp(trial_idx+count+1);
                dur_fut = en_fut-st_fut;
                en_pos_fut = prctile(tracesEvents.position_check(en_fut-floor(0.1*dur_fut):en_fut,1),95); %check if next epoch returns
            end
            if count>0 && abs(en_pos_fut - st_pos)>= th*abs(max_x - min_x)
                count = count-1;
            end
            if count == 0
                if cState_exp(trial_idx-1,3) == -1
                    cState_exp(trial_idx,3) = -2;
                elseif  trial_idx> 2
                    if cState_exp(trial_idx-2,3) == -1 && cState_exp(trial_idx-1,3) == 0
                        cState_exp(trial_idx,3) = -2;
                        cState_exp(trial_idx-1,:) = [];
                        trial_idx = trial_idx-1;
                    end
                else
                    cState_exp(trial_idx,3) = -1;
                end
            else
                cState_exp(trial_idx+1:trial_idx+count, :) = [];
                cState_exp(trial_idx,3) = -1;
            end
        end
    end
    trial_idx = trial_idx+1;
end

%% Plot 
t = (0:size(tracesEvents.position,1)-1)'./20;
figure
ax1 = subplot(2,1,1);
hold on;
for ii = 1:size(cState_exp,1)-1
    if cState_exp(ii,3) == 1
        if tracesEvents.position_check(cState_exp(ii,1))>tracesEvents.position_check(cState_exp(ii+1,1)-1)
                col = '#FEB24C';
        else
                col = '#3C93C2';
        end
    elseif cState_exp(ii,3) == -1
        col = '#CD071E';
    elseif cState_exp(ii,3) == -2
        col = '#F76DBF';
    else
        col = [0.7,0.7,0.7];
    end
    area([cState_exp(ii)/20, cState_exp(ii+1)/20], [800, 800], 'FaceAlpha', 0.75, 'FaceColor', col)
end
plot(t, tracesEvents.position_check(:,1), 'k', 'LineWidth', 2)
xlim([0, 100])
ylim([0, 800])
xlabel('Time (s)')
ylabel('X position (mm)')
set(gca,'FontSize',20)
set(gca,'TickDir','out')
ax2 = subplot(2,1,2);
hold on;
for ii = 1:size(cState_exp,1)-1
    if cState_exp(ii,3) == 1
        if tracesEvents.position_check(cState_exp(ii,1))>tracesEvents.position_check(cState_exp(ii+1,1)-1)
                col = '#FEB24C';
        else
                col = '#3C93C2';
        end
    elseif cState_exp(ii,3) == -1
        col = '#CD071E';
    elseif cState_exp(ii,3) == -2
        col = '#F76DBF';
    else
        col = [0.7,0.7,0.7];
    end
    area([cState_exp(ii)/20, cState_exp(ii+1)/20], [800, 800], 'FaceAlpha', 0.75, 'FaceColor', col)
end
plot(t, tracesEvents.velocity_check, 'k', 'LineWidth', 2)
xlim([0, 10])
ylim([0, 20])
xlabel('Time (s)')
ylabel('Velocity (cm/s)')
set(gca,'FontSize',20)
set(gca,'TickDir','out')
linkaxes([ax1,ax2],'x');

%% 
clearvars -except tracesEvents cState_exp acc min_x max_x
Fs = 20;
for ii = 1:size(cState_exp,1)-1
    trial_data(ii).mouse = 'Mouse2021';
    trial_data(ii).date = '2015';
    trial_data(ii).task = 'Linear-Track';
    trial_data(ii).trial_id = ii;
    trial_data(ii).mov = cState_exp(ii,2);
    trial_data(ii).cross_middle = cState_exp(ii,3);
    
    st_pos = tracesEvents.position_check(cState_exp(ii,1));
    en_pos = tracesEvents.position_check(cState_exp(ii+1,1)-1);
    if cState_exp(ii,3) == 1
        if st_pos>en_pos
            trial_data(ii).dir = 'L';
        else
            trial_data(ii).dir = 'R';
        end
    elseif cState_exp(ii,3) == -1 %if fail trial

        if abs(st_pos-en_pos)<0.1*abs(max_x + min_x)/2 %return trial
            if st_pos>abs(max_x + min_x)/2 %return right
                trial_data(ii).dir = 'FRR';
            else
                trial_data(ii).dir = 'FRL';
            end
        elseif cState_exp(ii+1,3) == -2 %not returning
            if st_pos>abs(max_x + min_x)/2 %return right
                trial_data(ii).dir = 'FR';
            else
                trial_data(ii).dir = 'FL';
            end
        end
    elseif cState_exp(ii,3) == -2 %success after fail trial
        if st_pos>abs(max_x + min_x)/2 %return right
            trial_data(ii).dir = 'FSR';
        else
            trial_data(ii).dir = 'FSL';
        end
    else
        trial_data(ii).dir = 'N';
    end
    trial_data(ii).Fs =Fs;
    trial_data(ii).bin_size = 1/trial_data(ii).Fs;
    trial_data(ii).idx_trial_start = cState_exp(ii,1);
    [~ , trial_data(ii).idx_peak_speed] = max(tracesEvents.velocity_check(cState_exp(ii,1):cState_exp(ii+1,1)-1));
    trial_data(ii).idx_trial_end = cState_exp(ii+1,1)-1;
    trial_data(ii).pos = tracesEvents.position_check(cState_exp(ii,1):cState_exp(ii+1,1)-1,:);
    trial_data(ii).vel = tracesEvents.velocity_check(cState_exp(ii,1):cState_exp(ii+1,1)-1);
    trial_data(ii).acc = acc(cState_exp(ii,1):cState_exp(ii+1,1)-1);
    trial_data(ii).rawProb = tracesEvents.rawProb(cState_exp(ii,1):cState_exp(ii+1,1)-1,:);
    if isfield(tracesEvents, 'zProb')
        trial_data(ii).zProb = tracesEvents.zProb(cState_exp(ii,1):cState_exp(ii+1,1)-1,:);
    else
        trial_data(ii).zProb = [];
    end
    trial_data(ii).deconvProb = tracesEvents.spikeDeconvTrace(cState_exp(ii,1):cState_exp(ii+1,1)-1,:);
    trial_data(ii).th_spikes = tracesEvents.tresholdEvents(cState_exp(ii,1):cState_exp(ii+1,1)-1,:);
    trial_data(ii).deconv_spikes= tracesEvents.spikeDeconv(cState_exp(ii,1):cState_exp(ii+1,1)-1,:);
    trial_data(ii).ML_spikes = tracesEvents.spikeML(cState_exp(ii,1):cState_exp(ii+1,1)-1,:);
    trial_data(ii).cellAnaLoc = tracesEvents.cellAnatomicLocat;
end

count = 1;
while count<=size(trial_data,2)
    if size(trial_data(count).pos,1)<=1 
        trial_data(count) = [];
    else
        count = count+1;
    end
end
save('M2021_20150301_151117_LT_PyalData_struct.mat', 'trial_data');

