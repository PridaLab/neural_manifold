load('CalbCharly13_concat_pre_events_s2.mat')


tracesEvents_og = tracesEvents;
fields = fieldnames(tracesEvents_og);
l = size(tracesEvents_og.raw_traces,1);

c1_s = 1;
c1_e = 11964;
c2_s = c1_e+1;
c2_e = l;

tracesEvents = tracesEvents_og;
tracesEvents.test = 'lt';
tracesEvents.session = 2;
for ii = 1:length(fields)
    if size(tracesEvents_og.(fields{ii}),1) == l
        tracesEvents.(fields{ii}) = tracesEvents_og.(fields{ii})(c1_s: c1_e,:);
    end
end
save('CalbCharly13_veh_lt_events_s2.mat', 'tracesEvents');

tracesEvents = tracesEvents_og;
tracesEvents.test = 'rot';
tracesEvents.session = 4;
for ii = 1:length(fields)
    if size(tracesEvents_og.(fields{ii}),1) == l
        % tracesEvents.(fields{ii})(c2_e:end,:) = [];
        tracesEvents.(fields{ii})(1:c2_s,:) = [];
    end
end
save('CalbCharly13_veh_rot_events_s4.mat', 'tracesEvents');