load('DD2_LTb_events_s2.mat')
tracesEvents_og = tracesEvents;
fields = fieldnames(tracesEvents_og);
l = size(tracesEvents.raw_traces,1);

c1_s = 1;
c1_e = 23477;
c2_s = 23479;
c2_e = 47838;
tracesEvents = tracesEvents_og;
tracesEvents.test = 'lt';
tracesEvents.session = 5;
for ii = 1:length(fields)
    if size(tracesEvents_og.(fields{ii}),1) == l
        tracesEvents.(fields{ii}) = tracesEvents_og.(fields{ii})(c1_s: c1_e,:);
    end
end
save('DD2_lt_events_s5.mat', 'tracesEvents');

tracesEvents = tracesEvents_og;
tracesEvents.test = 'rot';
tracesEvents.session = 7;
for ii = 1:length(fields)
    if size(tracesEvents_og.(fields{ii}),1) == l
        tracesEvents.(fields{ii})(c2_e:end,:) = [];
        tracesEvents.(fields{ii})(1:c2_s,:) = [];
    end
end
save('DD2_rot_events_s7.mat', 'tracesEvents');