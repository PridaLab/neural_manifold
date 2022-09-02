load('DD2_veh_LTm_events_s3.mat')
tracesEvents_og = tracesEvents;
fields = fieldnames(tracesEvents_og);
l = size(tracesEvents.raw_traces,1);

c1_s = 1;
c1_e = 23021;
c2_s = 23044;
c2_e = 47053;
tracesEvents = tracesEvents_og;
tracesEvents.test = 'lt';
tracesEvents.session = 3;
for ii = 1:length(fields)
    if size(tracesEvents_og.(fields{ii}),1) == l
        tracesEvents.(fields{ii}) = tracesEvents_og.(fields{ii})(c1_s: c1_e,:);
    end
end
save('DD2_veh_lt_events_s3.mat', 'tracesEvents');

tracesEvents = tracesEvents_og;
tracesEvents.test = 'rot';
tracesEvents.session = 3;
for ii = 1:length(fields)
    if size(tracesEvents_og.(fields{ii}),1) == l
        tracesEvents.(fields{ii})(c2_e:end,:) = [];
        tracesEvents.(fields{ii})(1:c2_s,:) = [];
    end
end
save('DD2_veh_rot_events_s3.mat', 'tracesEvents');