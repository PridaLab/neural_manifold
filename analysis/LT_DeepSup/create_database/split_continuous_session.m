

load('CZ6_ltrot_events_s1.mat')
tracesEvents_og = tracesEvents;
fields = fieldnames(tracesEvents_og);
l = size(tracesEvents.raw_traces,1);

c = 12486;
tracesEvents = tracesEvents_og;
tracesEvents.test = 'lt';
tracesEvents.session = 5;
for ii = 1:length(fields)
    if size(tracesEvents_og.(fields{ii}),1) == l
        tracesEvents.(fields{ii})(c+1:end,:) = [];
    end
end
save('CZ6_lt_events_s5.mat', 'tracesEvents');

tracesEvents = tracesEvents_og;
tracesEvents.test = 'rot';
tracesEvents.session = 7;
for ii = 1:length(fields)
    if size(tracesEvents_og.(fields{ii}),1) == l
        tracesEvents.(fields{ii})(1:c,:) = [];
    end
end
save('CZ6_rot_events_s7.mat', 'tracesEvents');