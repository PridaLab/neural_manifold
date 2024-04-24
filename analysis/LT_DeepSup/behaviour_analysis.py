from os.path import join
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from neural_manifold import general_utils as gu

def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))
base_dir = '/home/julio/Documents/DeepSup_project/'


palette_deepsup = ["#cc9900ff", "#9900ffff"]
palette_deepsup_strain = ["#f5b800ff", "#b48700ff", "#9900ffff"]
palette_dual = ["gray"]+palette_deepsup

#mouse folders
Calb_mice = ['CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1','CalbCharly2', 'CalbCharly11_concat', 'CalbV23', 'DD2']
ThyCalbRCaM_mice = ['ThyCalbRCaMP2']
Thy1hRGECO_mice = ['Thy1jRGECO22','Thy1jRGECO23']
ChRNA7_mice = ['ChZ4', 'ChZ7', 'ChZ8', 'ChRNA7Charly1', 'ChRNA7Charly2']
Thy1_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'GC7']


#mouse classification
deepsup_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','ChZ7', 'ChZ8', 'GC7','CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
deepsup_rot_mice = ['GC2','GC3','GC5_nvista', 'TGrin1', 'ChZ4','CZ3', 'CZ6', 'CZ8', 'CZ9', 'CGrin1']
dual_mice = ['Thy1jRGECO22','Thy1jRGECO23','ThyCalbRCaMP2']
dreadds_mice = ['CalbCharly2', 'CalbCharly11_concat', 'CalbV23', 'DD2','ChRNA7Charly1', 'ChRNA7Charly2']

#__________________________________________________________________________
#|                                                                        |#
#|                          DEEP/SUP PRE SESSION                          |#
#|________________________________________________________________________|#
save_dir = '/home/julio/Documents/DeepSup_project/DeepSup/behaviour/'
deepsup_dir = join(base_dir, 'DeepSup')


name_list = list()
correct_list = list()
fail_list = list()
performance_list = list()
layer_list = list()
type_list = list()
duration_list = list()
speed_list = list()

for mouse in deepsup_mice:
    if mouse in Calb_mice:
        mouse_dir = join(deepsup_dir, 'data', 'Calb', mouse)
        type_list.append('Calb')        
        layer_list.append('Sup')
    elif mouse in Thy1_mice:
        mouse_dir = join(deepsup_dir, 'data', 'Thy1', mouse)
        type_list.append('Thy1')
        layer_list.append('Deep')

    elif mouse in ChRNA7_mice:
        mouse_dir = join(deepsup_dir, 'data', 'ChRNA7', mouse)
        type_list.append('ChRNA7')
        layer_list.append('Deep')


    mouse_dict = gu.load_files(mouse_dir,'*_PyalData_struct*.mat',verbose=False,struct_type="PyalData")
    sessions = list(mouse_dict.keys())
    pre_session = [f for f in sessions if 'lt' in f][0]
    session_dict = copy.deepcopy(mouse_dict[pre_session])
    correct_trials = gu.select_trials(session_dict,"dir == ['L','R']")
    fail_trials = gu.select_trials(session_dict,"dir != ['L','R','N']")

    num_correct = correct_trials.shape[0]
    num_fail = fail_trials.shape[0]

    name_list.append(mouse)
    correct_list.append(num_correct)

    fail_list.append(num_fail)
    performance_list.append(1 - (num_fail/(num_correct+num_fail)))

    speed = get_signal(correct_trials, 'vel')
    speed_list.append(np.nanmean(speed))
    duration_list.append(get_signal(session_dict, 'vel').shape[0]/20)


beh_pd = pd.DataFrame(data={'layer': layer_list,
                            'strain': type_list,
                            'mouse': name_list,
                            'correct': correct_list,
                            'fail': fail_list,
                            'performance': performance_list,
                            'speed': speed_list,
                            'duration': duration_list})

beh_pd['trial/min'] = beh_pd['correct']/(beh_pd['duration']/60)


fig = plt.figure(figsize=(6,6))

ax = plt.subplot(2,3,1)
sns.boxplot(data = beh_pd, x ='layer', y='performance',
    palette= palette_deepsup,ax= ax)
sns.scatterplot(data = beh_pd, x ='layer', y='performance',
    color='grey', linewidths = 0, ax= ax)
deep_performance = beh_pd[beh_pd['layer']=='Deep']['performance'].to_list()
sup_performance = beh_pd[beh_pd['layer']=='Sup']['performance'].to_list()
deep_shapiro = stats.shapiro(deep_performance)
sup_shapiro = stats.shapiro(sup_performance)
if deep_shapiro.pvalue<=0.05 or sup_shapiro.pvalue<=0.05:
    ax.set_title(f'ks_2samp p: {stats.ks_2samp(deep_performance, sup_performance)[1]:.4f}')
else:
    ax.set_title(f'ttest_ind p: {stats.ttest_ind(deep_performance, sup_performance)[1]:.4f}')

ax = plt.subplot(2,3,2)
sns.boxplot(data = beh_pd, x ='layer', y='speed',
    palette= palette_deepsup,ax= ax)
sns.scatterplot(data = beh_pd, x ='layer', y='speed',
    color='grey', linewidths = 0, ax= ax)
deep_speed = beh_pd[beh_pd['layer']=='Deep']['speed'].to_list()
sup_speed = beh_pd[beh_pd['layer']=='Sup']['speed'].to_list()
deep_shapiro = stats.shapiro(deep_speed)
sup_shapiro = stats.shapiro(sup_speed)
if deep_shapiro.pvalue<=0.05 or sup_shapiro.pvalue<=0.05:
    ax.set_title(f'ks_2samp p: {stats.ks_2samp(deep_speed, sup_speed)[1]:.4f}')
else:
    ax.set_title(f'ttest_ind p: {stats.ttest_ind(deep_speed, sup_speed)[1]:.4f}')

ax = plt.subplot(2,3,3)
sns.boxplot(data = beh_pd, x ='layer', y='trial/min',
    palette= palette_deepsup,ax= ax)
sns.scatterplot(data = beh_pd, x ='layer', y='trial/min',
    color='grey', linewidths = 0, ax= ax)
deep_trial_min = beh_pd[beh_pd['layer']=='Deep']['trial/min'].to_list()
sup_trial_min = beh_pd[beh_pd['layer']=='Sup']['trial/min'].to_list()
deep_shapiro = stats.shapiro(deep_trial_min)
sup_shapiro = stats.shapiro(sup_trial_min)
if deep_shapiro.pvalue<=0.05 or sup_shapiro.pvalue<=0.05:
    ax.set_title(f'ks_2samp p: {stats.ks_2samp(deep_trial_min, sup_trial_min)[1]:.4f}')
else:
    ax.set_title(f'ttest_ind p: {stats.ttest_ind(deep_trial_min, sup_trial_min)[1]:.4f}')

ax = plt.subplot(2,3,4)
sns.boxplot(data = beh_pd, x ='strain', y='performance',
    palette= palette_deepsup_strain,ax= ax)
sns.scatterplot(data = beh_pd, x ='strain', y='performance',
    color='grey', linewidths = 0, ax= ax)
ax.set_ylim([0.5, 1.02])
Thy1_performance = beh_pd[beh_pd['strain']=='Thy1']['performance'].to_list()
ChRNA7_performance = beh_pd[beh_pd['strain']=='ChRNA7']['performance'].to_list()
Calb_performance = beh_pd[beh_pd['strain']=='Calb']['performance'].to_list()
Thy1_shapiro = stats.shapiro(Thy1_performance)
ChRNA7_shapiro = stats.shapiro(ChRNA7_performance)
Calb_shapiro = stats.shapiro(Calb_performance)
title_str = []
if Thy1_shapiro.pvalue<=0.05 or ChRNA7_shapiro.pvalue<=0.05:
    title_str.append(f'Thy1-ChRNA7: ks_2samp p: {stats.ks_2samp(Thy1_performance, ChRNA7_performance)[1]:.4f}')
else:
    title_str.append(f'Thy1-ChRNA7: ttest_ind p: {stats.ttest_ind(Thy1_performance, ChRNA7_performance)[1]:.4f}')

if Thy1_shapiro.pvalue<=0.05 or Calb_shapiro.pvalue<=0.05:
    title_str.append(f'Thy1-Calb: ks_2samp p: {stats.ks_2samp(Thy1_performance, Calb_performance)[1]:.4f}')
else:
    title_str.append(f'Thy1-Calb: ttest_ind p: {stats.ttest_ind(Thy1_performance, Calb_performance)[1]:.4f}')

if Calb_shapiro.pvalue<=0.05 or ChRNA7_shapiro.pvalue<=0.05:
    title_str.append(f'Calb-ChRNA7: ks_2samp p: {stats.ks_2samp(Calb_performance, ChRNA7_performance)[1]:.4f}')
else:
    title_str.append(f'Calb-ChRNA7: ttest_ind p: {stats.ttest_ind(Calb_performance, ChRNA7_performance)[1]:.4f}')
ax.set_title(title_str, fontsize=6)

ax = plt.subplot(2,3,5)
sns.boxplot(data = beh_pd, x ='strain', y='speed',
    palette= palette_deepsup_strain,ax= ax)
sns.scatterplot(data = beh_pd, x ='strain', y='speed',
    color='grey', linewidths = 0, ax= ax)
ax.set_ylim([0, 50])

Thy1_speed = beh_pd[beh_pd['strain']=='Thy1']['speed'].to_list()
ChRNA7_speed = beh_pd[beh_pd['strain']=='ChRNA7']['speed'].to_list()
Calb_speed = beh_pd[beh_pd['strain']=='Calb']['speed'].to_list()
Thy1_shapiro = stats.shapiro(Thy1_speed)
ChRNA7_shapiro = stats.shapiro(ChRNA7_speed)
Calb_shapiro = stats.shapiro(Calb_speed)
title_str = []
if Thy1_shapiro.pvalue<=0.05 or ChRNA7_shapiro.pvalue<=0.05:
    title_str.append(f'Thy1-ChRNA7: ks_2samp p: {stats.ks_2samp(Thy1_speed, ChRNA7_speed)[1]:.4f}')
else:
    title_str.append(f'Thy1-ChRNA7: ttest_ind p: {stats.ttest_ind(Thy1_speed, ChRNA7_speed)[1]:.4f}')

if Thy1_shapiro.pvalue<=0.05 or Calb_shapiro.pvalue<=0.05:
    title_str.append(f'Thy1-Calb: ks_2samp p: {stats.ks_2samp(Thy1_speed, Calb_speed)[1]:.4f}')
else:
    title_str.append(f'Thy1-Calb: ttest_ind p: {stats.ttest_ind(Thy1_speed, Calb_speed)[1]:.4f}')

if Calb_shapiro.pvalue<=0.05 or ChRNA7_shapiro.pvalue<=0.05:
    title_str.append(f'Calb-ChRNA7: ks_2samp p: {stats.ks_2samp(Calb_speed, ChRNA7_speed)[1]:.4f}')
else:
    title_str.append(f'Calb-ChRNA7: ttest_ind p: {stats.ttest_ind(Calb_speed, ChRNA7_speed)[1]:.4f}')
ax.set_title(title_str, fontsize=6)

ax = plt.subplot(2,3,6)
sns.boxplot(data = beh_pd, x ='strain', y='trial/min',
    palette= palette_deepsup_strain,ax= ax)
sns.scatterplot(data = beh_pd, x ='strain', y='trial/min',
    color='grey', linewidths = 0, ax= ax)
Thy1_trial_min = beh_pd[beh_pd['strain']=='Thy1']['trial/min'].to_list()
ChRNA7_trial_min = beh_pd[beh_pd['strain']=='ChRNA7']['trial/min'].to_list()
Calb_trial_min = beh_pd[beh_pd['strain']=='Calb']['trial/min'].to_list()
Thy1_shapiro = stats.shapiro(Thy1_trial_min)
ChRNA7_shapiro = stats.shapiro(ChRNA7_trial_min)
Calb_shapiro = stats.shapiro(Calb_trial_min)
title_str = []
if Thy1_shapiro.pvalue<=0.05 or ChRNA7_shapiro.pvalue<=0.05:
    title_str.append(f'Thy1-ChRNA7: ks_2samp p: {stats.ks_2samp(Thy1_trial_min, ChRNA7_trial_min)[1]:.4f}')
else:
    title_str.append(f'Thy1-ChRNA7: ttest_ind p: {stats.ttest_ind(Thy1_trial_min, ChRNA7_trial_min)[1]:.4f}')

if Thy1_shapiro.pvalue<=0.05 or Calb_shapiro.pvalue<=0.05:
    title_str.append(f'Thy1-Calb: ks_2samp p: {stats.ks_2samp(Thy1_trial_min, Calb_trial_min)[1]:.4f}')
else:
    title_str.append(f'Thy1-Calb: ttest_ind p: {stats.ttest_ind(Thy1_trial_min, Calb_trial_min)[1]:.4f}')

if Calb_shapiro.pvalue<=0.05 or ChRNA7_shapiro.pvalue<=0.05:
    title_str.append(f'Calb-ChRNA7: ks_2samp p: {stats.ks_2samp(Calb_trial_min, ChRNA7_trial_min)[1]:.4f}')
else:
    title_str.append(f'Calb-ChRNA7: ttest_ind p: {stats.ttest_ind(Calb_trial_min, ChRNA7_trial_min)[1]:.4f}')
ax.set_title(title_str, fontsize=6)

plt.savefig(os.path.join(save_dir,'deep_sup_pre_behaviour.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'deep_sup_pre_behaviour.png'), dpi = 400,bbox_inches="tight")



from statsmodels.formula.api import ols
import statsmodels.api as sm
#__________________________________________________________________________
#|                                                                        |#
#|                            DREADDS SESSION                             |#
#|________________________________________________________________________|#
save_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb/behaviour/'
dreadds_dir = join(base_dir, 'DREADDs')


name_list = list()
correct_list = list()
fail_list = list()
performance_list = list()
condition_list = list()
type_list = list()
duration_list = list()
speed_list = list()
session_list = list()
for mouse in dreadds_mice:
    if mouse in Calb_mice:
        mouse_dir = join(dreadds_dir, 'Calb', 'data', mouse)
        type_mouse = 'Calb'
    elif mouse in ChRNA7_mice:
        mouse_dir = join(dreadds_dir, 'ChRNA7', 'data', mouse)
        type_mouse = 'ChRNA7'


    for condition in ['veh', 'CNO']:
        condition_dir = join(mouse_dir, mouse+'_'+condition)
        mouse_dict = gu.load_files(condition_dir,'*_PyalData_struct*.mat',verbose=False,struct_type="PyalData")
        sessions = list(mouse_dict.keys())
        for session in ['lt', 'rot']:
            session_name = [f for f in sessions if session in f][0]
            session_dict = copy.deepcopy(mouse_dict[session_name])
            correct_trials = gu.select_trials(session_dict,"dir == ['L','R']")
            fail_trials = gu.select_trials(session_dict,"dir != ['L','R','N']")

            num_correct = correct_trials.shape[0]
            num_fail = fail_trials.shape[0]
            # if ('ChRNA7Charly2' in mouse) and ('CNO' in condition) and ('rot' in session):
            #     num_fail = 8
            #     num_correct = 47
            name_list.append(mouse)
            correct_list.append(num_correct)
            fail_list.append(num_fail)
            performance_list.append(1 - (num_fail/(num_correct+num_fail)))

            speed = get_signal(correct_trials, 'vel')
            speed_list.append(np.nanmean(speed))
            duration_list.append(get_signal(session_dict, 'vel').shape[0]/20)
            condition_list.append(condition)
            session_list.append(session)
            type_list.append(type_mouse)


#add chrna7 only beh
for mouse in ['Chrna7Dreadd2', 'Chrna7Dreadd3']:
    mouse_dir = join(dreadds_dir, 'ChRNA7', 'data', 'only_behaviour')
    type_mouse = 'ChRNA7'
    for condition in ['veh', 'CNO']:

        mouse_dict = gu.load_files(mouse_dir,f'*{mouse}_{condition}*_PyalData_struct*.mat',verbose=False,struct_type="PyalData")
        sessions = list(mouse_dict.keys())
        for session in ['lt', 'rot']:
            session_name = [f for f in sessions if session in f][0]
            session_dict = copy.deepcopy(mouse_dict[session_name])
            correct_trials = gu.select_trials(session_dict,"dir == ['L','R']")
            fail_trials = gu.select_trials(session_dict,"dir != ['L','R','N']")

            num_correct = correct_trials.shape[0]
            num_fail = fail_trials.shape[0]

            name_list.append(mouse)
            correct_list.append(num_correct)
            fail_list.append(num_fail)
            performance_list.append(1 - (num_fail/(num_correct+num_fail)))

            speed = get_signal(correct_trials, 'vel')
            speed_list.append(np.nanmean(speed))
            duration_list.append(get_signal(session_dict, 'vel').shape[0]/20)
            condition_list.append(condition)
            session_list.append(session)
            type_list.append(type_mouse)


beh_pd = pd.DataFrame(data={'condition': condition_list,
                            'strain': type_list,
                            'mouse': name_list,
                            'correct': correct_list,
                            'fail': fail_list,
                            'performance': performance_list,
                            'speed': speed_list,
                            'duration': duration_list,
                            'session': session_list})

beh_pd['trial_min'] = beh_pd['correct']/(beh_pd['duration']/60)



#calb
fig = plt.figure(figsize=(6,6))

calb_pd = beh_pd.loc[beh_pd['strain']=='Calb']
ax = plt.subplot(1,3,1)
sns.boxplot(data = calb_pd, x ='condition', y='performance', hue ='session',ax= ax)
sns.scatterplot(data = calb_pd, x ='condition', y='performance', hue ='session', ax= ax)
model = ols('performance ~ C(condition) + C(session) + C(condition):C(session)', data=calb_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

ax = plt.subplot(1,3,2)
sns.boxplot(data = calb_pd, x ='condition', y='speed', hue ='session',ax= ax)
sns.scatterplot(data = calb_pd, x ='condition', y='speed', hue ='session', ax= ax)
model = ols('speed ~ C(condition) + C(session) + C(condition):C(session)', data=calb_pd).fit()
sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

ax = plt.subplot(1,3,3)
sns.boxplot(data = calb_pd, x ='condition', y='trial_min', hue ='session',ax= ax)
sns.scatterplot(data = calb_pd, x ='condition', y='trial_min', hue ='session', ax= ax)
model = ols('trial_min ~ C(condition) + C(session) + C(condition):C(session)', data=calb_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

plt.savefig(os.path.join(save_dir,'dreadds_calb_behaviour.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir,'dreadds_calb_behaviour.png'), dpi = 400,bbox_inches="tight")

#chrna7
fig = plt.figure(figsize=(6,6))
chrna7_pd = beh_pd.loc[beh_pd['strain']=='ChRNA7']
ax = plt.subplot(1,3,1)
sns.boxplot(data = chrna7_pd, x ='condition', y='performance', hue ='session',ax= ax)
sns.scatterplot(data = chrna7_pd, x ='condition', y='performance', hue ='session', ax= ax)
model = ols('performance ~ C(condition) + C(session) + C(condition):C(session)', data=chrna7_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))
ax.set_ylim(0.48,1.02)

ax = plt.subplot(1,3,2)
sns.boxplot(data = chrna7_pd, x ='condition', y='speed', hue ='session',ax= ax)
sns.scatterplot(data = chrna7_pd, x ='condition', y='speed', hue ='session', ax= ax)
model = ols('speed ~ C(condition) + C(session) + C(condition):C(session)', data=chrna7_pd).fit()
sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))


ax = plt.subplot(1,3,3)
sns.boxplot(data = chrna7_pd, x ='condition', y='trial_min', hue ='session',ax= ax)
sns.scatterplot(data = chrna7_pd, x ='condition', y='trial_min', hue ='session', ax= ax)
model = ols('trial_min ~ C(condition) + C(session) + C(condition):C(session)', data=chrna7_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

plt.savefig(os.path.join('/home/julio/Documents/DeepSup_project/DREADDs/ChRNA7/behaviour/','dreadds_chrna7_behaviour.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join('/home/julio/Documents/DeepSup_project/DREADDs/ChRNA7/behaviour/','dreadds_chrna7_behaviour.png'), dpi = 400,bbox_inches="tight")


#speed CNO pareado
chrna7_CNO_pd = chrna7_pd.loc[chrna7_pd['condition']=='CNO']
chrna7_CNO_lt_pd = chrna7_CNO_pd.loc[chrna7_CNO_pd['session']=='lt']
chrna7_CNO_rot_pd = chrna7_CNO_pd.loc[chrna7_CNO_pd['session']=='rot']
stats.ttest_rel(chrna7_CNO_lt_pd['performance'].to_list(), chrna7_CNO_rot_pd['performance'].to_list())
stats.ks_2samp(chrna7_CNO_lt_pd['performance'].to_list(), chrna7_CNO_rot_pd['performance'].to_list())

#__________________________________________________________________________
#|                                                                        |#
#|                          DEEP/SUP PRE SESSION                          |#
#|________________________________________________________________________|#
save_dir = '/home/julio/Documents/DeepSup_project/DeepSup/behaviour/'
deepsup_dir = join(base_dir, 'DeepSup')

num_trials = 10

name_list = list()
correct_list = list()
fail_list = list()
performance_list = list()
layer_list = list()
type_list = list()
duration_list = list()
speed_list = list()
still_duration_list = list()
session_list = list()
for mouse in deepsup_mice:
    if mouse in Calb_mice:
        mouse_dir = join(deepsup_dir, 'data', 'Calb', mouse)
    elif mouse in Thy1_mice:
        mouse_dir = join(deepsup_dir, 'data', 'Thy1', mouse)
    elif mouse in ChRNA7_mice:
        mouse_dir = join(deepsup_dir, 'data', 'ChRNA7', mouse)

    mouse_dict = gu.load_files(mouse_dir,'*_PyalData_struct*.mat',verbose=False,struct_type="PyalData")
    sessions = list(mouse_dict.keys())
    for session in sessions:
        if 'lt' in session:
            session_list.append('pre')
        elif 'rot' in session:
            session_list.append('rot')
        if mouse in Calb_mice:
            mouse_dir = join(deepsup_dir, 'data', 'Calb', mouse)
            type_list.append('Calb')        
            layer_list.append('Sup')
        elif mouse in Thy1_mice:
            mouse_dir = join(deepsup_dir, 'data', 'Thy1', mouse)
            type_list.append('Thy1')
            layer_list.append('Deep')

        elif mouse in ChRNA7_mice:
            mouse_dir = join(deepsup_dir, 'data', 'ChRNA7', mouse)
            type_list.append('ChRNA7')
            layer_list.append('Deep')
        session_dict = copy.deepcopy(mouse_dict[session])
        correct_trials = gu.select_trials(session_dict,"dir == ['L','R']", reset_index=False)
        fail_trials = gu.select_trials(session_dict,"dir != ['L','R','N']", reset_index=False)
        still_trials = gu.select_trials(session_dict,"dir != ['N']", reset_index=False)
        #keep only the first 10 trials
        trial_limit = correct_trials.index[num_trials-1]
        first_correct_trials = correct_trials.loc[:trial_limit]
        first_fail_trials = fail_trials.loc[:trial_limit]
        first_still_trials = still_trials.loc[:trial_limit]

        num_correct = first_correct_trials.shape[0]
        num_fail = first_fail_trials.shape[0]


        name_list.append(mouse)
        correct_list.append(num_correct)
        fail_list.append(num_fail)
        performance_list.append(1 - (num_fail/(num_correct+num_fail)))

        speed = get_signal(first_correct_trials, 'vel')
        speed_list.append(np.nanmean(speed))

        duration_list.append(get_signal(session_dict.loc[:trial_limit], 'vel').shape[0]/20)
        still_duration_list.append(get_signal(first_still_trials, 'vel').shape[0]/(20*first_still_trials.shape[0]))

beh_pd = pd.DataFrame(data={'layer': layer_list,
                            'session': session_list,
                            'strain': type_list,
                            'mouse': name_list,
                            'correct': correct_list,
                            'fail': fail_list,
                            'performance': performance_list,
                            'speed': speed_list,
                            'duration': duration_list,
                            'still_duration': still_duration_list})

beh_pd['trial/min'] = beh_pd['correct']/(beh_pd['duration']/60)



fig = plt.figure(figsize=(6,6))

ax = plt.subplot(1,3,1)
sns.boxplot(data = beh_pd, x ='layer', y='performance', hue='session',
    palette= ['blue', 'red'] ,ax= ax)
sns.scatterplot(data = beh_pd, x ='layer', y='performance', hue='session',
    color='grey', linewidths = 0, ax= ax)

model = ols('performance ~ C(layer) + C(session) + C(layer):C(session)', data=beh_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

ax = plt.subplot(1,3,2)
sns.boxplot(data = beh_pd, x ='layer', y='speed', hue='session',
    palette= ['blue', 'red'] ,ax= ax)
sns.scatterplot(data = beh_pd, x ='layer', y='speed', hue='session',
    color='grey', linewidths = 0, ax= ax)

model = ols('speed ~ C(layer) + C(session) + C(layer):C(session)', data=beh_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))


ax = plt.subplot(1,3,3)
sns.boxplot(data = beh_pd, x ='layer', y='still_duration', hue='session'
,    palette= ['blue', 'red'] ,ax= ax)
sns.scatterplot(data = beh_pd, x ='layer', y='still_duration', hue='session',
    color='grey', linewidths = 0, ax= ax)

model = ols('still_duration ~ C(layer) + C(session) + C(layer):C(session)', data=beh_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))
plt.suptitle(f"{num_trials} first correct trials only")

plt.savefig(join(save_dir,'deep_sup_behaviour_first_trials.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(join(save_dir,'deep_sup_behaviour_first_trials.png'), dpi = 400,bbox_inches="tight")


#__________________________________________________________________________
#|                                                                        |#
#|                           DREADDS first trials                         |#
#|________________________________________________________________________|#
save_dir = '/home/julio/Documents/DeepSup_project/DREADDs/Calb/behaviour/'
dreadds_dir = join(base_dir, 'DREADDs')

num_trials = 10

name_list = list()
correct_list = list()
fail_list = list()
performance_list = list()
condition_list = list()
type_list = list()
duration_list = list()
speed_list = list()
session_list = list()
still_duration_list = list()

for mouse in dreadds_mice:
    if mouse in Calb_mice:
        mouse_dir = join(dreadds_dir, 'Calb', 'data', mouse)
        type_mouse = 'Calb'
    elif mouse in ChRNA7_mice:
        mouse_dir = join(dreadds_dir, 'ChRNA7', 'data', mouse)
        type_mouse = 'ChRNA7'

    for condition in ['veh', 'CNO']:
        condition_dir = join(mouse_dir, mouse+'_'+condition)
        mouse_dict = gu.load_files(condition_dir,'*_PyalData_struct*.mat',verbose=False,struct_type="PyalData")
        sessions = list(mouse_dict.keys())
        for session in ['lt', 'rot']:
            session_name = [f for f in sessions if session in f][0]
            session_dict = copy.deepcopy(mouse_dict[session_name])
            correct_trials = gu.select_trials(session_dict,"dir == ['L','R']")
            fail_trials = gu.select_trials(session_dict,"dir != ['L','R','N']", reset_index=False)
            still_trials = gu.select_trials(session_dict,"dir != ['N']", reset_index=False)

            trial_limit = correct_trials.index[num_trials-1]
            first_correct_trials = correct_trials.loc[:trial_limit]
            first_fail_trials = fail_trials.loc[:trial_limit]
            first_still_trials = still_trials.loc[:trial_limit]


            num_correct = first_correct_trials.shape[0]
            num_fail = first_fail_trials.shape[0]

            name_list.append(mouse)
            correct_list.append(num_correct)
            fail_list.append(num_fail)
            performance_list.append(1 - (num_fail/(num_correct+num_fail)))

            speed = get_signal(first_correct_trials, 'vel')
            speed_list.append(np.nanmean(speed))
            duration_list.append(get_signal(session_dict, 'vel').shape[0]/20)
            still_duration_list.append(get_signal(first_still_trials, 'vel').shape[0]/(20*first_still_trials.shape[0]))

            condition_list.append(condition)
            session_list.append(session)
            type_list.append(type_mouse)


#add chrna7 only beh
for mouse in ['Chrna7Dreadd2', 'Chrna7Dreadd3']:
    mouse_dir = join(dreadds_dir, 'ChRNA7', 'data', 'only_behaviour')
    type_mouse = 'ChRNA7'
    for condition in ['veh', 'CNO']:

        mouse_dict = gu.load_files(mouse_dir,f'*{mouse}_{condition}*_PyalData_struct*.mat',verbose=False,struct_type="PyalData")
        sessions = list(mouse_dict.keys())
        for session in ['lt', 'rot']:
            session_name = [f for f in sessions if session in f][0]
            session_dict = copy.deepcopy(mouse_dict[session_name])
            correct_trials = gu.select_trials(session_dict,"dir == ['L','R']")
            fail_trials = gu.select_trials(session_dict,"dir != ['L','R','N']", reset_index=False)
            still_trials = gu.select_trials(session_dict,"dir != ['N']", reset_index=False)

            trial_limit = correct_trials.index[num_trials-1]
            first_correct_trials = correct_trials.loc[:trial_limit]
            first_fail_trials = fail_trials.loc[:trial_limit]
            first_still_trials = still_trials.loc[:trial_limit]


            num_correct = first_correct_trials.shape[0]
            num_fail = first_fail_trials.shape[0]

            name_list.append(mouse)
            correct_list.append(num_correct)
            fail_list.append(num_fail)
            performance_list.append(1 - (num_fail/(num_correct+num_fail)))

            speed = get_signal(first_correct_trials, 'vel')
            speed_list.append(np.nanmean(speed))
            duration_list.append(get_signal(session_dict, 'vel').shape[0]/20)
            still_duration_list.append(get_signal(first_still_trials, 'vel').shape[0]/(20*first_still_trials.shape[0]))

            condition_list.append(condition)
            session_list.append(session)
            type_list.append(type_mouse)



beh_pd = pd.DataFrame(data={'condition': condition_list,
                            'strain': type_list,
                            'mouse': name_list,
                            'correct': correct_list,
                            'fail': fail_list,
                            'performance': performance_list,
                            'speed': speed_list,
                            'duration': duration_list,
                            'session': session_list,
                            'still_duration': still_duration_list})

beh_pd['trial_min'] = beh_pd['correct']/(beh_pd['duration']/60)



calb_pd = beh_pd.loc[beh_pd['strain']=='Calb']
fig = plt.figure(figsize=(12,6))
ax = plt.subplot(1,3,1)
sns.boxplot(data = calb_pd, x ='condition', y='performance', hue ='session',ax= ax)
sns.scatterplot(data = calb_pd, x ='condition', y='performance', hue ='session', ax= ax)
model = ols('performance ~ C(condition) + C(session) + C(condition):C(session)', data=calb_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

ax = plt.subplot(1,3,2)
sns.boxplot(data = calb_pd, x ='condition', y='speed', hue ='session',ax= ax)
sns.scatterplot(data = calb_pd, x ='condition', y='speed', hue ='session', ax= ax)
model = ols('speed ~ C(condition) + C(session) + C(condition):C(session)', data=calb_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

ax = plt.subplot(1,3,3)
sns.boxplot(data = calb_pd, x ='condition', y='still_duration', hue ='session',ax= ax)
sns.scatterplot(data = calb_pd, x ='condition', y='still_duration', hue ='session', ax= ax)
model = ols('still_duration ~ C(condition) + C(session) + C(condition):C(session)', data=calb_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

plt.savefig(join(save_dir,'dreadds_calb_behaviour_first_trials.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(join(save_dir,'dreadds_calb_behaviour_first_trials.png'), dpi = 400,bbox_inches="tight")


#speed CNO pareado
calb_CNO_pd = calb_pd.loc[calb_pd['condition']=='CNO']
calb_CNO_lt_pd = calb_CNO_pd.loc[calb_CNO_pd['session']=='lt']
calb_CNO_rot_pd = calb_CNO_pd.loc[calb_CNO_pd['session']=='rot']
stats.ttest_rel(calb_CNO_lt_pd['speed'].to_list(), calb_CNO_rot_pd['speed'].to_list())


chrna7_pd = beh_pd.loc[beh_pd['strain']=='ChRNA7']
fig = plt.figure(figsize=(12,6))
ax = plt.subplot(1,3,1)
sns.boxplot(data = chrna7_pd, x ='condition', y='performance', hue ='session',ax= ax)
sns.scatterplot(data = chrna7_pd, x ='condition', y='performance', hue ='session', ax= ax)
model = ols('performance ~ C(condition) + C(session) + C(condition):C(session)', data=chrna7_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))
ax.set_ylim(0.48,1.02)

ax = plt.subplot(1,3,2)
sns.boxplot(data = chrna7_pd, x ='condition', y='speed', hue ='session',ax= ax)
sns.scatterplot(data = chrna7_pd, x ='condition', y='speed', hue ='session', ax= ax)
model = ols('speed ~ C(condition) + C(session) + C(condition):C(session)', data=chrna7_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

ax = plt.subplot(1,3,3)
sns.boxplot(data = chrna7_pd, x ='condition', y='still_duration', hue ='session',ax= ax)
sns.scatterplot(data = chrna7_pd, x ='condition', y='trial_min', hue ='session', ax= ax)
model = ols('still_duration ~ C(condition) + C(session) + C(condition):C(session)', data=chrna7_pd).fit()
a = sm.stats.anova_lm(model, typ=2)
ax.set_title(str(a['PR(>F)']))

#speed CNO pareado
chrna7_CNO_pd = chrna7_pd.loc[chrna7_pd['condition']=='CNO']
chrna7_CNO_lt_pd = chrna7_CNO_pd.loc[chrna7_CNO_pd['session']=='lt']
chrna7_CNO_rot_pd = chrna7_CNO_pd.loc[chrna7_CNO_pd['session']=='rot']
stats.ttest_rel(chrna7_CNO_lt_pd['performance'].to_list(), chrna7_CNO_rot_pd['performance'].to_list())


plt.savefig(join('/home/julio/Documents/DeepSup_project/DREADDs/ChRNA7/behaviour/','dreadds_chrna7_behaviour_first_trials.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(join('/home/julio/Documents/DeepSup_project/DREADDs/ChRNA7/behaviour/','dreadds_chrna7_behaviour_first_trials.png'), dpi = 400,bbox_inches="tight")