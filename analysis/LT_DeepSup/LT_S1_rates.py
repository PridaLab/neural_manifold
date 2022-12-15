from neural_manifold import general_utils as gu
import sys, os, timeit
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mouse = sys.argv[1]
data_dir = sys.argv[2]
save_dir = sys.argv[3]

kernel_std = 0.25
assymetry = True
num_std = 5
continuous = True

#__________________________________________________________________________
#|                                                                        |#
#|                            1. COMPUTE RATES                            |#
#|________________________________________________________________________|#

save_dir_step = os.path.join(save_dir, 'S1_add_rates')
if not os.path.exists(save_dir_step):
    os.makedirs(save_dir_step)

f = open(os.path.join(save_dir_step,mouse + '_logFile.txt'), 'w')
original = sys.stdout
sys.stdout = gu.Tee(sys.stdout, f)

global_starttime = timeit.default_timer()

print(f"Working on mouse {mouse}:")
print(f"\tdata_dir: {data_dir}")
print(f"\tsave_dir: {save_dir}")
print(f"\tDate: {datetime.now():%Y-%m-%d %H:%M}")

#1. Load data
local_starttime = timeit.default_timer()
print('### 1. LOAD DATA ###')
load_dir = os.path.join(data_dir, mouse)
print('1 Searching & loading data in directory:\n', load_dir)
mouse_pd = gu.load_files(load_dir, '*_PyalData_struct.mat', verbose=True, 
                                                    struct_type = "PyalData")
gu.print_time_verbose(local_starttime, global_starttime)


#2. Compute rates
local_starttime = timeit.default_timer()
print('### 2. COMPUTE RATES ###')
params = {
    'std': kernel_std,
    'num_std': num_std,
    'continuous': continuous,
    'assymetry': assymetry
}

print(f'\tParams: {params}')
mouse_pd = gu.apply_to_dict(gu.add_firing_rates, mouse_pd, 'smooth', **params)

#3. Save data
print(f'\tSaving data on {save_dir_step}')

#save rates dict
file_name = os.path.join(save_dir_step, mouse+ "_rates_dict.pkl")
save_df = open(file_name, "wb")
pickle.dump(mouse_pd, save_df)
save_df.close()

#save params
file_name = os.path.join(save_dir_step, mouse+ "_rates_params.pkl")
save_params = open(file_name, "wb")
pickle.dump(params, save_params)
save_params.close()

gu.print_time_verbose(local_starttime, global_starttime)

#%% PLOT
fnames = list(mouse_pd.keys())
rates_pre = np.nanmean(np.concatenate(mouse_pd[fnames[0]]['rates_SNR3'].values, axis= 0), axis = 0)
rates_rot = np.nanmean(np.concatenate(mouse_pd[fnames[1]]['rates_SNR3'].values, axis= 0), axis = 0)

revents_pre = np.nanmean(np.concatenate(mouse_pd[fnames[0]]['revents_SNR3'].values, axis= 0), axis = 0)
revents_rot = np.nanmean(np.concatenate(mouse_pd[fnames[1]]['revents_SNR3'].values, axis= 0), axis = 0)

pre_label = ['pre']*rates_pre.shape[0]
rot_label = ['rot']*rates_rot.shape[0]

rates_struct = pd.DataFrame(data={'rates':np.hstack((rates_pre, rates_rot)), 
                                  'revents': np.hstack((revents_pre, revents_rot)),
                                  'session_type':np.hstack((pre_label, rot_label))})

# Update default settings
sns.set(palette=['#62C370', '#C360B4', '#6083C3', '#C3A060'])

lims = [np.min(rates_struct['rates']), np.percentile(rates_struct['rates'], 98)]

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

sns.kdeplot(x='rates', data=rates_struct, hue='session_type', shade=True, 
    common_norm=False, clip=[0, None], common_grid=True, ax=ax[0,0])  
ax[0,0].set_title("Spike rates")
ax[0,0].set_xlim(lims)
ax[0,0].set_xlabel("Spike rates")

ax[0,1].plot(lims, lims, 'k--')
ax[0,1].scatter(rates_pre, rates_rot,10)
ax[0,1].set_xlabel('Pre Spike rates')
ax[0,1].set_ylabel('Rot Spike rates')
ax[0,1].set_xlim(lims)
ax[0,1].set_ylim(lims)

lims = [np.min(rates_struct['revents']), np.percentile(rates_struct['revents'], 98)]

sns.kdeplot(x='revents', data=rates_struct, hue='session_type', shade=True, 
    common_norm=False, clip=[0, None], common_grid=True, ax=ax[1,0])  
ax[1,0].set_title("Event rates")
ax[1,0].set_xlim(lims)
ax[1,0].set_xlabel("Event rates")

ax[1,1].plot(lims, lims, 'k--')
ax[1,1].scatter(revents_pre, revents_rot,10)
ax[1,1].set_xlabel('Pre event rates')
ax[1,1].set_ylabel('Rot event rates')
ax[1,1].set_xlim(lims)
ax[1,1].set_ylim(lims)
plt.tight_layout()

figures_dir = os.path.join(save_dir_step, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

plt.savefig(os.path.join(figures_dir, mouse + '_rates_plot.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(figures_dir,mouse + '_rates_plot.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)

#End
sys.stdout = original
f.close()