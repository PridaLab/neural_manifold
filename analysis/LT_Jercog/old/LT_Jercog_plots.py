#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:49:59 2022

@author: julio
"""
import os
from neural_manifold import general_utils as gu
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

#%%
#Data Folders
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results_pipeline/move_data'
mouse_M2019 = 'M2019'
results_dir_M2019 = os.path.join(save_dir, "M2019_170122_100510")
name_sessions_M2019 = ['Day 1 eve', 'Day 2 mor', 'Day 2 eve part 1', 'Day 2 eve part 2', 'Day 7 mor part 1' , 'Day 7 mor part 2', 'Day 7 mor part 3']

mouse_M2021 = 'M2021'
results_dir_M2021 = os.path.join(save_dir, "M2021_170122_131036")
name_sessions_M2021 = ['Day 1 eve', 'Day 2 mor', 'Day 2 eve', 'Day 7 mor']

mouse_M2022 = 'M2022'
results_dir_M2022 = os.path.join(save_dir, "M2022_170122_132357")
name_sessions_M2022 = ['Day 1 eve', 'Day 2 mor', 'Day 2 eve part 1', 'Day 2 eve part 2', 'Day 7 mor part 1' , 'Day 7 mor part 2']

mouse_M2023 = 'M2023'
results_dir_M2023 = os.path.join(save_dir, "M2023_170122_134606")
name_sessions_M2023 = ['Day 1 eve', 'Day 2 mor', 'Day 2 eve part 1', 'Day 2 eve part 2', 'Day 2 eve part 3',
                       'Day 7 mor part 1' , 'Day 7 mor part 2', 'Day 7 mor part 3' ,'Day 7 mor part 4']

mouse_M2024 = 'M2024'
results_dir_M2024 = os.path.join(save_dir, "M2024_170122_141956")
name_sessions_M2024 = ['Day 1 eve', 'Day 2 mor', 'Day 2 eve','Day 4 mor part 1' , 'Day 4 mor part 2']

mouse_M2025 = 'M2025'
results_dir_M2025 = os.path.join(save_dir, "M2025_170122_143435")
name_sessions_M2025 = ['Day 1 eve', 'Day 2 mor', 'Day 2 eve', 'Day 7 mor part 1' , 
                       'Day 4 mor part 2', 'Day 4 mor part 3' ,'Day 4 mor part 4', 'Day 4 mor part 5']

mouse_M2026 = 'M2026'
results_dir_M2026 = os.path.join(save_dir, "M2026_170122_150559")
name_sessions_M2026 = ['Day 1 eve', 'Day 2 mor', 'Day 2 eve ', 
                       'Day 4 mor part 1' , 'Day 4 mor part 2', 'Day 4 mor part 3']
#%%
#Create html header
html = '<HTML>\n'
html = html + '<style>\n'
html = html + 'h1 {text-align: center;}\n'
html = html + 'h2 {text-align: center;}\n'
html = html + 'img {display: block; width: 70%; margin-left: auto; margin-right: auto;}'
html = html + '</style>\n'
#Add title
html = html + '<h1>Linear Track Jercog Data</h1>\n<br>\n'
#Manifold Shape

#Add subtitle
html = html + '<br><h2>Manifold shape</h2><br>\n'
#Load data
M2019_dict = gu.load_files(results_dir_M2019, '*_move_data_dict.pkl', verbose=False, struct_type = "pickle")
M2021_dict = gu.load_files(results_dir_M2021, '*_move_data_dict.pkl', verbose=False, struct_type = "pickle")
M2022_dict = gu.load_files(results_dir_M2022, '*_move_data_dict.pkl', verbose=False, struct_type = "pickle")
M2023_dict = gu.load_files(results_dir_M2023, '*_move_data_dict.pkl', verbose=False, struct_type = "pickle")
M2024_dict = gu.load_files(results_dir_M2024, '*_move_data_dict.pkl', verbose=False, struct_type = "pickle")
M2025_dict = gu.load_files(results_dir_M2025, '*_move_data_dict.pkl', verbose=False, struct_type = "pickle")
M2026_dict = gu.load_files(results_dir_M2026, '*_move_data_dict.pkl', verbose=False, struct_type = "pickle")

varList = ["dir_mat", "posx", "posy", "index_mat"]
f = gu.plot_3D_embedding(M2019_dict, 'ML_umap', varList = varList, name_sessions = name_sessions_M2019)
f.suptitle('M2019 Firing Rates Umap ')
tmpfile = BytesIO()
f.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(f)

f = gu.plot_3D_embedding(M2021_dict, 'ML_umap', varList = varList, name_sessions = name_sessions_M2021)
f.suptitle('M2021 Firing Rates Umap ')
tmpfile = BytesIO()
f.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(f)

f = gu.plot_3D_embedding(M2022_dict, 'ML_umap', varList = varList, name_sessions = name_sessions_M2022)
f.suptitle('M2022 Firing Rates Umap ')
tmpfile = BytesIO()
f.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(f)


f = gu.plot_3D_embedding(M2023_dict, 'ML_umap', varList = varList, name_sessions = name_sessions_M2023)
f.suptitle('M2023 Firing Rates Umap ')
tmpfile = BytesIO()
f.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(f)

f = gu.plot_3D_embedding(M2024_dict, 'ML_umap', varList = varList, name_sessions = name_sessions_M2024)
f.suptitle('M2024 Firing Rates Umap ')
tmpfile = BytesIO()
f.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(f)

f = gu.plot_3D_embedding(M2025_dict, 'ML_umap', varList = varList, name_sessions = name_sessions_M2025)
f.suptitle('M2025 Firing Rates Umap ')
tmpfile = BytesIO()
f.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(f)

f = gu.plot_3D_embedding(M2026_dict, 'ML_umap', varList = varList, name_sessions = name_sessions_M2026)
f.suptitle('M2026 Firing Rates Umap ')
tmpfile = BytesIO()
f.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(f)

#Save html file
with open(os.path.join(save_dir, "LT_analysis.html"),'w') as f:
    f.write(html)
    
    
#%%
#Create html header
html = '<HTML>\n'
html = html + '<style>\n'
html = html + 'h1 {text-align: center;}\n'
html = html + 'h2 {text-align: center;}\n'
html = html + 'img {display: block; width: 70%; margin-left: auto; margin-right: auto;}'
html = html + '</style>\n'
#Add title
html = html + '<h1>Linear Track Jercog Data</h1>\n<br>\n'
#Manifold Shape

#Add subtitle
html = html + '<br><h2>M2019 number of cells study</h2><br>\n'
#Load data
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results_pipeline/move_data'
mouse_M2019 = 'M2019'
results_dir_M2019 = os.path.join(save_dir, "M2019_170122_100510")
M2019_nn = gu.load_files(results_dir_M2019, '*_nneurons_manifold_dict.pkl', verbose=True, struct_type = "pickle")

##Plot

varList = ["dir_mat", "posx", "posy", "index_mat"]


fig, ax = plt.subplots(1,1,figsize=(6,3))
color_code = [[2, 62, 138], [0, 119, 182], 
              [0, 150, 199], [0, 180, 216],[72, 202, 228],
              [144, 224, 239],[173, 232, 244],[202, 240, 248]]

color_code = [np.array(cc)/255 for cc in color_code]
for f_idx, fname in enumerate(np.sort(list(M2019_nn.keys()))):
    nn_cells = np.zeros((len(M2019_nn[fname]["cells_picked"])))
    for nn_it in range(nn_cells.shape[0]):
        nn_cells[nn_it] = len(M2019_nn[fname]["cells_picked"][f"{nn_it}"])
    
    m = np.nanmean(M2019_nn[fname]["num_trust"][:,0,:], axis=1)
    sd = np.nanstd(M2019_nn[fname]["num_trust"][:,0,:], axis=1)/np.sqrt(M2019_nn[fname]["num_trust"].shape[2])
    ax.plot(nn_cells, m, color = color_code[f_idx], label = fname)
    ax.fill_between(nn_cells, m-sd, m+sd, color = color_code[f_idx], alpha = 0.25)
    
ax.set_ylabel('Dimensionality',fontsize=12)
ax.set_ylim([0,5])
plt.setp(ax.spines.values(), linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)        
ax.legend(fontsize=6)
ax.set_title('Num trust dimensionality',fontsize=14)
fig.tight_layout()

tmpfile = BytesIO()
fig.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(fig)


fig, ax = plt.subplots(1,1,figsize=(6,3))
color_code = [[2, 62, 138], [0, 119, 182], 
              [0, 150, 199], [0, 180, 216],[72, 202, 228],
              [144, 224, 239],[173, 232, 244],[202, 240, 248]]
color_code = [np.array(cc)/255 for cc in color_code]
for f_idx, fname in enumerate(np.sort(list(M2019_nn.keys()))):
    nn_cells = np.zeros((len(M2019_nn[fname]["cells_picked"])))
    for nn_it in range(nn_cells.shape[0]):
        nn_cells[nn_it] = len(M2019_nn[fname]["cells_picked"][f"{nn_it}"])
    
    m = np.nanmean(M2019_nn[fname]["cI_val"][:,:], axis=1)
    sd = np.nanstd(M2019_nn[fname]["cI_val"][:,:], axis=1)/np.sqrt(M2019_nn[fname]["cI_val"].shape[1])
    ax.plot(nn_cells, m, color = color_code[f_idx], label = fname)
    ax.fill_between(nn_cells, m-sd, m+sd, color = color_code[f_idx], alpha = 0.25)
    
ax.set_ylabel('Structure Index',fontsize=12)
ax.set_ylim([0,1])
plt.setp(ax.spines.values(), linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)        
ax.legend(fontsize=6)
ax.set_title('Structure Index pos-x',fontsize=14)

tmpfile = BytesIO()
fig.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(fig)


fig, ax = plt.subplots(1,1,figsize=(6,3))
color_code = [[2, 62, 138], [0, 119, 182], 
              [0, 150, 199], [0, 180, 216],[72, 202, 228],
              [144, 224, 239],[173, 232, 244],[202, 240, 248]]
color_code = [np.array(cc)/255 for cc in color_code]
for f_idx, fname in enumerate(np.sort(list(M2019_nn.keys()))):
    nn_cells = np.zeros((len(M2019_nn[fname]["cells_picked"])))
    for nn_it in range(nn_cells.shape[0]):
        nn_cells[nn_it] = len(M2019_nn[fname]["cells_picked"][f"{nn_it}"])
    
    m = np.nanmean(M2019_nn[fname]["inner_dim"][:,:], axis=1)
    sd = np.nanstd(M2019_nn[fname]["inner_dim"][:,:], axis=1)/np.sqrt(M2019_nn[fname]["inner_dim"].shape[1])
    ax.plot(nn_cells, m, color = color_code[f_idx], label = fname)
    ax.fill_between(nn_cells, m-sd, m+sd, color = color_code[f_idx], alpha = 0.25)
    
ax.set_ylabel('Dimensionality',fontsize=12)
ax.set_ylim([0,10])
plt.setp(ax.spines.values(), linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)        
ax.legend(fontsize=6)
ax.set_title('Inner dimensionality',fontsize=14)
fig.tight_layout()

tmpfile = BytesIO()
fig.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
html = html + '<br>\n' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>\n'
plt.close(fig)

#Save html file
with open(os.path.join(save_dir, "LT_number_of_cells_analysis.html"),'w') as f:
    f.write(html)
#%% Kernel Study plot - 
#Load data
save_dir = '/media/julio/DATOS/spatial_navigation/Jercog_data/LT/results_pipeline/move_data'
mouse_M2019 = 'M2019'
results_dir_M2019 = os.path.join(save_dir, "M2019_170122_100510")
M2019_kernel = gu.load_files(results_dir_M2019, '*_decoder_kernel_dict.pkl', verbose=True, struct_type = "pickle")

## Kernel study