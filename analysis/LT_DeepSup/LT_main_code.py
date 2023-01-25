import os, subprocess
# mice = ["CZ3","CZ4", "CZ6", "CZ7", "CZ8", "CZ9", "GC2", "GC3", "GC4_nvista", "GC5_nvista", "DDC", "ChZ2", "ChZ4", "ChZ6", "DDA"]
data_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/data/'
save_dir = '/media/julio/DATOS/spatial_navigation/JP_data/LT_inscopix/results/'
#mice = ["CZ3","CZ4", "CZ6", "CZ7", "CZ8"]
mice = ["CZ9", "GC2", "GC3", "GC4_nvista", "GC5_nvista", "DDC", "ChZ2", "ChZ4", "ChZ6", "DDA"]

print(os.path.abspath(''))
os.chdir('/media/julio/DATOS/spatial_navigation/GitHub_SN/neural_manifold/analysis/LT_DeepSup/')

# #STEP 1: RATES
# for mouse in mice:
#     subprocess.call(["python", "LT_S1_rates.py", mouse, data_dir, save_dir])


#STEP 2: INTRINSIC DIMENSION
signal = 'revents_SNR3'
load_data = os.path.join(save_dir,'S1_add_rates')
step_save_dir = os.path.join(save_dir,'S2_ID')
if not os.path.exists(step_save_dir):
    os.makedirs(step_save_dir)
step_save_dir = os.path.join(step_save_dir,signal)
for mouse in mice:
    subprocess.call(["python", "LT_S2_ID.py", mouse, load_data, step_save_dir, signal])


# #STEP 3: COMPUTE EMBEDDINGS
# signal = 'revents_SNR3'
# dim = 3

# load_data = os.path.join(save_dir,'S1_add_rates')
# step_save_dir = os.path.join(save_dir,'S3_embs')
# if not os.path.exists(step_save_dir):
#     os.makedirs(step_save_dir)
# step_save_dir = os.path.join(step_save_dir,signal)
# for mouse in mice:
#     subprocess.call(["python", "LT_S3_embs.py", mouse, load_data, step_save_dir, signal, str(dim)])


#STEP 4: ROTATION


#STEP 5: DECODERS (error vs position)


#STEP 6: SI (OG & EMB)


#STEP 7: EMB RESOLUTION (position vs size on emb)


#STEP 8: PLACE CELLS