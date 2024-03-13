'''
No@
Mar 13th, 2024
'''

import os

import glob

folder_path = '../results_test/21-scenes-less_resolution'
files = glob.glob(folder_path + '/**/*.png', recursive=True)
files = sorted(files)

for i, f in enumerate(files)

scenes = []
for f in files:
    if 'Unet' in f and 'buffers' not in f: 
        scenes.append(os.path.split(os.path.split(f)[0])[1])
scenes = sorted(list(set(scenes)))

for scene in scenes:
    scene_files = []
    # =========== MATCH FILES
    for f in files:
        if scene in f and 'buffers' not in f: 
            scene_files.append(f)
    
    # =========== GRAB HH, HV, GT, AND IRGS
    for i in scene_files: print(i)
    break

'''
results_test/21-scenes-less_resolution/Unet/model_0/20100418/CNN_colored_m_v_per_CC.png

results_test/21-scenes-less_resolution/IRGS_trans_superpixels/end_to_end/Loss_end_to_end/model_0/20100418/cnn/colored_gts.png
'''