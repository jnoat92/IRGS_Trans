'''
No@
Mar 13th, 2024
'''

import os
from PIL import Image

import glob

folder_path = '../results_test'

files_ = glob.glob(folder_path + '/**/*.png', recursive=True)

files = []
for i, f in enumerate(files_):
    if 'buffers' not in f: files.append(f)

scenes = []
for f in files:
    if 'Unet' in f: scenes.append(os.path.split(os.path.split(f)[0])[1])
scenes = sorted(list(set(scenes)))

for scene in scenes:
    output_folder = os.path.join(folder_path, 'images_for_latex', scene)
    os.makedirs(output_folder + '/soft-labels', exist_ok=True)
    os.makedirs(output_folder + '/hard-labels', exist_ok=True)

    # =========== MATCH FILES
    scene_files = []
    for f in files:
        if scene in f: 
            scene_files.append(f)
    
    for i in scene_files: 
        if 'Unet' in i:
        # =========== HH, HV, GT, AND IRGS
            if 'HH.png' in i: Image.open(i).save(output_folder + '/HH.png')
            elif 'HV.png' in i: Image.open(i).save(output_folder + '/HV.png')
            elif 'colored_gts.png' in i: Image.open(i).save(output_folder + '/GT.png')
            elif 'colored_irgs_output.png' in i: Image.open(i).save(output_folder + '/IRGS.png')
        # =========== P1
            elif 'colored_predict_cnn.png'  in i: Image.open(i).save(output_folder + '/hard-labels/P1.png')
            elif 'soft_lbl_cnn.png'         in i: Image.open(i).save(output_folder + '/soft-labels/P1.png')
        # =========== P2
        elif 'end_to_end/Loss_transformer' in i:
            if 'colored_predict_transformer.png'  in i: Image.open(i).save(output_folder + '/hard-labels/P2.png')
            elif 'soft_lbl_transformer.png'         in i: Image.open(i).save(output_folder + '/soft-labels/P2.png')
        # =========== P3
        elif 'multi_stage/Loss_transformer' in i:
            if 'colored_predict_transformer.png'  in i: Image.open(i).save(output_folder + '/hard-labels/P3.png')
            elif 'soft_lbl_transformer.png'         in i: Image.open(i).save(output_folder + '/soft-labels/P3.png')
        # =========== P4
        elif 'end_to_end/Loss_end_to_end' in i:
            if 'colored_predict_cnn.png'  in i: Image.open(i).save(output_folder + '/hard-labels/P4_cnn.png')
            elif 'soft_lbl_cnn.png'         in i: Image.open(i).save(output_folder + '/soft-labels/P4_cnn.png')
            elif 'colored_predict_transformer.png'  in i: Image.open(i).save(output_folder + '/hard-labels/P4_trans.png')
            elif 'soft_lbl_transformer.png'         in i: Image.open(i).save(output_folder + '/soft-labels/P4_trans.png')
            elif 'combined output/0.7_hard_lbl.png' in i: Image.open(i).save(output_folder + '/hard-labels/P4_comb.png')
            elif 'combined output/0.7_soft_lbl.png' in i: Image.open(i).save(output_folder + '/soft-labels/P4_comb.png')
        # =========== P5
        elif 'multi_stage/Loss_end_to_end' in i:
            if 'colored_predict_cnn.png'  in i: Image.open(i).save(output_folder + '/hard-labels/P5_cnn.png')
            elif 'soft_lbl_cnn.png'         in i: Image.open(i).save(output_folder + '/soft-labels/P5_cnn.png')
            elif 'colored_predict_transformer.png'  in i: Image.open(i).save(output_folder + '/hard-labels/P5_trans.png')
            elif 'soft_lbl_transformer.png'         in i: Image.open(i).save(output_folder + '/soft-labels/P5_trans.png')
            elif 'combined output/0.7_hard_lbl.png' in i: Image.open(i).save(output_folder + '/hard-labels/P5_comb.png')
            elif 'combined output/0.7_soft_lbl.png' in i: Image.open(i).save(output_folder + '/soft-labels/P5_comb.png')

